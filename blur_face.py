import os
from datetime import datetime

import cv2

WINDOW_NAME = "Video"
MIN_FACE_SIZE = 28
DETECTION_SCALE = 0.5
PROCESS_EVERY_N_FRAMES = 2

# UI now shows constant levels 1..6
PIXEL_LEVEL_LABELS = [1, 2, 3, 4, 5, 6]
PIXEL_BLOCK_SIZES = [4, 8, 12, 18, 26, 36]
DEFAULT_PIXEL_INDEX = 2  # level 3

ui_state = {
    "pixel_idx": DEFAULT_PIXEL_INDEX,
    "recording": False,
    "writer": None,
    "record_path": None,
    "circle_centers": [],
    "record_rect": (0, 0, 0, 0),
    "latest_frame_size": (640, 480),
    "fps": 20.0,
    "exempt_enabled": True,
    "exempt_targets": [],  # [{"box": (x, y, w, h), "lost": int}, ...]
    "clickable_faces": [],
    "frame_count": 0,
    "cached_faces": [],
    "circle_hit_radius": 18,
    "mouse_pos": None,
}

# Haar cascade detectors (frontal + profile) for speed + side-face support.
frontal_default_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
frontal_alt2_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
)
profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)

if (
    frontal_default_cascade.empty()
    or frontal_alt2_cascade.empty()
    or profile_cascade.empty()
):
    raise RuntimeError("Could not load Haar cascade XML files for face detection.")


def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = aw * ah
    area_b = bw * bh
    return inter_area / float(area_a + area_b - inter_area)


def merge_overlaps(faces, iou_threshold=0.35):
    merged = []
    for face in faces:
        x, y, w, h = [int(v) for v in face]
        was_merged = False
        for idx, (mx, my, mw, mh) in enumerate(merged):
            if iou_xywh((x, y, w, h), (mx, my, mw, mh)) > iou_threshold:
                nx = min(x, mx)
                ny = min(y, my)
                nx2 = max(x + w, mx + mw)
                ny2 = max(y + h, my + mh)
                merged[idx] = (nx, ny, nx2 - nx, ny2 - ny)
                was_merged = True
                break
        if not was_merged:
            merged.append((x, y, w, h))
    return merged


def detect_faces(gray):
    faces = []

    small_gray = cv2.resize(gray, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)

    frontal_default = frontal_default_cascade.detectMultiScale(
        small_gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(max(10, int(MIN_FACE_SIZE * DETECTION_SCALE)), max(10, int(MIN_FACE_SIZE * DETECTION_SCALE))),
    )
    faces.extend(frontal_default)

    frontal_alt2 = frontal_alt2_cascade.detectMultiScale(
        small_gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(max(10, int(MIN_FACE_SIZE * DETECTION_SCALE)), max(10, int(MIN_FACE_SIZE * DETECTION_SCALE))),
    )
    faces.extend(frontal_alt2)

    profile = profile_cascade.detectMultiScale(
        small_gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(max(10, int(MIN_FACE_SIZE * DETECTION_SCALE)), max(10, int(MIN_FACE_SIZE * DETECTION_SCALE))),
    )
    faces.extend(profile)

    # Detect opposite profile by mirroring.
    small_gray_flipped = cv2.flip(small_gray, 1)
    profile_flipped = profile_cascade.detectMultiScale(
        small_gray_flipped,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(max(10, int(MIN_FACE_SIZE * DETECTION_SCALE)), max(10, int(MIN_FACE_SIZE * DETECTION_SCALE))),
    )
    small_w = small_gray.shape[1]
    for x, y, w, h in profile_flipped:
        faces.append((small_w - (x + w), y, w, h))

    # Back to full resolution.
    inv = 1.0 / DETECTION_SCALE
    full_faces = []
    for x, y, w, h in merge_overlaps(faces):
        full_faces.append((int(x * inv), int(y * inv), int(w * inv), int(h * inv)))

    return merge_overlaps(full_faces)


def clamp_face_box(x, y, w, h, frame_shape):
    h_img, w_img = frame_shape[:2]
    left = max(0, x)
    top = max(0, y)
    right = min(w_img, x + w)
    bottom = min(h_img, y + h)
    if right <= left or bottom <= top:
        return None
    return top, right, bottom, left


def pixelate_region(frame, top, right, bottom, left, block):
    top = max(0, top)
    left = max(0, left)
    bottom = min(frame.shape[0], bottom)
    right = min(frame.shape[1], right)
    if right <= left or bottom <= top:
        return

    roi = frame[top:bottom, left:right]
    if roi.size == 0:
        return

    block = max(1, int(block))
    small_w = max(1, roi.shape[1] // block)
    small_h = max(1, roi.shape[0] // block)

    reduced = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(
        reduced,
        (roi.shape[1], roi.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    frame[top:bottom, left:right] = pixelated


def create_video_writer(width, height, fps):
    os.makedirs("recordings", exist_ok=True)
    filename = datetime.now().strftime("blur_capture_%Y%m%d_%H%M%S.mp4")
    output_path = os.path.join("recordings", filename)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if writer.isOpened():
        return writer, output_path

    filename_avi = datetime.now().strftime("blur_capture_%Y%m%d_%H%M%S.avi")
    output_path_avi = os.path.join("recordings", filename_avi)
    fourcc_avi = cv2.VideoWriter_fourcc(*"XVID")
    writer_avi = cv2.VideoWriter(output_path_avi, fourcc_avi, fps, (width, height))
    if writer_avi.isOpened():
        return writer_avi, output_path_avi

    return None, None


def toggle_recording():
    width, height = ui_state["latest_frame_size"]
    fps = max(10.0, float(ui_state["fps"]))

    if ui_state["recording"]:
        if ui_state["writer"] is not None:
            ui_state["writer"].release()
        ui_state["writer"] = None
        ui_state["recording"] = False
        ui_state["record_path"] = None
        return

    writer, output_path = create_video_writer(width, height, fps)
    if writer is not None:
        ui_state["writer"] = writer
        ui_state["recording"] = True
        ui_state["record_path"] = output_path


def toggle_exempt_target(face):
    for idx, target in enumerate(ui_state["exempt_targets"]):
        if iou_xywh(target["box"], face) > 0.4:
            # Click same face again -> remove from exempt list.
            del ui_state["exempt_targets"][idx]
            return
    ui_state["exempt_targets"].append({"box": face, "lost": 0})


def update_exempt_targets(current_faces):
    if not ui_state["exempt_targets"]:
        return

    next_targets = []
    for target in ui_state["exempt_targets"]:
        best_face = None
        best_iou = 0.0
        for face in current_faces:
            score = iou_xywh(target["box"], face)
            if score > best_iou:
                best_iou = score
                best_face = face

        if best_face is not None and best_iou > 0.1:
            next_targets.append({"box": best_face, "lost": 0})
        else:
            lost = target["lost"] + 1
            if lost <= 20:
                next_targets.append({"box": target["box"], "lost": lost})

    ui_state["exempt_targets"] = next_targets


def draw_ui(frame):
    h, w = frame.shape[:2]
    # Responsive UI for bigger frames/screens.
    ui_scale = max(1.0, min(4.0, min(w / 640.0, h / 480.0)))

    bar_left = int(50 * ui_scale)
    bar_right = max(bar_left + int(250 * ui_scale), w - int(210 * ui_scale))
    bar_y = int(45 * ui_scale)
    line_thickness = max(2, int(2 * ui_scale))
    title_scale = 0.6 * ui_scale
    label_scale = 0.55 * ui_scale
    circle_radius = max(14, int(14 * ui_scale))
    ui_state["circle_hit_radius"] = circle_radius + max(4, int(4 * ui_scale))

    level_count = len(PIXEL_LEVEL_LABELS)
    span = max(1, bar_right - bar_left)
    step = span / float(level_count - 1)

    cv2.line(frame, (bar_left, bar_y), (bar_right, bar_y), (255, 255, 255), line_thickness)
    cv2.putText(
        frame,
        "Pixel Level",
        (bar_left, bar_y - int(15 * ui_scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        title_scale,
        (255, 255, 255),
        max(1, int(2 * ui_scale)),
        cv2.LINE_AA,
    )

    centers = []
    for idx, level in enumerate(PIXEL_LEVEL_LABELS):
        cx = int(round(bar_left + idx * step))
        cy = bar_y
        centers.append((cx, cy))

        selected = idx == ui_state["pixel_idx"]
        color = (0, 255, 0) if selected else (230, 230, 230)
        thickness = -1 if selected else max(2, int(2 * ui_scale))
        cv2.circle(frame, (cx, cy), circle_radius, color, thickness)

        cv2.putText(
            frame,
            f"{level}",
            (cx - int(8 * ui_scale), bar_y + int(34 * ui_scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            label_scale,
            (255, 255, 255),
            max(1, int(2 * ui_scale)),
            cv2.LINE_AA,
        )

    ui_state["circle_centers"] = centers

    # Record button
    btn_w, btn_h = int(140 * ui_scale), int(44 * ui_scale)
    btn_x1 = w - btn_w - int(20 * ui_scale)
    btn_y1 = int(22 * ui_scale)
    btn_x2 = btn_x1 + btn_w
    btn_y2 = btn_y1 + btn_h
    ui_state["record_rect"] = (btn_x1, btn_y1, btn_x2, btn_y2)

    if ui_state["recording"]:
        btn_color = (20, 20, 220)
        label = "STOP"
    else:
        btn_color = (40, 180, 40)
        label = "REC"

    cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), btn_color, -1)
    cv2.rectangle(
        frame,
        (btn_x1, btn_y1),
        (btn_x2, btn_y2),
        (255, 255, 255),
        max(2, int(2 * ui_scale)),
    )
    cv2.putText(
        frame,
        label,
        (btn_x1 + int(42 * ui_scale), btn_y1 + int(30 * ui_scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8 * ui_scale,
        (255, 255, 255),
        max(1, int(2 * ui_scale)),
        cv2.LINE_AA,
    )

    exempt_state = "ON" if ui_state["exempt_enabled"] else "OFF"
    status_text = f"Exempt: {exempt_state} ({len(ui_state['exempt_targets'])})"
    status_color = (0, 220, 0) if ui_state["exempt_enabled"] else (0, 180, 255)
    cv2.putText(
        frame,
        status_text,
        (int(20 * ui_scale), h - int(20 * ui_scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65 * ui_scale,
        status_color,
        max(1, int(2 * ui_scale)),
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "Keys: q=quit, r=record, e=exempt on/off, c=clear all exempt",
        (int(20 * ui_scale), h - int(50 * ui_scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52 * ui_scale,
        (255, 255, 255),
        max(1, int(1 * ui_scale)),
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Click face: add exempt, click again: remove exempt",
        (int(20 * ui_scale), h - int(72 * ui_scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52 * ui_scale,
        (255, 255, 255),
        max(1, int(1 * ui_scale)),
        cv2.LINE_AA,
    )


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        ui_state["mouse_pos"] = (x, y)

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    for idx, (cx, cy) in enumerate(ui_state["circle_centers"]):
        if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= ui_state["circle_hit_radius"] ** 2:
            ui_state["pixel_idx"] = idx
            return

    x1, y1, x2, y2 = ui_state["record_rect"]
    if x1 <= x <= x2 and y1 <= y <= y2:
        toggle_recording()
        return

    # Select exempt person by clicking inside one detected face box.
    for fx, fy, fw, fh in ui_state["clickable_faces"]:
        if fx <= x <= fx + fw and fy <= y <= fy + fh:
            toggle_exempt_target((fx, fy, fw, fh))
            ui_state["exempt_enabled"] = True
            return


# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
# Prefer larger capture size so UI can scale on bigger displays.
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

fps = video_capture.get(cv2.CAP_PROP_FPS)
if fps and fps > 1.0:
    ui_state["fps"] = fps

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW_NAME, on_mouse)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    ui_state["latest_frame_size"] = (frame.shape[1], frame.shape[0])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Performance optimization: run heavy detector every N frames and reuse last result.
    if ui_state["frame_count"] % PROCESS_EVERY_N_FRAMES == 0:
        ui_state["cached_faces"] = detect_faces(gray)

    faces = list(ui_state["cached_faces"])
    ui_state["clickable_faces"] = faces
    update_exempt_targets(faces)

    pixel_block = PIXEL_BLOCK_SIZES[ui_state["pixel_idx"]]

    for x, y, w, h in faces:
        trbl = clamp_face_box(x, y, w, h, frame.shape)
        if trbl is None:
            continue

        top, right, bottom, left = trbl
        is_exempt = (
            ui_state["exempt_enabled"]
            and any(iou_xywh(target["box"], (x, y, w, h)) > 0.2 for target in ui_state["exempt_targets"])
        )

        if not is_exempt:
            pixelate_region(frame, top, right, bottom, left, pixel_block)
        else:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "EXEMPT",
                (left, max(20, top - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    # Hover preview: show a box on the face under mouse cursor before clicking.
    hovered_face = None
    if ui_state["mouse_pos"] is not None:
        mx, my = ui_state["mouse_pos"]
        for fx, fy, fw, fh in ui_state["clickable_faces"]:
            if fx <= mx <= fx + fw and fy <= my <= fy + fh:
                hovered_face = (fx, fy, fw, fh)
                break

    if hovered_face is not None:
        fx, fy, fw, fh = hovered_face
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 255), 2)
        cv2.putText(
            frame,
            "CLICK TO TOGGLE EXEMPT",
            (fx, max(24, fy - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    draw_ui(frame)

    if ui_state["recording"] and ui_state["writer"] is not None:
        ui_state["writer"].write(frame)

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("r"):
        toggle_recording()
    if key == ord("e"):
        ui_state["exempt_enabled"] = not ui_state["exempt_enabled"]
    if key == ord("c"):
        ui_state["exempt_targets"] = []

    ui_state["frame_count"] += 1

# Release resources
if ui_state["writer"] is not None:
    ui_state["writer"].release()
video_capture.release()
cv2.destroyAllWindows()
