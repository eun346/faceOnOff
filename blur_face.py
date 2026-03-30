import os
from datetime import datetime
import cv2

# This script opens the default webcam, detects faces in real time, and applies
# a pixelation effect to protect privacy. The user can interactively control
# blur strength, choose faces that should stay unblurred, and record the
# processed output video.
#
# High-level flow:
# 1) Capture a frame from the webcam.
# 2) Detect possible faces with multiple Haar cascade models.
# 3) Filter and stabilize detections to reduce noise and flicker.
# 4) Pixelate every detected face except the user-selected exempt ones.
# 5) Draw the on-screen UI (pixel level slider, record button, help text).
# 6) React to mouse and keyboard input.

# OpenCV window title shown at the top of the preview window.
WINDOW_NAME = "Blur Face Video"
# Camera capture size. Lower input resolution reduces the amount of work done
# by both the webcam pipeline and the face detector.
CAPTURE_WIDTH = 960
CAPTURE_HEIGHT = 540
# Minimum candidate face size. Very tiny detections are usually false positives,
# so this threshold removes boxes that are too small to be trustworthy.
MIN_FACE_SIZE = 20
# Detection is run on a resized grayscale frame for speed. A smaller value means
# faster detection, but also less detail for the classifiers to work with.
DETECTION_SCALE = 0.5
# Run the heavy detector every N frames. Cached results are reused in between to
# keep the live preview responsive.
PROCESS_EVERY_N_FRAMES = 3
# Allowed width/height ratio range for a "reasonable" face box.
MIN_FACE_ASPECT = 0.55
MAX_FACE_ASPECT = 1.85
# Minimum box area as a fraction of the whole frame. This filters out detections
# that are so small that they are unlikely to be real faces.
MIN_FACE_AREA_RATIO = 0.001
# A face candidate must be detected in this many consecutive updates before it
# is treated as stable and shown/blurred in the main loop.
CONFIRM_HITS = 2
# Number of consecutive misses allowed before a tracked candidate is discarded.
MAX_CANDIDATE_MISS = 2

# The UI exposes friendly blur levels 1 through 6. Each visible level maps to a
# concrete pixel block size used by the pixelation algorithm below.
PIXEL_LEVEL_LABELS = [1, 2, 3, 4, 5, 6]
PIXEL_BLOCK_SIZES = [4, 8, 12, 18, 26, 36]
DEFAULT_PIXEL_INDEX = 2  # level 3

# Shared runtime state used by the video loop, mouse callback, and recording
# logic. Keeping everything in one dictionary is simple and works well for this
# single-file interactive application.
ui_state = {
    # Currently selected blur strength index from PIXEL_LEVEL_LABELS.
    "pixel_idx": DEFAULT_PIXEL_INDEX,
    # Whether output recording is currently active.
    "recording": False,
    # Active cv2.VideoWriter object while recording, otherwise None.
    "writer": None,
    # Path of the file currently being recorded.
    "record_path": None,
    # Screen coordinates of the blur-level circles for mouse hit detection.
    "circle_centers": [],
    # Bounding rectangle of the record button: (x1, y1, x2, y2).
    "record_rect": (0, 0, 0, 0),
    # Most recent frame size as (width, height). Used when opening the writer.
    "latest_frame_size": (640, 480),
    # Camera FPS estimate used for saved video output.
    "fps": 20.0,
    # Global switch to enable or disable exempt face handling.
    "exempt_enabled": True,
    # Faces that should not be blurred.
    # Each entry stores the last known face box and a "lost" counter so we can
    # keep following a person even if detection misses them briefly.
    "exempt_targets": [],  # [{"box": (x, y, w, h), "lost": int}, ...]
    # Stable face boxes for the current frame that can respond to mouse clicks.
    "clickable_faces": [],
    # Global frame counter used for frame-skipping logic.
    "frame_count": 0,
    # Latest raw detections reused between detection passes.
    "cached_faces": [],
    # Latest stabilized faces reused between detection passes.
    "stable_faces": [],
    # Temporary tracked detections used to stabilize boxes across frames before
    # they are accepted as reliable faces.
    "face_candidates": [],  # [{"box": (x, y, w, h), "hits": int, "misses": int}]
    # Mouse click radius for choosing a blur-level circle.
    "circle_hit_radius": 18,
    # Latest cursor position for hover previews.
    "mouse_pos": None,
}

# Haar Cascade XML files are included with OpenCV and provide pre-trained models for face detection. 

# Multiple Haar cascades are combined so the app can detect front-facing faces,
# slightly different frontal variations, and profile views. Using several
# detectors increases coverage because each model has different strengths.
frontal_default_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
frontal_alt2_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
)
frontal_alt_tree_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt_tree.xml"
)
profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)
FACE_CASCADES = (
    (frontal_default_cascade, 1.08, 4),
    (frontal_alt2_cascade, 1.08, 3),
    (profile_cascade, 1.1, 3),
)

if (
    frontal_default_cascade.empty()
    or frontal_alt2_cascade.empty()
    or frontal_alt_tree_cascade.empty()
    or profile_cascade.empty()
):
    # Fail fast if OpenCV cannot load the XML classifier files.
    raise RuntimeError("Could not load Haar cascade XML files for face detection.")


def iou_xywh(a, b):
    """
    Return the Intersection-over-Union (IoU) score for two boxes.

    The boxes use the format (x, y, w, h). IoU measures how much two
    rectangles overlap:
    - 0.0 means no overlap at all
    - 1.0 means the boxes match perfectly

    This score is used in several places:
    - merging duplicate detections from different cascade models
    - deciding whether a tracked face candidate matches a new detection
    - deciding whether a clicked face matches an existing exempt target
    """
    # (x, y, width, height)
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    # A Box:
    # Left top = (ax, ay)
    # Right bottom = (ax2, ay2)
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    # Find the coordinates of the intersection rectangle.
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    # Compute the area of intersection and the area of both boxes to calculate IoU.
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    # If there is no intersection, return 0.0 immediately to avoid division by zero.
    if inter_area == 0:
        return 0.0

    # If not zero, compute the area of both boxes and return the IoU score.
    area_a = aw * ah
    area_b = bw * bh
    return inter_area / float(area_a + area_b - inter_area)


def merge_overlaps(faces, iou_threshold=0.35):
    """
    Merge boxes that overlap enough to likely represent the same face.

    Because multiple detectors are run, one real face may produce several
    nearby rectangles. This function combines them into a single larger box so
    later logic only works with one detection per face.
    """
    merged = []
    for face in faces:
        x, y, w, h = [int(v) for v in face]
        was_merged = False
        # Check if this face overlaps enough with any existing merged box. 
        # If so, merge them by expanding the existing box to cover the union of both.
        for idx, (mx, my, mw, mh) in enumerate(merged):
            if iou_xywh((x, y, w, h), (mx, my, mw, mh)) > iou_threshold: # If IoU > 0.35, consider as the same face
                # Expand the merged box so it covers the union of both
                # overlapping detections.
                nx = min(x, mx)
                ny = min(y, my)
                nx2 = max(x + w, mx + mw)
                ny2 = max(y + h, my + mh)
                merged[idx] = (nx, ny, nx2 - nx, ny2 - ny)
                was_merged = True
                break
        
        # If this face was not merged with any existing box, add it as a new face in the list.
        if not was_merged:
            merged.append((x, y, w, h))
    return merged


def detect_with_cascade(cascade, gray_frame, scale_factor, min_neighbors, min_face_size):
    """
    Run one Haar cascade with shared detection settings.
    """
    return cascade.detectMultiScale(
        gray_frame,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_face_size, min_face_size),
    )


def detect_faces(gray):
    """
    Detect faces in a grayscale frame and return full-resolution boxes.

    The workflow is:
    1) Downscale the frame to reduce detector cost.
    2) Run several frontal/profile Haar cascades.
    3) Mirror the frame to detect the opposite profile direction.
    4) Merge overlapping results.
    5) Scale coordinates back to the original frame size.
    """
    faces = []

    # Detection on a smaller image is much faster and good enough for live use.
    small_gray = cv2.resize(gray, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
    min_face_size = max(10, int(MIN_FACE_SIZE * DETECTION_SCALE))

    # Detect with each frontal cascade and collect all the results together. 
    for cascade, scale_factor, min_neighbors in FACE_CASCADES:
        faces.extend(
            detect_with_cascade(
                cascade,
                small_gray,
                scale_factor,
                min_neighbors,
                min_face_size,
            )
        )

    # The profile cascade is direction-sensitive. Running it on a horizontally
    # flipped frame lets us detect faces looking the opposite way as well.
    small_gray_flipped = cv2.flip(small_gray, 1)
    profile_flipped = detect_with_cascade(
        profile_cascade,
        small_gray_flipped,
        1.1,
        4,
        min_face_size,
    )
    small_w = small_gray.shape[1]
    for x, y, w, h in profile_flipped:
        faces.append((small_w - (x + w), y, w, h))

    # Convert boxes from downscaled coordinates back into the full frame.
    inv = 1.0 / DETECTION_SCALE
    full_faces = []
    for x, y, w, h in merge_overlaps(faces):
        full_faces.append((int(x * inv), int(y * inv), int(w * inv), int(h * inv)))

    # It merges overlapping bounding boxes that likely represent the same face into a single box.
    return merge_overlaps(full_faces)


def is_reasonable_face_box(face, frame_shape):
    """
    Filter out detections that do not look like realistic faces.

    Haar cascades are fast but can generate false positives. This function
    removes obviously bad boxes using simple geometric checks:
    - width and height must be positive
    - the aspect ratio must be within a face-like range
    - the area must not be too tiny relative to the whole frame
    """
    x, y, w, h = face
    if w <= 0 or h <= 0:
        return False
    aspect = w / float(h)
    if aspect < MIN_FACE_ASPECT or aspect > MAX_FACE_ASPECT:
        return False
    frame_area = frame_shape[0] * frame_shape[1]
    if w * h < frame_area * MIN_FACE_AREA_RATIO:
        return False
    return True


def stabilize_faces(detected_faces):
    """
    Stabilize face detections across frames before accepting them.

    Real-time detectors often flicker, especially with slight movement or
    lighting changes. Instead of trusting one-frame detections immediately, we
    maintain a list of temporary candidates:
    - when a new detection repeatedly overlaps a candidate, its hit count grows
    - once the hit count reaches CONFIRM_HITS, the box becomes "stable"
    - if a candidate stops matching anything for too long, it is dropped

    This greatly reduces one-frame false positives and makes the blur overlay
    feel steadier.
    """
    candidates = ui_state["face_candidates"]
    unmatched = set(range(len(detected_faces)))

    for cand in candidates:
        best_idx = -1
        best_iou = 0.0
        for idx in unmatched:
            score = iou_xywh(cand["box"], detected_faces[idx])
            if score > best_iou:
                best_iou = score
                best_idx = idx

        if best_idx >= 0 and best_iou >= 0.25:
            # Refresh an existing candidate with a matching new detection.
            cand["box"] = detected_faces[best_idx]
            cand["hits"] += 1
            cand["misses"] = 0
            unmatched.remove(best_idx)
        else:
            # No good match this frame, so increment the miss counter.
            cand["misses"] += 1

    # Any remaining unmatched detections are treated as new candidates that
    # still need confirmation in future frames.
    for idx in sorted(unmatched):
        candidates.append({"box": detected_faces[idx], "hits": 1, "misses": 0})

    # If a candidate has too many misses, it is probably not a real face or has disappeared, 
    # so we stop tracking it.
    ui_state["face_candidates"] = [c for c in candidates if c["misses"] <= MAX_CANDIDATE_MISS]
    
    # return only the boxes that have reached the confirmation threshold, 
    # so the main loop can use them for blurring and interaction.
    return [c["box"] for c in ui_state["face_candidates"] if c["hits"] >= CONFIRM_HITS]


def is_exempt_face(face):
    """
    Return whether a face overlaps with any tracked exempt target.
    """
    return any(iou_xywh(target["box"], face) > 0.2 for target in ui_state["exempt_targets"])


def clamp_face_box(x, y, w, h, frame_shape):
    """
    Clamp a face box to the frame bounds and convert it to TRBL order.

    The returned value is (top, right, bottom, left), which makes slicing the
    frame easy and safe even when a detection partially extends outside the
    image.
    """
    h_img, w_img = frame_shape[:2] # get frame height and width 
    left = max(0, x)
    top = max(0, y)
    right = min(w_img, x + w)
    bottom = min(h_img, y + h)
    if right <= left or bottom <= top: # if the box is completely outside the frame, return None to indicate it should be ignored
        return None
    return top, right, bottom, left


def pixelate_region(frame, top, right, bottom, left, block):
    """
    Pixelate a rectangular region of the frame in place.

    The effect is created by:
    1) extracting the region of interest (ROI)
    2) shrinking it to a much smaller image
    3) scaling it back up using nearest-neighbor interpolation

    Larger block values make the intermediate image smaller and therefore
    produce a stronger blocky censorship effect.
    """
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

    # Loss of detail is what creates the blur effect, 
    # so the intermediate size is based on the block size.
    small_w = max(1, roi.shape[1] // block)
    small_h = max(1, roi.shape[0] // block)

    # resize back up to keep the same size while destroying detail
    reduced = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(
        reduced,
        (roi.shape[1], roi.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    frame[top:bottom, left:right] = pixelated


def create_video_writer(width, height, fps):
    """
    Create a VideoWriter for saving the processed webcam feed.

    Files are saved to the current Windows user's Downloads folder by default.
    MP4 is tried first because it is convenient for playback and sharing. If
    the current environment does not support that codec, the code falls back to
    AVI/XVID. The function returns both the writer object and the output path.
    """
    downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
    os.makedirs(downloads_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for extension, codec in (("mp4", "mp4v"), ("avi", "XVID")):
        filename = f"blur_capture_{timestamp}.{extension}"
        output_path = os.path.join(downloads_dir, filename)
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer, output_path

    return None, None


def toggle_recording():
    """
    Start or stop recording the processed video stream.

    When starting, the writer is opened using the latest known frame size and a
    safe minimum FPS. When stopping, the writer is released so the video file
    is finalized correctly.
    """
    width, height = ui_state["latest_frame_size"]
    fps = max(10.0, float(ui_state["fps"]))

    if ui_state["recording"]:
        # Finalize the current output file before turning recording off.
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
    """
    Toggle a face inside the exempt list.

    Exempt faces are intentionally left unblurred. If the user clicks roughly
    the same face again, it is removed from the exempt list.
    """
    for idx, target in enumerate(ui_state["exempt_targets"]):
        if iou_xywh(target["box"], face) > 0.4:
            # Click same face again -> remove from exempt list.
            del ui_state["exempt_targets"][idx]
            return
    ui_state["exempt_targets"].append({"box": face, "lost": 0})


def update_exempt_targets(current_faces):
    """
    Keep exempt face tracking aligned with the latest stable detections.

    Each exempt target is matched to the current face with the best overlap.
    If a temporary miss occurs, the target is kept for a short grace period so
    the exemption does not disappear immediately due to detector flicker.
    """
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
            # Strong enough match: refresh the target with the new box.
            next_targets.append({"box": best_face, "lost": 0})
        else:
            # Weak or missing match: keep the old box for a few frames.
            lost = target["lost"] + 1
            if lost <= 20:
                next_targets.append({"box": target["box"], "lost": lost})

    ui_state["exempt_targets"] = next_targets


def draw_ui(frame):
    """
    Draw the full on-screen overlay.

    The overlay contains:
    - a blur-strength selector
    - a recording button
    - exempt mode status
    - keyboard/mouse usage hints

    The layout scales with frame size so the UI remains readable across
    different camera resolutions and window sizes.
    """
    h, w = frame.shape[:2]
    # Scale UI elements based on the current frame dimensions.
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

    # Draw the slider baseline and its title.
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

        # Highlight the selected blur level so the current strength is obvious.
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

    # Record button area is stored so the mouse callback can detect clicks.
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

    # Bottom-left status text shows whether exempt mode is active and how many
    # exempt faces are currently being tracked.
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
        "Keys: q=quit, r=record, c=clear all exempt",
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
    """
    Handle mouse movement and clicks on the preview window.

    Supported interactions:
    - moving the cursor stores the current mouse position for hover previews
    - clicking a level circle changes blur strength
    - clicking the REC button starts or stops recording
    - clicking a detected face toggles whether that face is exempt from blur
    """
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


# Open webcam device 0, which is usually the default built-in camera.
video_capture = cv2.VideoCapture(0)
# Use a moderate capture resolution so the preview still looks good without
# making real-time detection unnecessarily expensive.
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
# Keep the internal buffer small to reduce visible latency.
video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Reuse the camera FPS when available so the saved recording plays back with a
# timing that matches the live preview more closely.
fps = video_capture.get(cv2.CAP_PROP_FPS) # Get FPS from the camera
if fps and fps > 1.0: # If FPS=1, it usually means the camera does not provide a valid FPS value
    ui_state["fps"] = fps # Set the playback speed of the video as the camera FPS for smoother recording

# Create a resizable OpenCV window and connect the mouse event handler.
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW_NAME, on_mouse)

while True:
    # Main processing loop runs until the user quits (line 760) or the camera stops
    # providing frames.
    ret, frame = video_capture.read()
    if not ret:
        break

    # Store the latest frame size (wdth, height))
    ui_state["latest_frame_size"] = (frame.shape[1], frame.shape[0])

    # Run expensive detection only at the configured interval. In between, reuse
    # the most recent result to reduce CPU usage while keeping interaction fast.
    if ui_state["frame_count"] % PROCESS_EVERY_N_FRAMES == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale only when face detection runs.
        gray = cv2.equalizeHist(gray) # Improve contrast to make detection more robust in poor lighting.
        raw_faces = detect_faces(gray)
        ui_state["cached_faces"] = [f for f in raw_faces if is_reasonable_face_box(f, frame.shape)]
        ui_state["stable_faces"] = stabilize_faces(list(ui_state["cached_faces"]))

    # Only stabilized boxes are used for blur and UI interactions so noisy
    # one-frame detections do not immediately affect the output.
    faces = list(ui_state["stable_faces"])
    ui_state["clickable_faces"] = faces # update the list of faces that can be clicked for exemption toggling
    update_exempt_targets(faces) # keep the exempt target tracking aligned with the latest stable detections

    # Translate the selected UI level to the actual block size passed into the
    # pixelation function.

    # Change the pixelation block size based on the selected pixel level index
    pixel_block = PIXEL_BLOCK_SIZES[ui_state["pixel_idx"]]
    
    # Assume that no face is hovered until we check the mouse position 
    # against the detected face boxes. This allows us to provide 
    # a visual preview of what would happen if the user clicks on a face 
    # before they actually click.
    hovered_face = None 
    hovered_is_exempt = False

    # Check if the mouse is currently hovering over any of the detected face boxes. 
    # If so, store that face as the hovered_face and whether it is currently exempt. 
    if ui_state["mouse_pos"] is not None:
        mx, my = ui_state["mouse_pos"]
        for fx, fy, fw, fh in ui_state["clickable_faces"]:
            if fx <= mx <= fx + fw and fy <= my <= fy + fh:
                hovered_face = (fx, fy, fw, fh)
                hovered_is_exempt = is_exempt_face(hovered_face)
                break

    for x, y, w, h in faces:
        trbl = clamp_face_box(x, y, w, h, frame.shape)
        if trbl is None:
            continue

        top, right, bottom, left = trbl
        is_exempt = ui_state["exempt_enabled"] and is_exempt_face((x, y, w, h))

        if not is_exempt:
            # Only non-exempt faces are censored.
            pixelate_region(frame, top, right, bottom, left, pixel_block)

    if hovered_face is not None:
        # Hover preview explains what a click would do before the user commits:
        # Green means the face is already exempt and clicking will remove it.
        # Yellow means the face is not exempt yet and clicking will add it.
        fx, fy, fw, fh = hovered_face
        hover_text = "EXEMPT (CLICK TO REMOVE)" if hovered_is_exempt else "CLICK TO ADD EXEMPT"
        hover_color = (0, 255, 0) if hovered_is_exempt else (0, 255, 255)
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), hover_color, 2)
        cv2.putText(
            frame,
            hover_text,
            (fx, max(24, fy - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            hover_color,
            2,
            cv2.LINE_AA,
        )

    draw_ui(frame)

    # Record the processed frame after blur/UI overlays are applied so the saved
    # video matches what the user sees on screen.
    if ui_state["recording"] and ui_state["writer"] is not None:
        ui_state["writer"].write(frame)

    cv2.imshow(WINDOW_NAME, frame)

    # Keyboard shortcuts provide quick control without relying only on the mouse.
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("r"):
        toggle_recording()
    if key == ord("c"):
        ui_state["exempt_targets"] = []

    ui_state["frame_count"] += 1

# Always release the writer, webcam, and UI window before exit so files are
# finalized correctly and the camera is not left locked by the process.
if ui_state["writer"] is not None:
    ui_state["writer"].release()
video_capture.release()
cv2.destroyAllWindows()
