# faceOnOff

**faceOnOff** is a real-time webcam application that detects human faces and applies pixelation to protect privacy.
Unlike traditional face recognition projects, this system focuses on **live interaction, user control, and performance optimization** rather than identity recognition.

The application allows users to selectively hide faces, adjust pixelation strength, and record processed video output in real time.

---

## 📝 Project Overview

faceOnOff is a **real-time computer vision application** built with OpenCV that:

* Detects faces from webcam input
* Stabilizes detection across frames
* Applies pixelation-based blur
* Allows user interaction through mouse and keyboard

The system is designed to be:

* **Interactive**
* **Stable**
* **Fast (real-time capable)**
* **User-controlled**

---

## 🧠 Relation to face_recognition (ageitgey)

This project is inspired by the face detection workflow shown in the
**ageitgey face_recognition repository**.

### Key relationship:

* face_recognition demonstrates:

  * Webcam input processing
  * Frame resizing for performance
  * Face detection pipeline

* faceOnOff extends this idea by:

  * Replacing heavy recognition with **fast Haar cascade detection**
  * Adding **multi-detector fusion** (frontal + profile)
  * Introducing **temporal stabilization**
  * Supporting **interactive face selection**
  * Implementing **real-time pixelation UI system**

### Why not use face_recognition directly?

* face_recognition:

  * Accurate but **slow for real-time UI-heavy interaction**
  * Focused on **identity recognition**

* faceOnOff:

  * Optimized for **speed + UX**
  * Focused on **privacy control**

---

## ✨ Features

### 1. Multi-Model Face Detection

* Uses multiple Haar cascades:

  * Frontal (default, alt2, alt_tree)
  * Profile (left + mirrored right)
* Merges overlapping detections using IoU

---

### 2. Detection Stabilization

* Faces must appear across multiple frames before being accepted
* Reduces flickering and false positives

---

### 3. Selective Face Exemption (Core Feature)

* Click a face → mark as **EXEMPT**
* Click again → remove exemption
* Only non-exempt faces are blurred

---

### 4. Adjustable Pixelation Levels

* 6 blur levels (UI circles)
* Real-time switching

---

### 5. Hover Interaction Feedback

* Yellow box → will be exempted
* Green box → already exempt

---

### 6. Recording System

* Start/Stop recording
* Saves to:

  * `.mp4` (preferred)
  * `.avi` fallback

---

### 7. Real-Time UI System

* Drawn directly on OpenCV frame
* Includes:

  * Pixel selector
  * Record button
  * Status indicators

---

### 8. Performance Optimization

* Frame resizing for detection
* Cached detection results
* Detection every N frames

---

## 🛠 Core Architecture

### Pipeline

1. Capture webcam frame
2. Convert to grayscale
3. Detect faces (multi-cascade)
4. Merge overlapping boxes
5. Filter invalid detections
6. Stabilize across frames
7. Apply pixelation
8. Render UI
9. Handle input

---

### Key Techniques

* IoU-based tracking
* Temporal filtering (`hits / misses`)
* ROI pixelation (downscale → upscale)
* Mouse-based interaction system

---

## 🎮 Controls

| Action             | Key / Input      |
| ------------------ | ---------------- |
| Quit               | `q`              |
| Toggle Recording   | `r`              |
| Clear Exempt Faces | `c`              |
| Change Blur Level  | Click UI circles |
| Select Face        | Mouse click      |

---

## 🚀 Tech Stack

### Core

* Python
* OpenCV (cv2)

### Algorithms

* Haar Cascade Detection
* IoU (Intersection over Union)
* Temporal Stabilization

---

## 🎯 Goal

faceOnOff aims to move beyond basic face blurring by creating a system where:

* Privacy is **user-controlled**
* Detection is **stable and reliable**
* Interaction is **intuitive and real-time**

Ultimately, this project explores how computer vision can be combined with UI/UX to create **interactive privacy tools** instead of passive filters.

---

## ❗ Limitations (Critical View)

* Haar cascade is less accurate than deep learning models
* No face identity tracking (only bounding box tracking)
* Performance depends on lighting conditions
* False positives still possible

---

## 🔮 Future Improvements

* Integrate deep learning detector (e.g., RetinaFace, YOLO-face)
* Add face recognition for persistent identity
* Improve tracking with Kalman Filter / SORT
* GPU acceleration
* Web-based version (WebRTC + WASM)

---

## 🧩 Suggested Project Extensions (for you)

* OBS plugin integration (streaming use-case)
* VR/AR privacy filter (your MR 관심이랑 연결 가능)
* Multi-user tagging system
* Real-time anonymization for datasets

