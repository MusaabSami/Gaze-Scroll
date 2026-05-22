# Tech Stack — GazeClick

> Face-controlled cursor movement and eyebrow-triggered clicking, built for learning.

---

## Language & Runtime

| Layer | Choice | Reason |
|---|---|---|
| Core language | Python 3.11+ | Best ecosystem for CV/ML; you're already fluent |
| Package manager | `pip` + `venv` | Simple, no overhead for a learning project |

---

## Computer Vision & Face Tracking

| Library | Version | Role |
|---|---|---|
| `mediapipe` | 0.10.x | Face Mesh — 478 3D landmarks, head pose, eyebrow geometry |
| `opencv-python` | 4.x | Camera capture, frame display, debug overlays |
| `numpy` | latest | Landmark math, coordinate transforms |

**Why MediaPipe over dlib/OpenCV haarcascade?**
MediaPipe Face Mesh runs on CPU comfortably, gives 3D landmark positions, and has built-in head pose geometry — no manual PnP solver needed for a first pass.

---

## Screen & Input Control

| Library | Role |
|---|---|
| `pyautogui` | Move cursor, fire mouse clicks |
| `screeninfo` | Fetch monitor resolution at runtime |

`pyautogui` is simple and cross-platform. For lower latency later, `pynput` is a drop-in upgrade.

---

## Smoothing & Signal Processing

| Library | Role |
|---|---|
| `collections.deque` (stdlib) | Rolling window for landmark smoothing |
| `scipy` (optional, later) | Low-pass filter if jitter becomes an issue |

Cursor jitter from raw head pose is significant — a simple moving average over 5–10 frames handles most of it without added dependencies.

---

## Settings & Config

| Tool | Role |
|---|---|
| `json` (stdlib) | Persist user sensitivity settings |
| `tkinter` (stdlib) | Lightweight settings GUI (sensitivity slider, threshold tuner) |

No config library needed — plain JSON file is fine at this scale.

---

## Dev Tools

| Tool | Role |
|---|---|
| VS Code | Editor |
| `black` | Formatter |
| `loguru` | Cleaner logging than stdlib `logging` |
| Git | Version control |

---

## Hardware Assumptions

- Windows 10/11 desktop
- GPU available (not required for MediaPipe CPU inference, but useful if you later add gaze regression)
- Standard webcam (720p minimum recommended)

---

## What's Deliberately Left Out

| Thing | Why excluded |
|---|---|
| Gaze regression model | Eliminates calibration complexity; head pose is sufficient |
| Deep learning eye tracker | Overkill for this learning goal; hard calibration problem |
| Electron/web UI | Python-native is faster to iterate |
| Docker | Not needed locally |
