# GazeClick

Hands-free cursor control using eye tracking via webcam. No extra hardware required — just a standard webcam and Python.

## How It Works

MediaPipe Face Mesh detects 478 facial landmarks including iris centres in real time. The iris position within each eye socket is normalized to a ratio, smoothed over a moving average window, and compared against a neutral baseline. The deviation from neutral drives the cursor at a speed proportional to how far you look — the further from centre, the faster it moves.

## Project Phases

| Phase | File | Status |
|-------|------|--------|
| 1 | `landmark_demo.py` | Done — live face mesh + eyebrow overlay |
| 2 | `cursor_move.py` | Done — iris-driven cursor movement |
| 3 | `eyebrow_click.py` | Planned — eyebrow raise → left click |
| 4 | `gazeclick.py` | Planned — integrated app |
| 5 | `settings.py` + `config.json` | Planned — settings UI |
| 6 | Polish | Planned — cooldowns, refined dead zone |

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install opencv-python mediapipe pyautogui screeninfo
```

## Running

```bash
# Phase 1 — landmark visualiser
python landmark_demo.py

# Phase 2 — cursor control
python cursor_move.py
```

## Controls (cursor_move.py)

| Key | Action |
|-----|--------|
| `R` | Reset neutral — recalibrate from current eye position |
| `D` | Toggle debug overlay (iris tracking lines + eye displacement values) |
| `Q` | Quit |

## Current Settings

These are the tuned values in `cursor_move.py` as of Phase 2:

| Constant | Value | Description |
|----------|-------|-------------|
| `SMOOTHING` | `15` | Moving-average window in frames — higher = smoother but more lag |
| `SPEED_H` | `4000.0` | Horizontal cursor speed — pixels/second per unit of eye displacement |
| `SPEED_V` | `8000.0` | Vertical cursor speed — higher than horizontal to compensate for the narrower vertical range of eye movement |
| `DEAD_ZONE` | `0.008` | Eye displacement fraction — cursor stops when gaze is within this radius of neutral |

### Movement Model

**Velocity mode**: eye displacement from neutral drives cursor *speed*, not absolute position. Looking slightly off-centre moves the cursor slowly; looking further moves it faster. Looking back to neutral stops it. The cursor stays where it is when your gaze returns to the dead zone.

### Calibration Tips

- Run `cursor_move.py` and look straight at the camera for ~1 second before moving your eyes — neutral is captured automatically on the first stable frame.
- Press `R` any time to recalibrate from your current gaze.
- If the cursor moves too fast or slow, adjust `SPEED_H` / `SPEED_V` at the top of `cursor_move.py`.
- If the cursor drifts when your eyes are still, increase `DEAD_ZONE` slightly (e.g. `0.012`).

## Tech Stack

- [MediaPipe](https://developers.google.com/mediapipe) — face mesh + iris landmarks (landmark 468/473)
- [OpenCV](https://opencv.org/) — webcam capture and debug overlay
- [pyautogui](https://pyautogui.readthedocs.io/) — cursor movement
- [screeninfo](https://github.com/rr-/screeninfo) — monitor resolution detection
