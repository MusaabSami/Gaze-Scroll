# Face Scroll

Hands-free cursor control and clicking using head pose via webcam. No extra hardware required — just a standard webcam and Python.

## How It Works

MediaPipe Face Mesh detects 478 facial landmarks in real time. The nose tip landmark is used as a stable proxy for head orientation — its normalised position in the frame shifts predictably as you turn or tilt your head. That displacement from a neutral baseline drives the cursor at a speed proportional to how far you move — the further from centre, the faster it moves. Returning to the neutral zone stops the cursor. Raising both eyebrows triggers a left click.

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
python face_move.py
```

Hold your head level and look straight at the camera for ~1 second on startup — neutral is captured automatically.

## Controls

| Key | Action |
|-----|--------|
| `R` | Reset neutral — recalibrate from current head position |
| `D` | Toggle debug overlay |
| `Q` | Quit |

## Clicking

Raise both eyebrows and hold for `0.15 s` to fire a left click. The cursor freezes as soon as a raise is detected so the click always lands on the intended target.

## Settings

| Constant | Value | Description |
|----------|-------|-------------|
| `SMOOTHING` | `15` | Moving-average window in frames — higher = smoother but more lag |
| `SPEED_H` | `4000.0` | Horizontal cursor speed (pixels/second per unit of head displacement) |
| `SPEED_V` | `4000.0` | Vertical cursor speed |
| `DEAD_ZONE` | `0.008` | Head displacement fraction — cursor stops within this radius of neutral |
| `DWELL_FREEZE` | `0.5` | Seconds in dead zone before neutral re-anchors to current head position |
| `BROW_THRESHOLD` | `0.03` | Eyebrow raise delta (fraction of face height) required to trigger a click |
| `BROW_FREEZE_THRESH` | `0.022` | Lower threshold at which cursor freezes in anticipation of a click |
| `BROW_HOLD` | `0.15` | Seconds eyebrows must stay raised before click fires |
| `BROW_COOLDOWN` | `0.8` | Minimum seconds between clicks |

### Movement Model

**Velocity mode**: head displacement from neutral drives cursor *speed*, not absolute position. A small tilt moves the cursor slowly; a larger tilt moves it faster. Returning to the dead zone stops it immediately. After `DWELL_FREEZE` seconds of stillness the neutral point re-anchors, so the cursor stays put without any sustained effort.

### Calibration Tips

- If the cursor drifts when your head is still, increase `DEAD_ZONE` slightly (e.g. `0.012`).
- If small movements feel unresponsive, lower `DEAD_ZONE` (e.g. `0.006`).
- If the cursor moves too fast or slow, adjust `SPEED_H` / `SPEED_V`.
- If clicks fire accidentally, raise `BROW_THRESHOLD` (e.g. `0.04`) or increase `BROW_HOLD`.
- Press `R` any time to recalibrate if the cursor starts drifting after repositioning.

## Tech Stack

- [MediaPipe](https://developers.google.com/mediapipe) — face mesh landmarks (nose tip + eyebrow points)
- [OpenCV](https://opencv.org/) — webcam capture and debug overlay
- [pyautogui](https://pyautogui.readthedocs.io/) — cursor movement and clicking
- [screeninfo](https://github.com/rr-/screeninfo) — monitor resolution detection
