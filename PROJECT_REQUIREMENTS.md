# Project Requirements — GazeClick

> A hands-free cursor control system: head movement drives the cursor, eyebrow raises trigger clicks.

---

## Project Goal

Build a Python application that lets a user control their computer cursor entirely through facial movement — no hands, no calibration ceremony. This is a personal learning project exploring computer vision, signal processing, and human-computer interaction.

---

## Core Concepts

| Concept | Description |
|---|---|
| **Head Pose → Cursor** | Yaw (left/right) and pitch (up/down) of the head maps to X/Y screen coordinates |
| **Eyebrow Raise → Click** | A deliberate eyebrow raise (held for ~200ms) fires a left mouse click |
| **No Fixed Calibration** | The user naturally moves to a comfortable neutral head position; the system adapts |

---

## Functional Requirements

### FR-1 — Camera Input
- Capture live webcam feed at 30fps minimum
- Support standard USB and built-in webcams
- Graceful error if no camera is detected

### FR-2 — Face Detection & Landmark Tracking
- Detect a single face in the frame
- Extract 3D facial landmarks via MediaPipe Face Mesh (478 points)
- Handle brief occlusion (glasses, hand passing in front) without crashing

### FR-3 — Head Pose Estimation
- Derive yaw (horizontal) and pitch (vertical) angles from facial landmarks
- Map yaw/pitch range to full screen X/Y coordinates
- Dead zone in the center (~5° around neutral) to prevent drift while stationary

### FR-4 — Cursor Movement
- Move system cursor smoothly based on head pose
- Apply moving-average smoothing (configurable window, default 7 frames)
- Sensitivity is adjustable (how many degrees of head tilt = full screen width)

### FR-5 — Eyebrow Click Detection
- Track vertical distance between eyebrow landmarks and eye landmarks (both brows)
- Trigger click when raise exceeds threshold AND is held for a dwell period (default 200ms)
- Reset detection state after each click (prevent repeat-fire)
- Cooldown period after click (default 500ms) to prevent accidental double-clicks

### FR-6 — Neutral Position Reset
- Keyboard shortcut (`R` key) resets the neutral head position to wherever the user is currently facing
- Allows users to recenter without restarting the app

### FR-7 — Debug Overlay
- Optional OpenCV window showing:
  - Live face mesh
  - Current yaw/pitch values
  - Eyebrow raise level (bar or numeric)
  - Click indicator flash
- Toggled on/off with `D` key

### FR-8 — Settings
- Sensitivity (cursor speed per degree of head tilt)
- Eyebrow raise threshold
- Dwell time before click fires
- Smoothing window size
- Settings saved to `config.json` and loaded on startup

### FR-9 — Clean Exit
- `Q` key or window close stops tracking and releases the camera

---

## Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-1 | End-to-end latency < 100ms (camera → cursor move) on target hardware |
| NFR-2 | CPU usage < 40% on a modern desktop (MediaPipe CPU inference) |
| NFR-3 | No internet connection required at runtime |
| NFR-4 | Works at normal desk-to-screen distance (50–80cm from webcam) |
| NFR-5 | Graceful degradation if face is lost (cursor freezes, does not jump) |

---

## Out of Scope

- Eye gaze regression (iris tracking → cursor)
- Multi-monitor support (single primary monitor only, for now)
- Right-click, scroll, or drag gestures (Phase 2)
- Mobile or web deployment
- Multi-user profiles

---

## Milestones

| Phase | Goal | Deliverable |
|---|---|---|
| **Phase 1** | Get MediaPipe running, visualize landmarks | `landmark_demo.py` |
| **Phase 2** | Head pose → cursor movement | `cursor_move.py` |
| **Phase 3** | Eyebrow raise detection | `eyebrow_click.py` |
| **Phase 4** | Integrate into single app with smoothing | `gazeclick.py` |
| **Phase 5** | Settings UI + config persistence | `settings.py` + `config.json` |
| **Phase 6** | Polish: dead zone, cooldowns, debug overlay | Final app |

---

## Success Criteria

- User can navigate to and click a button on screen without touching input devices
- Cursor does not jitter noticeably during stationary head pose
- Eyebrow click does not fire accidentally during normal expression changes
- App runs stably for 10+ minutes without crash or memory growth
