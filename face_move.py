import os
import sys

# MediaPipe's C++ backend writes warnings directly to fd 2 on every frame.
# Redirect fd 2 → devnull permanently, but keep Python's sys.stderr working
# on the saved fd so real Python errors are still visible.
_saved_stderr_fd = os.dup(2)
_devnull_fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull_fd, 2)
os.close(_devnull_fd)
sys.stderr = os.fdopen(_saved_stderr_fd, "w", buffering=1)

import time
import collections

import cv2
import mediapipe as mp
import pyautogui
from screeninfo import get_monitors

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# ---------------------------------------------------------------------------
# Nose tip landmark — tracks head yaw (H) and pitch (V)
# When head turns right: nose.x decreases in the unmirrored image
# When head tilts down:  nose.y increases in the image
# ---------------------------------------------------------------------------
_NOSE_TIP = 1

# ---------------------------------------------------------------------------
# Tunable defaults — identical to cursor_move.py
# ---------------------------------------------------------------------------
SMOOTHING    = 15      # moving-average window (frames)
SPEED_H      = 6000.0  # pixels/second per unit of horizontal head displacement
SPEED_V      = 10000.0 # pixels/second per unit of vertical head displacement
DEAD_ZONE    = 0.008   # head displacement fraction — cursor stops inside this radius
DWELL_FREEZE = 0.5     # seconds head must stay in dead zone to re-anchor neutral

mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils


def _screen_size():
    m = get_monitors()[0]
    return m.width, m.height


def _nose_position(lm):
    """Return (h_ratio, v_ratio) as nose tip normalised x,y in image space.

    h_ratio: decreases when head turns right (nose moves left in unmirrored image)
    v_ratio: increases when head tilts down  (nose moves toward bottom of image)
    """
    return lm[_NOSE_TIP].x, lm[_NOSE_TIP].y


def main():
    sw, sh = _screen_size()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: no camera detected.")
        return

    h_buf = collections.deque(maxlen=SMOOTHING)
    v_buf = collections.deque(maxlen=SMOOTHING)

    neutral_h    = None
    neutral_v    = None
    cur_x        = sw // 2
    cur_y        = sh // 2
    stable_since = None
    pyautogui.moveTo(cur_x, cur_y)

    debug     = True
    prev_time = time.time()

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            now       = time.time()
            dt        = now - prev_time
            prev_time = now
            fps       = 1.0 / max(dt, 1e-6)

            face_ok = results.multi_face_landmarks is not None

            if face_ok:
                lm               = results.multi_face_landmarks[0].landmark
                h_ratio, v_ratio = _nose_position(lm)

                h_buf.append(h_ratio)
                v_buf.append(v_ratio)

                smooth_h = sum(h_buf) / len(h_buf)
                smooth_v = sum(v_buf) / len(v_buf)

                if neutral_h is None:
                    neutral_h = smooth_h
                    neutral_v = smooth_v

                # rel_h < 0 → head turned right   rel_h > 0 → head turned left
                # rel_v < 0 → head tilted up       rel_v > 0 → head tilted down
                rel_h = smooth_h - neutral_h
                rel_v = smooth_v - neutral_v

                in_dead_zone = abs(rel_h) < DEAD_ZONE and abs(rel_v) < DEAD_ZONE

                if in_dead_zone:
                    if stable_since is None:
                        stable_since = now
                    elif now - stable_since >= DWELL_FREEZE:
                        # Re-anchor: cursor is locked here, neutral resets to current head pose
                        neutral_h    = smooth_h
                        neutral_v    = smooth_v
                        stable_since = None
                else:
                    stable_since = None
                    if abs(rel_h) > DEAD_ZONE:
                        cur_x = max(0, min(sw - 1, cur_x + int(-rel_h * SPEED_H * dt)))
                    if abs(rel_v) > DEAD_ZONE:
                        cur_y = max(0, min(sh - 1, cur_y + int(rel_v * SPEED_V * dt)))
                pyautogui.moveTo(cur_x, cur_y)

                if debug:
                    cv2.putText(frame, f"Head H: {rel_h:+.3f}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(frame, f"Head V: {rel_v:+.3f}", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    if stable_since is not None:
                        pct = min(1.0, (now - stable_since) / DWELL_FREEZE)
                        cv2.rectangle(frame, (10, 145), (130, 160), (60, 60, 60), -1)
                        cv2.rectangle(frame, (10, 145), (10 + int(120 * pct), 160), (0, 210, 0), -1)
                        cv2.putText(frame, "Freeze", (135, 158),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 0), 1)

                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=results.multi_face_landmarks[0],
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=(0, 200, 100), thickness=1),
                    )

            status       = "Face detected" if face_ok else "No face"
            status_color = (0, 255, 0) if face_ok else (0, 0, 255)
            hf            = frame.shape[0]

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, status, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame, "R-reset  D-debug  Q-quit", (10, hf - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

            cv2.imshow("GazeClick - Face Move", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                neutral_h    = None
                neutral_v    = None
                stable_since = None
                h_buf.clear()
                v_buf.clear()
            elif key == ord("d"):
                debug = not debug

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
