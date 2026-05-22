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
# Iris landmark indices (require refine_landmarks=True)
# Right eye = subject's right = LEFT side of unmirrored image
# Left eye  = subject's left  = RIGHT side of unmirrored image
# ---------------------------------------------------------------------------
_R_OUTER = 33    # right eye temporal (outer) corner
_R_INNER = 133   # right eye nasal (inner) corner
_R_TOP   = 159   # right eye upper lid
_R_BOT   = 145   # right eye lower lid
_R_IRIS  = 468   # right iris centre

_L_INNER = 362   # left eye nasal (inner) corner
_L_OUTER = 263   # left eye temporal (outer) corner
_L_TOP   = 386   # left eye upper lid
_L_BOT   = 374   # left eye lower lid
_L_IRIS  = 473   # left iris centre

# ---------------------------------------------------------------------------
# Tunable defaults
# ---------------------------------------------------------------------------
SMOOTHING = 15      # moving-average window (frames)
SPEED_H   = 5500.0  # pixels/second per unit of horizontal eye displacement
SPEED_V   = 16000.0  # pixels/second per unit of vertical eye displacement
DEAD_ZONE = 0.008   # eye displacement fraction — cursor stops inside this radius

mp_face_mesh      = mp.solutions.face_mesh
mp_drawing        = mp.solutions.drawing_utils


def _screen_size():
    m = get_monitors()[0]
    return m.width, m.height


def _iris_position(lm):
    """Return (h_ratio, v_ratio) normalised within each eye, or (None, None) on blink.

    h_ratio: decreases when looking right (iris moves left in unmirrored image)
    v_ratio: decreases when looking up   (iris moves toward upper lid)
    """
    r_eye_w = lm[_R_INNER].x - lm[_R_OUTER].x
    l_eye_w = lm[_L_OUTER].x - lm[_L_INNER].x
    r_eye_h = lm[_R_BOT].y   - lm[_R_TOP].y
    l_eye_h = lm[_L_BOT].y   - lm[_L_TOP].y

    if r_eye_h < 0.15 * r_eye_w or l_eye_h < 0.15 * l_eye_w:
        return None, None

    r_h = (lm[_R_IRIS].x - lm[_R_OUTER].x) / r_eye_w
    l_h = (lm[_L_IRIS].x - lm[_L_INNER].x) / l_eye_w
    r_v = (lm[_R_IRIS].y - lm[_R_TOP].y)   / r_eye_h
    l_v = (lm[_L_IRIS].y - lm[_L_TOP].y)   / l_eye_h

    return (r_h + l_h) / 2, (r_v + l_v) / 2


def main():
    sw, sh = _screen_size()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: no camera detected.")
        return

    h_buf = collections.deque(maxlen=SMOOTHING)
    v_buf = collections.deque(maxlen=SMOOTHING)

    neutral_h = None
    neutral_v = None
    cur_x     = sw // 2
    cur_y     = sh // 2
    pyautogui.moveTo(cur_x, cur_y)

    debug     = True
    prev_time = time.time()

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
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
                h_ratio, v_ratio = _iris_position(lm)

                if h_ratio is not None:
                    h_buf.append(h_ratio)
                    v_buf.append(v_ratio)

                    smooth_h = sum(h_buf) / len(h_buf)
                    smooth_v = sum(v_buf) / len(v_buf)

                    if neutral_h is None:
                        neutral_h = smooth_h
                        neutral_v = smooth_v

                    # rel_h < 0 → looking right   rel_h > 0 → looking left
                    # rel_v < 0 → looking up       rel_v > 0 → looking down
                    rel_h = smooth_h - neutral_h
                    rel_v = smooth_v - neutral_v

                    # Velocity: eye displacement drives cursor speed, dead zone stops it
                    if abs(rel_h) > DEAD_ZONE:
                        cur_x = max(0, min(sw - 1, cur_x + int(-rel_h * SPEED_H * dt)))
                    if abs(rel_v) > DEAD_ZONE:
                        cur_y = max(0, min(sh - 1, cur_y + int(-rel_v * SPEED_V * dt)))
                    pyautogui.moveTo(cur_x, cur_y)

                    if debug:
                        cv2.putText(frame, f"Eye H: {rel_h:+.3f}", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        cv2.putText(frame, f"Eye V: {rel_v:+.3f}", (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if debug:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=results.multi_face_landmarks[0],
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=(0, 255, 255), thickness=1),
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

            cv2.imshow("GazeClick - Cursor Move", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                neutral_h = None
                neutral_v = None
                h_buf.clear()
                v_buf.clear()
            elif key == ord("d"):
                debug = not debug

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
