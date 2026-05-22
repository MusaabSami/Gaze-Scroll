import os
os.environ["GLOG_minloglevel"] = "2"  # suppress MediaPipe C++ backend warnings

import time

import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

EYEBROW_SPEC = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: no camera detected.")
        return

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

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            face_detected = results.multi_face_landmarks is not None

            if face_detected:
                for landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                    )
                    # Eyebrows highlighted in cyan — these drive the click gesture
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=landmarks,
                        connections=mp_face_mesh.FACEMESH_LEFT_EYEBROW,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=EYEBROW_SPEC,
                    )
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=landmarks,
                        connections=mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=EYEBROW_SPEC,
                    )

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            status = "Face detected" if face_detected else "No face"
            status_color = (0, 255, 0) if face_detected else (0, 0, 255)
            h = frame.shape[0]

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, status, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame, "Q - quit", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

            cv2.imshow("GazeClick - Landmark Demo", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
