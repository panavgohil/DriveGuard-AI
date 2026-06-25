import cv2
import mediapipe as mp
import os
import time

# -----------------------------
# Create Dataset Folders
# -----------------------------
os.makedirs("my_dataset/awake", exist_ok=True)
os.makedirs("my_dataset/sleepy", exist_ok=True)

# -----------------------------
# MediaPipe Setup
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# -----------------------------
# Camera
# -----------------------------
cap = cv2.VideoCapture(0)

# -----------------------------
# Counters
# -----------------------------
awake_count = len(os.listdir("my_dataset/awake"))
sleepy_count = len(os.listdir("my_dataset/sleepy"))

mode = "PAUSED"

SAVE_INTERVAL = 0.05      # 50 ms
last_save_time = time.time()

# -----------------------------
# Eye Crop Function
# -----------------------------
def get_eye_crop(face_landmarks, eye_indices, frame):

    h, w, _ = frame.shape

    points = []

    for idx in eye_indices:
        x = int(face_landmarks.landmark[idx].x * w)
        y = int(face_landmarks.landmark[idx].y * h)
        points.append((x, y))

    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    padding = 25

    x_min = max(0, min(x_coords) - padding)
    x_max = min(w, max(x_coords) + padding)

    y_min = max(0, min(y_coords) - padding)
    y_max = min(h, max(y_coords) + padding)

    eye_crop = frame[y_min:y_max, x_min:x_max]

    return eye_crop

# -----------------------------
# Main Loop
# -----------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        h, w, _ = frame.shape

        for face_landmarks in results.multi_face_landmarks:

            left_eye = get_eye_crop(
                face_landmarks,
                LEFT_EYE,
                frame
            )

            if left_eye.size > 0:

                left_eye_gray = cv2.cvtColor(
                    left_eye,
                    cv2.COLOR_BGR2GRAY
                )

                left_eye_resized = cv2.resize(
                    left_eye_gray,
                    (86, 86)
                )

                cv2.imshow(
                    "Processed Eye",
                    left_eye_resized
                )

                current_time = time.time()

                if current_time - last_save_time >= SAVE_INTERVAL:

                    if mode == "AWAKE":

                        filename = f"my_dataset/awake/awake_{awake_count}.jpg"

                        cv2.imwrite(
                            filename,
                            left_eye_resized
                        )

                        awake_count += 1

                    elif mode == "SLEEPY":

                        filename = f"my_dataset/sleepy/sleepy_{sleepy_count}.jpg"

                        cv2.imwrite(
                            filename,
                            left_eye_resized
                        )

                        sleepy_count += 1

                    last_save_time = current_time

            # Draw Left Eye
            for idx in LEFT_EYE:

                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    2,
                    (0, 255, 0),
                    -1
                )

            # Draw Right Eye
            for idx in RIGHT_EYE:

                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    2,
                    (255, 0, 0),
                    -1

                )

    # -----------------------------
    # UI
    # -----------------------------
    cv2.putText(
        frame,
        f"MODE : {mode}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        f"Awake : {awake_count}",
        (20,70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,0),
        2
    )

    cv2.putText(
        frame,
        f"Sleepy : {sleepy_count}",
        (20,105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,0,255),
        2
    )

    cv2.putText(
        frame,
        "1=Awake  2=Sleepy  0=Pause  Q=Quit",
        (20,140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    cv2.imshow(
        "Eye Landmarks",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord('1'):
        mode = "AWAKE"
        print("Collecting AWAKE images...")

    elif key == ord('2'):
        mode = "SLEEPY"
        print("Collecting SLEEPY images...")

    elif key == ord('0'):
        mode = "PAUSED"
        print("Collection Paused")

    elif key == ord('q'):
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()

print("\nCollection Complete!")
print("Awake Images :", awake_count)
print("Sleepy Images:", sleepy_count)