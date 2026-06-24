import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import numpy as np
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)
model=load_model("drowsiness_model.keras")
print("model loaded")
cap = cv2.VideoCapture(0)
def get_eye_crop(face_landmarks, eye_indices, frame):
    h,w,_=frame.shape
    points=[]
    for idx in eye_indices:
        x=int(face_landmarks.landmark[idx].x*w)
        y=int(face_landmarks.landmark[idx].y*h)
        points.append((x,y))
    x_coords=[p[0] for p in points]
    y_coords=[p[1] for p in points]

    x_min=min(x_coords)
    x_max=max(x_coords)
    y_min=min(y_coords)
    y_max=max(y_coords)
    padding=10
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)

    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)

    eye_crop=frame[y_min:y_max, x_min:x_max]
    return eye_crop

while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        h, w, _ = frame.shape
        for face_landmarks in results.multi_face_landmarks:
            left_eye=get_eye_crop(
                face_landmarks,
                LEFT_EYE,
                frame
            )
            if left_eye.size > 0:
                left_eye_gray=cv2.cvtColor(
                    left_eye,
                    cv2.COLOR_BGR2GRAY
                )
                left_eye_resized=cv2.resize(
                    left_eye_gray,
                    (86,86)

                )
                eye_input=left_eye_resized.astype("float32")/255.0
                eye_input=np.expand_dims(
                    eye_input,
                    axis=-1
                )
                eye_input=np.expand_dims(
                    eye_input,
                    axis=0
                
                )
                print(eye_input.shape)
                prediction=model.predict(
                    eye_input,
                    verbose=0
                )[0][0]
                print("Prediction:", prediction)
                cv2.imshow(
                    "processed eye",
                    left_eye_resized
                )
                if prediction > 0.5:
                    state = f"SLEEPY {prediction:.2f}"
                else:
                    state = f"AWAKE {(1-prediction):.2f}"
                cv2.putText(
                    frame,
                    state,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
            # Left eye
            for idx in LEFT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            # Right eye
            for idx in RIGHT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
    cv2.imshow("Eye Landmarks", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()