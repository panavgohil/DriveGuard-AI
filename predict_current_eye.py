import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("drowsiness_model.keras")

img = cv2.imread(
    "current_eye.jpg",
    cv2.IMREAD_GRAYSCALE
)

img = cv2.resize(img, (86, 86))

img = img.astype("float32") / 255.0

img = np.expand_dims(img, axis=-1)
img = np.expand_dims(img, axis=0)

prediction = model.predict(
    img,
    verbose=0
)[0][0]

if prediction > 0.5:
    print("SLEEPY")
else:
    print("AWAKE")

print("Raw Prediction:", prediction)