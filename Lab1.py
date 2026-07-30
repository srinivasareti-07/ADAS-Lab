import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions
)

model = MobileNetV2(weights="imagenet")

image = cv2.imread("day.jpg")

if image is None:
    print("ERROR: Image could not be loaded.")
else:
    img = cv2.resize(image, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)

    predictions = model.predict(x)
    results = decode_predictions(predictions, top=3)[0]

    for _, label, prob in results:
        print(f"{label}: {prob:.2%}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    print("Ambient Brightness:", brightness)

    if brightness < 80:
        print("Headlights ON")
    else:
        print("Headlights OFF")