from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
from io import BytesIO
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

interpreter = tf.lite.Interpreter(model_path="sign_language_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Updated label list
labels = [
    'A', 'B', 'C', 'D','E','F','G','H', 'Hello' ,'I', 'I Love You' ,'J',
    'K','L','M','N','O','P','Q','R','S','T', 'Thank You','U','V','W','X','Y'
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        h, w, _ = img.shape
        hand = results.multi_hand_landmarks[0]
        x_min = w
        y_min = h
        x_max = y_max = 0

        for lm in hand.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)

        padding = 20
        x_min = max(x_min - padding, 0)
        y_min = max(y_min - padding, 0)
        x_max = min(x_max + padding, w)
        y_max = min(y_max + padding, h)

        cropped = img[y_min:y_max, x_min:x_max]
        resized = cv2.resize(cropped, (96, 96))
        normalized = resized.astype(np.float32) / 255.0
        input_data = np.expand_dims(normalized, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        prediction = labels[np.argmax(output)]

        return {"label": prediction}
    else:
        return {"label": "No hand detected"}
