import base64
import numpy as np
import cv2
import mediapipe as mp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="sign_language_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Constants
gesture_classes = ['A', 'B', 'C', 'D','E','F','G','H', 'Hello' ,'I', 'I Love You' ,'J','K','L','M','N','O','P','Q','R','S','T', 'Thank You','U','V','W','X','Y']
IMAGE_SIZE = 96

# Init FastAPI app
app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Request schema
class ImageData(BaseModel):
    image: str  # base64-encoded image

@app.get("/")
def root():
    return {"message": "🚀 TFLite FastAPI backend is running"}

@app.post("/predict")
def predict(data: ImageData):
    print("✅ /predict called")
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(data.image.split(',')[-1])
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        # MediaPipe processing
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_img)

        if not results.multi_hand_landmarks:
            return {"label": "-"}


        # Get bounding box
        h, w, _ = img.shape
        x_min, y_min = w, h
        x_max, y_max = 0, 0

        for lm in results.multi_hand_landmarks[0].landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x), max(y_max, y)

        padding = 20
        x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
        x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)

        # Preprocess image
        cropped = img[y_min:y_max, x_min:x_max]
        resized = cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE))
        normalized = resized.astype(np.float32) / 255.0
        input_data = np.expand_dims(normalized, axis=0)

        # Predict with TFLite
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_index = np.argmax(output_data)
        predicted_label = gesture_classes[predicted_index]

        return {"label": predicted_label}

    except Exception as e:
        print("❌ Prediction error:", e)
        raise HTTPException(status_code=500, detail="Internal Server Error: " + str(e))
