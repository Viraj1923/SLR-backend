import base64
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import cv2
import mediapipe as mp

app = FastAPI()

# Allow CORS from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ImageData(BaseModel):
    image: str

# Model and labels
MODEL_PATH = "sign_language_light_model.h5"
gesture_classes = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
    'Hello', 'I', 'I Love You', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'Thank You',
    'U', 'V', 'W', 'X', 'Y'
]

model = None
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

@app.on_event("startup")
def load_model_on_startup():
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

@app.post("/predict")
async def predict_image(data: ImageData):
    try:
        # Decode base64 image
        base64_str = data.image.split(',')[-1]
        img_bytes = base64.b64decode(base64_str)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.flip(img, 1)

        if img is None:
            raise ValueError("Failed to decode image!")

        # Detect hand using MediaPipe
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_img)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                x_min, y_min = img.shape[1], img.shape[0]
                x_max, y_max = 0, 0
                for landmark in landmarks.landmark:
                    x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
                    x_min, y_min = min(x_min, x), min(y_min, y)
                    x_max, y_max = max(x_max, x), max(y_max, y)

                # Crop hand with padding
                padding = 20
                x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
                x_max, y_max = min(img.shape[1], x_max + padding), min(img.shape[0], y_max + padding)

                hand_img = img[y_min:y_max, x_min:x_max]
                if hand_img.size == 0:
                    raise ValueError("Invalid cropped hand image")

                hand_img = cv2.resize(hand_img, (96, 96))
                hand_img = hand_img / 255.0
                hand_img = np.expand_dims(hand_img, axis=0)

                prediction = model.predict(hand_img)
                predicted_label = np.argmax(prediction[0])
                label = gesture_classes[predicted_label]
                print(f"✅ Prediction: {label}")

                return {
                    "prediction": int(predicted_label),
                    "label": label
                }

        raise ValueError("No hands detected in the image.")

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return {"error": str(e)}

@app.get("/")
def root():
    return {"message": "🚀 FastAPI backend for FingerTalk is running!"}
