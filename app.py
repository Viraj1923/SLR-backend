import os
import warnings
import base64
import numpy as np
import cv2
import gdown
import mediapipe as mp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model

# Environment settings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
warnings.filterwarnings("ignore")

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
MODEL_PATH = "sign_language_light_model.h5"
gesture_classes = ['A', 'B', 'C', 'D','E','F','G','H', 'Hello' ,'I', 'I Love You' ,'J','K','L','M','N','O','P','Q','R','S','T', 'Thank You','U','V','W','X','Y']
IMAGE_SIZE = 96

# Load model at startup
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Model failed to load:", str(e))
    model = None

# Request body schema
class ImageData(BaseModel):
    image: str

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

@app.post("/predict")
def predict(data: ImageData):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded.")

        # Decode image
        image_bytes = base64.b64decode(data.image)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image.")

        # Convert to RGB and detect hand
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_img)

        if not results.multi_hand_landmarks:
            raise HTTPException(status_code=400, detail="No hand detected.")

        # Bounding box
        h, w, _ = img.shape
        x_min, y_min, x_max, y_max = w, h, 0, 0

        for lm in results.multi_hand_landmarks[0].landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x), max(y_max, y)

        padding = 20
        x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
        x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)

        cropped = img[y_min:y_max, x_min:x_max]
        resized = cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE))
        normalized = resized / 255.0
        input_img = np.expand_dims(normalized, axis=0)

        # Predict
        predictions = model.predict(input_img)
        predicted_class = gesture_classes[np.argmax(predictions)]

        return {"label": predicted_class}

    except Exception as e:
        print("Prediction error:", e)
        raise HTTPException(status_code=500, detail="Internal Server Error: " + str(e))

@app.get("/")
def home():
    return {"message": "🚀 FastAPI backend is running!"}
