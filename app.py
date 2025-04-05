import os
import warnings
import base64
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set model path (adjust if using models/ folder)
MODEL_PATH = "sign_language_light_model.h5"

gesture_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Hello', 'I', 'I Love You',
                   'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'Thank You',
                   'U', 'V', 'W', 'X', 'Y']

# Request model
class ImageData(BaseModel):
    image: str

# Load model on startup
@app.on_event("startup")
def load_model_once():
    global model
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")

@app.get("/")
def home():
    return {"message": "Backend is running!"}

@app.post("/predict")
def predict(data: ImageData):
    try:
        if len(data.image) > 2_000_000:
            raise HTTPException(status_code=413, detail="Image payload too large.")

        image_bytes = base64.b64decode(data.image)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data.")

        # Preprocessing
        img = cv2.resize(img, (96, 96))
        img = img / 255.0
        img = img.reshape(1, 96, 96, 3)


        # Prediction
        prediction = model.predict(img)
        predicted_class = gesture_classes[np.argmax(prediction)]

        return {"label": predicted_class}

    except Exception as e:
        print("❌ Error:", e)
        raise HTTPException(status_code=500, detail="Internal Server Error: " + str(e))
