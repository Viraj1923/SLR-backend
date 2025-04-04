import os
import warnings
import base64
import numpy as np
import cv2
import gdown
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model

# Suppress warnings and TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://slr-frontenddd.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Google Drive file ID and model path
FILE_ID = "1uQGu9Prwp9hvSIKZTR5S8E7gRLrKKz1-"
MODEL_PATH = "keras_model.h5"

# Function to download model if not exists
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("Model downloaded.")

# Download and load the model
download_model()
model = load_model(MODEL_PATH)

# Gesture class labels
gesture_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Hello', 'I', 'I Love You',
                   'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'Thank You',
                   'U', 'V', 'W', 'X', 'Y']

# Request model class
class ImageData(BaseModel):
    image: str

@app.get("/")
def home():
    return {"message": "Backend is running!"}

@app.post("/predict")
def predict(data: ImageData):
    try:
        image_bytes = base64.b64decode(data.image)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image. Check base64 format.")

        img = cv2.resize(img, (224, 224))

        img = img / 255.0
        img = img.reshape(1, 224, 224, 3)

        prediction = model.predict(img)
        predicted_class = gesture_classes[np.argmax(prediction)]

        return {"label": predicted_class}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
