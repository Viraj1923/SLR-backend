import base64
import io
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
import cv2

app = FastAPI()

# Allow CORS from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageData(BaseModel):
    image: str

# Load model safely
MODEL_PATH = "sign_language_light_model.h5"
model = None

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
        # Remove base64 header if present
        base64_str = data.image.split(',')[-1]
        img_bytes = base64.b64decode(base64_str)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image!")

        # Resize and preprocess
        img = cv2.resize(img, (96, 96))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        print("🔍 Image shape for prediction:", img.shape)

        prediction = model.predict(img)
        predicted_label = np.argmax(prediction[0])
        print("✅ Prediction successful:", predicted_label)

        return {"prediction": int(predicted_label)}

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return {"error": str(e)}
