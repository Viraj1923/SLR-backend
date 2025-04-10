import base64
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
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

# Request model
class ImageData(BaseModel):
    image: str

# Model path and label mapping
MODEL_PATH = "sign_language_light_model.h5"
gesture_classes = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
    'Hello', 'I', 'I Love You', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'Thank You',
    'U', 'V', 'W', 'X', 'Y'
]

# Load model once at startup
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
        # Decode base64 image
        base64_str = data.image.split(',')[-1]
        img_bytes = base64.b64decode(base64_str)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image!")

        # Resize and preprocess image
        img = cv2.resize(img, (96, 96))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        print("🔍 Image shape for prediction:", img.shape)

        # Predict
        prediction = model.predict(img)
        predicted_label = np.argmax(prediction[0])
        label = gesture_classes[predicted_label]
        print(f"✅ Prediction successful: {label}")
        print("🧠 Raw prediction:", prediction[0])
        print("🧠 Label from list:", gesture_classes[predicted_label])


        return {
            "prediction": int(predicted_label),
            "label": label
        }

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return {"error": str(e)}

@app.get("/")
def root():
    return {"message": "🚀 FastAPI backend for FingerTalk is running!"}
