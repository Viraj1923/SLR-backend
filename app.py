from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
import base64
import gdown
import os
from io import BytesIO
from PIL import Image
import mediapipe as mp

# === CONFIGURATION ===
MODEL_PATH = "model.tflite"
DRIVE_FILE_ID = "1MuNjBliVKTLiDwux2MVdUrYsTyeSY596"  # Replace this with your file ID
LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Hello', 'I', 'I Love You',
          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'Thank You',
          'U', 'V', 'W', 'X', 'Y']

# === DOWNLOAD MODEL FROM DRIVE ===
def download_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("✅ Model downloaded from Google Drive.")

# === INITIALIZE ===
download_model()
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

app = FastAPI()

# === CORS SETUP ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === REQUEST SCHEMA ===
class ImageData(BaseModel):
    image: str  # base64 encoded image

# === HELPER FUNCTIONS ===
def preprocess_image(base64_str):
    # ✅ Strip header if present
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",")[1]

    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    img_np = np.array(image)

    # Detect hand and crop
    results = hands.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    if results.multi_hand_landmarks:
        h, w, _ = img_np.shape
        hand = results.multi_hand_landmarks[0]
        x_coords = [lm.x * w for lm in hand.landmark]
        y_coords = [lm.y * h for lm in hand.landmark]
        x1, y1 = int(max(min(x_coords) - 20, 0)), int(max(min(y_coords) - 20, 0))
        x2, y2 = int(min(max(x_coords) + 20, w)), int(min(max(y_coords) + 20, h))
        img_np = img_np[y1:y2, x1:x2]

    # Resize and normalize
    resized = cv2.resize(img_np, (224, 224))  # change to your model's input size
    normalized = resized.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=0)


def predict(image_np):
    interpreter.set_tensor(input_details[0]['index'], image_np)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred_index = np.argmax(output_data)
    return LABELS[pred_index]

# === PREDICT ENDPOINT ===
@app.post("/predict")
async def predict_letter(data: ImageData):
    try:
        image_np = preprocess_image(data.image)
        letter = predict(image_np)
        return {"prediction": letter}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
