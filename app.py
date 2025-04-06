import base64
import numpy as np
import cv2
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load TFLite model
TFLITE_MODEL_PATH = "sign_language_model.tflite"
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Label classes
gesture_classes = ['A', 'B', 'C', 'D','E','F','G','H', 'Hello' ,'I', 'I Love You' ,'J','K','L','M','N','O','P','Q','R','S','T', 'Thank You','U','V','W','X','Y']

# Input schema
class ImageData(BaseModel):
    image: str  # Base64 encoded image

@app.post("/predict")
async def predict(data: ImageData):
    try:
        # Decode base64 string
        base64_str = data.image.split(',')[-1]
        image_bytes = base64.b64decode(base64_str)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        # Preprocess
        img = cv2.resize(img, (96, 96))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        # Set tensor and run inference
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        predicted_index = int(np.argmax(output))
        predicted_label = gesture_classes[predicted_index]

        return {"label": predicted_label}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.get("/")
def root():
    return {"message": "🚀 FastAPI with TFLite is running!"}
