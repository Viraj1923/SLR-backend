from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import base64
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import tensorflow as tf
import mediapipe as mp

# Initialize FastAPI app
app = FastAPI()

# CORS (optional but needed for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or set to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Labels
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Hello', 'I', 'I Love You',
          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'Thank You', 'U', 'V', 'W', 'X', 'Y']

# Input size (get from model)
input_shape = input_details[0]['shape']  # Usually (1, 224, 224, 3)
input_size = input_shape[1]  # 224 or whatever your model expects


# Request model
class ImageRequest(BaseModel):
    image: str


# Utility: Decode base64, preprocess image using MediaPipe (if needed)
def preprocess_base64_image(image_base64):
    # Decode image
    image_data = base64.b64decode(image_base64.split(',')[-1])
    image = Image.open(BytesIO(image_data)).convert("RGB")
    image = np.array(image)

    # Optionally use MediaPipe to crop hand region
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            h, w, _ = image.shape
            x_coords = [int(landmark.x * w) for landmark in hand_landmarks.landmark]
            y_coords = [int(landmark.y * h) for landmark in hand_landmarks.landmark]
            x_min, x_max = max(min(x_coords) - 20, 0), min(max(x_coords) + 20, w)
            y_min, y_max = max(min(y_coords) - 20, 0), min(max(y_coords) + 20, h)
            image = image[y_min:y_max, x_min:x_max]

    # Resize to input size and normalize
    image = cv2.resize(image, (input_size, input_size))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    return image


# Route: POST /predict
@app.post("/predict")
def predict(data: ImageRequest):
    image = preprocess_base64_image(data.image)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_index = int(np.argmax(output_data[0]))
    predicted_label = labels[predicted_index]

    return {"label": predicted_label}
