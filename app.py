import os
import warnings
from flask_cors import CORS
from flask import Flask, jsonify, request
import gdown
import cv2
import numpy as np
import base64
from tensorflow.keras.models import load_model

# Suppress warnings and TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# Google Drive file ID
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

# Define gesture classes
gesture_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Hello', 'I', 'I Love You', 
                   'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'Thank You', 
                   'U', 'V', 'W', 'X', 'Y']

@app.route('/')
def home():
    return jsonify({"message": "Backend is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data.get("image")  # Expecting a base64 encoded image

        if not image_data:
            return jsonify({"error": "No image provided"}), 400

        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Load as RGB
        img = cv2.resize(img, (64, 64))               # Resize               # Expecting 3 channels
        img = img / 255.0 
        img = img.reshape(1, -1)


        # Predict
        prediction = model.predict(img)
        predicted_class = gesture_classes[np.argmax(prediction)]

        return jsonify({"label": predicted_class})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
