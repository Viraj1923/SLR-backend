import os
import warnings
import requests
from flask_cors import CORS
from flask import Flask, jsonify

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses TensorFlow warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow frontend to access backend

# Google Drive model link (change FILE_ID)
MODEL_URL = "https://drive.google.com/file/d/1uQGu9Prwp9hvSIKZTR5S8E7gRLrKKz1-/view?usp=sharing"  

MODEL_PATH = "keras_model.h5"

def download_model():
    """Download the model from Google Drive if not already present."""
    if not os.path.exists(MODEL_PATH):
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as file:
            file.write(response.content)

# Download model before running
download_model()

# Load trained model
model = load_model(MODEL_PATH)

gesture_classes = ['A', 'B', 'C', 'D','E','F','G','H', 'Hello' ,'I', 'I Love You' ,'J','K','L','M','N','O','P','Q','R','S','T', 'Thank You','U','V','W','X','Y']

@app.route('/')
def home():
    return jsonify({"message": "Backend is running!"})

@app.route('/predict/<int:input_value>')
def predict(input_value):
    """Dummy prediction API (Modify this based on real input)."""
    input_array = np.array([input_value]).reshape(1, -1)
    prediction = model.predict(input_array)
    predicted_class = gesture_classes[np.argmax(prediction)]
    
    return jsonify({"label": predicted_class})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Get Render-assigned port
    app.run(host="0.0.0.0", port=port)
