import os
import warnings
from flask_cors import CORS
from flask import Flask, jsonify
import gdown

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
CORS(app)

# Google Drive file ID (not full link)
FILE_ID = "1uQGu9Prwp9hvSIKZTR5S8E7gRLrKKz1-"
MODEL_PATH = "keras_model.h5"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("Model downloaded.")

# Download and load model
download_model()
model = load_model(MODEL_PATH)

gesture_classes = ['A', 'B', 'C', 'D','E','F','G','H', 'Hello' ,'I', 'I Love You' ,'J','K','L','M','N','O','P','Q','R','S','T', 'Thank You','U','V','W','X','Y']

@app.route('/')
def home():
    return jsonify({"message": "Backend is running!"})

@app.route('/predict/<int:input_value>')
def predict(input_value):
    input_array = np.array([input_value]).reshape(1, -1)
    prediction = model.predict(input_array)
    predicted_class = gesture_classes[np.argmax(prediction)]
    return jsonify({"label": predicted_class})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
