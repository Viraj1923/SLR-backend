import os
import warnings
import requests
from flask_cors import CORS
from flask import Flask, jsonify

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
CORS(app)

# Google Drive file ID (extracted from shareable link)
FILE_ID = "1uQGu9Prwp9hvSIKZTR5S8E7gRLrKKz1-"
MODEL_PATH = "keras_model.h5"

def download_file_from_google_drive(file_id, destination):
    """Download a large file from Google Drive with confirmation token."""
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    with open(destination, 'wb') as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# Download model only if not already present
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    download_file_from_google_drive(FILE_ID, MODEL_PATH)

# Load the model
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
