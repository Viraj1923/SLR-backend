from fastapi import FastAPI
from pydantic import BaseModel
import base64
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from io import BytesIO

app = FastAPI()

# Load your trained model
model = tf.keras.models.load_model("your_model_path.h5")  # or .tflite loader if using TFLite

# Your gesture labels
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Hello', 'I', 'I Love You', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'Thank You', 'U', 'V', 'W', 'X', 'Y']

class ImageRequest(BaseModel):
    image: str  # base64 encoded image

@app.post("/predict")
def predict(data: ImageRequest):
    # Decode base64 image
    image_data = base64.b64decode(data.image.split(',')[-1])
    img = Image.open(BytesIO(image_data)).convert("RGB")
    
    # Resize to model input (224x224 in your case)
    img = img.resize((224, 224))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    predictions = model.predict(img)
    predicted_index = np.argmax(predictions[0])
    predicted_label = labels[predicted_index]

    return {"label": predicted_label}
