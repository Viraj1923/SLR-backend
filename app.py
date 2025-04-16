import os
import cv2
import numpy as np
import gdown
import mediapipe as mp
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from starlette.background import BackgroundTask

# Constants
IMAGE_SIZE = 224
MODEL_PATH = "sign_language_model.h5"
GDRIVE_FILE_ID = "1L0pN3R6ldTO_2h2XNZ4HVxE2YzZ2w_2u"  # 🔁 Replace with your file ID
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("[INFO] Downloading model from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# Load gesture labels
gesture_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Hello', 'I', 'I Love You',
                   'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'Thank You',
                   'U', 'V', 'W', 'X', 'Y']
label_dict = {idx: name for idx, name in enumerate(gesture_classes)}

# Load model
print("[INFO] Loading model...")
model = load_model(MODEL_PATH)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# FastAPI setup
app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Camera capture
cap = cv2.VideoCapture(0)
detected_label = "No Detection"

def generate_frames():
    global detected_label
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                h, w, _ = frame.shape
                x_min, y_min, x_max, y_max = w, h, 0, 0

                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min, y_min = min(x_min, x), min(y_min, y)
                    x_max, y_max = max(x_max, x), max(y_max, y)

                padding = 20
                x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
                x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)

                hand_img = frame[y_min:y_max, x_min:x_max]
                if hand_img.size == 0:
                    continue

                hand_img = cv2.resize(hand_img, (IMAGE_SIZE, IMAGE_SIZE))
                hand_img = hand_img.astype("float32") / 255.0
                hand_img = np.expand_dims(hand_img, axis=0)

                predictions = model.predict(hand_img, verbose=0)
                predicted_index = (np.argmax(predictions) - 1) % len(gesture_classes)
                detected_label = label_dict.get(predicted_index, "Unknown")

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, detected_label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/get_label")
def get_label():
    return JSONResponse({"label": detected_label})
