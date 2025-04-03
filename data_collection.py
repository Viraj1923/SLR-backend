import cv2
import os
import numpy as np
import mediapipe as mp

# Constants
MAX_IMAGES = 200
IMAGE_SIZE = 224
DATASET_DIR = "dataset"

# Function to create gesture folder
def create_folder(gesture_name):
    gesture_folder = os.path.join(DATASET_DIR, gesture_name)
    os.makedirs(gesture_folder, exist_ok=True)
    return gesture_folder

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Webcam setup
cap = cv2.VideoCapture(0)

gesture_name = input("Enter gesture name (e.g., 'ThankYou'): ")
gesture_folder = create_folder(gesture_name)

count = 0
print(f"[INFO] Starting data collection for '{gesture_name}'. Press 's' to save, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    hand_img = None

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            x_min, y_min = frame.shape[1], frame.shape[0]
            x_max, y_max = 0, 0
            for landmark in landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            # Add padding
            padding = 20
            x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
            x_max, y_max = min(frame.shape[1], x_max + padding), min(frame.shape[0], y_max + padding)

            # Draw bounding box on the frame
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Crop and resize hand region
            hand_img = frame[y_min:y_max, x_min:x_max]
            hand_img = cv2.resize(hand_img, (IMAGE_SIZE, IMAGE_SIZE))

    cv2.putText(frame, f"Captured: {count}/{MAX_IMAGES}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and hand_img is not None:
        image_path = os.path.join(gesture_folder, f"{gesture_name}_{count}.jpg")
        cv2.imwrite(image_path, hand_img)
        count += 1
        print(f"[INFO] Captured: {image_path}")

        if count >= MAX_IMAGES:
            print(f"[INFO] Reached {MAX_IMAGES} images. Exiting...")
            break

    elif key == ord('q'):
        print("[INFO] Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
