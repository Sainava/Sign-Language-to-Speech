import cv2
import torch
import numpy as np
import pyttsx3
import time
import mediapipe as mp
import json
from ultralytics import YOLO
from src.model import LSTMClassifier

# === CONFIG ===
YOLO_MODEL_PATH = "yolov8n.pt"
LSTM_CHECKPOINT = "models/best_model.pt"
LABEL_MAP_PATH = "data/processed/label_map.json"
RECORD_SECONDS = 2
FPS = 10
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# === Load class labels ===
with open(LABEL_MAP_PATH) as f:
    idx_to_label = json.load(f)
CLASSES = [idx_to_label[str(i)] for i in range(len(idx_to_label))]

# === Initialize models ===
yolo_model = YOLO(YOLO_MODEL_PATH)
lstm_model = LSTMClassifier(input_dim=144, hidden_dim=512, num_classes=len(CLASSES))
lstm_model.load_state_dict(torch.load(LSTM_CHECKPOINT, map_location=torch.device('cpu')))
lstm_model.eval()

# === Initialize text-to-speech ===
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# === Mediapipe setup ===
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# === Hybrid Feature Extraction ===
def extract_hybrid_features(frame):
    # YOLO detection
    yolo_results = yolo_model(frame, verbose=False)
    detections = yolo_results[0].boxes
    yolo_feats = []
    for box in detections.data[:3]:  # max 3 detections
        yolo_feats.extend(box[:6].tolist())
    while len(yolo_feats) < 18:
        yolo_feats.extend([0.0] * 6)

    # Mediapipe landmarks
    lm_feats = []
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks[:1]:  # only 1 hand
            for lm in hand_landmarks.landmark:
                lm_feats.extend([lm.x, lm.y, lm.z])
    while len(lm_feats) < 126:
        lm_feats.extend([0.0] * 3)

    return np.array(yolo_feats + lm_feats)  # (144,)

# === Start webcam ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

print("✅ Press 'r' to record gesture for 2 seconds.")
print("❌ Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed.")
        break

    cv2.putText(frame, "Press 'r' to record | 'q' to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Sign Language Capture", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'):
        print("📹 Recording started...")
        frames = []
        start_time = time.time()
        while time.time() - start_time < RECORD_SECONDS:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            time.sleep(1 / FPS)

        print(f"📦 Captured {len(frames)} frames. Extracting features...")

        sequence = []
        for f in frames:
            features = extract_hybrid_features(f)
            sequence.append(features)

        sequence = np.stack(sequence)  # (T, 144)
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # (1, T, 144)
        lengths = torch.tensor([sequence_tensor.shape[1]])

        with torch.no_grad():
            logits = lstm_model(sequence_tensor, lengths)
            pred = torch.argmax(logits, dim=1).item()
            predicted_word = CLASSES[pred]

        print(f"🗣️ Predicted: {predicted_word}")
        engine.say(predicted_word)
        engine.runAndWait()

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
mp_hands.close()
