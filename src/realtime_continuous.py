import cv2
import torch
import numpy as np
import time
import collections
import mediapipe as mp
import json
from ultralytics import YOLO
from src.model import LSTMClassifier

# === CONFIG ===
YOLO_MODEL_PATH = "yolov8n.pt"
LSTM_CHECKPOINT = "models/best_model.pt"
LABEL_MAP_PATH = "data/processed/label_map.json"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Sliding window settings
WINDOW_SIZE = 32        # Number of frames in buffer
STRIDE = 8              # Run inference every STRIDE frames
THRESHOLD = 0.5         # Confidence threshold for display (optional)

# === Load class labels ===
with open(LABEL_MAP_PATH) as f:
    idx_to_label = json.load(f)
CLASSES = [idx_to_label[str(i)] for i in range(len(idx_to_label))]

# === Initialize models ===
yolo_model = YOLO(YOLO_MODEL_PATH)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lstm_model = LSTMClassifier(input_dim=144, hidden_dim=512, num_classes=len(CLASSES))
lstm_model.load_state_dict(torch.load(LSTM_CHECKPOINT, map_location=device))
lstm_model.to(device).eval()

# === Mediapipe Hands ===
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# === Hybrid feature extractor ===
def extract_hybrid_features(frame):
    # YOLO features
    results = yolo_model(frame, verbose=False)
    detections = results[0].boxes
    yolo_feats = []
    for box in detections.data[:3]:
        yolo_feats.extend(box[:6].tolist())
    while len(yolo_feats) < 18:
        yolo_feats.extend([0.0]*6)
    # MediaPipe landmarks
    lm_feats = []
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mp_hands.process(rgb)
    if res.multi_hand_landmarks:
        for hand_landmarks in res.multi_hand_landmarks[:1]:
            for lm in hand_landmarks.landmark:
                lm_feats.extend([lm.x, lm.y, lm.z])
    while len(lm_feats) < 126:
        lm_feats.extend([0.0]*3)
    return np.array(yolo_feats + lm_feats)

# === Initialize webcam and buffer ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
buffer = collections.deque(maxlen=WINDOW_SIZE)
frame_count = 0

print("Starting continuous real-time sign recognition...")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Extract features and append to buffer
    feat = extract_hybrid_features(frame)
    buffer.append(feat)
    frame_count += 1

    # Every STRIDE frames, run inference if buffer is full
    predicted = None
    if frame_count % STRIDE == 0 and len(buffer) == WINDOW_SIZE:
        seq = np.stack(buffer)                     # (WINDOW_SIZE, 144)
        tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1, W, 144)
        lengths = torch.tensor([WINDOW_SIZE]).to(device)
        with torch.no_grad():
            logits = lstm_model(tensor, lengths)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            idx = np.argmax(probs)
            if probs[idx] >= THRESHOLD:
                predicted = CLASSES[idx]
            else:
                predicted = None

    # Display result
    display_text = predicted if predicted else "..."
    cv2.putText(frame, f"Prediction: {display_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0) if predicted else (0, 0, 255), 2)
    cv2.imshow("Continuous ISL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
