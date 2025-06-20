import cv2
import torch
import numpy as np
import pyttsx3
import time
from ultralytics import YOLO
from src.model import LSTMClassifier  
import json

# === CONFIG ===
YOLO_MODEL_PATH = "yolov8n.pt"
LSTM_CHECKPOINT = "models/best_model.pt"
RECORD_SECONDS = 2
FPS = 10
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


LABEL_MAP_PATH = "data/processed/label_map.json"

with open(LABEL_MAP_PATH) as f:
    idx_to_label = json.load(f)

# Sort keys numerically because JSON keys are strings like "0", "1", ...
CLASSES = [idx_to_label[str(i)] for i in range(len(idx_to_label))]

# print("Loaded class labels:")
# for i, label in enumerate(CLASSES):
#     print(f"{i}: {label}")


# === Initialize models ===
yolo_model = YOLO(YOLO_MODEL_PATH)

lstm_model = LSTMClassifier(input_dim=18, hidden_dim=512, num_classes=len(CLASSES))
lstm_model.load_state_dict(torch.load(LSTM_CHECKPOINT, map_location=torch.device('cpu')))
lstm_model.eval()

# === Initialize text-to-speech ===
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# === Feature extraction ===
def extract_features_from_frame(frame):
    results = yolo_model(frame, verbose=False)
    detections = results[0].boxes

    if detections is None or detections.shape[0] == 0:
        return np.zeros(18)

    features = []
    for box in detections.data[:3]:  # max 3 detections
        features.extend(box[:6].tolist())

    while len(features) < 18:
        features.extend([0.0] * 6)

    return np.array(features)

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
            features = extract_features_from_frame(f)
            sequence.append(features)

        sequence = np.stack(sequence)  # (T, 18)
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # (1, T, 18)
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
