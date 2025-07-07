import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from sklearn.preprocessing import LabelEncoder
from collections import deque
from models.sign_model import SignLanguageModel
import mediapipe as mp

# ----------------------------
# Config
# ----------------------------
unique_words = [
    'alright', 'bad', 'beautiful', 'blind', 'deaf', 'good', 'good_afternoon',
    'good_morning', 'happy', 'he', 'hello', 'how_are_you',
    'i', 'it', 'loud', 'no_sign', 'quiet', 'sad', 'she', 'you'
]
SEQUENCE_LENGTH = 64  # Your training sequence length
CONFIDENCE_THRESHOLD = 0.5

encoder = LabelEncoder()
encoder.fit(unique_words)

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

model = SignLanguageModel(
    cnn_out=128,
    lstm_hidden=160,
    num_classes=len(encoder.classes_),
    bidirectional=True
).to(device)

model.classifier = nn.Sequential(
    nn.Linear(160 * 2 * 4, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, len(encoder.classes_))
).to(device)

model.load_state_dict(torch.load("models/best_web_model.pth", map_location=device))
model.eval()


# ----------------------------
# YOLO & MediaPipe
# ----------------------------
yolo_model = YOLO("models/yolo11n.pt")
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ----------------------------
# Buffers
# ----------------------------
pose_seq = deque(maxlen=SEQUENCE_LENGTH)
face_seq = deque(maxlen=SEQUENCE_LENGTH)
lhand_seq = deque(maxlen=SEQUENCE_LENGTH)
rhand_seq = deque(maxlen=SEQUENCE_LENGTH)

pose_mask_seq = deque(maxlen=SEQUENCE_LENGTH)
face_mask_seq = deque(maxlen=SEQUENCE_LENGTH)
lhand_mask_seq = deque(maxlen=SEQUENCE_LENGTH)
rhand_mask_seq = deque(maxlen=SEQUENCE_LENGTH)

# ----------------------------
# Helper: Extract
# ----------------------------
def extract_landmarks(frame, holistic):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = holistic.process(rgb)
    rgb.flags.writeable = True

    def get_data(landmarks, size, include_visibility=False):
        if landmarks:
            if include_visibility:
                return np.array([[l.x, l.y, l.z, l.visibility] for l in landmarks.landmark])
            else:
                return np.array([[l.x, l.y, l.z] for l in landmarks.landmark])
        else:
            return np.zeros(size)

    pose = get_data(results.pose_landmarks, (33, 4), include_visibility=True)
    face = get_data(results.face_landmarks, (468, 3))
    lhand = get_data(results.left_hand_landmarks, (21, 3))
    rhand = get_data(results.right_hand_landmarks, (21, 3))

    return pose, face, lhand, rhand, results

# ----------------------------
# Main
# ----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Webcam not found.")
    exit()

with mp_holistic.Holistic(
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    predicted_word = "..."

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # ----------------------------
        # YOLO: detect person box
        # ----------------------------
        results_yolo = yolo_model.predict(source=frame, conf=0.3, classes=[0], verbose=False)
        detections = results_yolo[0].boxes.xyxy.cpu().numpy() if len(results_yolo) > 0 else []

        if len(detections) > 0:
            x1, y1, x2, y2 = detections[0].astype(int)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            cropped = frame[y1:y2, x1:x2]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            cropped = frame  # fallback

        pose, face, lhand, rhand, results = extract_landmarks(cropped, holistic)

        pose_seq.append(pose)
        face_seq.append(face)
        lhand_seq.append(lhand)
        rhand_seq.append(rhand)

        pose_mask_seq.append(1.0 if np.any(pose) else 0.0)
        face_mask_seq.append(1.0 if np.any(face) else 0.0)
        lhand_mask_seq.append(1.0 if np.any(lhand) else 0.0)
        rhand_mask_seq.append(1.0 if np.any(rhand) else 0.0)

        # ----------------------------
        # Predict when buffer full
        # ----------------------------
        if len(pose_seq) == SEQUENCE_LENGTH:
            pose_tensor = torch.tensor(np.stack(pose_seq), dtype=torch.float32).unsqueeze(0).to(device)
            face_tensor = torch.tensor(np.stack(face_seq), dtype=torch.float32).unsqueeze(0).to(device)
            lhand_tensor = torch.tensor(np.stack(lhand_seq), dtype=torch.float32).unsqueeze(0).to(device)
            rhand_tensor = torch.tensor(np.stack(rhand_seq), dtype=torch.float32).unsqueeze(0).to(device)

            pose_mask = torch.tensor(np.stack(pose_mask_seq), dtype=torch.float32).unsqueeze(0).to(device)
            face_mask = torch.tensor(np.stack(face_mask_seq), dtype=torch.float32).unsqueeze(0).to(device)
            lhand_mask = torch.tensor(np.stack(lhand_mask_seq), dtype=torch.float32).unsqueeze(0).to(device)
            rhand_mask = torch.tensor(np.stack(rhand_mask_seq), dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(pose_tensor, face_tensor, lhand_tensor, rhand_tensor,
                               pose_mask, face_mask, lhand_mask, rhand_mask)
                probs = torch.softmax(logits, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_idx].item()

                if confidence >= CONFIDENCE_THRESHOLD:
                    predicted_word = encoder.inverse_transform([pred_idx])[0].upper()
                    print(f"[PREDICT] {predicted_word} ({confidence:.2f})")

        # ----------------------------
        # Visualize
        # ----------------------------
        mp_drawing.draw_landmarks(cropped, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(cropped, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        mp_drawing.draw_landmarks(cropped, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(cropped, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cropped_resized = cv2.resize(cropped, (w, h))
        combined = np.hstack((frame, cropped_resized))

        cv2.putText(combined, f"{predicted_word}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow("YOLO + Holistic", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
