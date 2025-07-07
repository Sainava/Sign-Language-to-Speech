from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from web_app.model_def import SignLanguageModel

import cv2
import numpy as np
import torch
from collections import deque
from ultralytics import YOLO
import mediapipe as mp
import base64

# ------------------------------
# App init & static mount
# ------------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="web_app/static"), name="static")

# ------------------------------
# Load YOLOv8, TorchScript model & Mediapipe Holistic
# ------------------------------
yolo_model = YOLO("models/yolo11n.pt")

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = SignLanguageModel(
    pose_dim=4,
    face_dim=3,
    hand_dim=3,
    cnn_out=128,          
    lstm_hidden=160,     
    bidirectional=True,   
    num_classes=20
)
model.load_state_dict(torch.load("models/best_web_model.pth", map_location=device))
model.eval()


mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Label encoder for classes
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.classes_ = np.array([
    'alright', 'bad', 'beautiful', 'blind', 'deaf', 'good',
    'good_afternoon', 'good_morning', 'happy', 'he', 'hello',
    'how_are_you', 'i', 'it', 'loud', 'no_sign', 'quiet',
    'sad', 'she', 'you'
])

# ------------------------------
# Buffers for sequence data
# ------------------------------
SEQUENCE_LENGTH = 64
pose_seq, face_seq, lhand_seq, rhand_seq = [deque(maxlen=SEQUENCE_LENGTH) for _ in range(4)]
pose_mask_seq, face_mask_seq, lhand_mask_seq, rhand_mask_seq = [deque(maxlen=SEQUENCE_LENGTH) for _ in range(4)]

CONFIDENCE_THRESHOLD = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ------------------------------
# Request schema
# ------------------------------
class FrameRequest(BaseModel):
    image: str  # base64-encoded PNG from client

# ------------------------------
# Serve index.html
# ------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("web_app/static/index.html") as f:
        html = f.read()
    return HTMLResponse(content=html)

# ------------------------------
# Predict API
# ------------------------------
@app.post("/predict")
async def predict(req: FrameRequest):
    # Decode base64 image to OpenCV
    img_data = base64.b64decode(req.image.split(",")[1])
    np_img = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    h, w, _ = frame.shape

    # Detect person with YOLOv8
    results_yolo = yolo_model.predict(source=frame, conf=0.3, classes=[0], verbose=False)
    detections = results_yolo[0].boxes.xyxy.cpu().numpy() if len(results_yolo) > 0 else []

    if len(detections) > 0:
        x1, y1, x2, y2 = detections[0].astype(int)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        cropped = frame[y1:y2, x1:x2]
    else:
        cropped = frame

    # Run Holistic on cropped region
    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = holistic.process(rgb)
    rgb.flags.writeable = True

    # Extract landmark arrays
    def extract_landmarks(landmarks, size, include_visibility=False):
        if landmarks:
            if include_visibility:
                return np.array([[l.x, l.y, l.z, l.visibility] for l in landmarks.landmark])
            else:
                return np.array([[l.x, l.y, l.z] for l in landmarks.landmark])
        else:
            return np.zeros(size)

    pose = extract_landmarks(results.pose_landmarks, (33, 4), include_visibility=True)
    face = extract_landmarks(results.face_landmarks, (468, 3))
    lhand = extract_landmarks(results.left_hand_landmarks, (21, 3))
    rhand = extract_landmarks(results.right_hand_landmarks, (21, 3))

    pose_seq.append(pose)
    face_seq.append(face)
    lhand_seq.append(lhand)
    rhand_seq.append(rhand)
    pose_mask_seq.append(1.0 if np.any(pose) else 0.0)
    face_mask_seq.append(1.0 if np.any(face) else 0.0)
    lhand_mask_seq.append(1.0 if np.any(lhand) else 0.0)
    rhand_mask_seq.append(1.0 if np.any(rhand) else 0.0)

    predicted_word = "..."

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
            logits = model(
                pose_tensor, face_tensor, lhand_tensor, rhand_tensor,
                pose_mask, face_mask, lhand_mask, rhand_mask
            )
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()

            if confidence >= CONFIDENCE_THRESHOLD:
                predicted_word = encoder.inverse_transform([pred_idx])[0]

    return JSONResponse(content={"word": predicted_word})
