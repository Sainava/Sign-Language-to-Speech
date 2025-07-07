from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import torch
from collections import deque
from ultralytics import YOLO
import mediapipe as mp
import base64

from web_app.model_def import SignLanguageModel
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

app.mount("/static", StaticFiles(directory="web_app/static"), name="static")

# === Load YOLO ===
yolo_model = YOLO("models/yolo11n.pt")

# === Load SignLanguageModel ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignLanguageModel(
    cnn_out=128,
    lstm_hidden=160,
    num_classes=20,
    bidirectional=True
).to(device)
model.load_state_dict(torch.load("models/best_web_model.pth", map_location=device))
model.eval()

# === Mediapipe ===
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# === Label encoder ===
encoder = LabelEncoder()
encoder.classes_ = np.array([
    'alright', 'bad', 'beautiful', 'blind', 'deaf', 'good',
    'good_afternoon', 'good_morning', 'happy', 'he', 'hello',
    'how_are_you', 'i', 'it', 'loud', 'no_sign', 'quiet',
    'sad', 'she', 'you'
])

SEQUENCE_LENGTH = 64
CONFIDENCE_THRESHOLD = 0.5

pose_seq = deque(maxlen=SEQUENCE_LENGTH)
face_seq = deque(maxlen=SEQUENCE_LENGTH)
lhand_seq = deque(maxlen=SEQUENCE_LENGTH)
rhand_seq = deque(maxlen=SEQUENCE_LENGTH)
pose_mask_seq = deque(maxlen=SEQUENCE_LENGTH)
face_mask_seq = deque(maxlen=SEQUENCE_LENGTH)
lhand_mask_seq = deque(maxlen=SEQUENCE_LENGTH)
rhand_mask_seq = deque(maxlen=SEQUENCE_LENGTH)

inference_enabled = False  # <-- FLAG

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("web_app/static/index.html") as f:
        return HTMLResponse(f.read())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global inference_enabled

    await websocket.accept()
    last_final_word = "..."

    try:
        while True:
            data = await websocket.receive_text()

            if data == "RESET" or data == "START":
                # Always clear buffers for both
                pose_seq.clear()
                face_seq.clear()
                lhand_seq.clear()
                rhand_seq.clear()
                pose_mask_seq.clear()
                face_mask_seq.clear()
                lhand_mask_seq.clear()
                rhand_mask_seq.clear()
                last_final_word = "..."

                if data == "START":
                    inference_enabled = True
                    print("[INFO] Inference triggered.")
                else:
                    inference_enabled = False
                    print("[INFO] Buffer cleared manually.")

                continue

            # === Otherwise treat as frame ===
            img_data = base64.b64decode(data.split(",")[1])
            np_img = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            h, w, _ = frame.shape

            # === YOLO detects person ===
            # results_yolo = yolo_model.predict(
            #     source=frame,
            #     conf=0.3,
            #     classes=[0],
            #     verbose=False
            # )
            # detections = results_yolo[0].boxes.xyxy.cpu().numpy() if len(results_yolo) > 0 else []

            # if len(detections) > 0:
            #     x1, y1, x2, y2 = detections[0].astype(int)
            #     x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            #     cropped = frame[y1:y2, x1:x2]
            # else:
            #     cropped = frame

            cropped = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            def get_data(landmarks, size, vis=False):
                if landmarks:
                    if vis:
                        return np.array([[l.x, l.y, l.z, l.visibility] for l in landmarks.landmark])
                    else:
                        return np.array([[l.x, l.y, l.z] for l in landmarks.landmark])
                else:
                    return np.zeros(size)

            pose = get_data(results.pose_landmarks, (33, 4), True)
            face = get_data(results.face_landmarks, (468, 3))
            lhand = get_data(results.left_hand_landmarks, (21, 3))
            rhand = get_data(results.right_hand_landmarks, (21, 3))

            pose_seq.append(pose)
            face_seq.append(face)
            lhand_seq.append(lhand)
            rhand_seq.append(rhand)
            pose_mask_seq.append(1.0 if np.any(pose) else 0.0)
            face_mask_seq.append(1.0 if np.any(face) else 0.0)
            lhand_mask_seq.append(1.0 if np.any(lhand) else 0.0)
            rhand_mask_seq.append(1.0 if np.any(rhand) else 0.0)

            if inference_enabled and len(pose_seq) == SEQUENCE_LENGTH:
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
                    else:
                        predicted_word = "no_sign"

                await websocket.send_json({"word": predicted_word})
                inference_enabled = False  # Done until next START

    except WebSocketDisconnect:
        print("[INFO] WebSocket disconnected cleanly.")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if not websocket.client_state.name == "DISCONNECTED":
            await websocket.close()
