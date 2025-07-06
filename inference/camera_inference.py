import cv2
import mediapipe as mp
import numpy as np
import torch
from collections import deque
from sklearn.preprocessing import LabelEncoder
from models.sign_model import SignLanguageModel

# ----------------------------
# Model & Label Encoder
# ----------------------------

# Load LabelEncoder classes
unique_words = [
    'Alright', 'Bad', 'Beautiful', 'Blind', 'Deaf', 'Good', 'Good_afternoon',
    'Good_morning', 'He', 'Hello', 'How_are_you', 'I', 'IT', 'Loud',
    'Quiet', 'Sad', 'She', 'Ugly', 'You', 'happy'
]
encoder = LabelEncoder()
encoder.fit(unique_words)

# Setup PyTorch device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Instantiate your trained model
model = SignLanguageModel(
    pose_dim=4,
    face_dim=3,
    hand_dim=3,
    cnn_out=96,
    lstm_hidden=64,
    num_classes=len(encoder.classes_),
    bidirectional=True
).to(device)


# Load weights
model.load_state_dict(torch.load("models/best_sign_model_hp_tuned.pth", map_location=device))
model.eval()

# ----------------------------
# MediaPipe
# ----------------------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ----------------------------
# Landmark extraction helper
# ----------------------------
def extract_landmarks(frame, holistic_model):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False 
    results = holistic_model.process(rgb_frame)
    rgb_frame.flags.writeable = True

    pose_size = (33, 4)   # x, y, z, visibility
    face_size = (468, 3)  # x, y, z
    hand_size = (21, 3)

    def get_data(landmarks, size, include_visibility=False):
        if landmarks:
            if include_visibility:
                data = np.array([[l.x, l.y, l.z, l.visibility] for l in landmarks.landmark])
            else:
                data = np.array([[l.x, l.y, l.z] for l in landmarks.landmark])
        else:
            data = np.zeros(size)
        return data

    pose = get_data(results.pose_landmarks, pose_size, include_visibility=True)
    face = get_data(results.face_landmarks, face_size)
    lhand = get_data(results.left_hand_landmarks, hand_size)
    rhand = get_data(results.right_hand_landmarks, hand_size)

    return pose, face, lhand, rhand, results

# ----------------------------
# Rolling buffer
# ----------------------------
SEQUENCE_LENGTH = 60
pose_seq = deque(maxlen=SEQUENCE_LENGTH)
face_seq = deque(maxlen=SEQUENCE_LENGTH)
lhand_seq = deque(maxlen=SEQUENCE_LENGTH)
rhand_seq = deque(maxlen=SEQUENCE_LENGTH)

pose_mask_seq = deque(maxlen=SEQUENCE_LENGTH)
face_mask_seq = deque(maxlen=SEQUENCE_LENGTH)
lhand_mask_seq = deque(maxlen=SEQUENCE_LENGTH)
rhand_mask_seq = deque(maxlen=SEQUENCE_LENGTH)

# ----------------------------
# Main loop
# ----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

with mp_holistic.Holistic(
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    predicted_word = "..."
    confidence_threshold = 0.5

    print("Starting live inference. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        pose, face, lhand, rhand, results = extract_landmarks(frame, holistic)

        pose_seq.append(pose)
        face_seq.append(face)
        lhand_seq.append(lhand)
        rhand_seq.append(rhand)

        pose_mask_seq.append(1.0 if np.any(pose) else 0.0)
        face_mask_seq.append(1.0 if np.any(face) else 0.0)
        lhand_mask_seq.append(1.0 if np.any(lhand) else 0.0)
        rhand_mask_seq.append(1.0 if np.any(rhand) else 0.0)

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

                if confidence > confidence_threshold:
                    predicted_word = encoder.inverse_transform([pred_idx])[0].upper()
                    print(f"Predicted: {predicted_word} ({confidence:.2f})")

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.putText(frame, f"{predicted_word}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow("Live Sign Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
