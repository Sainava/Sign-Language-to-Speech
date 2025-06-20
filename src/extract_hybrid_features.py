import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
import mediapipe as mp

# === Configurations ===
EXCEL_PATH = "data/raw/ISL_CSLRT_Corpus/corpus_csv_files/ISL_CSLRT_Corpus_word_details.xlsx"
FRAME_ROOT = "data/raw/ISL_CSLRT_Corpus/Frames_Word_Level"
SAVE_DIR = "data/processed/hybrid_features"
MODEL_PATH = "yolov8n.pt"

# === Create save directory if not exists ===
os.makedirs(SAVE_DIR, exist_ok=True)

# === Load metadata ===
df = pd.read_excel(EXCEL_PATH)
print("Loaded samples:", len(df))

# === Initialize YOLO model ===
yolo = YOLO(MODEL_PATH)

# === Initialize MediaPipe Hands ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# === Helper: YOLO features ===
def extract_yolo_features(frame):
    results = yolo(frame, verbose=False)
    detections = results[0].boxes

    if detections is None or detections.shape[0] == 0:
        return np.zeros(18)

    features = []
    for box in detections.data[:3]:  # max 3 detections
        features.extend(box[:6].tolist())

    while len(features) < 18:
        features.extend([0.0] * 6)

    return np.array(features)

# === Helper: MediaPipe hand landmarks ===
def extract_hand_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if not results.multi_hand_landmarks:
        return np.zeros(63 * 2)  # 2 hands * 21 landmarks * (x,y,z) ~= 126

    features = []
    for hand_landmarks in results.multi_hand_landmarks[:2]:
        for lm in hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])

    while len(features) < 126:
        features.extend([0.0] * 3)

    return np.array(features)

# === Combine YOLO and hand features ===
def extract_combined_features(frame):
    yolo_feat = extract_yolo_features(frame)
    hand_feat = extract_hand_landmarks(frame)
    return np.concatenate([yolo_feat, hand_feat])

# === Loop over all rows ===
for i, row in tqdm(df.iterrows(), total=len(df)):
    label = row['Word']
    raw_path = row['Frames path']
    rel_path = raw_path.replace("ISL_CSLRT_Corpus\\", "").replace("\\", "/")
    folder_parts = rel_path.split('/')
    if len(folder_parts) < 2:
        print(f"[!] Skipping badly formatted path: {raw_path}")
        continue
    full_dir = os.path.join(FRAME_ROOT, folder_parts[1])

    if not os.path.exists(full_dir):
        print(f"[!] Skipping missing folder: {full_dir}")
        continue

    frame_files = sorted([f for f in os.listdir(full_dir) if f.endswith(".jpg")])
    if len(frame_files) == 0:
        print(f"[!] No frames in: {full_dir}")
        continue

    sequence_features = []
    for fname in frame_files:
        frame_path = os.path.join(full_dir, fname)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        features = extract_combined_features(frame)
        sequence_features.append(features)

    if len(sequence_features) == 0:
        continue

    first_shape = sequence_features[0].shape
    if not all(f.shape == first_shape for f in sequence_features):
        print(f"[WARN] Inconsistent feature shapes at index {i}, skipping.")
        continue

    features_np = np.stack(sequence_features)
    save_path = os.path.join(SAVE_DIR, f"sample_{i}_{label.replace(' ', '_')}.npy")
    np.save(save_path, features_np)

print("\n✅ Hybrid feature extraction complete. Saved to:", SAVE_DIR)
