import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

# === Configurations ===
EXCEL_PATH = "data/raw/ISL_CSLRT_Corpus/corpus_csv_files/ISL_CSLRT_Corpus_word_details.xlsx"
FRAME_ROOT = "data/raw/ISL_CSLRT_Corpus/Frames_Word_Level"
SAVE_DIR = "data/processed/features"
MODEL_PATH = "yolov8n.pt"  # You can replace with yolov8s.pt or yolov8x.pt if you want

# === Create save directory if not exists ===
os.makedirs(SAVE_DIR, exist_ok=True)

# === Load metadata ===
df = pd.read_excel(EXCEL_PATH)
print("Loaded samples:", len(df))

# === Initialize YOLO model ===
model = YOLO(MODEL_PATH)

# === Helper: Run YOLO on image and return a flattened feature vector ===
def extract_features_from_frame(frame):
    results = model(frame, verbose=False)
    detections = results[0].boxes

    if detections is None or detections.shape[0] == 0:
        return np.zeros(20)  # fallback if no detection

    # Extract up to N boxes (x1, y1, x2, y2, conf, class)
    features = []
    for box in detections.data[:3]:  # max 3 detections (hands/face/body)
        features.extend(box[:6].tolist())

    # Pad with zeros if fewer than 3 detections
    while len(features) < 18:
        features.extend([0.0] * 6)

    return np.array(features)

# === Loop over all rows ===
for i, row in tqdm(df.iterrows(), total=len(df)):
    label = row['Word']
    raw_path = row['Frames path']

    # Normalize path from Windows-style to POSIX
    # Fix Windows path and extract just the folder part (up to the word)
    rel_path = raw_path.replace("ISL_CSLRT_Corpus\\", "").replace("\\", "/")
    # Only keep the folder up to the word (first two segments)
    folder_parts = rel_path.split('/')
    full_dir = os.path.join("data/raw/ISL_CSLRT_Corpus", "Frames_Word_Level", folder_parts[1])

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
        features = extract_features_from_frame(frame)
        sequence_features.append(features)

    if len(sequence_features) == 0:
        continue

    # Check that all feature vectors have the same shape
    first_shape = sequence_features[0].shape
    if not all(f.shape == first_shape for f in sequence_features):
        print(f"[WARN] Inconsistent feature shapes at index {i}, skipping.")
        continue  # Skip this sample

    # Save sequence as numpy array
    features_np = np.stack(sequence_features)  # Shape: (T, F)
    save_path = os.path.join(SAVE_DIR, f"sample_{i}_{label.replace(' ', '_')}.npy")
    np.save(save_path, features_np)

print("\n✅ Feature extraction complete. Saved to:", SAVE_DIR)
