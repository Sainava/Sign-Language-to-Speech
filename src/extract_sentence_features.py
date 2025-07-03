import os
import cv2
import numpy as np
import json
from tqdm import tqdm
from ultralytics import YOLO
from mediapipe.python.solutions.hands import Hands
import mediapipe as mp

# === Paths ===
VIDEO_DIR = "data/raw/ISL_CSLRT_Corpus/Videos_Sentence_Level"
GLOSS_FILE = "data/raw/ISL_CSLRT_Corpus/corpus_csv_files/ISL Corpus sign glosses.csv"
OUTPUT_FEATURE_DIR = "data/processed/hybrid_features_sentences"
OUTPUT_LABELS_DIR = "data/processed/hybrid_labels_sentences"
YOLO_MODEL_PATH = "yolov8n.pt"

# === Setup ===
os.makedirs(OUTPUT_FEATURE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)
yolo = YOLO(YOLO_MODEL_PATH)
mp_hands = Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# === Helper: Hybrid feature from a frame ===
def extract_hybrid_features(frame):
    yolo_features = []
    results = yolo(frame, verbose=False)[0]

    if results.boxes is not None and results.boxes.data.shape[0] > 0:
        for box in results.boxes.data[:3]:
            yolo_features.extend(box[:6].tolist())
    while len(yolo_features) < 18:
        yolo_features.extend([0.0] * 6)

    hand_results = mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    landmarks = []
    if hand_results.multi_hand_landmarks:
        for lm in hand_results.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    while len(landmarks) < 63:
        landmarks.extend([0.0] * 3)

    return np.array(yolo_features + landmarks)

# === Load sentence-gloss mappings ===
import pandas as pd
gloss_df = pd.read_csv(GLOSS_FILE)
sentence_to_gloss = dict(zip(gloss_df['Sentence'], gloss_df['SIGN GLOSSES']))

# === Process each sentence folder ===
print("Extracting hybrid features from sentence-level videos...")
for sentence_name in tqdm(os.listdir(VIDEO_DIR)):
    sentence_path = os.path.join(VIDEO_DIR, sentence_name)

    if not os.path.isdir(sentence_path):
        continue

    for video_file in os.listdir(sentence_path):
        if not video_file.lower().endswith(".mp4"):
            continue

        video_path = os.path.join(sentence_path, video_file)
        cap = cv2.VideoCapture(video_path)
        features = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            f = extract_hybrid_features(frame)
            features.append(f)

        cap.release()

        features = np.stack(features)
        sentence_id = os.path.splitext(video_file)[0].replace(" ", "_")
        out_feat_path = os.path.join(OUTPUT_FEATURE_DIR, f"{sentence_id}.npy")
        out_label_path = os.path.join(OUTPUT_LABELS_DIR, f"{sentence_id}.txt")

        np.save(out_feat_path, features)

        sentence_key = sentence_name.lower().strip()
        if sentence_key in sentence_to_gloss:
            label_str = sentence_to_gloss[sentence_key]
            with open(out_label_path, "w") as f:
                f.write(label_str)

print("✅ Sentence-level hybrid feature extraction complete.")
