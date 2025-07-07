# scripts/extract_both_hands.py

import os
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# CONFIG
INPUT_DIR = Path("data/augmented_crops")
OUTPUT_DIR = Path("data/hand_landmark_sequences")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

# Loop over sign folders
for sign_folder in tqdm(list(INPUT_DIR.iterdir()), desc="Signs"):
    if not sign_folder.is_dir():
        continue

    frames = []

    # Sort frames to keep order
    frame_files = sorted(sign_folder.glob("*.jpg"))

    for img_file in frame_files:
        img = cv2.imread(str(img_file))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # Init both hands as zeros
        left_hand = np.zeros((21, 3))
        right_hand = np.zeros((21, 3))

        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = results.multi_handedness[idx].classification[0].label
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
                landmarks = np.array(landmarks)

                if label == "Left":
                    left_hand = landmarks
                else:
                    right_hand = landmarks

        frame_landmarks = np.stack([left_hand, right_hand])  # shape: (2, 21, 3)
        frames.append(frame_landmarks)

    frames = np.stack(frames)  # shape: (num_frames, 2, 21, 3)

    out_path = OUTPUT_DIR / f"{sign_folder.name}.npy"
    np.save(str(out_path), frames)

hands.close()
print(f"Saved multi-hand sequences in {OUTPUT_DIR}")
