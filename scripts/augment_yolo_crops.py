# scripts/augment_yolo_crops.py

import os
import cv2
import random
import numpy as np

from pathlib import Path
from tqdm import tqdm

# -------- CONFIG --------
INPUT_DIR = Path("data/yolo_crops")
OUTPUT_DIR = Path("data/augmented_crops")
AUG_PER_IMAGE = 5  # How many new images per original?

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def random_augment(image):
    # 1. Flip
    if random.random() < 0.5:
        image = cv2.flip(image, 1)

    # 2. Rotate
    angle = random.uniform(-10, 10)
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    image = cv2.warpAffine(image, M, (w, h))

    # 3. Brightness
    value = random.randint(-30, 30)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], value)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return image

# -------- MAIN --------
sign_folders = [f for f in INPUT_DIR.iterdir() if f.is_dir()]

for sign_folder in tqdm(sign_folders, desc="Signs"):
    out_folder = OUTPUT_DIR / sign_folder.name
    out_folder.mkdir(parents=True, exist_ok=True)

    frames = list(sign_folder.glob("*.jpg"))
    for frame_path in tqdm(frames, desc=f"{sign_folder.name}", leave=False):
        img = cv2.imread(str(frame_path))

        for i in range(AUG_PER_IMAGE):
            aug_img = random_augment(img)
            out_name = f"{frame_path.stem}_aug_{i}.jpg"
            cv2.imwrite(str(out_folder / out_name), aug_img)

print(f"Augmented images saved in {OUTPUT_DIR}")
