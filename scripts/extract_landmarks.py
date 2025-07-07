#extract_landmarks.py

import os
import cv2
import numpy as np
from tqdm import tqdm
from src.landmarks.landmark_extractor import MediaPipeHolisticExtractor

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--crops_dir", type=str, required=True, help="Path to YOLO cropped frames dir")
parser.add_argument("--output_dir", type=str, required=True, help="Path to save output .npy files")
args = parser.parse_args()

crops_dir = args.crops_dir
output_dir = args.output_dir

extractor = MediaPipeHolisticExtractor()

X_pose = []
X_face = []
X_lhand = []
X_rhand = []
y = []

for sign_folder in tqdm(os.listdir(crops_dir)):
    folder_path = os.path.join(crops_dir, sign_folder)
    if not os.path.isdir(folder_path):
        continue

    if '_' in sign_folder and sign_folder[-3] == '_' and sign_folder[-2:].isdigit():
        label = sign_folder[:-3].lower()
    else:
        label = sign_folder.lower()

    frame_files = sorted(os.listdir(folder_path))

    pose_frames = []
    face_frames = []
    lhand_frames = []
    rhand_frames = []

    for frame_file in frame_files:
        frame_path = os.path.join(folder_path, frame_file)
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"Warning: could not read {frame_path}")
            continue

        landmarks = extractor.extract_from_frame(frame)

        pose_frames.append(landmarks['pose'])
        face_frames.append(landmarks['face'])
        lhand_frames.append(landmarks['left_hand'])
        rhand_frames.append(landmarks['right_hand'])

    if len(pose_frames) == 0:
        continue

    X_pose.append(np.stack(pose_frames))
    X_face.append(np.stack(face_frames))
    X_lhand.append(np.stack(lhand_frames))
    X_rhand.append(np.stack(rhand_frames))
    y.append(label)

extractor.close()

X_pose = np.stack(X_pose)
X_face = np.stack(X_face)
X_lhand = np.stack(X_lhand)
X_rhand = np.stack(X_rhand)
y = np.array(y)

os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, "X_pose.npy"), X_pose)
np.save(os.path.join(output_dir, "X_face.npy"), X_face)
np.save(os.path.join(output_dir, "X_lhand.npy"), X_lhand)
np.save(os.path.join(output_dir, "X_rhand.npy"), X_rhand)
np.save(os.path.join(output_dir, "y.npy"), y)

print(f"Saved pose: {X_pose.shape}")
print(f"Saved face: {X_face.shape}")
print(f"Saved left hand: {X_lhand.shape}")
print(f"Saved right hand: {X_rhand.shape}")
print(f"Saved labels: {y.shape}")
