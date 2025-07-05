import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from src.landmarks.landmark_extractor import MediaPipeHolisticExtractor

def normalize_label(raw_folder_name):
    """
    Normalize folder name: 
    E.g., 'Good_afternoon_01' -> 'Good afternoon'
    """
    base = raw_folder_name.rsplit('_', 1)[0]  # removes trailing number
    return base.replace('_', ' ')

def process_cropped_frames(crops_dir, output_dir, sequence_length=60):
    extractor = MediaPipeHolisticExtractor(sequence_length=sequence_length)
    X, y = [], []

    images = [
        os.path.join(dp, f)
        for dp, dn, filenames in os.walk(crops_dir)
        for f in filenames if f.endswith('.jpg')
    ]

    videos_by_folder = {}
    for path in images:
        parts = path.split(os.sep)
        video = parts[-2]  # e.g., 'Good_afternoon_01'
        word = normalize_label(video)  # ✅ FIXED
        videos_by_folder.setdefault((word, video), []).append(path)

    for (word, video), frames in tqdm(videos_by_folder.items(), desc="Processing Cropped Frames"):
        frames = sorted(frames)
        sequence = []

        for frame_path in frames[:sequence_length]:
            frame = cv2.imread(frame_path)
            landmarks = extractor.extract_from_frame(frame)
            sequence.append(landmarks)

        if len(sequence) < sequence_length:
            padding = sequence_length - len(sequence)
            sequence += [np.zeros_like(sequence[0]) for _ in range(padding)]

        X.append(sequence)
        y.append(word)

    X = np.array(X)  # shape: (N, T, L, 4)
    y = np.array(y)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'y.npy'), y)

    extractor.close()

    print(f"Saved landmarks → {output_dir}")
    print(f"Shape of X: {X.shape} | y: {y.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--crops_dir', type=str, default='data/yolo_crops', help="Input dir with YOLO crops")
    parser.add_argument('--output_dir', type=str, default='data/landmarks', help="Output dir for landmarks")
    parser.add_argument('--sequence_length', type=int, default=60)

    args = parser.parse_args()
    process_cropped_frames(args.crops_dir, args.output_dir, args.sequence_length)
