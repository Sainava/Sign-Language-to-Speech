# scripts/extract_frames.py

import os
import argparse
from src.preprocessing.frame_extractor import extract_frames_from_video
import cv2

def main(input_dir, output_dir, max_frames=None):
    os.makedirs(output_dir, exist_ok=True)

    # Loop over all videos in input_dir
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.mp4', '.mov', '.MOV')):
                video_path = os.path.join(root, file)
                video_name = os.path.splitext(file)[0]

                # Create subfolder for each video’s frames
                video_output_dir = os.path.join(output_dir, video_name)
                os.makedirs(video_output_dir, exist_ok=True)

                print(f"Extracting frames for {video_name}...")

                for idx, frame in enumerate(extract_frames_from_video(video_path, max_frames)):
                    frame_file = os.path.join(video_output_dir, f"frame_{idx:04d}.jpg")
                    cv2.imwrite(frame_file, frame)

                print(f"Saved frames to {video_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/raw', help="Input directory with videos")
    parser.add_argument('--output_dir', type=str, default='data/frames', help="Output directory for extracted frames")
    parser.add_argument('--max_frames', type=int, default=None, help="Optional max frames per video")

    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.max_frames)
