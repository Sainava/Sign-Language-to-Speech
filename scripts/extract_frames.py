
import os
import argparse
from tqdm import tqdm
import cv2

from src.preprocessing.frame_extractor import extract_frames_from_video

def main(input_dir, output_dir, max_frames=None):
    os.makedirs(output_dir, exist_ok=True)

    # Collect all videos first
    video_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.mp4', '.mov', '.MOV')):
                video_files.append(os.path.join(root, file))

    print(f"Found {len(video_files)} videos.")

    # Loop over videos with tqdm
    for video_path in tqdm(video_files, desc="Processing Videos"):
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        video_output_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        print(f"\nExtracting frames for {video_name}...")

        # Wrap frame extraction with tqdm
        for idx, frame in enumerate(
            tqdm(
                extract_frames_from_video(video_path, max_frames),
                desc=f"Frames ({video_name})",
                leave=False
            )
        ):
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
