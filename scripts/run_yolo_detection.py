# scripts/run_yolo_detection.py

import os
import cv2
from tqdm import tqdm
import argparse
from src.detection.yolo_detector import YOLODetector

def crop_and_save(frame, bbox, save_path):
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    cv2.imwrite(save_path, crop)

def main(frames_dir, output_dir, model_path="yolo11n.pt"):
    os.makedirs(output_dir, exist_ok=True)

    detector = YOLODetector(model_path=model_path, conf=0.3)

    videos = os.listdir(frames_dir)
    for video_folder in tqdm(videos, desc="Processing Videos"):
        video_input_dir = os.path.join(frames_dir, video_folder)
        video_output_dir = os.path.join(output_dir, video_folder)
        os.makedirs(video_output_dir, exist_ok=True)

        for frame_file in sorted(os.listdir(video_input_dir)):
            frame_path = os.path.join(video_input_dir, frame_file)
            frame = cv2.imread(frame_path)

            detections = detector.detect_objects(frame)

            # For sanity: Save ONLY the largest box or all boxes? Let’s do largest for now.
            if detections:
                areas = [(b[2]-b[0]) * (b[3]-b[1]) for b in detections]
                largest_box = detections[areas.index(max(areas))]
                save_path = os.path.join(video_output_dir, frame_file)
                crop_and_save(frame, largest_box, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_dir', type=str, default='data/frames', help="Folder of extracted frames")
    parser.add_argument('--output_dir', type=str, default='data/yolo_crops', help="Where to save YOLO crops")
    parser.add_argument('--model_path', type=str, default='yolo11n.pt', help="YOLOv11 detection model path")

    args = parser.parse_args()
    main(args.frames_dir, args.output_dir, args.model_path)
