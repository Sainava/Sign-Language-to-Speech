# src/preprocessing/frame_extractor.py

import cv2

def extract_frames_from_video(video_path, max_frames=None):
    """
    Extract frames from a video file.

    Args:
        video_path (str): Path to video file.
        max_frames (int, optional): Max frames to extract.

    Yields:
        np.ndarray: Frame image.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
        count += 1
        if max_frames and count >= max_frames:
            break

    cap.release()
