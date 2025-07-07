import cv2
import numpy as np

def extract_fixed_frames_from_video(video_path, target_frame_count=64):
    """
    Extract a fixed number of frames from video:
    - If longer: sample evenly
    - If shorter: pad by repeating last frame

    Args:
        video_path (str): Path to video.
        target_frame_count (int): Number of frames wanted.

    Yields:
        np.ndarray: Frame image.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    total_frames = len(frames)

    if total_frames == 0:
        raise ValueError(f"No frames found in {video_path}")

    if total_frames == target_frame_count:
        selected_frames = frames

    elif total_frames > target_frame_count:
        # Sample evenly
        indices = np.linspace(0, total_frames - 1, target_frame_count).astype(int)
        selected_frames = [frames[i] for i in indices]

    else:
        # Pad by repeating last frame
        pad_count = target_frame_count - total_frames
        selected_frames = frames + [frames[-1]] * pad_count

    for frame in selected_frames:
        yield frame
