import cv2
import numpy as np

def extract_frames_from_video(video_path, max_frames=60):
    """
    Extracts exactly `max_frames` from the video:
    - If video has enough frames → samples evenly spaced frames.
    - If video is shorter → repeats the last valid frame until it reaches `max_frames`.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        raise ValueError(f"No frames found in {video_path}")

    # If enough frames, sample evenly spaced indices
    if total_frames >= max_frames:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    else:
        # If not enough, just take all available frames then pad later
        indices = np.arange(total_frames)

    frames = []
    current_idx = 0
    target_idx = indices[0] if len(indices) > 0 else 0
    next_target_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx == target_idx:
            frames.append(frame)
            if next_target_idx >= len(indices):
                break
            target_idx = indices[next_target_idx]
            next_target_idx += 1
        current_idx += 1

    cap.release()

    # If padding needed, repeat last frame
    if len(frames) < max_frames:
        last_frame = frames[-1]
        while len(frames) < max_frames:
            frames.append(last_frame)

    return frames
