# scripts/record_sign.py

import cv2
import time
from pathlib import Path

# === CONFIG ===
SAVE_TO = Path("datasets/raw/Deaf/Deaf_02.mov")  # Change this for each clip!
RECORD_SECONDS = 3  # Or 5 if you want longer clips
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 20.0
COUNTDOWN = 3  # seconds before recording starts


# Ensure output folder exists
SAVE_TO.parent.mkdir(parents=True, exist_ok=True)

# === Setup webcam ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# === Setup video writer ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(SAVE_TO), fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

print(f"[INFO] Press 'r' to start recording immediately for {RECORD_SECONDS} sec")
print(f"[INFO] Press 'd' to start with {COUNTDOWN} sec delay")
print("[INFO] Press 'q' to quit without recording")

recording = False
start_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF

    # Start recording immediately
    if key == ord('r') and not recording:
        print("[INFO] Recording started immediately...")
        recording = True
        start_time = time.time()

    # Start recording with countdown
    elif key == ord('d') and not recording:
        print(f"[INFO] Starting recording in {COUNTDOWN} seconds...")
        for i in reversed(range(1, COUNTDOWN + 1)):
            ret, frame = cap.read()
            cv2.putText(frame, f"Starting in {i}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.imshow('Recording', frame)
            cv2.waitKey(1000)
        print("[INFO] Recording started!")
        recording = True
        start_time = time.time()

    elif key == ord('q'):
        print("[INFO] Quitting without saving.")
        break

    if recording:
        # Add REC text overlay
        cv2.putText(frame, "REC", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        out.write(frame)

        elapsed = time.time() - start_time
        if elapsed >= RECORD_SECONDS:
            print("[INFO] Recording finished.")
            break

    cv2.imshow('Recording', frame)

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"[INFO] Video saved to {SAVE_TO}")
