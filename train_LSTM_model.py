import os
import mediapipe as mp
import cv2
import numpy as np

KAGGLE_DATA_PATH = '/kaggle/input/include/Adjectives_1of8'

WORKING_DIR = '/kaggle/working/'
CROPPED_FRAMES_DIR = os.path.join(WORKING_DIR, 'yolo_cropped_frames')
LANDMARKS_DIR = os.path.join(WORKING_DIR, 'extracted_landmarks')

os.makedirs(CROPPED_FRAMES_DIR, exist_ok=True)
os.makedirs(LANDMARKS_DIR, exist_ok=True)

print(f"Kaggle Data Path: {KAGGLE_DATA_PATH}")
print(f"Working Directory: {WORKING_DIR}")

file_paths = []

for root, dirs, files in os.walk(KAGGLE_DATA_PATH):
    for file in files:
        if file.endswith('.MOV'):
            file_path = os.path.join(root, file)
            path_components = file_path.split('/')    
            for component in reversed(path_components):
                if '. ' in component:
                 word = component.split('. ')[1]
            file_paths.append([word.lower(), file_path])

file_paths = np.array(file_paths)
file_paths.shape

def extract_landmarks(frame, pose):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.pose_landmarks.landmark])
        return landmarks
    else:
        return None
    
def process_videos(file_paths):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    landmarks = []
    labels = []
    
    for (word, file_path) in file_paths:
        print(f"Prcessing {file_path} for landmarks")
        cap = cv2.VideoCapture(file_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            landmarks_frame = extract_landmarks(frame, pose)
            if landmarks_frame is not None:
                landmarks.append(landmarks_frame)
                labels.append(word) 

        cap.release()
                    
    landmarks = np.array(landmarks)
    labels = np.array(labels)

    np.save('/kaggle/working/extracted_landmarks/pose_landmarks_landmarks.npy', landmarks)
    np.save('/kaggle/working/extracted_landmarks/pose_landmarks_labels.npy', labels)

    pose.close()

process_videos(file_paths)