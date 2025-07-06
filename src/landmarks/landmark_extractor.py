#landmark_extractor.py

import numpy as np
import cv2
import mediapipe as mp

class MediaPipeHolisticExtractor:
    def __init__(self):
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_from_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)

        POSE = 33
        FACE = 468
        HAND = 21

        def get_pose_data(landmarks_obj, num_landmarks):
            if landmarks_obj:
                return np.array([[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in landmarks_obj.landmark])
            return np.zeros((num_landmarks, 4))

        def get_xyz_data(landmarks_obj, num_landmarks):
            if landmarks_obj:
                return np.array([[lmk.x, lmk.y, lmk.z] for lmk in landmarks_obj.landmark])
            return np.zeros((num_landmarks, 3))

        return {
            'pose': get_pose_data(results.pose_landmarks, POSE),
            'face': get_xyz_data(results.face_landmarks, FACE),
            'left_hand': get_xyz_data(results.left_hand_landmarks, HAND),
            'right_hand': get_xyz_data(results.right_hand_landmarks, HAND),
        }

    def close(self):
        self.holistic.close()
