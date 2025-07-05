import numpy as np
import cv2
import mediapipe as mp

class MediaPipeHolisticExtractor:
    def __init__(self, sequence_length=60):
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
        TOTAL = POSE + FACE + HAND + HAND

        # (TOTAL, 3) instead of 4
        frame_landmarks = np.zeros((TOTAL, 3))

        def get_landmarks_data(landmarks_obj, num_landmarks):
            if landmarks_obj:
                coords = np.array([[lmk.x, lmk.y, lmk.z] for lmk in landmarks_obj.landmark])
                return coords
            return np.zeros((num_landmarks, 3))

        idx = 0
        pose = get_landmarks_data(results.pose_landmarks, POSE)
        frame_landmarks[idx:idx+POSE] = pose
        idx += POSE

        face = get_landmarks_data(results.face_landmarks, FACE)
        frame_landmarks[idx:idx+FACE] = face
        idx += FACE

        left_hand = get_landmarks_data(results.left_hand_landmarks, HAND)
        frame_landmarks[idx:idx+HAND] = left_hand
        idx += HAND

        right_hand = get_landmarks_data(results.right_hand_landmarks, HAND)
        frame_landmarks[idx:idx+HAND] = right_hand

        return frame_landmarks

    def close(self):
        self.holistic.close()
