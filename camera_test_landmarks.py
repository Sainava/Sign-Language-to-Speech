import cv2
import mediapipe as mp
import numpy as np

def extract_landmarks(frame, holistic_model):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False 
    
    results = holistic_model.process(rgb_frame)
    rgb_frame.flags.writeable = True

    expected_pose_size = 33 * 4
    expected_face_size = 468 * 3
    expected_hand_size = 21 * 3

    def get_landmarks_data(landmarks_obj, expected_size, include_visibility=False):
        if landmarks_obj:
            if include_visibility:
                data = np.array([[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in landmarks_obj.landmark]).flatten()
            else:
                data = np.array([[lmk.x, lmk.y, lmk.z] for lmk in landmarks_obj.landmark]).flatten()
            return data
        return np.zeros(expected_size)

    pose_data = get_landmarks_data(results.pose_landmarks, expected_pose_size, include_visibility=True)
    face_data = get_landmarks_data(results.face_landmarks, expected_face_size)
    left_hand_data = get_landmarks_data(results.left_hand_landmarks, expected_hand_size)
    right_hand_data = get_landmarks_data(results.right_hand_landmarks, expected_hand_size)

    full_landmark_vector = np.concatenate([pose_data, face_data, left_hand_data, right_hand_data])
    print(full_landmark_vector.shape)
    return full_landmark_vector, results

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        full_landmark_vector, results = extract_landmarks(frame, holistic)
        
        mp_drawing.draw_landmarks(
            frame, 
            results.face_landmarks, 
            mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        )
        
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

        mp_drawing.draw_landmarks(
            frame, 
            results.left_hand_landmarks, 
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        )

        mp_drawing.draw_landmarks(
            frame, 
            results.right_hand_landmarks, 
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
        
        cv2.imshow('MediaPipe Holistic Landmark Demonstration', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()