import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import time
from collections import deque

unique_words = ['beautiful', 'blind', 'deaf', 'happy', 'loud', 'quiet', 'sad', 'ugly']
label_encoder = LabelEncoder()
label_encoder.fit(unique_words)
print("Label encoder initialized with classes:", label_encoder.classes_)

model_filename = 'demo_lstm_85.h5'
try:
    model = load_model(model_filename)
    print(f"Successfully loaded the model '{model_filename}'.")
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Please make sure '{model_filename}' is in the same directory.")
    exit()

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

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
    
    return full_landmark_vector, results

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream. Make sure your webcam is connected.")
        return

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic_model:

        SEQUENCE_LENGTH = 60
        sequence = deque(maxlen=SEQUENCE_LENGTH)
        
        predicted_word = "Waiting for gesture..."
        last_prediction_time = time.time()
        prediction_display_duration = 3 
        
        frame_delay_ms = 1
        
        confidence_threshold = 0.8

        last_time = time.time()
        fps = 0
        
        frame_count = 0
        frame_interval = 3  # Process one frame out of every 5. Increase this number to slow down further.

        print("Starting live inference demo. Perform a gesture. Press 'q' to quit.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            frame_count += 1
            if frame_count % frame_interval == 0:
                landmark_vector, results = extract_landmarks(frame, holistic_model)
                sequence.append(landmark_vector)
            else:
                landmark_vector, results = extract_landmarks(frame, holistic_model)


            if len(sequence) == SEQUENCE_LENGTH:
                input_sequence = np.expand_dims(np.array(sequence), axis=0)
                
                prediction = model.predict(input_sequence, verbose=0)
                
                predicted_class_index = np.argmax(prediction)
                confidence = prediction[0][predicted_class_index]

                if confidence > confidence_threshold:
                    predicted_word = label_encoder.inverse_transform([predicted_class_index])[0]
                    last_prediction_time = time.time()
                    print(f"Predicted word: {predicted_word} (Confidence: {confidence:.2f})")
                
                sequence.clear()
            
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1)
            )
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            if time.time() - last_prediction_time < prediction_display_duration:
                cv2.putText(frame, predicted_word.upper(), (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Performing gesture...", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            current_time = time.time()
            fps = 1 / (current_time - last_time) if (current_time - last_time) > 0 else 0
            last_time = current_time
            
            cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('Live Gesture Recognition Demo', frame)

            if cv2.waitKey(frame_delay_ms) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()