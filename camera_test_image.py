import cv2
from ultralytics import YOLO

# Replace 'best.pt' with the path to your trained model file
# The model will automatically recognize the classes from your training configuration
model = YOLO('best.pt')

# You can change the index if you have multiple cameras (e.g., 1, 2, etc.)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened successfully. Press 'q' to quit.")

while cap.isOpened():
    # Read a frame from the camera
    success, frame = cap.read()

    if not success:
        print("Error: Could not read frame from webcam.")
        break

    # The `stream=True` argument makes the process more efficient for video streams
    # The `show=True` argument displays the annotated frame in a pop-up window
    # The `conf` parameter sets the confidence threshold for detections
    results = model(frame, conf=0.5, stream=True)

    # The `plot()` method from ultralytics automatically draws bounding boxes and labels
    for r in results:
        annotated_frame = r.plot()
        
        # Display the annotated frame
        cv2.imshow("YOLOv11 Live Camera Detection", annotated_frame)

    # 6. Break the loop if 'q' is pressed
    # `cv2.waitKey(1)` waits for 1 millisecond for a key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 7. Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()