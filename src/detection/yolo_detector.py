from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="yolo11n.pt", conf=0.3):
        self.model = YOLO(model_path)
        self.model_path = model_path  # ✅ keep this for printing
        self.conf = conf

    def detect_objects(self, frame):
        #print(f"[YOLO] Using model: {self.model_path} | Confidence: {self.conf}")
        results = self.model.predict(source=frame, conf=self.conf, verbose=False)
        detections = []
        for box in results[0].boxes.xyxy:
            detections.append(box.cpu().numpy())
        return detections
