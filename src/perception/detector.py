import cv2
import numpy as np
from ultralytics import YOLO

class VehicleDetector:
    """
    YOLO-based Perception module to count vehicles from camera frames
    for adaptive traffic signal observation.
    """
    def __init__(self, model_weight="yolov8n.pt", conf_threshold=0.35):
        # yolov8n auto-download ho jayega first run par
        self.model = YOLO(model_weight)
        self.conf_threshold = conf_threshold
        # COCO class IDs: 2: car, 3: motorcycle, 5: bus, 7: truck
        self.vehicle_classes = [2, 3, 5, 7]

    def process_frame(self, frame):
        """
        Takes raw BGR image/frame and returns:
        - vehicle_count: Total vehicles detected
        - annotated_frame: Frame with bounding boxes drawn
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        
        vehicle_count = 0
        boxes = results.boxes

        for box in boxes:
            cls_id = int(box.cls[0].item())
            if cls_id in self.vehicle_classes:
                vehicle_count += 1

        annotated_frame = results.plot()
        return vehicle_count, annotated_frame

    def process_multi_camera_streams(self, camera_frames: list):
        """
        Takes 4 camera inputs (North, South, East, West approaches)
        and formats them into the 8-dim RL state vector.
        """
        counts = []
        annotated_screens = []

        for frame in camera_frames:
            count, visual = self.process_frame(frame)
            counts.append(float(count))
            annotated_screens.append(visual)

        # 4 approaches (2 lanes each -> estimate per-lane density)
        # Pad or expand to 8-dim vector required by SumoTrafficEnv
        rl_state = []
        for c in counts:
            rl_state.extend([c / 2.0, c / 2.0])

        return np.array(rl_state[:8], dtype=np.float32), annotated_screens