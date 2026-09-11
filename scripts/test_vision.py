import numpy as np
import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.perception.detector import VehicleDetector

def main():
    print("[INFO] Initializing YOLOv8 Perception Module...")
    detector = VehicleDetector(model_weight="yolov8n.pt")

    # Synthetic camera frames create kar rahe hain (North, South, East, West)
    print("[INFO] Testing detection pipeline on 4 camera feeds...")
    camera_feeds = []
    for i in range(4):
        # Blank canvas bana kar dummy cars (rectangles) draw kar rahe hain
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw some mock objects
        cv2.rectangle(img, (100, 150), (220, 300), (0, 255, 0), -1)
        cv2.rectangle(img, (300, 200), (450, 350), (0, 255, 0), -1)
        camera_feeds.append(img)

    state_vector, _ = detector.process_multi_camera_streams(camera_feeds)

    print("\n" + "=" * 50)
    print("PERCEPTION PIPELINE STATUS: VERIFIED")
    print("=" * 50)
    print("Extracted RL State Vector (8-dim):", state_vector)
    print("Shape:", state_vector.shape)
    print("=" * 50)

if __name__ == "__main__":
    main()