import cv2
import numpy as np
import tensorflow as tf

class Camera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        # Load YOLO and Gender Model (similar to the main.py)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        # Add YOLO and Gender Classification Logic (similar to main.py)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
