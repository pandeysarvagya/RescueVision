import numpy as np
import cv2

class KalmanTracker:
    def __init__(self, dt=1.0/30):
        self.dt = dt
        self.kf = cv2.KalmanFilter(6, 4)

        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0, 0, 0],
            [0, 1, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], np.float32)

        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.1
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0

        self.initialized = False

    def init(self, measurement):
        self.kf.statePost = np.array([
            measurement[0], measurement[1],
            0, 0,
            measurement[2], measurement[3]
        ], np.float32).reshape(-1, 1)
        self.initialized = True

    def predict(self):
        if not self.initialized:
            return None
        return self.kf.predict()

    def update(self, measurement):
        if not self.initialized:
            self.init(measurement)
        else:
            self.kf.correct(measurement)
