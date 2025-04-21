import numpy as np
import cv2

class KalmanTracker3D:
    def __init__(self, dt=1.0/30, track_id=None):
        self.dt = dt
        self.track_id = track_id
        self.age = 0
        self.hits = 0
        self.time_since_update = 0
        self.hit_streak = 0
        
        self.kf = cv2.KalmanFilter(9, 5)
        
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, dt, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, dt, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, dt, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
        ], np.float32)
        
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
        ], np.float32)
        
        process_noise = np.eye(9, dtype=np.float32)
        process_noise[0:3, 0:3] *= 0.1
        process_noise[3:6, 3:6] *= 1.0
        process_noise[6:9, 6:9] *= 0.1
        self.kf.processNoiseCov = process_noise
        
        self.kf.measurementNoiseCov = np.eye(5, dtype=np.float32) * 1.0
        
        self.kf.errorCovPost = np.eye(9, dtype=np.float32) * 10
        
        self.initialized = False
        self.last_measurement = None
        
    def estimate_z(self, intensity):
        z = 100.0 - (intensity / 255.0 * 90.0)
        return max(z, 1.0)
    
    def init(self, measurement):
        x, y, width, height, intensity = measurement
        z = self.estimate_z(intensity)
        self.kf.statePost = np.array([
            x, y, z,
            0, 0, 0,
            width, height, intensity
        ], np.float32).reshape(-1, 1)
        self.initialized = True
        self.last_measurement = (x, y, width, height, intensity)
        
    def predict(self):
        if not self.initialized:
            return None
        state = self.kf.predict()
        self.time_since_update += 1
        self.age += 1
        x, y = state[0, 0], state[1, 0]
        width, height = state[6, 0], state[7, 0]
        intensity = state[8, 0]
        return np.array([x, y, width, height, intensity])
        
    def update(self, measurement):
        if not self.initialized:
            self.init(measurement)
            self.hits = 1
            self.hit_streak = 1
            return
        measurement_array = np.array(measurement, dtype=np.float32).reshape(-1, 1)
        self.kf.correct(measurement_array)
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        self.last_measurement = tuple(measurement)
        
    def get_state(self):
        if not self.initialized:
            return None
        return self.kf.statePost.flatten()
    
    def get_position_3d(self):
        if not self.initialized:
            return None
        state = self.kf.statePost.flatten()
        return state[0:3]


class MultiObjectTracker:
    def __init__(self, max_age=20, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.next_id = 0
    
    def update(self, detections):
        predicted_locations = []
        for tracker in self.trackers:
            prediction = tracker.predict()
            if prediction is not None:
                predicted_locations.append(prediction)
            else:
                predicted_locations.append(None)
        
        if len(predicted_locations) > 0 and len(detections) > 0:
            detection_list = []
            for obj in detections:
                detection_list.append([
                    obj['x'], 
                    obj['y'], 
                    obj['width'], 
                    obj['height'], 
                    obj['intensity']
                ])
            
            iou_matrix = np.zeros((len(predicted_locations), len(detection_list)))
            
            for t, prediction in enumerate(predicted_locations):
                if prediction is None:
                    continue
                    
                for d, detection in enumerate(detection_list):
                    x1, y1 = prediction[0] - prediction[2]/2, prediction[1] - prediction[3]/2
                    x2, y2 = prediction[0] + prediction[2]/2, prediction[1] + prediction[3]/2
                    
                    d_x1 = detection[0] - detection[2]/2
                    d_y1 = detection[1] - detection[3]/2
                    d_x2 = detection[0] + detection[2]/2
                    d_y2 = detection[1] + detection[3]/2
                    
                    iou_matrix[t, d] = self._calculate_iou(
                        [x1, y1, x2, y2],
                        [d_x1, d_y1, d_x2, d_y2]
                    )
            
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matched_indices = list(zip(row_ind, col_ind))
            
            unmatched_detections = list(range(len(detection_list)))
            for t, d in matched_indices:
                if iou_matrix[t, d] >= self.iou_threshold:
                    self.trackers[t].update(detection_list[d])
                    if d in unmatched_detections:
                        unmatched_detections.remove(d)
                else:
                    unmatched_detections.append(d)
                    
            for idx in unmatched_detections:
                new_tracker = KalmanTracker3D(track_id=self.next_id)
                detection = detections[idx]
                measurement = [
                    detection['x'], 
                    detection['y'], 
                    detection['width'], 
                    detection['height'], 
                    detection['intensity']
                ]
                new_tracker.init(measurement)
                self.trackers.append(new_tracker)
                self.next_id += 1
        else:
            for detection in detections:
                new_tracker = KalmanTracker3D(track_id=self.next_id)
                measurement = [
                    detection['x'], 
                    detection['y'], 
                    detection['width'], 
                    detection['height'], 
                    detection['intensity']
                ]
                new_tracker.init(measurement)
                self.trackers.append(new_tracker)
                self.next_id += 1
                
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        
        results = []
        for tracker in self.trackers:
            if tracker.hit_streak >= self.min_hits or tracker.hits >= self.min_hits:
                state = tracker.get_state()
                if state is not None:
                    pos_3d = tracker.get_position_3d()
                    results.append({
                        'id': tracker.track_id,
                        'x': state[0],
                        'y': state[1],
                        'z': state[2],
                        'width': state[6],
                        'height': state[7],
                        'intensity': state[8],
                        'confirmed': True
                    })
                    
        return results
    
    def _calculate_iou(self, bbox1, bbox2):
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_max < x_min or y_max < y_min:
            return 0.0
            
        intersection = (x_max - x_min) * (y_max - y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        iou = intersection / (area1 + area2 - intersection + 1e-6)
        return iou
