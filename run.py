from thermal_tracker.image_processor import ThermalImageProcessor
from thermal_tracker.tracking import MultiObjectTracker
import cv2
import numpy as np
import os
import time

def process_video(video_path):
    processor = ThermalImageProcessor()
    tracker = MultiObjectTracker(max_age=10, min_hits=3, iou_threshold=0.3)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_path = "output_tracked.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        processed = processor.preprocess_image(frame)
        detected_objects = processor.detect_objects(processed)
        
        tracked_objects = tracker.update(detected_objects)
        
        output_frame = frame.copy()
        
        for obj in detected_objects:
            x, y = int(obj['x']), int(obj['y'])
            w, h = int(obj['width']), int(obj['height'])
            cv2.rectangle(output_frame, (x-w//2, y-h//2), (x+w//2, y+h//2), (255, 0, 0), 1)
        
        for obj in tracked_objects:
            x, y = int(obj['x']), int(obj['y'])
            w, h = int(obj['width']), int(obj['height'])
            z = obj['z']
            track_id = obj['id']
            
            cv2.rectangle(output_frame, (x-w//2, y-h//2), (x+w//2, y+h//2), (0, 255, 0), 2)
            
            label = f"ID:{track_id} Z:{z:.1f} I:{obj['intensity']:.1f}"
            cv2.putText(output_frame, label, (x-w//2, y-h//2-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.putText(output_frame, f"Frame: {frame_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
        out.write(output_frame)
        
        cv2.imshow("Thermal Tracking", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output video saved to {output_path}")

def process_image(image_path):
    processor = ThermalImageProcessor()
    tracker = MultiObjectTracker(max_age=10, min_hits=1, iou_threshold=0.3)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    processed = processor.preprocess_image(image)
    detected_objects = processor.detect_objects(processed)
    
    tracked_objects = tracker.update(detected_objects)
    
    output = image.copy()
    
    for obj in detected_objects:
        x, y = int(obj['x']), int(obj['y'])
        w, h = int(obj['width']), int(obj['height'])
        cv2.rectangle(output, (x-w//2, y-h//2), (x+w//2, y+h//2), (255, 0, 0), 1)
        cv2.putText(output, f"I:{obj['intensity']:.1f}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    for obj in tracked_objects:
        x, y = int(obj['x']), int(obj['y'])
        w, h = int(obj['width']), int(obj['height'])
        z = obj['z']
        track_id = obj['id']
        
        cv2.rectangle(output, (x-w//2, y-h//2), (x+w//2, y+h//2), (0, 255, 0), 2)
        label = f"ID:{track_id} Z:{z:.1f} I:{obj['intensity']:.1f}"
        cv2.putText(output, label, (x-w//2, y-h//2-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    output_path = "output_image_tracked.jpg"
    cv2.imwrite(output_path, output)
    print(f"Output saved to {output_path}")
    
    cv2.imshow("Thermal Tracking", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Process single image
    test_image = "test.jpg"
    if os.path.exists(test_image):
        process_image(test_image)
    else:
        print(f"Test image {test_image} not found.")
    
    # Process video if available
    test_video = "test.mp4"
    if os.path.exists(test_video):
        process_video(test_video)
    else:
        print(f"Test video {test_video} not found. Skipping video processing.")