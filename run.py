from thermal_tracker.image_processor import ThermalImageProcessor
from thermal_tracker.tracking import KalmanTracker
import cv2
import numpy as np
import os

def process_image(image_path):
    processor = ThermalImageProcessor()
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    processed = processor.preprocess_image(image)
    objects = processor.detect_objects(processed)

    output = image.copy()
    for obj in objects:
        x, y = int(obj['x']), int(obj['y'])
        w, h = int(obj['width']), int(obj['height'])
        cv2.rectangle(output, (x-w//2, y-h//2), (x+w//2, y+h//2), (0, 255, 0), 2)
        cv2.putText(output, f"{obj['intensity']:.1f}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    output_path = "output_image.jpg"
    cv2.imwrite(output_path, output)
    print(f"Output saved to {output_path}")
    cv2.imshow("Detection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Replace with the path to any test image you have locally
    test_image = "test.jpg"
    process_image(test_image)
