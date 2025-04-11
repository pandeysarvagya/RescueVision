import cv2
import numpy as np

class ThermalImageProcessor:
    def __init__(self, min_threshold=150):
        self.min_threshold = min_threshold

    def preprocess_image(self, image):
        # Convert to grayscale if it's a color image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold the image to highlight hot regions
        _, thresh = cv2.threshold(blurred, self.min_threshold, 255, cv2.THRESH_BINARY)

        return thresh

    def detect_objects(self, processed_image):
        contours, _ = cv2.findContours(processed_image,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        objects = []
        for contour in contours:
            if cv2.contourArea(contour) < 100:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w / 2
            center_y = y + h / 2

            # Create a mask and calculate mean intensity
            mask = np.zeros_like(processed_image)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            intensity = cv2.mean(processed_image, mask=mask)[0]

            objects.append({
                'x': center_x,
                'y': center_y,
                'width': w,
                'height': h,
                'intensity': intensity,
                'contour': contour
            })

        return objects
