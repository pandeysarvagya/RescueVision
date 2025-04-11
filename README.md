# 🔥 RescueVision: Thermal-Based Human Detection & Tracking

RescueVision is a Python-based project designed to aid **search and rescue operations** using thermal imaging data. It helps detect and track humans trapped in disaster scenarios using **thermal images or videos**, leveraging computer vision and Kalman filtering.

---

## 📦 Features

- ✅ Thermal image preprocessing & binary thresholding
- 🎯 Object detection using contours
- 🔁 Object tracking using a Kalman Filter
- 🖼 Works with both thermal images and videos
- 🧱 Clean, modular structure for easy expansion and testing

---

## 🗂 Project Structure

```
thermal_tracker_project/
├── thermal_tracker/
│   ├── __init__.py
│   ├── processor.py       # Handles image preprocessing and object detection
│   ├── tracking.py        # Kalman Filter logic
│   └── utils.py           # Helper functions (optional)
│
├── run.py                 # Main script to run image/video tracking
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation (this file)
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/pandeysarvagya/RescueVision.git
cd RescueVision
```

### 2. (Optional) Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# or
source venv/bin/activate     # macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Usage

### ▶️ Process a Thermal Image

```bash
python run.py --mode image --input path/to/image.jpg
```

### 📹 Process a Thermal Video

```bash
python run.py --mode video --input path/to/video.mp4
```

The processed image/video will be saved in your current directory and optionally displayed (if using in Jupyter/Colab).

---

## 🧠 How It Works

1. **Preprocessing:** Converts to grayscale, blurs, and thresholds to isolate hot regions.
2. **Detection:** Finds contours around these regions and draws bounding boxes.
3. **Tracking:** Uses a Kalman Filter to estimate and follow object movement frame-by-frame.
4. **Output:** Annotated frames with object positions and intensity.

---

## 🔧 Requirements

- Python 3.7+
- OpenCV (cv2)
- NumPy

(For notebooks/Colab: `cv2_imshow`, `IPython.display`, etc.)

---

## 🛠 Future Ideas

- Integrate deep learning models (e.g., YOLOv8 for thermal person detection)
- Add real-time video stream support (e.g., from a drone camera)
- Export detected positions to GPS/map overlays

---

## 🤝 Contributing

Want to improve RescueVision? Fix bugs? Add enhancements?
Feel free to fork, open issues, and submit PRs.

---
