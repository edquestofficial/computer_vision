# computer_vision
computer vision related use cases

# YOLOv8 Real-Time Person Counter

A Python project using **YOLOv8** and **OpenCV** to detect and count people crossing a virtual line in real-time via webcam.

---

## 📝 Overview

This project captures video from a camera, detects people using the YOLOv8 object detection model, and counts the number of people crossing a horizontal line. It draws bounding boxes around detected persons and displays their centroids. The total count is shown on the video feed.

---

## ⚙️ Features

- Real-time person detection with **YOLOv8**.
- Person counting when crossing a horizontal line.
- Bounding boxes and centroids drawn on detected persons.
- Simple visualization of count on video feed.
- Works with webcam or video input.

---

## 📦 Requirements

- Python 3.8+
- [Ultralytics YOLOv8](https://pypi.org/project/ultralytics/)
- OpenCV
- NumPy

Install dependencies with:

```bash
pip install ultralytics opencv-python numpy
