#  Wheel Detection using YOLOv8 and OpenCV

This project detects and counts **wheels** in videos using a custom-trained **YOLOv8 model**.  
Each frame of the video is processed to draw bounding boxes around detected wheels and display the total wheel count dynamically.  
The output video with detections is saved automatically.

---

#  Features

- Detects wheels in real-time from a video.
- Counts the total number of detected wheels per frame.
- Draws bounding boxes and confidence scores.
- Saves the processed output as a new video file.
- Displays live detection using OpenCV.

---

# Prerequisites

Make sure the following are installed before running the project:

- Python 3.8 or higher  
- [Ultralytics YOLO](https://docs.ultralytics.com/)  
- [OpenCV](https://opencv.org/)  
- GPU (optional but recommended for faster inference)

---

#  Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/wheel-detection.git
   cd wheel-detection
