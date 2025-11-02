import cv2
from PIL import Image

class FaceDetector:
    """
    Lightweight CPU-based face detector.
    Uses OpenCV's built-in Haar Cascade for simplicity and portability.
    """

    def __init__(self):
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect_faces(self, frame):
        """
        Input:
            frame (numpy array, BGR)
        Output:
            List of cropped PIL.Image objects (faces)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        crops = []
        for (x, y, w, h) in faces:
            crop = frame[y:y + h, x:x + w]
            crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        return crops
