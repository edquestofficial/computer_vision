import cv2
from face_detector import FaceDetector

RTSP = "rtsp://username:password@camera-ip/stream"
RTSP = "rtsp://admin:admin%40123@192.168.1.106:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"
cap = cv2.VideoCapture(RTSP)
ret, frame = cap.read()
cap.release()

if ret:
    fd = FaceDetector()
    faces = fd.detect_faces(frame)
    print(f"Detected faces: {len(faces)}")
    for i, f in enumerate(faces):
        f.save(f"face_rtsp_{i}.jpg")
else:
    print("⚠️ Could not fetch RTSP frame.")
