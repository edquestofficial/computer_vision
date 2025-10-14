from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
from threading import Thread, Lock

# Load YOLOv8 model (person detection)
model = YOLO("yolov8n.pt")

# Create a global DeepSORT tracker (shared across both streams)
tracker = DeepSort(max_age=30)

# RTSP streams
cam1_url = "rtsp://admin:admin%40123@192.168.1.104:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"
cam2_url = "rtsp://admin:admin%40123@192.168.1.106:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"

resize_dim = (640, 480)

class RTSPStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.lock = Lock()
        self.stopped = False
        Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame

    def read(self):
        with self.lock:
            if self.ret and self.frame is not None:
                return True, self.frame.copy()
            else:
                return False, None

    def release(self):
        self.stopped = True
        self.cap.release()

# Initialize both streams
stream1 = RTSPStream(cam1_url)
stream2 = RTSPStream(cam2_url)

# Global set to keep unique IDs across both cameras
counted_ids = set()

while True:
    # Read frames from both cameras
    frames = []
    for stream in [stream1, stream2]:
        ret, frame = stream.read()
        if ret:
            frame = cv2.resize(frame, resize_dim)
            frames.append(frame)
        else:
            frames.append(np.zeros((resize_dim[1], resize_dim[0], 3), dtype=np.uint8))

    # Run YOLO on each frame and collect detections globally
    all_detections = []
    for frame in frames:
        results = model(frame, verbose=False)[0]
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # person class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                conf = float(box.conf[0])
                all_detections.append(([x1, y1, w, h], conf, 'person'))

    # Update one global tracker with all detections
    # (frame=frames[0] is required by deep_sort_realtime but doesn't affect global tracking)
    tracks = tracker.update_tracks(all_detections, frame=frames[0])

    # Draw tracks on BOTH camera frames
    for t in tracks:
        if not t.is_confirmed():
            continue

        track_id = t.track_id
        ltrb = t.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Count unique persons globally
        if track_id not in counted_ids:
            counted_ids.add(track_id)

        # Draw bounding box + ID on both frames for visibility
        for frame in frames:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display total count on top of the first frame
    total_count = len(counted_ids)
    cv2.putText(frames[0], f"Total Unique Persons: {total_count}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Combine both camera feeds side by side
    combined = np.hstack(frames)
    cv2.imshow("Multi-Camera Person Counting", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
stream1.release()
stream2.release()
cv2.destroyAllWindows()
