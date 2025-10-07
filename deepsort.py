from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Path to input video
video_path = r"C:\Users\ayush kumar dixit\OneDrive\Desktop\DRF CODE1\yolo1\vehicle_count\demo video.mp4"

# Open video
cap = cv2.VideoCapture(video_path)

# Initialize DeepSORT
tracker = DeepSort(max_age=30)

# Vehicle counter
vehicle_count = 0
line1_y = 250
line2_y = 400

# Keep track of IDs that have crossed
counted_ids = set()

# COCO vehicle classes: car, motorcycle, bus, truck
vehicle_classes = [2, 3, 5, 7]

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO detection (only for vehicles)
    results = model.predict(source=frame, classes=vehicle_classes, verbose=False)
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        if cls in vehicle_classes:  # vehicle classes
            detections.append([[x1, y1, x2 - x1, y2 - y1], conf, "vehicle"])

    # Update DeepSORT tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w, h = track.to_ltwh()
        cx, cy = int(l + w / 2), int(t + h / 2)

        # Draw bounding box + ID
        cv2.rectangle(frame, (int(l), int(t)), (int(l + w), int(t + h)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (int(l), int(t) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # Count vehicle only once when crossing both lines downward
        if cy > line1_y and cy < line2_y and track_id not in counted_ids:
            vehicle_count += 1
            counted_ids.add(track_id)

    # Draw two lines
    cv2.line(frame, (0, line1_y), (frame.shape[1], line1_y), (255, 0, 0), 2)
    cv2.line(frame, (0, line2_y), (frame.shape[1], line2_y), (0, 0, 255), 2)

    # Show count
    cv2.putText(frame, f'Vehicle Count: {vehicle_count}', (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show video feed
    cv2.imshow("YOLOv8 + DeepSORT Vehicle Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
