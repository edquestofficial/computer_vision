from ultralytics import YOLO
import cv2

# Load YOLOv8 pretrained on COCO
model = YOLO("yolov8n.pt")

# Vehicle classes in COCO: car=2, motorbike=3, bus=5, truck=7
vehicle_classes = [2, 3, 5, 7]

# Video path
video_path = r"C:\Users\ayush kumar dixit\OneDrive\Desktop\DRF CODE1\yolo1\vehicle_count\demo6.mp4"
cap = cv2.VideoCapture(video_path)

# Line positions
entry_line_y = 300  # Line to count vehicles
exit_line_y = 500   # Optional visualization line

# Track counted vehicle IDs
counted_entry_ids = set()

# Vehicle counter
entry_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 tracking mode (persistent IDs)
    results = model.track(
        source=frame,
        persist=True,
        classes=vehicle_classes,
        verbose=False
    )

    # Draw entry and exit lines for visualization
    cv2.line(frame, (0, entry_line_y), (frame.shape[1], entry_line_y), (0, 255, 0), 2)  # Entry (green)
    cv2.line(frame, (0, exit_line_y), (frame.shape[1], exit_line_y), (0, 0, 255), 2)    # Optional line (red)

    # Process detected vehicles
    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xyxy.cpu().tolist()

        for box, obj_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Draw bounding box + ID + center
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, f'ID:{obj_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Count only once when crossing entry line downward
            if cy > entry_line_y and obj_id not in counted_entry_ids:
                counted_entry_ids.add(obj_id)
                entry_count += 1

    # Show entry count
    cv2.putText(frame, f'Entry Count: {entry_count}', (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display video
    cv2.imshow("Vehicle Entry Counter", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n Total Vehicles Counted (Entry Only): {entry_count}")
