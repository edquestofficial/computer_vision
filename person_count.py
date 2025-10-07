from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 model pretrained on COCO
model = YOLO("yolov8n.pt")

# Person class in COCO
person_class = [0]  # 0 = person

# Open camera
cap = cv2.VideoCapture(0)

# Person counter
person_count = 0

# Line position for counting
line_y = 300

# Store previous centroids
prev_centroids = []

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Failed to grab frame")
        break

    # Run YOLO detection
    results = model.predict(source=frame, classes=person_class, verbose=False)

    # Current frame centroids
    current_centroids = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)  # centroid
        current_centroids.append((cx, cy))

        # Draw box + center
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    # Convert to NumPy arrays for easy comparison
    current_centroids = np.array(current_centroids)
    prev_centroids = np.array(prev_centroids)

    if len(prev_centroids) > 0 and len(current_centroids) > 0:
        for (cx, cy) in current_centroids:
            # Find closest previous centroid
            distances = np.linalg.norm(prev_centroids - np.array([cx, cy]), axis=1)
            nearest_idx = np.argmin(distances)

            prev_cy = prev_centroids[nearest_idx][1]

            # Detect crossing (downward movement through the line)
            if prev_cy < line_y and cy >= line_y:
                person_count += 1

    # Update prev centroids
    prev_centroids = current_centroids.tolist()

    # Draw line
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255, 0, 0), 2)

    # Show count
    cv2.putText(frame, f'Count: {person_count}', (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show video feed
    cv2.imshow("Person Counter", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
