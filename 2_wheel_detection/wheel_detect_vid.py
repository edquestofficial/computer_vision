

from ultralytics import YOLO
import cv2
import os

#  Load your trained model
model = YOLO("runs/detect/train3/weights/best.pt")  

 
video_path ="D:/Trolly/data/vid.mp4"   
cap = cv2.VideoCapture(video_path)

# 3 Video writer setup
save_path = "output_detected.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

#  Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model.predict(source=frame, conf=0.25, imgsz=640, verbose=False)

    wheel_count = 0

    # Loop through detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            label = f"{model.names[cls]} {conf:.2f}"

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            wheel_count += 1

    # Display wheel count on the frame
    cv2.putText(frame, f"Wheels: {wheel_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Write the frame into the output video
    out.write(frame)

    # Show frame live
    cv2.imshow("Wheel Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 5️⃣ Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

print(f" Detection complete! Video saved at: {os.path.abspath(save_path)}")
