from ultralytics import YOLO
import cv2

# Load  trained model
model = YOLO("runs/detect/train3/weights/best.pt")    

 
img_path = "D:/Trolly/data/104.jpg"

 
results = model.predict(source=img_path, conf=0.25)

# Load image with OpenCV
img = cv2.imread(img_path)

# Loop over detections
for result in results:
    for box in result.boxes:
      
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0].cpu().numpy())
        cls = int(box.cls[0].cpu().numpy())
        label = f"{model.names[cls]} {conf:.2f}"

        # Draw rectangle + label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

# Show result
cv2.imshow("Detections", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
