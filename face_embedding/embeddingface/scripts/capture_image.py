import cv2
import os





# === Step 1: Input Employee Details ===
employee_id = input("Enter Employee ID: ").strip()
username = input("Enter Employee Name: ").strip()




# Directory to save images
base_dir = r"C:\Users\edquestofficial\Desktop\Yogi\embeddingface\data\photos"
save_dir = os.path.join(base_dir, f"{employee_id}_{username}")
os.makedirs(save_dir, exist_ok=True)

# === Step 2: Initialize Webcam ===
cap = cv2.VideoCapture(index=1)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# === Step 3: Capture Configuration ===
angles = ["Front", "Left", "Right", "Up"]
count = 0

print("\nInstructions:")
print("- Position your face according to the prompt.")
print("- Press 'C' to capture the image for that angle.")
print("- Press 'Q' to quit anytime.\n")

# === Step 4: Capture Images ===
for angle in angles:
    print(f"Prepare for {angle} view...")
    captured = False
    while not captured:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.namedWindow("Capture Face", cv2.WINDOW_NORMAL)
        cv2.imshow("Capture Face", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):  # Capture
            count += 1
            filename = os.path.join(save_dir, f"{count}_{angle}.jpg")
            cv2.imwrite(filename, frame)
            print(f" Saved {filename}")
            captured = True

        elif key == ord('q'):  # Quit
            print("Quitting capture...")
            cap.release()
            cv2.destroyAllWindows()
            exit()

            



# === Step 5: Cleanup ===
cap.release()
cv2.destroyAllWindows()
print(f"\n Done! Captured {count} images for Employee ID: {employee_id}, Username: {username}")




