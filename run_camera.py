import cv2
import time
from routes.vector_store import recognize_faces_in_frame, face_app
# Import the functions and models from your first file
# (Assuming your first file is named vector_store.py)
# try:
    
# except ImportError:
#     print("Error: Could not import from vector_store.py")
#     print("Make sure your first file is named vector_store.py and is in the same directory.")
#     exit()

# --- Configuration ---
RTSP_STREAM_URL = "rtsp://admin:admin%40123@192.168.1.106:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif" # ⚠️ UPDATE THIS
# To use a webcam, uncomment this line:
# RTSP_STREAM_URL = 0 

def main():
    if face_app is None:
        print("FaceAnalysis model failed to initialize. Exiting.")
        return

    cap = cv2.VideoCapture(RTSP_STREAM_URL)

    if not cap.isOpened():
        print(f"Error: Could not open video stream at {RTSP_STREAM_URL}")
        return

    print("🎥 Camera stream opened. Starting recognition...")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame. Reconnecting...")
            time.sleep(2)
            cap.release()
            cap = cv2.VideoCapture(RTSP_STREAM_URL)
            continue

        # --- This is the magic ---
        # Pass the whole frame to your recognition function
        recognition_results = recognize_faces_in_frame(frame)
        # -------------------------

        # Loop through the results and draw on the frame
        for res in recognition_results:
            x1, y1, x2, y2 = res['box']
            
            if res['status'] == 'match':
                name = res['data']['name']
                color = (0, 255, 0) # Green for match
                # This is the data you would send to your frontend
                frontend_data = res['data'] 
                print(f"SENDING TO FRONTEND (Presence): {frontend_data}")
                
            else: # 'unknown'
                name = "Unknown"
                color = (0, 0, 255) # Red for unknown

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Draw label
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Show the video feed
        # cv2.imshow("Face Recognition", frame)
        cv2.imwrite("Image", frame)
        # cv2.waitKey(0)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Stream stopped.")

if __name__ == "__main__":
    main()