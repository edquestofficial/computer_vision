from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import threading
import numpy as np
from scipy.spatial.distance import cosine
import time

# ---------------- YOLO MODEL ----------------
model = YOLO("yolov8n.pt")  # Pretrained YOLOv8

# ---------------- DEEPSORT TRACKERS ----------------
tracker_cam1 = DeepSort(max_age=30)
tracker_cam2 = DeepSort(max_age=30)

# ---------------- RTSP URLs ----------------
CAMERA_URLS = [
    "rtsp://admin:admin%40123@192.168.1.106:554/cam/realmonitor?channel=1&subtype=1",
    "rtsp://admin:admin%40123@192.168.1.104:554/cam/realmonitor?channel=1&subtype=1"
]

frames = [None, None]
lock = threading.Lock()

# ---------------- GLOBAL PERSON REGISTRY ----------------
global_embeddings = []
next_global_id = 1
global_ids_seen = set()

# ---------------- SKIP SETTINGS ----------------
SKIP_INTERVAL = 20  # process every N-th frame

# ---------------- REID THRESHOLD ----------------
REID_THRESHOLD = 0.3  # lower = stricter match

# ---------------- SMOOTHING ----------------
prev_positions = [{}, {}]  # previous bbox per track for each camera

# ---------------- FPS TARGET ----------------
TARGET_FPS = 30
FRAME_INTERVAL = 0.5 / TARGET_FPS


def match_embedding(embedding, threshold=REID_THRESHOLD):
    """Match new embedding to existing global embeddings using cosine distance."""
    global global_embeddings
    for vec, gid in global_embeddings:
        dist = cosine(embedding, vec)
        if dist < threshold:
            return gid
    return None


def process_camera(index, url, tracker):
    global frames, next_global_id, global_embeddings, global_ids_seen

    cap = cv2.VideoCapture(url)
    frame_count = 0

    fps_counter = 0
    fps_start = time.time()

    while True:
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.resize(frame, (640, 480))

        fps_counter += 1
        if time.time() - fps_start >= 1:
            print(f"Camera {index} FPS: {fps_counter}")
            fps_counter = 0
            fps_start = time.time()

        # ---------- Frame Skipping for speed ----------
        if frame_count % SKIP_INTERVAL != 0:
            # Update tracker without YOLO (predict only)
            tracks = tracker.update_tracks([], frame=frame)
        else:
            # --- YOLO Detection ---
            results = model(frame, verbose=False)[0]
            detections = []
            for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
                if int(cls) == 0 and conf > 0.5:  # person class
                    x1, y1, x2, y2 = map(int, box)
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf.item(), 'person'))

            tracks = tracker.update_tracks(detections, frame=frame)

        # --- Person Count in Current Frame ---
        frame_person_count = 0
        matched_in_frame = set()

        for t in tracks:
            if not t.is_confirmed():
                continue

            ltrb = t.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            # Smooth bounding boxes
            if t.track_id in prev_positions[index]:
                px1, py1, px2, py2 = prev_positions[index][t.track_id]
                x1 = int(0.5 * x1 + 0.5 * px1)
                y1 = int(0.5 * y1 + 0.5 * py1)
                x2 = int(0.5 * x2 + 0.5 * px2)
                y2 = int(0.5 * y2 + 0.5 * py2)
            prev_positions[index][t.track_id] = (x1, y1, x2, y2)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Frame person count
            frame_person_count += 1

            # --- Embedding & Global ID Matching ---
            emb = t.get_feature()
            if emb is not None:
                emb = emb / np.linalg.norm(emb)
                gid = match_embedding(emb)
                if gid is None and id(emb) not in matched_in_frame:
                    gid = next_global_id
                    next_global_id += 1
                    global_embeddings.append((emb, gid))
                if gid is not None:
                    global_ids_seen.add(gid)
                    matched_in_frame.add(gid)
                    cv2.putText(frame, f"GID:{gid}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- Overlay FPS & Frame Person Count ---
        cv2.putText(frame, f"FPS: {fps_counter}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Persons in Frame: {frame_person_count}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        with lock:
            frames[index] = frame

        # --- FPS Regulation ---
        elapsed = time.time() - loop_start
        sleep_time = FRAME_INTERVAL - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()


# ---------------- THREADS ----------------
t1 = threading.Thread(target=process_camera, args=(0, CAMERA_URLS[0], tracker_cam1), daemon=True)
t2 = threading.Thread(target=process_camera, args=(1, CAMERA_URLS[1], tracker_cam2), daemon=True)
t1.start()
t2.start()

# ---------------- DISPLAY LOOP ----------------
while True:
    with lock:
        if all(f is not None for f in frames):
            combined = np.hstack(frames)
            # Overlay total unique persons
            cv2.rectangle(combined, (0, 0), (1280, 40), (0, 0, 0), -1)
            cv2.putText(combined, f"Total Unique Persons: {len(global_ids_seen)}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Multi-Camera Person Counting", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

