import os
import numpy as np
import cv2
from glob import glob
from insightface.app import FaceAnalysis

# Initialize FaceAnalysis model
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Use 'CUDAExecutionProvider' for GPU
face_app.prepare(ctx_id=-1)  # ctx_id=-1 for CPU, 0 for GPU

# Paths
DATA_DIR = r"C:\Users\edquestofficial\Desktop\Yogi\embeddingface\data"
PHOTOS_DIR = os.path.join(DATA_DIR, "photos")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.npy")

# Load existing data if available
if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(METADATA_PATH):
    embeddings = np.load(EMBEDDINGS_PATH)
    metadata = np.load(METADATA_PATH, allow_pickle=True).tolist()
    print(f"Loaded {len(metadata)} existing embeddings.")
else:
    embeddings = np.empty((0, 512), dtype="float32")
    metadata = []

# Iterate through each folder (format: empID_username)
for folder_name in os.listdir(PHOTOS_DIR):
    folder_path = os.path.join(PHOTOS_DIR, folder_name)
    if not os.path.isdir(folder_path):
        continue

    # Extract emp_id and username from folder name
    try:
        emp_id, username = folder_name.split("_", 1)
    except ValueError:
        print(f" Skipping folder '{folder_name}' — expected format 'empid_username'")
        continue

    images = glob(os.path.join(folder_path, "*.jpg"))
    print(f" Found {len(images)} images for Employee: {emp_id} ({username})")

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print(f" Unable to read {img_path}")
            continue

        faces = face_app.get(img)
        if faces:
            emb = faces[0].embedding.astype("float32")
            embeddings = np.vstack([embeddings, emb])
            metadata.append([emp_id, username])
        else:
            print(f" No face detected in {img_path}")

# Save embeddings and metadata
np.save(EMBEDDINGS_PATH, embeddings)
np.save(METADATA_PATH, np.array(metadata, dtype=object))

print(f"\n Embeddings generated and saved:")
print(f" - Embeddings shape: {embeddings.shape}")
print(f" - Metadata entries: {len(metadata)}")
print(f" - Saved to: {EMBEDDINGS_PATH} and {METADATA_PATH}")
