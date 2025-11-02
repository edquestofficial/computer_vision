import numpy as np
import os
import insightface
import numpy as np
from insightface.app import FaceAnalysis
from glob import glob

# face_app = FaceAnalysis(name='buffalo')
# face_app.prepare(ctx_id=0, nms=0.4)

# Initialize face analysis model
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Use 'CUDAExecutionProvider' for GPU
face_app.prepare(ctx_id=-1)  # ctx_id=-1 for CPU, 0 for GPU

embeddings_file = "E:/facerecognition_embedding/data/embeddings.npy"
metadata_file = "E:/facerecognition_embedding/data/embeddings.npy"

if os.path.exists(embeddings_file):
    embeddings = np.load(embeddings_file)
    metadata = np.load(metadata_file)
else:
    embeddings = np.empty((0,512))
    metadata = []

    photos_dir = "E:\\facerecognition_embedding\\data\\photos"
    files = os.listdir(photos_dir)
    print("dir data", files )
    for emp_id in os.listdir(photos_dir):
        images = glob(f"{photos_dir}/{emp_id}/*.jpg")
        print(len(images), "images found for Employee ID:", emp_id)
        for img_path in images:
            import cv2
            img = cv2.imread(img_path)
            faces = face_app.get(img)
            if faces:
                emb = faces[0].embedding
                embeddings = np.vstack([embeddings, emb])
                print("embeddings -------------- ", embeddings)
                metadata.append(emp_id)

    np.save("E:/facerecognition_embedding/data/embeddings.npy", embeddings.astype('float32'))
    np.save("E:/facerecognition_embedding/data/metadata.npy", np.array(metadata))
    print("Embeddings generated and saved.")