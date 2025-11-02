# from insightface.app import FaceAnalysis
# import cv2
# import numpy as np
# import os
# import faiss 
# async def facegenerating_embedding(id,username,img_path):
#     print("func callling step 1")
#     face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Use 'CUDAExecutionProvider' for GPU
#     face_app.prepare(ctx_id=-1)  
#     img = cv2.imread(img_path)
#     print("func callling step 2")
#     if img is None:
#         print(f" Unable to read {img_path}")
#         return
#     embeddings = np.empty((0, 512), dtype="float32")
#     metadata = []
#     print("func callling step 3")
#     faces = face_app.get(img)
#     if faces:
#         emb = faces[0].embedding.astype("float32")
#         embeddings = np.vstack([embeddings, emb])
#         metadata.append([id, username])
#         print("func callling step 4")
#     else:
#         print(f" No face detected in {img_path}")

# # Save embeddings and metadata
#     print("func callling step 5")
#     DATA_DIR = "C:\\Users\\edquestofficial\\Desktop\\Yogi\\embeddingface\\data"
#     EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
#     METADATA_PATH = os.path.join(DATA_DIR, "metadata.npy")
#     np.save(EMBEDDINGS_PATH, embeddings)
#     np.save(METADATA_PATH, np.array(metadata, dtype=object))

#     print(f"\n Embeddings generated and saved:")
#     print(f" - Embeddings shape: {embeddings.shape}")
#     print(f" - Metadata entries: {len(metadata)}")
#     print(f" - Saved to: {EMBEDDINGS_PATH} and {METADATA_PATH}")


# def face_embedding_search(image_path):
   

#     # Initialize the face analysis model
#     face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
#     face_app.prepare(ctx_id=-1)  # CPU mode; use ctx_id=0 for GPU

#     # Load the query image
#     img = cv2.imread(image_path)
#     if img is None:
#         return {"status": "error", "message": f"Unable to read {image_path}"}

#     faces = face_app.get(img)
#     if not faces:
#         return {"status": "error", "message": f"No face detected in {image_path}"}

#     # Get query face embedding
#     query_emb = faces[0].embedding.astype("float32").reshape(1, -1)

#     # Paths to stored data
#     DATA_DIR = "C:/Users/edquestofficial/Desktop/Yogi/embeddingface/data"
#     EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
#     METADATA_PATH = os.path.join(DATA_DIR, "metadata.npy")

#     # Check if embedding/metadata files exist and are not empty
#     if (
#         not os.path.exists(EMBEDDINGS_PATH)
#         or not os.path.exists(METADATA_PATH)
#         or os.path.getsize(EMBEDDINGS_PATH) == 0
#         or os.path.getsize(METADATA_PATH) == 0
#     ):
#         return {
#             "status": "error",
#             "message": "Embeddings or metadata file is empty. Please generate embeddings first."
#         }

#     # Load embeddings and metadata
#     stored_embeddings = np.load(EMBEDDINGS_PATH)
#     metadata = np.load(METADATA_PATH, allow_pickle=True)

#     # Build FAISS index
#     index = faiss.IndexFlatL2(stored_embeddings.shape[1])
#     index.add(stored_embeddings)

#     # Perform similarity search
#     k = min(3, len(stored_embeddings))  # top 3 matches or less
#     D, I = index.search(query_emb, k=k)

#     # Compute similarity scores (higher = more similar)
#     similarities = 1 / (1 + D[0])

#     # Prepare results
#     results = []
#     for rank, (idx, sim) in enumerate(zip(I[0], similarities), start=1):
#         id_val, username = metadata[idx]
#         results.append({
#             "rank": rank,
#             "id": id_val,
#             "username": username,
#             "similarity": round(float(sim), 4)
#         })
    
#     return results




from insightface.app import FaceAnalysis
import cv2
import numpy as np
import os
import faiss

# -----------------------------
# CONFIG PATHS
# -----------------------------
DATA_DIR = "C:\\Users\\edquestofficial\\Desktop\\Yogi\\embeddingface\\data"
FAISS_PATH = os.path.join(DATA_DIR, "face_index.faiss")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.npy")

os.makedirs(DATA_DIR, exist_ok=True)


# =============================
# GENERATE + STORE EMBEDDING
# =============================
async def facegenerating_embedding(id, username, img_path):
    print("func calling step 1")

    # Initialize face model
    face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=-1)  # CPU mode; use ctx_id=0 for GPU

    # Load image
    img = cv2.imread(img_path)
    print("func calling step 2")
    if img is None:
        print(f" Unable to read {img_path}")
        return

    faces = face_app.get(img)
    if not faces:
        print(f" No face detected in {img_path}")
        return

    print("func calling step 3")
    emb = faces[0].embedding.astype("float32").reshape(1, -1)

    # Load existing FAISS index or create new
    dim = emb.shape[1]
    if os.path.exists(FAISS_PATH):
        index = faiss.read_index(FAISS_PATH)
        print(" Loaded existing FAISS index")
    else:
        index = faiss.IndexFlatL2(dim)
        print(" Created new FAISS index")

    # Add embedding to index
    index.add(emb)
    faiss.write_index(index, FAISS_PATH)
    print(" Embedding added to FAISS index")

    # Load or create metadata
    if os.path.exists(METADATA_PATH):
        metadata = np.load(METADATA_PATH, allow_pickle=True).tolist()
    else:
        metadata = []

    # Append new metadata
    metadata.append({"id": id, "username": username})
    np.save(METADATA_PATH, np.array(metadata, dtype=object))

    print("\n Embedding stored successfully in FAISS!")
    print(f" - Total vectors: {index.ntotal}")
    print(f" - Metadata entries: {len(metadata)}")


# =============================
# SEARCH FACE IN FAISS INDEX
# =============================
def face_embedding_search(image_path):
    # Initialize model
    face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=-1)

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return {"status": "error", "message": f"Unable to read {image_path}"}

    faces = face_app.get(img)
    if not faces:
        return {"status": "error", "message": "No face detected"}

    query_emb = faces[0].embedding.astype("float32").reshape(1, -1)

    # Check FAISS & metadata existence
    if not (os.path.exists(FAISS_PATH) and os.path.exists(METADATA_PATH)):
        return {"status": "error", "message": "Vector database not found"}

    index = faiss.read_index(FAISS_PATH)
    metadata = np.load(METADATA_PATH, allow_pickle=True).tolist()

    # Search top-k
    k = min(3, index.ntotal)
    D, I = index.search(query_emb, k)
    similarities = 1 / (1 + D[0])

    # Prepare results
    results = []
    for rank, (idx, sim) in enumerate(zip(I[0], similarities), start=1):
        meta = metadata[idx]
        results.append({
            "rank": rank,
            "id": meta["id"],
            "username": meta["username"],
            "similarity": round(float(sim), 4)
        })

    return {"status": "success", "results": results}
