import cv2
import os
import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

# ===== PATHS =====
video_path = "data/video.mp4"
frames_folder = "frames"
index_folder = "index"

os.makedirs(frames_folder, exist_ok=True)
os.makedirs(index_folder, exist_ok=True)

# ===== STEP 1: EXTRACT FRAMES =====
print("Extracting frames...")
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
interval = int(fps * 2)

count = 0
frame_id = 0
frame_paths = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if count % interval == 0:
        filename = os.path.join(frames_folder, f"frame_{frame_id}.jpg")
        cv2.imwrite(filename, frame)
        frame_paths.append(filename)
        frame_id += 1

    count += 1

cap.release()
print(f"{len(frame_paths)} frames extracted")

# ===== LOAD MODEL =====
print("Loading model...")
model = SentenceTransformer('clip-ViT-B-32')

# ===== CREATE EMBEDDINGS =====
print("Creating embeddings...")
embeddings = []

for path in frame_paths:
    image = Image.open(path).convert("RGB")
    emb = model.encode(image)
    embeddings.append(emb)

embeddings = np.array(embeddings).astype("float32")

print("Embedding shape:", embeddings.shape)

# ===== FAISS =====
print("Building FAISS index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, os.path.join(index_folder, "faiss.index"))
np.save(os.path.join(index_folder, "paths.npy"), frame_paths)

print("Indexing completed successfully!")