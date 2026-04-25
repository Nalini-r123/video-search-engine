import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json

print("Loading model...")
model = SentenceTransformer('clip-ViT-B-32')

index = faiss.read_index("index/faiss.index")
frame_paths = np.load("index/paths.npy", allow_pickle=True)

print("System ready!")

FPS = 30  # assume ~30 FPS video (approx)

while True:
    query = input("\nEnter your query (or 'exit'): ")

    if query.lower() == "exit":
        break

    query_emb = model.encode(query).astype("float32").reshape(1, -1)

    D, I = index.search(query_emb, k=5)

    results = []

    print("\nTop results:")
    for rank, idx in enumerate(I[0]):
        frame_path = frame_paths[idx]

        # 🔥 TIMESTAMP CALCULATION
        frame_number = int(frame_path.split("_")[-1].split(".")[0])
        seconds = frame_number * 2  # because we sampled every 2 sec

        hh = seconds // 3600
        mm = (seconds % 3600) // 60
        ss = seconds % 60

        timestamp = f"{hh:02}:{mm:02}:{ss:02}"

        score = float(D[0][rank])

        print(f"{timestamp} | {frame_path} | score: {score:.4f}")

        results.append({
            "query": query,
            "timestamp": timestamp,
            "frame": frame_path,
            "score": score
        })

    # 🔥 SAVE RESULTS
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nResults saved to results.json")