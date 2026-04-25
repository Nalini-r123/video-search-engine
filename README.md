# 🎥 Intelligent Video Search Engine

## 📌 Overview

This project implements an intelligent video search system that allows users to query video content using natural language. It retrieves the most relevant frames along with timestamps and similarity scores.

---

## 🚀 Features

* Natural language query support
* Frame-level video search
* Timestamp extraction
* Similarity score ranking
* JSON result export

---

## 🧠 Architecture

* Video → Frame extraction using OpenCV
* Frames → Embeddings using CLIP (Sentence Transformers)
* Embeddings stored in FAISS
* Query → embedding → similarity search

---

## ⚙️ How to Run

### Install dependencies

pip install opencv-python faiss-cpu sentence-transformers pillow

### Run indexing

python indexer.py

### Run search

python app.py

---

## 🔍 Example Query

person walking
car
people talking

---

## 📊 Output

Each result includes:

* Timestamp (HH:MM:SS)
* Frame path
* Similarity score

Example:
00:01:02 | frames/frame_31.jpg | score: 134.86

---

## 📁 Output File

Results are saved in:
results.json

---

## 🧩 Design Decisions

* Used frame sampling (every 2 seconds)
* Used CLIP for semantic understanding
* Used FAISS for fast retrieval

---

## ⚠️ Limitations

* No temporal reasoning
* Approximate timestamps
* Basic ranking

---

## 🚀 Future Improvements

* Add UI (Streamlit)
* Add temporal filtering
* Improve ranking

---

## 🎥 Demo Video

(Add your video link here)

---
