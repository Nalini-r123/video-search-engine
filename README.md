# Intelligent Video Search Engine

## Overview
This project implements a video search system that allows users to query video content using natural language. It retrieves the most relevant frames along with timestamps and similarity scores.

## Features
- Natural language query support
- Frame-level video search
- Timestamp extraction
- Similarity score ranking
- JSON result export

## Architecture
1. Video ingestion and frame extraction using OpenCV
2. Frame embeddings generated using CLIP (via Sentence Transformers)
3. FAISS used for efficient similarity search
4. Query is converted to embedding and matched with frame embeddings

## How It Works
- Video is processed offline
- Frames are sampled every 2 seconds
- Each frame is converted into an embedding
- Query is converted into embedding
- FAISS retrieves top-K similar frames

## How to Run

### Install dependencies
