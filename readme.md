# RAG-Based AI Teaching Assistant
## Overview
This project demonstrates how RAG can be used to build a domain-specific AI tutor. The assistant processes course videos, converts them into embeddings, and uses Llama 3.2 as the reasoning engine to answer user questions with precise context from the training material.

---

## How It Works
1. **Collectig Videos:**

    First, the lecture videos are collected.

2. **Converting:**

    The videos are converted to `.mp3` files with the help of **FFmpeg**

3. **Transcription & Chunking:**

    Audio files are transcribed into text using **OpenAI Whisper** and divided into chunks which stored as `.json` files.

4. **Merging Chunks:**

    To increase the accuracy of the model the chunks are merged

5. **Embedding & Knowledge Base Creation:**

    The JSON transcripts are converted into vector embeddings using the **bge-m3** embedding model. The embeddings are stored as a **DataFrame** and serialized using `joblib` for fast retrieval.

6. **Question Answeing**

    - User queries are converted into embeddings.
    - The system finds the most relevant video chunks using **cosine similarity**.
    - The context is then sent to **Llama 3.2** to generate a detailed answer.

---

## Tech Stack
| Component | Technology |
| --------- | ---------- |
| Backend | Python, Flask |
| Embeddings | bge-m3 |
| AI Model | Llama 3.2 |
| Video Processing | FFmpeg |
| Speech to Text | OpenAI Whisper |
| Similarity Search | Cosine Similarity |
| Storage | JSON, Joblib |

---

## Setup and Usage
### Step 1 — Collect Your Videos
Move all your course videos into the `videos/` folder.

### Step 2 — Convert to MP3

Convert all the video files to mp3 by running `process_video.py`

### Step 3 — Transcribe Audio to JSON

Convert all the mp3 files to json by running `create_chunks.py` and merge them by running `merge_chunks.py`

### Step 4 — Generate Embeddings

Convert the JSON transcripts into embeddings by running `read_chunks.py` and it will also save them as a `.joblib` file.

### Step 5 — Start the Web App

```bash
python app.py
```

Then open your browser and visit:
```
http://localhost:5000
```

Ask any question related to your course — the assistant will retrieve relevant content and answer using Llama 3.2.

---

## Author

**Debasish Sarkar**