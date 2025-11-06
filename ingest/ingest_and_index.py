"""
- Reads .txt files from docs/
- Splits into overlapping chunks
- Embeds chunks using sentence-transformers
- Stores metadata in SQLite (neurolens_meta.db)
- Writes FAISS index to neurolens.faiss
"""
import sqlite3
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import argparse
import json
import os
import math

MODEL_NAME = "all-MiniLM-L6-v2"  
DB_PATH = "neurolens_meta.db"
FAISS_INDEX_PATH = "neurolens.faiss"
META_JSON = "neurolens_meta_map.json" 

def create_meta_db(db_path = DB_PATH):
    # meta data stores vector idx -> (doc_id, chunk_idx, text[:200])
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id TEXT,
        chunk_idx INTEGER,
        text TEXT
    )
    """)
    conn.commit()
    conn.close()

def chunk_text(text, chunk_size=500, overlap = 50):
    if len(text)<=chunk_size:
        return [text]
    chunk = []
    start = 0
    while start < len(text):
        end = min(len(text), start+chunk_size)
        chunk.append(text[start:end])
        start = end  - overlap
        if end == len(text):
            break
    return chunk

def ingest_docs(docs_dir="docs", model_name=MODEL_NAME, db_path=DB_PATH, faiss_path=FAISS_INDEX_PATH, meta_json=META_JSON, chunk_size=500, overlap=50):
    model = SentenceTransformer(model_name)
    create_meta_db(db_path)
    conn = sqlite3.connect(db_path)
    c=conn.cursor()

    texts = []
    meta_map = []

    docs = sorted(Path(docs_dir).glob("*.txt"))
    if not docs:
        raise SystemExit(f"No txt files found in {docs_dir}. Add some sample docs and re-run.")
    
    for p in docs:
        doc_id = p.stem
        raw = p.read_text(encoding="utf-8")
        chunks = chunk_text(raw, chunk_size=chunk_size, overlap=overlap)
        for i, chunk in enumerate(chunks):
            # store meta
            c.execute("INSERT INTO chunks (doc_id, chunk_idx, text) VALUES (?,?,?)", (doc_id, i, chunk))
            texts.append(chunk)
            meta_map.append((doc_id, i, chunk[:200].replace("\n", " ")))
    conn.commit()
    conn.close()
    print(f"Inserted {len(texts)} chunks into SQLite at {db_path}")

    # encode embeddings
    print("Computing embeddings with model:", model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # L2 normalize for cosine similarity with FAISS index flat IP
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    print("Embedding dimension:", dim)
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors == cosine sim
    index.add(embeddings)
    faiss.write_index(index, faiss_path)
    print("Wrote FAISS index to", faiss_path)

    # write meta map
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta_map, f, indent=2, ensure_ascii=False)
    print("Wrote meta map to", meta_json)
    print("Ingestion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=str, default="docs", help="docs directory")
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()
    ingest_docs(docs_dir=args.docs, chunk_size=args.chunk_size, overlap=args.overlap)