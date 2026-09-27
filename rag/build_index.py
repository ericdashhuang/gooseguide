"""build the vector index chunk every file in data/raw/, embed each chunk with
chroma's bundled local embedding model (all-MiniLM-L6-v2 via onnxruntime, no
api key or gpu needed), and persist the collection to ./chroma_db

usage python rag/build_index.py
"""
from pathlib import Path

import chromadb

from chunking import chunk_raw_file

HERE = Path(__file__).parent
RAW_DIR = HERE.parent / "data" / "raw"
DB_DIR = HERE.parent / "chroma_db"
COLLECTION_NAME = "waterloo_first_year"


def main():
    raw_files = sorted(RAW_DIR.glob("*.txt"))
    if not raw_files:
        raise SystemExit(f"No files found in {RAW_DIR} -- run data/ingest.py first.")

    client = chromadb.PersistentClient(path=str(DB_DIR))
    # fresh build every run, so rerunning after editing the corpus doesn't
    # leave stale chunks behind
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    ids, documents, metadatas = [], [], []
    for path in raw_files:
        for chunk in chunk_raw_file(path):
            ids.append(f"{path.stem}-{chunk.chunk_index}")
            documents.append(chunk.text)
            metadatas.append({
                "source_url": chunk.source_url,
                "source_title": chunk.source_title,
                "chunk_index": chunk.chunk_index,
            })

    print(f"Embedding {len(documents)} chunks from {len(raw_files)} pages...")
    # chroma batches and embeds internally when you call add()
    batch = 100
    for i in range(0, len(ids), batch):
        collection.add(
            ids=ids[i:i + batch],
            documents=documents[i:i + batch],
            metadatas=metadatas[i:i + batch],
        )
        print(f"  {min(i + batch, len(ids))}/{len(ids)}")

    print(f"Done. Collection '{COLLECTION_NAME}' has {collection.count()} chunks, persisted to {DB_DIR}")


if __name__ == "__main__":
    main()
