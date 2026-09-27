"""retrieve the most relevant chunks for a question from the vector index

usage
  python rag/retrieval.py "how many coop work terms do i need to complete?"

prints the chunks it found so you can inspect retrieval quality on its own,
the browser demo (docs/app.js) is what actually generates answers
"""
import sys
from pathlib import Path

import chromadb

HERE = Path(__file__).parent
DB_DIR = HERE.parent / "chroma_db"
COLLECTION_NAME = "waterloo_first_year"
TOP_K = 5


def retrieve(question: str, k: int = TOP_K):
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    results = collection.query(query_texts=[question], n_results=k)
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        chunks.append({"text": doc, "meta": meta, "distance": dist})
    return chunks


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python rag/retrieval.py "your question"')
    question = " ".join(sys.argv[1:])

    chunks = retrieve(question)

    print(f"Question: {question}\n")
    print(f"Retrieved {len(chunks)} chunks:")
    for i, c in enumerate(chunks, 1):
        preview = c["text"][:150].replace("\n", " ")
        print(f"  [{i}] (distance={c['distance']:.3f}) {c['meta']['source_title']} -- {preview}...")
    print()


if __name__ == "__main__":
    main()
