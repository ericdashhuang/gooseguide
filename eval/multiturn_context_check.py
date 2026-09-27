"""one off measurement backing the "prepending the previous question fixes
multiturn context loss" claim in the readme and portfolio writeup

not part of the ci suite (see run_eval.py for that), this is a single,
hand picked representative example, not a statistical benchmark, it exists so
the improvement is a real, reproducible number instead of an assumed one

methodology
  1. load the already built chroma index (run rag/build_index.py first)
  2. take a representative follow up pair a first question ("how many coop
     work terms do i need to complete?") and a short, ambiguous follow up
     ("what about abroad?") that only makes sense given the first question
  3. embed the follow up two ways, alone, and with the first question
     prepended (exactly what docs/app.js does before calling retrieve())
     using the same embedding model and cosine similarity math the browser
     app uses (normalized vectors, dot product)
  4. compare each version's similarity to the chunks of the page that
     actually answers the follow up ("work abroad coop requirements") and
     show where that page lands in the top-5 results either way

usage python eval/multiturn_context_check.py
"""
import sys
from pathlib import Path

import numpy as np
import chromadb
from chromadb.utils import embedding_functions

sys.path.insert(0, str(Path(__file__).parent.parent / "rag"))

DB_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "waterloo_first_year"
TARGET_URL = "https://uwaterloo.ca/co-operative-education/work-abroad/before-you-go/co-op-requirements"
TARGET_TITLE = "Work abroad co-op requirements"

PRIOR_QUESTION = "How many co-op work terms do I need to complete?"
FOLLOWUP = "What about abroad?"


def main():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    ef = embedding_functions.DefaultEmbeddingFunction()

    all_docs = collection.get(include=["metadatas", "embeddings"])
    embs = np.array(all_docs["embeddings"])
    norm_embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    def top5(query_text):
        q = np.array(ef([query_text])[0])
        q = q / np.linalg.norm(q)
        sims = norm_embs @ q
        order = np.argsort(-sims)[:5]
        return [
            (all_docs["metadatas"][i]["source_title"], all_docs["metadatas"][i]["source_url"], round(float(sims[i]), 3))
            for i in order
        ]

    followup_with_context = f"{PRIOR_QUESTION}\n{FOLLOWUP}"

    print(f'Query: "{FOLLOWUP}" (embedded alone, the bug)')
    for title, url, sim in top5(FOLLOWUP):
        marker = "  <-- target page" if url.rstrip("/") == TARGET_URL else ""
        print(f"  {sim}  {title}{marker}")

    print(f'\nQuery: "{followup_with_context}" (prior question prepended, the fix)')
    for title, url, sim in top5(followup_with_context):
        marker = "  <-- target page" if url.rstrip("/") == TARGET_URL else ""
        print(f"  {sim}  {title}{marker}")


if __name__ == "__main__":
    main()
