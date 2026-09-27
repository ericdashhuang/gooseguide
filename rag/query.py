"""ask a question, retrieve the most relevant chunks from the vector index,
and (if an api key is available) ask an llm to answer using only those chunks

usage
  python rag/query.py "how many coop work terms do i need to complete?"

without ANTHROPIC_API_KEY set, this still runs the retrieval step and prints
the chunks it would have used, so you can inspect retrieval quality on its own
"""
import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv

HERE = Path(__file__).parent
DB_DIR = HERE.parent / "chroma_db"
COLLECTION_NAME = "waterloo_first_year"
TOP_K = 5

SYSTEM_PROMPT = """You are a helpful assistant answering questions about being a \
first-year student at the University of Waterloo, based only on the provided \
context from uwaterloo.ca. If the context doesn't contain the answer, say so \
plainly instead of guessing. Cite the source URL(s) you used at the end of \
your answer."""


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


def format_context(chunks) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[{i}] Source: {c['meta']['source_title']} ({c['meta']['source_url']})\n{c['text']}"
        )
    return "\n\n".join(parts)


def generate_answer(question: str, chunks) -> str:
    import anthropic

    client = anthropic.Anthropic()
    context = format_context(chunks)
    message = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Context:\n\n{context}\n\nQuestion: {question}",
        }],
    )
    return message.content[0].text


def main():
    load_dotenv()
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python rag/query.py "your question"')
    question = " ".join(sys.argv[1:])

    chunks = retrieve(question)

    print(f"Question: {question}\n")
    print(f"Retrieved {len(chunks)} chunks:")
    for i, c in enumerate(chunks, 1):
        preview = c["text"][:150].replace("\n", " ")
        print(f"  [{i}] (distance={c['distance']:.3f}) {c['meta']['source_title']} -- {preview}...")
    print()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("No ANTHROPIC_API_KEY set -- skipping generation step.")
        print("Set it (e.g. in a .env file) to get a generated answer instead of raw chunks.")
        return

    print("Generating answer...\n")
    answer = generate_answer(question, chunks)
    print(answer)


if __name__ == "__main__":
    main()
