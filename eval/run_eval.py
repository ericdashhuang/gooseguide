"""measure retrieval accuracy against eval/questions.json

for each question, retrieves the top k chunks from the vector index and
checks whether the expected source page shows up among them (recall@k)
a chunk from the right page is useful even if it's not the exact original
chunk, so page level recall is the metric that matters here

usage python eval/run_eval.py
requires rag/build_index.py has already been run (needs the index), and
eval/questions.json exists (run eval/generate_questions.py first)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "rag"))
from query import TOP_K, retrieve  # noqa: E402

HERE = Path(__file__).parent
QUESTIONS_PATH = HERE / "questions.json"


def main():
    if not QUESTIONS_PATH.exists():
        raise SystemExit(f"{QUESTIONS_PATH} not found -- run eval/generate_questions.py first.")
    questions = json.loads(QUESTIONS_PATH.read_text())

    hits = 0
    misses = []
    for q in questions:
        chunks = retrieve(q["question"], k=TOP_K)
        retrieved_urls = {c["meta"]["source_url"] for c in chunks}
        if q["source_url"] in retrieved_urls:
            hits += 1
        else:
            misses.append(q)

    total = len(questions)
    recall = hits / total if total else 0
    print(f"Recall@{TOP_K}: {hits}/{total} ({recall:.0%})\n")

    if misses:
        print("Missed questions:")
        for m in misses:
            print(f"  - \"{m['question']}\" (expected {m['source_title']})")


if __name__ == "__main__":
    main()
