"""export the chunked corpus (text + metadata, no vectors) as json for the
static web demo, the browser embeds these itself with transformers.js, so the
same embedding model produces both the corpus vectors and the query vector
no risk of a python export vs browser export mismatch

usage python rag/export_chunks.py
"""
import json
from pathlib import Path

from chunking import chunk_raw_file

HERE = Path(__file__).parent
RAW_DIR = HERE.parent / "data" / "raw"
OUT_PATH = HERE.parent / "docs" / "data" / "chunks.json"


def main():
    raw_files = sorted(RAW_DIR.glob("*.txt"))
    chunks = []
    for path in raw_files:
        for chunk in chunk_raw_file(path):
            chunks.append({
                "id": f"{path.stem}-{chunk.chunk_index}",
                "text": chunk.text,
                "source_url": chunk.source_url,
                "source_title": chunk.source_title,
            })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(chunks))
    print(f"Exported {len(chunks)} chunks to {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
