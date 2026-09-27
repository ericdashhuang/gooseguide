"""turn a raw page's text into overlapping chunks small enough to embed and
retrieve individually, while keeping each chunk on paragraph boundaries where
possible so we don't cut a sentence in half mid thought
"""
from dataclasses import dataclass

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


@dataclass
class Chunk:
    text: str
    source_url: str
    source_title: str
    chunk_index: int


def parse_raw_file(text: str) -> tuple[str, str, str]:
    """split a data/raw/*.txt file into (url, title, body)"""
    lines = text.splitlines()
    url = lines[0].removeprefix("URL: ").strip()
    title = lines[1].removeprefix("TITLE: ").strip()
    body = "\n".join(lines[3:]).strip()
    return url, title, body


def chunk_text(body: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    paragraphs = [p.strip() for p in body.split("\n") if p.strip()]

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n{para}".strip() if current else para
        if len(candidate) <= size:
            current = candidate
            continue

        if current:
            chunks.append(current)
        if len(para) <= size:
            current = para
        else:
            # a single paragraph longer than the chunk size hard split it
            for i in range(0, len(para), size - overlap):
                chunks.append(para[i:i + size])
            current = ""

    if current:
        chunks.append(current)

    # add overlap by prefixing each chunk (after the first) with the tail of
    # the previous one, so retrieval doesn't lose context at a chunk boundary
    overlapped = []
    for i, c in enumerate(chunks):
        if i == 0:
            overlapped.append(c)
        else:
            tail = chunks[i - 1][-overlap:]
            overlapped.append(f"{tail}\n{c}")
    return overlapped


def chunk_raw_file(path) -> list[Chunk]:
    url, title, body = parse_raw_file(path.read_text())
    return [
        Chunk(text=text, source_url=url, source_title=title, chunk_index=i)
        for i, text in enumerate(chunk_text(body))
    ]
