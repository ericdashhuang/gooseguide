"""unit tests for rag/chunking.py, the step between ingest and embedding

these are pure function tests no network, no embedding model, no api key
run with pytest
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "rag"))
from chunking import CHUNK_OVERLAP, CHUNK_SIZE, chunk_text, parse_raw_file  # noqa: E402


def test_parse_raw_file_extracts_url_title_body():
    raw = "URL: https://example.com/page\nTITLE: Example Page\n\nFirst paragraph.\nSecond paragraph."
    url, title, body = parse_raw_file(raw)
    assert url == "https://example.com/page"
    assert title == "Example Page"
    assert body == "First paragraph.\nSecond paragraph."


def test_short_paragraphs_combine_into_a_single_chunk():
    body = "Para one.\nPara two.\nPara three."
    chunks = chunk_text(body)
    assert chunks == ["Para one.\nPara two.\nPara three."]


def test_long_body_splits_at_paragraph_boundary_not_mid_sentence():
    para1 = "X" * 600
    para2 = "Y" * 600
    body = f"{para1}\n{para2}"

    chunks = chunk_text(body)

    assert len(chunks) == 2
    # first chunk is exactly the first paragraph, not truncated mid way
    assert chunks[0] == para1
    # second chunk contains the second paragraph fully intact
    assert chunks[1].endswith(para2)


def test_overlap_prefixes_each_chunk_after_the_first():
    para1 = "X" * 600
    para2 = "Y" * 600
    body = f"{para1}\n{para2}"

    chunks = chunk_text(body)

    # chunks[1] should start with the last CHUNK_OVERLAP characters of chunks[0]'s
    # source paragraph, so retrieval doesn't lose context at the boundary
    assert chunks[1].startswith(para1[-CHUNK_OVERLAP:])
    assert chunks[1][CHUNK_OVERLAP] == "\n"


def test_oversized_single_paragraph_gets_hard_split():
    # one "paragraph" (no newlines) well over the chunk size
    body = "Z" * (CHUNK_SIZE * 2 + 200)
    chunks = chunk_text(body)
    # a single huge paragraph must still become multiple chunks, not one giant blob
    assert len(chunks) > 1


def test_no_chunk_wildly_exceeds_size_plus_overlap():
    # a realistic ish multiparagraph body of varying paragraph lengths
    paragraphs = [f"Paragraph {i}: " + ("word " * (i * 20)) for i in range(1, 10)]
    body = "\n".join(paragraphs)
    chunks = chunk_text(body)
    # overlap prefixes at most CHUNK_OVERLAP chars plus one newline onto a
    # chunk that was already at most CHUNK_SIZE before the prefix was added
    for c in chunks:
        assert len(c) <= CHUNK_SIZE + CHUNK_OVERLAP + 1


def test_no_empty_chunks():
    paragraphs = [f"Paragraph {i} has some real content in it." for i in range(1, 8)]
    body = "\n".join(paragraphs)
    chunks = chunk_text(body)
    assert all(c.strip() for c in chunks)


def test_short_body_returns_exactly_one_chunk():
    chunks = chunk_text("Just one short paragraph, well under the size limit.")
    assert len(chunks) == 1
