"""Unit tests for eval/analyze_corpus.py.

Pure-function tests over small, hand-built DataFrames: no network, no
Chroma index, no LLM calls. Expected values are computed independently
(with the stdlib `statistics` module rather than numpy) so the test
isn't just re-stating the implementation.
"""
import statistics
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
from analyze_corpus import (  # noqa: E402
    compute_chunk_stats,
    compute_page_recall,
    flag_weak_pages,
    run_retrieval,
)


@pytest.fixture
def chunks_df():
    return pd.DataFrame([
        {"id": "a-0", "source_url": "https://example.com/a", "source_title": "Page A", "chunk_index": 0, "text": "x" * 100, "char_len": 100},
        {"id": "a-1", "source_url": "https://example.com/a", "source_title": "Page A", "chunk_index": 1, "text": "x" * 200, "char_len": 200},
        {"id": "b-0", "source_url": "https://example.com/b", "source_title": "Page B", "chunk_index": 0, "text": "x" * 300, "char_len": 300},
    ])


@pytest.fixture
def questions_df():
    return pd.DataFrame([
        {"question": "q about a (1)", "source_url": "https://example.com/a", "source_title": "Page A"},
        {"question": "q about a (2)", "source_url": "https://example.com/a", "source_title": "Page A"},
        {"question": "q about b", "source_url": "https://example.com/b", "source_title": "Page B"},
    ])


def fake_retrieve_always_hits_a_never_b(question: str, k: int):
    # Every question retrieves a chunk from page A only, regardless of the
    # question text -- so questions about A "hit" and questions about B miss.
    return [{"text": "x", "meta": {"source_url": "https://example.com/a", "source_title": "Page A"}, "distance": 0.1}]


def test_compute_chunk_stats_matches_independently_computed_values(chunks_df):
    stats = compute_chunk_stats(chunks_df)

    lengths = [100, 200, 300]
    assert stats["count"] == 3
    assert stats["mean_len"] == pytest.approx(statistics.mean(lengths))
    assert stats["std_len"] == pytest.approx(statistics.pstdev(lengths))
    assert stats["min_len"] == 100
    assert stats["max_len"] == 300


def test_run_retrieval_marks_hit_when_source_url_in_results(questions_df):
    results = run_retrieval(questions_df, fake_retrieve_always_hits_a_never_b, k=5)

    assert list(results["hit"]) == [True, True, False]
    assert set(results.columns) >= {"question", "source_url", "source_title", "hit", "best_distance"}


def test_compute_page_recall_aggregates_per_page(questions_df):
    results = run_retrieval(questions_df, fake_retrieve_always_hits_a_never_b, k=5)
    recall = compute_page_recall(results).set_index("source_url")

    assert recall.loc["https://example.com/a", "n_questions"] == 2
    assert recall.loc["https://example.com/a", "hits"] == 2
    assert recall.loc["https://example.com/a", "recall"] == pytest.approx(1.0)

    assert recall.loc["https://example.com/b", "n_questions"] == 1
    assert recall.loc["https://example.com/b", "hits"] == 0
    assert recall.loc["https://example.com/b", "recall"] == pytest.approx(0.0)


def test_flag_weak_pages_returns_only_pages_below_threshold(questions_df):
    results = run_retrieval(questions_df, fake_retrieve_always_hits_a_never_b, k=5)
    recall = compute_page_recall(results)

    weak = flag_weak_pages(recall, threshold=0.5)

    assert list(weak["source_url"]) == ["https://example.com/b"]
