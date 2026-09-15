"""Turn raw retrieval results into a structured report: per-page recall,
chunk-length statistics, and a flagged list of pages the index serves badly.

This is a reusable, parameterized replacement for the print-only summary in
run_eval.py -- point it at any chunk export + question set (not just this
corpus) via --chunks-file / --questions-file, and it writes a multi-sheet
Excel report plus a queryable SQLite database instead of just stdout.

Usage:
  python eval/analyze_corpus.py
  python eval/analyze_corpus.py --chunks-file docs/data/chunks.json \
      --questions-file eval/questions.json --output eval/corpus_report.xlsx

Requires: rag/build_index.py has already been run (needs the index).
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "rag"))
from query import TOP_K, retrieve  # noqa: E402

HERE = Path(__file__).parent
DEFAULT_CHUNKS_PATH = HERE.parent / "docs" / "data" / "chunks.json"
DEFAULT_QUESTIONS_PATH = HERE / "questions.json"
DEFAULT_DB_PATH = HERE / "corpus_analytics.db"
DEFAULT_OUTPUT_PATH = HERE / "corpus_report.xlsx"
DEFAULT_WEAK_THRESHOLD = 0.5


def load_chunks_df(path: Path) -> pd.DataFrame:
    """Load an exported chunk file (rag/export_chunks.py's output) into a
    DataFrame with a derived char_len column, one row per chunk."""
    records = json.loads(Path(path).read_text())
    df = pd.DataFrame(records)
    df["char_len"] = df["text"].str.len()
    return df


def load_questions_df(path: Path) -> pd.DataFrame:
    return pd.DataFrame(json.loads(Path(path).read_text()))


def compute_chunk_stats(chunks_df: pd.DataFrame) -> dict:
    """Derived variables over chunk length, using NumPy for the arithmetic."""
    lengths = chunks_df["char_len"].to_numpy()
    return {
        "count": int(lengths.size),
        "mean_len": float(np.mean(lengths)),
        "std_len": float(np.std(lengths)),
        "min_len": int(np.min(lengths)),
        "max_len": int(np.max(lengths)),
        "p95_len": float(np.percentile(lengths, 95)),
    }


def run_retrieval(questions_df: pd.DataFrame, retrieve_fn, k: int) -> pd.DataFrame:
    """Run retrieval for every question and record a hit/miss per row.

    retrieve_fn is injected (rather than imported directly) so this is
    testable with a fake retriever and no live Chroma index.
    """
    rows = []
    for q in questions_df.to_dict("records"):
        chunks = retrieve_fn(q["question"], k)
        retrieved_urls = {c["meta"]["source_url"] for c in chunks}
        best_distance = min((c["distance"] for c in chunks), default=float("nan"))
        rows.append({
            "question": q["question"],
            "source_url": q["source_url"],
            "source_title": q["source_title"],
            "hit": q["source_url"] in retrieved_urls,
            "best_distance": best_distance,
        })
    return pd.DataFrame(rows)


def compute_page_recall(results_df: pd.DataFrame) -> pd.DataFrame:
    """Per-page recall: of the questions whose ground-truth answer lives on
    this page, how many actually retrieved it?"""
    grouped = results_df.groupby(["source_url", "source_title"], as_index=False).agg(
        n_questions=("hit", "size"),
        hits=("hit", "sum"),
    )
    grouped["hits"] = grouped["hits"].astype(int)
    grouped["recall"] = grouped["hits"] / grouped["n_questions"]
    return grouped


def flag_weak_pages(page_recall_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Pages whose recall is below threshold -- candidates for re-chunking,
    re-titling, or otherwise improving retrievability."""
    weak = page_recall_df[page_recall_df["recall"] < threshold]
    return weak.sort_values("recall").reset_index(drop=True)


def persist_to_sqlite(chunks_df: pd.DataFrame, results_df: pd.DataFrame, db_path: Path) -> None:
    """Write chunk metadata and eval results as SQLite tables so they can be
    queried with SQL instead of only held in memory."""
    con = sqlite3.connect(db_path)
    try:
        chunks_df.assign(chunk_index=chunks_df.get("chunk_index", 0)).to_sql(
            "chunks", con, if_exists="replace", index=False
        )
        results_df.assign(hit=results_df["hit"].astype(int)).to_sql(
            "eval_results", con, if_exists="replace", index=False
        )
    finally:
        con.close()


SQL_SUMMARY_QUERY = """
SELECT
    c.source_url,
    c.source_title,
    COUNT(DISTINCT c.id)                AS n_chunks,
    AVG(c.char_len)                     AS avg_chunk_len,
    COALESCE(r.n_questions, 0)          AS n_questions,
    COALESCE(r.hits, 0)                 AS hits
FROM chunks c
LEFT JOIN (
    SELECT source_url, COUNT(*) AS n_questions, SUM(hit) AS hits
    FROM eval_results
    GROUP BY source_url
) r ON c.source_url = r.source_url
GROUP BY c.source_url, c.source_title
ORDER BY n_questions DESC, c.source_url
"""


def query_summary_from_sqlite(db_path: Path) -> pd.DataFrame:
    """Extract a per-page summary by joining the two SQLite tables with a
    single SQL query, rather than re-doing the join in pandas."""
    con = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(SQL_SUMMARY_QUERY, con)
    finally:
        con.close()


def write_excel_report(output_path: Path, sheets: dict) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks-file", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--questions-file", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--weak-threshold", type=float, default=DEFAULT_WEAK_THRESHOLD)
    args = parser.parse_args()

    if not args.chunks_file.exists():
        raise SystemExit(f"{args.chunks_file} not found -- run rag/export_chunks.py first.")
    if not args.questions_file.exists():
        raise SystemExit(f"{args.questions_file} not found -- run eval/generate_questions.py first.")

    chunks_df = load_chunks_df(args.chunks_file)
    questions_df = load_questions_df(args.questions_file)

    print(f"Retrieving top-{args.top_k} chunks for {len(questions_df)} questions...")
    results_df = run_retrieval(questions_df, retrieve, args.top_k)

    chunk_stats = compute_chunk_stats(chunks_df)
    page_recall_df = compute_page_recall(results_df)
    weak_pages_df = flag_weak_pages(page_recall_df, args.weak_threshold)

    persist_to_sqlite(chunks_df, results_df, args.db)
    sql_summary_df = query_summary_from_sqlite(args.db)

    overall_recall = results_df["hit"].mean()
    print(f"\nOverall recall@{args.top_k}: {results_df['hit'].sum()}/{len(results_df)} ({overall_recall:.0%})")
    print(f"Chunk stats: {chunk_stats['count']} chunks, mean {chunk_stats['mean_len']:.0f} chars "
          f"(std {chunk_stats['std_len']:.0f}, p95 {chunk_stats['p95_len']:.0f})")
    if len(weak_pages_df):
        print(f"\n{len(weak_pages_df)} page(s) below {args.weak_threshold:.0%} recall:")
        for row in weak_pages_df.itertuples():
            print(f"  - {row.source_title} ({row.recall:.0%}, {row.hits}/{row.n_questions})")

    write_excel_report(args.output, {
        "Summary": pd.DataFrame([{"overall_recall": overall_recall, **chunk_stats}]),
        "Per-Page Recall": page_recall_df,
        "Weak Pages": weak_pages_df,
        "Per-Question Results": results_df,
        "SQL Summary": sql_summary_df,
    })
    print(f"\nWrote report to {args.output}")
    print(f"Wrote queryable tables to {args.db}")


if __name__ == "__main__":
    main()
