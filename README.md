# GooseGuide
**Live demo:** https://ericdashhuang.github.io/gooseguide/

A retrieval-augmented generation (RAG) system that answers questions about starting first year at the University of Waterloo, including co-op, housing, orientation, and campus wellness.


## Why this exists

A plain LLM either doesn't know anything about a specific school's policies, or it guesses and gets details wrong.
This project fixes that by retrieving real content from uwaterloo.ca before generating an answer, so responses are grounded in actual source pages instead of the model's memory.

## How it works

1. **Ingest** (`data/ingest.py`) — fetches 73 public uwaterloo.ca pages (co-op, housing, orientation, international students, campus wellness, registrar) and strips them down to clean article text.
2. **Chunk** (`rag/chunking.py`) — splits each page into ~900-character overlapping chunks along paragraph boundaries, so a chunk rarely cuts a sentence in half.
3. **Embed + index** (`rag/build_index.py`) — converts every chunk into a vector using a local embedding model (`all-MiniLM-L6-v2`, bundled with Chroma, runs on CPU, no API key needed) and stores it in a persistent [Chroma](https://www.trychroma.com/) vector database.
4. **Retrieve + generate** (`rag/query.py`) — embeds the question the same way, finds the most similar chunks by vector search, and passes them to an LLM as context so it can answer from real source material and cite where the answer came from.

The live demo (see below) additionally supports follow-up questions: a short follow-up like "what about abroad?" gets the previous question prepended before it's embedded for retrieval, so retrieval has enough context to find the right page instead of searching on the fragment alone.

No LangChain, no LlamaIndex — every step above is under 100 lines of plain Python, on purpose.
The goal was to actually understand what a RAG pipeline does, not to call a framework method and trust that it works.

## Project structure

```
data/
  seed_urls.txt        curated list of source pages
  discover_urls.py      one-off crawler used to find candidate pages
  ingest.py              fetches + cleans pages into data/raw/
  raw/                    cleaned page text (checked into git)
rag/
  chunking.py            splits page text into overlapping chunks
  build_index.py         embeds chunks and builds the Chroma index
  query.py                retrieval + generation CLI
  export_chunks.py       exports chunk text (no vectors) for the web demo
eval/
  generate_questions.py  LLM-generates one ground-truth question per source page
  run_eval.py             measures retrieval recall against those questions
  analyze_corpus.py       turns eval results into per-page recall, chunk stats, and a SQLite + Excel report
docs/                    static client-side demo, served by GitHub Pages (see below)
  pure.js                 DOM-free retrieval/formatting helpers, unit tested directly
tests/                   unit tests for rag/chunking.py and eval/analyze_corpus.py
.github/workflows/       weekly corpus-refresh + CI test runs (see below)
```

## Running it locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Fetch and clean the source pages (already included in data/raw/, re-run to refresh)
python data/ingest.py

# 2. Build the vector index (~30 seconds)
python rag/build_index.py

# 3. Ask a question
python rag/query.py "How many co-op work terms do I need to complete?"
```

Retrieval works with no setup.
Generation needs an LLM key: copy `.env.example` to `.env` and add `ANTHROPIC_API_KEY=...`.
Without a key, `query.py` still prints the retrieved chunks so you can inspect retrieval quality on its own — that's deliberate, since retrieval and generation are genuinely separate steps worth being able to test independently.

## Live demo

**[ericdashhuang.github.io/gooseguide](https://ericdashhuang.github.io/gooseguide/)**

A static, browser-only version of this lives on GitHub Pages: see `docs/`.
It computes embeddings client-side (via [transformers.js](https://huggingface.co/docs/transformers.js)), so there's no backend server and no shared API key that a stranger could drain.
Retrieval works immediately with no setup.
To also get a generated written answer (not just the retrieved chunks), you paste your own Anthropic or Groq API key into the page — Groq has a free tier — and it's used directly from your browser to call the provider, never sent anywhere else or stored, unless you explicitly opt in to remembering it in that browser's local storage.

## Measuring retrieval accuracy

Instead of eyeballing a few questions by hand, `eval/` turns "does retrieval work" into a number:

```bash
# 1. Generate one ground-truth question per source page (needs ANTHROPIC_API_KEY or GROQ_API_KEY)
python eval/generate_questions.py

# 2. Check what fraction of them retrieve the right source page
python rag/build_index.py   # if you haven't already
python eval/run_eval.py
```

`run_eval.py` reports recall@5 (did the correct source page show up in the top 5 retrieved chunks) and lists which questions missed, so a regression in `rag/chunking.py` or a change to `TOP_K` shows up as a number going down instead of going unnoticed.

`run_eval.py`'s 73 generated questions are all single-turn, so they don't cover the follow-up-question fix described above (prepending the prior question before embedding a follow-up).
`eval/multiturn_context_check.py` is a separate, hand-picked measurement of that specific fix: it embeds a representative follow-up ("What about abroad?" after "How many co-op work terms do I need to complete?") both alone and with the prior question prepended, then shows where the actually-correct page ("Work abroad co-op requirements") lands in the top 5 either way.
Run it the same way as `run_eval.py`, after building the index.
This is a single illustrative example, not a statistical benchmark - see the script's own docstring for the exact methodology.

### Turning eval results into a report

`run_eval.py` prints one recall number to the terminal, which is enough to catch a regression but not enough to see *why* retrieval is weak on a specific page. `eval/analyze_corpus.py` takes the same inputs (the chunk export + the question set) and turns them into a structured report instead:

```bash
python rag/export_chunks.py   # if you haven't already - writes docs/data/chunks.json
python rag/build_index.py     # if you haven't already
python eval/analyze_corpus.py
```

It computes, with pandas/NumPy:
- per-page recall (which specific pages the index serves badly, not just the overall percentage)
- chunk-length statistics (mean, std, p95) across the corpus
- a flagged list of pages below a recall threshold, worth re-chunking or re-titling

then persists chunk metadata and per-question results as SQLite tables (`eval/corpus_analytics.db`), pulls a per-page summary back out with a single SQL join/aggregate query, and writes everything to a multi-sheet Excel workbook (`eval/corpus_report.xlsx`) plus the raw tables, instead of only stdout.

Every input path is a flag (`--chunks-file`, `--questions-file`, `--db`, `--output`, `--top-k`, `--weak-threshold`), so the same script works against a different corpus or question set without editing code - see `python eval/analyze_corpus.py --help`.

## Keeping the corpus fresh

uwaterloo.ca is a real site that changes — co-op requirements get updated, dates change year to year. `.github/workflows/refresh-corpus.yml` re-runs the ingest pipeline weekly and opens a PR if any of the 73 pages actually changed, so stale content gets caught automatically instead of silently going unnoticed. It never auto-merges — a human reviews the diff first, same as every other change to this repo.

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v            # rag/chunking.py (paragraph boundaries, overlap, hard-splitting oversized paragraphs)
                             # eval/analyze_corpus.py (recall aggregation, chunk stats, weak-page flagging)

node --test docs/pure.test.js   # docs/pure.js - similarity ranking, answer formatting/XSS-safety, conversation history
```

Both suites run automatically on every push and PR via `.github/workflows/tests.yml`. They're plain unit tests (no network, no API key, no browser) - separate from `eval/`, which measures retrieval *accuracy* rather than code correctness.

## Design decisions worth knowing about

- **Local embedding model, not an API.** Embeddings run through `onnxruntime` on CPU. This keeps the pipeline free to run and testable without any credentials, and it's genuinely fast enough at this corpus size (~400 chunks).
- **Chroma over a hosted vector DB.** It's a single persisted directory, not a service to run or an account to create — the simplest option that's still a real vector database with approximate nearest-neighbor search.
- **Bring-your-own-key for the live demo, not an embedded key.** A public site with an embedded API key is a real cost/abuse risk — anyone who finds it can run up charges. Letting each visitor use their own key removes that risk entirely instead of working around it.
- **Corpus scope.** Limited to official uwaterloo.ca pages under a handful of relevant sections. Reddit and other community sources were deliberately left out for v1, since scraping them reliably needs API credentials this project doesn't depend on.

## Possible next steps

- Add a re-ranking step after retrieval to improve answer quality on ambiguous questions.
- Expand the corpus with more sections (academic advising, clubs, transit).
- Add a browser smoke test (Playwright) that exercises the live demo end-to-end - the unit tests above cover the pure logic, not the actual UI.
