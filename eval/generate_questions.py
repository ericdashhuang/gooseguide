"""generate an evaluation question for each source page, using an llm

reads every file in data/raw/, asks an llm to write one realistic question a
first year student would ask that this specific page answers, and saves the
result as eval/questions.json, a small ground truth set for measuring
retrieval accuracy (see eval/run_eval.py)

needs an api key set ANTHROPIC_API_KEY or GROQ_API_KEY (eg in a .env
file), groq's free tier is enough for this, ~73 short requests, one per
source page

usage python eval/generate_questions.py
"""
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "rag"))
from chunking import parse_raw_file  # noqa: E402

HERE = Path(__file__).parent
RAW_DIR = HERE.parent / "data" / "raw"
OUT_PATH = HERE / "questions.json"

PROMPT = """Below is the text of a real page from the University of Waterloo's \
website, aimed at first-year students. Write ONE realistic question a \
first-year student would actually type into a search box, that this page \
specifically answers. Keep it under 20 words, and don't reference "this page" \
or "the text above". Return only the question itself, no quotes, no preamble.

Page title: {title}

Page text:
{body}"""


def ask_anthropic(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def ask_groq(prompt: str, max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
            json={
                "model": "openai/gpt-oss-120b",
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        if resp.status_code == 429 and attempt < max_retries - 1:
            # free tier throttles harder than you'd expect, back off and retry
            # instead of losing the whole run, respect retry after if groq sends one
            wait = float(resp.headers.get("retry-after", 2 ** attempt))
            print(f"    Rate limited, waiting {wait:.0f}s before retrying...", file=sys.stderr)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    raise SystemExit("Groq API kept rate-limiting after retries -- wait a bit and re-run (already-generated questions are saved, so it'll resume).")


def ask(title: str, body: str) -> str:
    prompt = PROMPT.format(title=title, body=body[:3000])
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ask_anthropic(prompt)
    if os.environ.get("GROQ_API_KEY"):
        return ask_groq(prompt)
    raise SystemExit("Set ANTHROPIC_API_KEY or GROQ_API_KEY (e.g. in a .env file) first.")


def main():
    load_dotenv()
    raw_files = sorted(RAW_DIR.glob("*.txt"))
    if not raw_files:
        raise SystemExit(f"No files found in {RAW_DIR} -- run data/ingest.py first.")

    # resume support if questions.json already has entries (eg a previous
    # run got rate limited partway through), skip pages already done instead
    # of respending api calls and overwriting good results
    questions = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else []
    done_urls = {q["source_url"] for q in questions}
    if done_urls:
        print(f"Resuming - {len(done_urls)} questions already generated.")

    for i, path in enumerate(raw_files, 1):
        url, title, body = parse_raw_file(path.read_text())
        if url in done_urls:
            continue
        question = ask(title, body)
        questions.append({"question": question, "source_url": url, "source_title": title})
        # save after every question, not just at the end, so a crash or
        # rate limit mid run only costs the one in flight request
        OUT_PATH.write_text(json.dumps(questions, indent=2))
        print(f"  [{i}/{len(raw_files)}] {title}: {question}")

    print(f"\nWrote {len(questions)} questions to {OUT_PATH}")


if __name__ == "__main__":
    main()
