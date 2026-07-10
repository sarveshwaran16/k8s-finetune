import os
import json
import time
import requests
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dataset.jsonl")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# NOTE: llama3-8b-8192 was deprecated by Groq (May 2025), and its successor
# llama-3.1-8b-instant was itself deprecated in June 2026. Groq's current
# recommended replacement is openai/gpt-oss-20b. Set GROQ_MODEL in your .env
# to override (e.g. "openai/gpt-oss-120b" for higher quality, slower/pricier).
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

SYSTEM_PROMPT = (
    "You are a Kubernetes expert creating a training dataset. "
    "You always respond with valid JSON only — no preamble, no markdown "
    "code fences, no commentary."
)

GENERATION_PROMPT = """Read the following Kubernetes documentation and generate exactly 5 question-answer pairs.

Rules:
- Questions must be specific and practical (troubleshooting, configuration, concepts)
- Answers must be accurate, thorough, and based ONLY on the provided text
- Answers can be a few sentences long where useful — don't artificially truncate them
- Format your response as valid JSON only, nothing else:

{{"pairs": [
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}}
]}}

Document title: {title}
Source: {source}

Text:
{text}

JSON:"""


MAX_RETRIES = 6          # how many times to retry a 429 before giving up on this doc
DEFAULT_BACKOFF = 5.0    # seconds, used if Groq doesn't tell us how long to wait
MAX_BACKOFF = 120.0      # cap for genuine short (TPM) waits only
DAILY_QUOTA_THRESHOLD = 150.0  # if Groq asks us to wait longer than this, treat it as RPD/TPD exhaustion, not a TPM blip
TOKEN_SAFETY_MARGIN = 200  # pause proactively once remaining TPM budget gets this tight


class DailyQuotaExhausted(Exception):
    """Raised when Groq's wait time implies we've hit a daily cap (RPD/TPD), not a per-minute one."""
    def __init__(self, wait_seconds):
        self.wait_seconds = wait_seconds
        super().__init__(f"Daily quota likely exhausted; Groq asked for a {wait_seconds:.0f}s wait")


def _maybe_wait_for_tpm(headers) -> None:
    """
    Groq returns x-ratelimit-remaining-tokens and x-ratelimit-reset-tokens on
    every response. If we're close to running out of this minute's token
    budget, sleep until it resets instead of firing another request and
    guaranteeing a 429.
    """
    remaining = headers.get("x-ratelimit-remaining-tokens")
    reset_in = headers.get("x-ratelimit-reset-tokens")
    if remaining is None or reset_in is None:
        return
    try:
        remaining = float(remaining)
        # reset_in can come as "7.66s" or a plain number of seconds
        reset_in = float(str(reset_in).rstrip("s"))
    except ValueError:
        return

    if remaining < TOKEN_SAFETY_MARGIN:
        wait = min(reset_in + 0.5, MAX_BACKOFF)
        print(f"[pacing] {remaining:.0f} tokens left this minute, waiting {wait:.1f}s to avoid a 429...")
        time.sleep(wait)


def _call_groq(prompt: str):
    """Single request to Groq, retrying on 429 with backoff. Raises on other errors."""
    attempt = 0
    while True:
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 1100,
                "response_format": {"type": "json_object"},
            },
            timeout=120
        )

        if resp.status_code == 429:
            attempt += 1

            # Groq sends Retry-After (seconds) on 429s; fall back to exponential backoff.
            retry_after = resp.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    raw_wait = float(retry_after)
                except ValueError:
                    raw_wait = DEFAULT_BACKOFF * (2 ** (attempt - 1))
            else:
                raw_wait = DEFAULT_BACKOFF * (2 ** (attempt - 1))

            # A genuinely long requested wait means we've hit a daily cap (RPD/TPD),
            # not a per-minute one — retrying every 120s into that is pointless and
            # just burns the rest of your quota window for nothing.
            if raw_wait > DAILY_QUOTA_THRESHOLD:
                raise DailyQuotaExhausted(raw_wait)

            if attempt > MAX_RETRIES:
                resp.raise_for_status()  # give up, let the caller's except handle it

            wait = min(raw_wait, MAX_BACKOFF)
            print(f"[rate-limit] 429 received, waiting {wait:.1f}s (attempt {attempt}/{MAX_RETRIES})...")
            time.sleep(wait)
            continue

        if not resp.ok:
            # Surface the actual error body (e.g. for 400s) instead of just the status code.
            print(f"[http-error] {resp.status_code} body: {resp.text[:500]}")
        resp.raise_for_status()

        # Show live remaining quota so you can watch your daily budget as it runs,
        # instead of checking the Groq console mid-run.
        rem_req = resp.headers.get("x-ratelimit-remaining-requests")
        lim_req = resp.headers.get("x-ratelimit-limit-requests")
        rem_tok = resp.headers.get("x-ratelimit-remaining-tokens")
        lim_tok = resp.headers.get("x-ratelimit-limit-tokens")
        if rem_req is not None and lim_req is not None:
            print(f"[quota] requests remaining: {rem_req}/{lim_req} (daily)  |  tokens remaining: {rem_tok}/{lim_tok} (per-minute)")

        # Proactively pace ourselves for the *next* call based on what's left this minute.
        _maybe_wait_for_tpm(resp.headers)
        return resp


def generate_pairs(doc: dict) -> list[dict]:
    # k8s_docs chunks are now section-sized (split by heading, not by whole
    # page), so most chunks should already fit comfortably. This cap is now
    # a safety net for outlier long sections (e.g. merged short sections, or
    # other sources like google_sre/opensre that may still be whole-page),
    # not a routine truncation like before. Raised from 2500 -> 6000 chars
    # (~1500 tokens) to avoid cutting off legitimate, right-sized chunks.
    text = doc["text"][:6000]
    prompt = GENERATION_PROMPT.format(
        title=doc["title"],
        source=doc["source"],
        text=text
    )
    try:
        resp = _call_groq(prompt)
        raw = resp.json()["choices"][0]["message"]["content"].strip()

        # Belt-and-braces cleanup in case a model ignores response_format and
        # wraps output in markdown fences anyway.
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        parsed = json.loads(raw)
        pairs = parsed.get("pairs", [])

        result = []
        for pair in pairs:
            if pair.get("question") and pair.get("answer"):
                result.append({
                    "instruction": pair["question"],
                    "response": pair["answer"],
                    "source": doc["source"],
                    "source_url": doc["source_url"],
                    "title": doc["title"]
                })
        return result

    except DailyQuotaExhausted:
        raise  # let this propagate up to run() so the whole run stops cleanly
    except Exception as e:
        print(f"[generate] Failed on {doc['title']}: {e}")
        return []


def run(limit: int = None):
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file: GROQ_API_KEY=gsk_..."
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".json")]
    if limit:
        files = files[:limit]

    # Resume: skip already processed titles
    processed_titles = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as existing:
            for line in existing:
                try:
                    pair = json.loads(line)
                    processed_titles.add(pair.get("title", ""))
                except:
                    pass
        if processed_titles:
            print(f"Resuming — {len(processed_titles)} titles already processed, skipping...")
            files = [f for f in files if json.load(open(os.path.join(RAW_DIR, f), encoding="utf-8"))["title"] not in processed_titles]

    print(f"Generating Q&A pairs from {len(files)} documents using Groq model '{GROQ_MODEL}'...")

    total_pairs = 0
    failed = 0
    docs_done = 0
    quota_exhausted = False

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
        for filename in tqdm(files):
            filepath = os.path.join(RAW_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                doc = json.load(f)

            try:
                pairs = generate_pairs(doc)
            except DailyQuotaExhausted as e:
                hours = e.wait_seconds / 3600
                print(f"\n⚠️  Hit a long-duration rate limit (Groq asked for a {e.wait_seconds:.0f}s / ~{hours:.1f}h wait).")
                print("   This is almost always the DAILY TOKEN cap (TPD), not the daily request")
                print("   cap (RPD) — Groq doesn't expose a 'remaining TPD' header, so the")
                print("   [quota] line above showing plenty of requests left can look misleading.")
                print("   Stopping here instead of retrying pointlessly.")
                print(f"   Progress so far is saved in {OUTPUT_FILE} — just re-run this script")
                print("   later (today, tomorrow, or whenever your quota resets) and it will")
                print("   automatically resume from where it left off.")
                quota_exhausted = True
                break

            docs_done += 1
            if pairs:
                for pair in pairs:
                    out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                out.flush()
                total_pairs += len(pairs)
            else:
                failed += 1

    if quota_exhausted:
        print(f"\n⏸️  Run paused early due to quota limits.")
    else:
        print(f"\n✅ Dataset generation complete!")
    print(f"   Documents processed this run: {docs_done}/{len(files)}")
    print(f"   Q&A pairs generated this run: {total_pairs}")
    print(f"   Failed: {failed}")
    print(f"   Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(limit=args.limit)