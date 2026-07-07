import os
import json
import requests
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dataset.jsonl")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")

GENERATION_PROMPT = """You are a Kubernetes expert creating a training dataset.

Read the following Kubernetes documentation and generate exactly 3 question-answer pairs.

Rules:
- Questions must be specific and practical (troubleshooting, configuration, concepts)
- Answers must be accurate, concise, and based ONLY on the provided text
- Format your response as valid JSON only, nothing else:

{{"pairs": [
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}}
]}}

Document title: {title}
Source: {source}

Text:
{text}

JSON:"""


def generate_pairs(doc: dict) -> list[dict]:
    text = doc["text"][:2000]
    prompt = GENERATION_PROMPT.format(
        title=doc["title"],
        source=doc["source"],
        text=text
    )
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_ctx": 2048, "num_predict": 500}
            },
            timeout=300
        )
        resp.raise_for_status()
        raw = resp.json()["response"].strip()

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

    except Exception as e:
        print(f"[generate] Failed on {doc['title']}: {e}")
        return []


def run(limit: int = None):
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

    print(f"Generating Q&A pairs from {len(files)} documents...")

    total_pairs = 0
    failed = 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
        for filename in tqdm(files):
            filepath = os.path.join(RAW_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                doc = json.load(f)

            pairs = generate_pairs(doc)

            if pairs:
                for pair in pairs:
                    out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                total_pairs += len(pairs)
            else:
                failed += 1

    print(f"\n✅ Dataset generation complete!")
    print(f"   Documents processed: {len(files)}")
    print(f"   Q&A pairs generated: {total_pairs}")
    print(f"   Failed: {failed}")
    print(f"   Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(limit=args.limit)