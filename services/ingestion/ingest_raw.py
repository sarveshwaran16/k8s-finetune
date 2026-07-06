import os
import sys
import json
import hashlib
from tqdm import tqdm
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))

from sources import k8s_docs, prometheus_runbooks, k8s_failures, opensre, google_sre

load_dotenv()

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")


def save_document(doc: dict, source_name: str):
    """Save a single scraped document as a JSON file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    doc_id = hashlib.md5(doc["metadata"]["source_url"].encode()).hexdigest()
    filename = f"{source_name}_{doc_id}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({
            "id": doc_id,
            "source": source_name,
            "title": doc["metadata"]["title"],
            "source_url": doc["metadata"]["source_url"],
            "text": doc["text"]
        }, f, indent=2, ensure_ascii=False)


def run():
    sources = [
        #("k8s_docs", k8s_docs.fetch_all()),
        #("prometheus_runbooks", prometheus_runbooks.fetch_all()),
        #("k8s_failures", k8s_failures.fetch_all()),
        #("opensre", opensre.fetch_all()),
         ("google_sre", google_sre.fetch_all()),
    ]

    total = 0
    for source_name, generator in sources:
        print(f"\n{'='*50}")
        print(f"Ingesting: {source_name}")
        print(f"{'='*50}")
        count = 0
        try:
            for doc in generator:
                save_document(doc, source_name)
                count += 1
                if count % 50 == 0:
                    print(f"[{source_name}] Saved {count} documents...")
        except Exception as e:
            print(f"[{source_name}] Error — skipping source: {e}")
        print(f"[{source_name}] Done — {count} documents saved")
        total += count

    print(f"\n✅ Ingestion complete — {total} total documents saved to data/raw/")


if __name__ == "__main__":
    run()