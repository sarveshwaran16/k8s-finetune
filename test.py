import os
import json
from collections import Counter

# Change this path if needed
RAW_DIR = r"D:\rag-assistant\k8s-finetune\data\raw"


def main():
    if not os.path.exists(RAW_DIR):
        print(f"❌ Directory not found: {RAW_DIR}")
        return

    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".json")]

    if not files:
        print("❌ No JSON files found.")
        return

    sources = []

    for filename in files:
        filepath = os.path.join(RAW_DIR, filename)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                sources.append(data.get("source", "Unknown"))
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    counts = Counter(sources)

    print("=" * 40)
    print(f"Total documents: {len(files)}")
    print("=" * 40)

    for source, count in sorted(counts.items()):
        print(f"{source:<25} : {count}")

    print("=" * 40)


if __name__ == "__main__":
    main()