import json
import os
from collections import Counter

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "dataset.jsonl")

pairs = []
with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        pairs.append(json.loads(line))

sources = Counter(p["source"] for p in pairs)
print(f"Total pairs: {len(pairs)}")
for source, count in sorted(sources.items()):
    print(f"  {source}: {count}")