import json
import os
from collections import Counter

INPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "dataset.jsonl")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "dataset_clean.jsonl")


def is_valid(pair: dict) -> tuple[bool, str]:
    instruction = pair.get("instruction", "").strip()
    response = pair.get("response", "")
    if isinstance(response, list):
        response = " ".join(str(r) for r in response)
    response = str(response).strip()

    if not instruction:
        return False, "empty instruction"
    if not response:
        return False, "empty response"
    if len(response) < 20:
        return False, f"response too short ({len(response)} chars)"
    if len(instruction) < 10:
        return False, "instruction too short"
    if instruction.lower() == response.lower():
        return False, "instruction equals response"
    if response.lower().startswith("i don't know") or response.lower().startswith("i do not know"):
        return False, "model admitted it doesn't know"
    if "as an ai" in response.lower() or "as a language model" in response.lower():
        return False, "model self-reference"

    return True, "ok"


def run():
    pairs = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                pairs.append(json.loads(line))
            except:
                pass

    valid = []
    invalid = []
    reasons = Counter()

    for pair in pairs:
        ok, reason = is_valid(pair)
        if ok:
            valid.append(pair)
        else:
            invalid.append(pair)
            reasons[reason] += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in valid:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Total pairs:   {len(pairs)}")
    print(f"Valid pairs:   {len(valid)}")
    print(f"Removed pairs: {len(invalid)}")
    print(f"\nRemoval reasons:")
    for reason, count in reasons.most_common():
        print(f"  {reason}: {count}")
    print(f"\nClean dataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    run()