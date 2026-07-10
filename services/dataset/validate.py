import json
import os
import re
from collections import Counter

INPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "dataset.jsonl")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "dataset_clean.jsonl")

# Phrases that signal the model is guessing/speculating rather than extracting
# a grounded fact from the source text — e.g. "It's not explicitly stated but
# given the context, it could imply..." This is worse than a short answer:
# it actively teaches a fine-tuned model to hedge and sound uncertain.
HEDGING_PATTERNS = [
    "not explicitly stated",
    "not explicitly mentioned",
    "isn't explicitly",
    "is not clear from the text",
    "the text does not specify",
    "it is unclear",
    "it's unclear",
    "could imply",
    "might suggest",
    "it is possible that",
    "presumably",
    "one could infer",
    "based on the context, it",
    "the text doesn't provide",
    "not directly stated",
]

SELF_REFERENCE_PATTERNS = [
    "as an ai",
    "as a language model",
    "i am an ai",
    "as an assistant",
]

DONT_KNOW_PREFIXES = (
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "unable to determine",
    "cannot determine",
)


def looks_truncated(response: str) -> bool:
    """Catch answers cut off mid-sentence by a max_tokens limit."""
    response = response.strip()
    if not response:
        return False
    # Ends mid-word/mid-clause: no terminal punctuation, or ends on a dangling
    # conjunction/preposition/article that clearly expects more text after it.
    if response[-1] not in ".!?\"')]}":
        return True
    dangling_endings = (
        " the", " a", " an", " and", " or", " but", " is", " are", " to",
        " of", " in", " on", " with", " for", " that", " which",
    )
    lowered = response.lower()
    if lowered.endswith(dangling_endings):
        return True
    return False


def is_valid(pair: dict) -> tuple[bool, str]:
    instruction = pair.get("instruction")
    response = pair.get("response")

    if instruction is None or response is None:
        return False, "missing instruction/response field"

    instruction = str(instruction).strip()
    if isinstance(response, list):
        response = " ".join(str(r) for r in response)
    response = str(response).strip()

    if not instruction:
        return False, "empty instruction"
    if not response:
        return False, "empty response"
    if len(instruction) < 10:
        return False, "instruction too short"
    if len(response) < 20:
        return False, f"response too short ({len(response)} chars)"
    if instruction.lower() == response.lower():
        return False, "instruction equals response"

    resp_lower = response.lower()

    if resp_lower.startswith(DONT_KNOW_PREFIXES):
        return False, "model admitted it doesn't know"

    if any(p in resp_lower for p in SELF_REFERENCE_PATTERNS):
        return False, "model self-reference"

    hedge_hit = next((p for p in HEDGING_PATTERNS if p in resp_lower), None)
    if hedge_hit:
        return False, f"hedging/speculative language ('{hedge_hit}')"

    if looks_truncated(response):
        return False, "response appears truncated (cut off mid-sentence)"

    return True, "ok"


def run():
    pairs = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[skip] line {line_num}: invalid JSON ({e})")

    valid = []
    invalid = []
    reasons = Counter()
    seen = set()
    duplicates = 0

    for pair in pairs:
        ok, reason = is_valid(pair)
        if not ok:
            invalid.append(pair)
            reasons[reason] += 1
            continue

        # Dedup on (instruction, response) — same Q&A pair generated twice
        # (e.g. across a resumed/re-run session) shouldn't appear twice in
        # the training set.
        key = (
            str(pair.get("instruction", "")).strip().lower(),
            str(pair.get("response", "")).strip().lower(),
        )
        if key in seen:
            duplicates += 1
            reasons["duplicate pair"] += 1
            continue
        seen.add(key)
        valid.append(pair)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in valid:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Total pairs:   {len(pairs)}")
    print(f"Valid pairs:   {len(valid)}")
    print(f"Removed pairs: {len(pairs) - len(valid)}")
    print(f"\nRemoval reasons:")
    for reason, count in reasons.most_common():
        print(f"  {reason}: {count}")
    print(f"\nClean dataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    run()