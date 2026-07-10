import os
import json
import hashlib
import argparse

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")


def content_hash(text: str) -> str:
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


def run(source_prefix: str, dry_run: bool = True):
    files = [f for f in os.listdir(RAW_DIR) if f.startswith(f"{source_prefix}_") and f.endswith(".json")]
    print(f"Found {len(files)} files with prefix '{source_prefix}_'")

    seen_hashes = {}   # content_hash -> first filename kept
    duplicates = []    # filenames to remove

    for filename in files:
        filepath = os.path.join(RAW_DIR, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception as e:
            print(f"[skip] {filename}: couldn't read ({e})")
            continue

        text = doc.get("text", "")
        h = content_hash(text)

        if h in seen_hashes:
            duplicates.append(filename)
        else:
            seen_hashes[h] = filename

    print(f"\nUnique documents: {len(seen_hashes)}")
    print(f"Duplicate files to remove: {len(duplicates)}")

    if not duplicates:
        print("Nothing to clean up.")
        return

    if dry_run:
        print("\n[DRY RUN] Would delete:")
        for f in duplicates[:20]:
            print(f"  {f}")
        if len(duplicates) > 20:
            print(f"  ... and {len(duplicates) - 20} more")
        print("\nRe-run with --apply to actually delete these files.")
    else:
        for f in duplicates:
            os.remove(os.path.join(RAW_DIR, f))
        print(f"\nDeleted {len(duplicates)} duplicate files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Source prefix to check, e.g. 'google_sre'")
    parser.add_argument("--apply", action="store_true", help="Actually delete duplicates (default is dry-run)")
    args = parser.parse_args()
    run(args.source, dry_run=not args.apply)