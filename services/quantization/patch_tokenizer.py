import os
import json
from transformers import AutoTokenizer

ADAPTER_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "k8s-tinyllama-final")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "k8s-tinyllama-merged")

TOKENIZER_CONFIG_PATH = os.path.join(ADAPTER_PATH, "tokenizer_config.json")


def patch_tokenizer_class():
    with open(TOKENIZER_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    if config.get("tokenizer_class") == "TokenizersBackend":
        # TinyLlama uses a Llama-style tokenizer -- this is the correct
        # transformers v4.x class name for it (the training environment on
        # Kaggle used transformers v5.x, which saved the newer, unified
        # "TokenizersBackend" name that v4.x doesn't recognize).
        print("Found 'tokenizer_class': 'TokenizersBackend' -- patching to 'LlamaTokenizerFast'.")
        config["tokenizer_class"] = "LlamaTokenizerFast"
        with open(TOKENIZER_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"Patched: {TOKENIZER_CONFIG_PATH}")
    else:
        print("'tokenizer_class' is not 'TokenizersBackend' -- nothing to patch.")


def save_tokenizer_only():
    # Model weights already saved successfully in the previous run --
    # this just finishes the one step that crashed.
    print("Loading tokenizer (patched)...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    print("Saving tokenizer into merged model folder...")
    tokenizer.save_pretrained(OUTPUT_PATH)
    print(f"Done: Tokenizer saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    patch_tokenizer_class()
    save_tokenizer_only()