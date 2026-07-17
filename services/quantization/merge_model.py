import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "k8s-tinyllama-final")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "k8s-tinyllama-merged")


def run():
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    print("Saving merged model...")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    model.save_pretrained(OUTPUT_PATH)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)

    print(f"Done: Merged model saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()