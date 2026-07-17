import os
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

MERGED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "k8s-tinyllama-merged")

# One question per actual PRD-scoped failure class (dropped CPU throttling
# and DNS failures -- neither is one of the 7 target classes).
EVAL_QUERIES = [
    "What causes OOMKilled errors?",                                   # OOMKilled
    "Why is my pod in CrashLoopBackOff?",                               # CrashLoopBackOff
    "How do I debug image pull failures?",                              # ImagePullBackOff
    "How do I troubleshoot NodeNotReady?",                              # NodeNotReady
    "Why am I getting a Forbidden error when accessing the API?",       # RBAC
    "Why is my Ingress returning 503 errors?",                          # Ingress/Service
    "Why is my PVC stuck in Pending?",                                  # PVC
]

def load_model():
    print("Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        do_sample=False,
    )
    print("Model loaded!")
    return pipe


def ask(pipe, query: str) -> tuple[str, float]:
    prompt = f"### Instruction:\n{query}\n\n### Response:\n"
    start = time.time()
    result = pipe(prompt)
    elapsed = round(time.time() - start, 1)
    generated = result[0]["generated_text"].replace(prompt, "").strip()
    return generated, elapsed


def run():
    pipe = load_model()
    results = []

    total = len(EVAL_QUERIES)
    print(f"\nRunning {total} evaluation queries...\n")

    for i, query in enumerate(EVAL_QUERIES, 1):
        print(f"[{i:02d}/{total}] {query}")
        answer, elapsed = ask(pipe, query)
        print(f"       Time: {elapsed}s")
        print(f"       Answer: {answer[:200]}")
        print()
        results.append({
            "query": query,
            "answer": answer,
            "response_time_sec": elapsed
        })

    output_path = os.path.join(os.path.dirname(__file__), "..", "..", "evaluation", "finetuned_results_tinyllama.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Done: Evaluation complete -- saved to {output_path}")


if __name__ == "__main__":
    run()
