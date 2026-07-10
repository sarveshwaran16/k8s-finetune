import os
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

MERGED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "k8s-tinyllama-merged")

EVAL_QUERIES = [
    "Why is my pod in CrashLoopBackOff?",
    "What causes OOMKilled errors?",
    "How do I troubleshoot NodeNotReady?",
    "Why is my PVC stuck in Pending?",
    "How do I diagnose Kubernetes DNS failures?",
    "Why is my deployment continuously restarting?",
    "How can I investigate CPU throttling?",
    "What are common causes of 5XX errors in Kubernetes?",
    "How do I debug image pull failures?",
    "What should I check when a service is unreachable?",
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

    print(f"\nRunning {len(EVAL_QUERIES)} evaluation queries...\n")

    for i, query in enumerate(EVAL_QUERIES, 1):
        print(f"[{i:02d}/10] {query}")
        answer, elapsed = ask(pipe, query)
        print(f"       Time: {elapsed}s")
        print(f"       Answer: {answer[:200]}")
        print()
        results.append({
            "query": query,
            "answer": answer,
            "response_time_sec": elapsed
        })

    output_path = os.path.join(os.path.dirname(__file__), "..", "..", "evaluation", "finetuned_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Evaluation complete — saved to {output_path}")


if __name__ == "__main__":
    run()