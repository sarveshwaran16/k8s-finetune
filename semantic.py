import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Path to your model's generated answers (from evaluate_qwen.py / evaluate.py)
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "evaluation", "finetuned_results.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "evaluation", "semantic_match_results.json")

# Reference answers for the 7 in-domain PRD failure classes -- these are
# the "correct" facts the judge checks the model's answer against. Kept
# short and factual, not styled like training data, so the judge is
# grading against ground truth, not phrasing similarity.
REFERENCE_ANSWERS = {
    "What causes OOMKilled errors?":
        "A container is killed by the kernel's OOM killer because it exceeded its memory limit. "
        "This is specifically a memory issue, not related to CPU.",
    "Why is my pod in CrashLoopBackOff?":
        "The container repeatedly crashes or exits, and Kubernetes keeps restarting it with an "
        "exponentially increasing backoff delay between attempts. Common causes include application "
        "errors, missing dependencies, misconfiguration, or failing health checks.",
    "How do I debug image pull failures?":
        "Use `kubectl describe pod <pod-name>` to see the exact pull error (e.g. image not found, "
        "wrong tag, missing registry credentials). Common causes are a typo in the image name/tag, "
        "a private registry without proper imagePullSecrets, or rate limiting from the registry.",
    "How do I troubleshoot NodeNotReady?":
        "Check `kubectl describe node <node-name>` for conditions and events. Common causes include "
        "kubelet not running or unreachable, network issues, resource pressure (disk/memory), or the "
        "node failing health checks.",
    "Why am I getting a Forbidden error when accessing the API?":
        "This is an RBAC authorization failure (HTTP 403), not an authentication failure. The "
        "requesting user or service account lacks a Role/ClusterRole and RoleBinding/ClusterRoleBinding "
        "granting permission for that action on that resource.",
    "Why is my Ingress returning 503 errors?":
        "Typically means there are no ready backend Pods matching the Service's selector, or the "
        "Service has no valid Endpoints. Can also result from misconfigured paths/backends in the "
        "Ingress resource itself.",
    "Why is my PVC stuck in Pending?":
        "Usually means no PersistentVolume is available that satisfies the PVC's request (size, access "
        "mode, StorageClass), or the StorageClass's provisioner is failing to dynamically provision one. "
        "Not related to Pod scheduling/resource requests.",
}

JUDGE_SYSTEM_PROMPT = (
    "You are an expert Kubernetes SRE grading a junior engineer's answer against a reference "
    "answer written by a senior engineer. You always respond with valid JSON only -- no preamble, "
    "no markdown fences, no commentary."
)

JUDGE_PROMPT_TEMPLATE = """Question: {question}

Reference answer (correct): {reference}

Model's answer (to be graded): {model_answer}

Score the model's answer against the reference on a 1-5 scale:
5 = fully correct, matches the key facts in the reference
4 = mostly correct, minor omission or imprecision
3 = partially correct, mixes accurate and inaccurate information
2 = mostly incorrect, but touches the right general topic
1 = completely incorrect, fabricated, or irrelevant

Respond with ONLY this JSON:
{{"score": <1-5>, "reasoning": "<one sentence explaining the score>"}}
"""


def judge_answer(question: str, reference: str, model_answer: str) -> dict:
    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, reference=reference, model_answer=model_answer)
    resp = requests.post(
        GROQ_API_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 150,
            "response_format": {"type": "json_object"},
        },
        timeout=60,
    )
    if not resp.ok:
        # Surface the ACTUAL reason instead of just the generic status code --
        # a 400 here almost always means the request body itself was rejected
        # (e.g. this specific model doesn't support response_format), not a
        # transient network issue.
        print(f"  [http-error {resp.status_code}] {resp.text[:500]}")
    resp.raise_for_status()

    raw = resp.json()["choices"][0]["message"]["content"].strip()
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    return json.loads(raw)


def run():
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in your .env file.")

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        model_results = json.load(f)

    scored = []
    for item in model_results:
        question = item["query"]
        model_answer = item["answer"]
        reference = REFERENCE_ANSWERS.get(question)

        if reference is None:
            print(f"[skip] No reference answer defined for: {question}")
            continue

        print(f"Judging: {question}")
        try:
            judged = judge_answer(question, reference, model_answer)
        except Exception as e:
            print(f"  [error] {e}")
            judged = {"score": None, "reasoning": f"judging failed: {e}"}

        scored.append({
            "query": question,
            "model_answer": model_answer,
            "reference_answer": reference,
            "score": judged.get("score"),
            "reasoning": judged.get("reasoning"),
        })
        time.sleep(1)  # light pacing, avoid hammering the judge API

    valid_scores = [s["score"] for s in scored if isinstance(s["score"], (int, float))]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    pass_count = sum(1 for s in valid_scores if s >= 4)  # score >=4 treated as "semantic match"
    pass_rate = (pass_count / len(valid_scores) * 100) if valid_scores else 0

    summary = {
        "results": scored,
        "average_score": round(avg_score, 2),
        "pass_rate_percent": round(pass_rate, 1),
        "prd_target_percent": 85,
        "meets_target": pass_rate >= 85,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nAverage score: {avg_score:.2f}/5")
    print(f"Pass rate (score >= 4): {pass_rate:.1f}%  (PRD target: >85%)")
    print(f"Meets target: {'YES' if summary['meets_target'] else 'NO'}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run()