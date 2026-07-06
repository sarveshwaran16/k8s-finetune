"""
evaluation/queries.py — 10 standard evaluation queries from PRD 
Run against the live API to validate retrieval quality before handoff.
"""

import time
import requests
import json
import os
output_path = os.path.join(os.path.dirname(__file__), "results.json")

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


def run_eval(api_url: str = "http://localhost:8000"):
    results = []
    for i, query in enumerate(EVAL_QUERIES, 1):
        print(f"[{i:02d}/10] {query}")
        start = time.time()
        try:
            resp = requests.post(f"{api_url}/query", json={"query": query}, timeout=600)
            resp.raise_for_status()
            result = resp.json()
            elapsed = round(time.time() - start, 1)
            results.append({
                "query": query,
                "answer": result["answer"],
                "sources": result["sources"],
                "response_time_sec": elapsed
            })
            print(f"       Time: {elapsed}s | Sources: {[s['title'] for s in result.get('sources', [])]}\n")
        except Exception as e:
            elapsed = round(time.time() - start, 1)
            results.append({
                "query": query,
                "answer": f"ERROR: {e}",
                "sources": [],
                "response_time_sec": elapsed
            })
            print(f"       FAILED after {elapsed}s: {e}\n")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    under_30 = sum(1 for r in results if r["response_time_sec"] <= 30)
    print(f"✅ Eval complete — {under_30}/10 queries under 30s")
    print(f"   Results saved to {output_path}")


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    run_eval(url)