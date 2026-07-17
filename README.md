# Kubernetes Specialist SLM — Project Documentation

End-to-end pipeline: scraping Kubernetes/SRE documentation, generating and validating a Q&A dataset targeted at 7 common failure classes, fine-tuning two candidate small language models (TinyLlama-1.1B and Qwen2.5-0.5B), evaluating them, and quantizing/deploying via Ollama.

---

## 1. Project Structure

```
K8S-FINETUNE/
├── data/
│   ├── processed/
│   │   ├── dataset_clean.jsonl      # Final validated Q&A pairs (post-filtering)
│   │   ├── dataset.jsonl            # Raw generated Q&A pairs (pre-validation)
│   │   └── dataset.jsonl.bak        # Backup, auto-created before destructive filters
│   └── raw/                         # Scraped raw document chunks (JSON, one per chunk)
├── evaluation/
│   ├── finetuned_results.json            # Qwen model-generated answers on the 7 in-domain test queries
│   ├── finetuned_results_tinyllama.json  # TinyLlama model-generated answers on the 7 in-domain test queries
│   └── semantic_match_results.json       # LLM-judged semantic scoring output
├── k8s/                             # (not part of the documented dataset pipeline)
├── llama-cpp-bin/                   # Prebuilt llama.cpp binaries (llama-quantize.exe, etc.)
├── llama.cpp/                       # Cloned llama.cpp source (used for HF -> GGUF conversion script)
├── models/
│   ├── k8s-qwen-final/              # Qwen LoRA adapter (downloaded from Kaggle)
│   ├── k8s-qwen-merged/             # Qwen base model + adapter merged (fp16, HF format)
│   ├── k8s-tinyllama-final/         # TinyLlama LoRA adapter (downloaded from Kaggle)
│   ├── k8s-tinyllama-merged/        # TinyLlama base model + adapter merged (fp16, HF format)
│   ├── k8s-qwen-fp16.gguf           # Qwen converted to GGUF, unquantized
│   ├── k8s-qwen-q4.gguf             # Qwen quantized (Q4_K_M) — final deployable artifact
│   ├── k8s-tinyllama-fp16.gguf      # TinyLlama converted to GGUF, unquantized
│   ├── k8s-tinyllama-q4.gguf        # TinyLlama quantized (Q4_K_M) — final deployable artifact
│   ├── Modelfile                    # Ollama Modelfile for the Qwen GGUF
│   └── Modelfile_tinyllama          # Ollama Modelfile for the TinyLlama GGUF
├── notebooks/
│   ├── k8s-finetune_qwen.ipynb      # Qwen2.5-0.5B fine-tuning notebook (Kaggle, T4)
│   └── k8s-finetune_tinyllama.ipynb # TinyLlama-1.1B fine-tuning notebook (Kaggle, T4)
├── services/
│   ├── dataset/
│   │   ├── generate.py              # Groq API-based Q&A pair generation from raw chunks
│   │   └── validate.py              # Filters hedging/truncated/duplicate pairs
│   ├── evaluation/
│   │   ├── evaluate.py              # Runs the Qwen merged model against the 7 in-domain test queries
│   │   └── evaluate_tinyllama.py    # Runs the TinyLlama merged model against the 7 in-domain test queries
│   ├── ingestion/
│   │   ├── sources/                 # Per-source scrapers (k8s_docs, google_sre, k8s_failures, prometheus_runbooks)
│   │   ├── dedupe.py                # Content-hash deduplication for raw chunks
│   │   └── ingest_raw.py            # Orchestrates scraping + saves standardized raw JSON chunks
│   └── quantization/
│       ├── gguf_conversion.py       # HF model -> GGUF -> Q4_K_M quantization
│       ├── merge_model.py           # Merges LoRA adapter into base model (fp16)
│       └── patch_tokenizer.py       # Fixes tokenizer_config.json version-mismatch bugs
├── .env                             # GROQ_API_KEY, GROQ_MODEL
├── requirements.txt
├── semantic.py                      # LLM-judged semantic match evaluation (Groq-based)
└── README.md
```

---

## 2. Prerequisites

Install all Python dependencies before running any stage:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` (or create `.env`) and populate your Groq API key:

```
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama3-70b-8192
```

---

## 3. Data Ingestion & Chunking

**Sources**: Kubernetes official docs (`kubernetes.io/docs`), Google SRE Book/Workbook, Kubernetes failure-mode writeups, Prometheus runbooks.

**Chunking strategy**: pages are split by heading structure (`h1`/`h2`/`h3`) into self-contained sections, rather than treating a whole page as one chunk. Short sections (<300 chars) merge forward rather than being dropped. Each chunk is tagged with heuristic failure-class labels (keyword-matched against the 7 target classes) and given an anchor-aware source URL for traceability.

**Why**: whole-page chunking, combined with downstream text truncation for LLM context limits, silently discarded most of the content on long pages before a single Q&A pair could be generated from it. Section-level chunking ensures every chunk is a complete, self-contained unit that fully reaches the generation step.

**Resulting corpus**: 7,503 raw chunks across all sources; 2,256 chunks tagged with at least one of the 7 target failure classes (OOMKilled, CrashLoopBackOff, ImagePullBackOff, NodeNotReady, RBAC, Ingress/Service, PVC).

**Run ingestion:**

```bash
py services/ingestion/ingest_raw.py
```

---

## 4. Dataset Generation & Validation

**Generation** (`services/dataset/generate.py`): uses the Groq API (rotating across multiple free-tier models to work around per-model daily token caps) to generate a natural 1–4 Q&A pairs per chunk — not a fixed count, so thin chunks aren't padded with trivial questions. Generation was prioritized to process failure-class-tagged chunks first.

- Failure-tagged chunks processed: 2,256
- Raw Q&A pairs generated: 7,277

**Run generation:**

```bash
py services/dataset/generate.py
```

**Validation** (`services/dataset/validate.py`): filters out pairs showing hedging/speculative language ("not explicitly stated," "could imply"), self-references ("as an AI"), admitted ignorance, apparent truncation (cut off mid-sentence), and exact duplicates.

- Final validated dataset: **7,204 pairs** (used for the TinyLlama v2 training run; the Qwen run's own load reported 7,024 pairs from the same filtered file)

**Run validation:**

```bash
py services/dataset/validate.py
```

---

## 5. Fine-Tuning

Both models use QLoRA (4-bit NF4 quantization), LoRA rank=16, alpha=32, applied to both attention layers (`q/k/v/o_proj`) and MLP layers (`gate/up/down_proj`), trained on Kaggle with a T4 GPU.

**Train/validation split** (set in the notebooks via `train_test_split`, `seed=42`):
- TinyLlama: 95% train / 5% validation
- Qwen: 92% train / 8% validation

**Run fine-tuning** by uploading and executing the Kaggle notebooks (T4 GPU required):

- `notebooks/k8s-finetune_tinyllama.ipynb` — TinyLlama-1.1B-Chat
- `notebooks/k8s-finetune_qwen.ipynb` — Qwen2.5-0.5B-Instruct

After training, download the saved LoRA adapters from Kaggle output and place them under `models/k8s-tinyllama-final/` and `models/k8s-qwen-final/` respectively.

### Fine-Tuning Results

#### TinyLlama-1.1B-Chat

| Run | LoRA scope | Best Val Loss | Best Step | Notes |
|---|---|---|---|---|
| v1 | Attention only | 1.196 | ~1200 (epoch ~3) | 7 epochs planned; overfitting visible past epoch 3 |
| v2 | Attention + MLP | **1.145** | 800 | 4 epochs planned; early stopping halted at step 1400 |

Trainable parameters (v2): 12.6M / 1.1B total (1.13%)

**Qualitative improvement (v1 → v2)**: PVC-Pending answer changed from incorrect/risky advice ("delete the PVC and recreate it") to factually correct ("PVC not bound to a PV — create one and bind it"). CPU throttling and CrashLoopBackOff answers remained weak in both versions.

#### Qwen2.5-0.5B-Instruct

| Run | Epochs planned | Best Val Loss | Best Step/Epoch | Notes |
|---|---|---|---|---|
| Run 1 | 4 (time-budget capped) | 1.532 | Step 624 (~epoch 1.7) | Completed in ~98 min |
| Run 2 | 15 | **1.519** | Epoch 3 | Early stopping halted at epoch 6; ~130 min |

Trainable parameters: 12.6M / 1.1B total (1.13%)

**Note on cross-model loss comparison**: Qwen's validation loss is not directly comparable to TinyLlama's, since Qwen's much larger vocabulary (~152K tokens vs. TinyLlama's ~32K) affects the scale of cross-entropy loss. Qualitative evaluation is the more reliable comparison between the two.

**Known technical finding**: Qwen2.5-0.5B trained noticeably slower than TinyLlama-1.1B despite having under half the parameters, because loss is computed over its much larger vocabulary at every step — largely offsetting the expected speed benefit of a smaller model.

---

## 6. Quantization (GGUF, Q4_K_M)

Quantization is a two-step process: patch the tokenizer configs saved from Kaggle, merge the LoRA adapter into the base model, then convert and quantize to GGUF.

### Paths to update before running

One path in this stage is machine-specific and must be updated before running:

| File | Variable | What to set it to |
|---|---|---|
| `services/quantization/gguf_conversion.py` | `LLAMA_QUANTIZE_EXE` (line 9) | Absolute path to `llama-quantize.exe` inside your local `llama-cpp-bin/` folder |

All other paths in the quantization scripts are relative to the script location and require no changes.

### Step 1 — Patch tokenizer configs

Fixes `tokenizer_config.json` field formats that are incompatible with the local `transformers` version:

```bash
py services/quantization/patch_tokenizer.py
```

### Step 2 — Merge LoRA adapter into base model

Produces a full fp16 HuggingFace model under `models/k8s-tinyllama-merged/` and `models/k8s-qwen-merged/`:

```bash
py services/quantization/merge_model.py
```

### Step 3 — Convert & quantize to GGUF

Converts the merged HF model to GGUF (fp16) and then quantizes to Q4_K_M:

```bash
py services/quantization/gguf_conversion.py
```

### Quantization Results

| Model | Quantized Size | PRD Target (<300MB) |
|---|---|---|
| Qwen2.5-0.5B | **379.4 MB** | Misses by ~79 MB |
| TinyLlama-1.1B | **636.9 MB** | Misses by ~337 MB |

Neither model hits the <300MB target at Q4_K_M — this holds even at the lowest standard quantization level (Q2_K) for both model families, based on published quantizations of the same base models. Qwen2.5-0.5B is meaningfully closer to the target, consistent with its smaller parameter count.

**Known issue resolved during conversion**: both models' tokenizer configs, saved from a newer `transformers` version on Kaggle, used field formats (`extra_special_tokens` as a list, `tokenizer_class: "TokenizersBackend"`) incompatible with the older `transformers` version installed locally. Fixed via `patch_tokenizer.py`, which corrects these fields directly in the saved config files.

---

## 7. Evaluation

### Step 1 — Run inference on the 7 in-domain queries

Outputs are saved to `evaluation/finetuned_results_tinyllama.json` and `evaluation/finetuned_results.json`:

```bash
# TinyLlama merged model
py services/evaluation/evaluate_tinyllama.py

# Qwen merged model
py services/evaluation/evaluate.py
```


### Qualitative — 7 in-domain queries (one per target failure class)

Both merged models were run against the same 7 queries via `evaluate.py` / `evaluate_tinyllama.py`.

| Query (class) | TinyLlama-1.1B (merged) | Qwen2.5-0.5B (merged) |
|---|---|---|
| OOMKilled | **Correct** — memory limit exceeded; cites memory leaks/overprovisioning as causes | Partially correct (conflates memory with CPU) |
| CrashLoopBackOff | Incorrect — invents "unable to access the host filesystem" as the cause | Incorrect (misstates the mechanism) |
| ImagePullBackOff | Incorrect/unhelpful — suggests `kubectl logs` (container never started, so no logs exist) | Incorrect (suggests `kubectl debug`/`exec`, not applicable) |
| NodeNotReady | **Correct** — `kubectl get node <name> -o yaml`, check Ready condition | Incorrect (invents a non-existent `rollout restart node` command) |
| RBAC Forbidden | Vague but not wrong — generic "check your credentials," no mention of RBAC/roles | Incorrect (conflates authentication with authorization) |
| Ingress 503 | Vague — "configured incorrectly, check the docs," no real diagnostic content | Incorrect mechanism (garbled explanation) |
| PVC Pending | **Correct** — PVC not bound to a PV, PV unavailable or mismatched | Incorrect domain (conflates scheduling with storage binding) |

**Score**: TinyLlama-1.1B — 3 correct, 2 vague/unhelpful, 2 incorrect. Qwen2.5-0.5B — 0 fully correct, 7 incorrect/partially incorrect. TinyLlama-1.1B is the qualitatively stronger model on this evaluation set.

### Inference Latency (PRD target: >10 tokens/sec on T4)

Measured via a warm-up-excluded, averaged run across the 7 in-domain queries on the Kaggle T4 session:

| Query | Qwen2.5-0.5B (tok/s) | TinyLlama-1.1B (tok/s) |
|---|---|---|
| What causes OOMKilled errors? | 27.83 | 34.20 |
| Why is my pod in CrashLoopBackOff? | 28.93 | 33.80 |
| How do I debug image pull failures? | 29.35 | 32.31 |
| How do I troubleshoot NodeNotReady? | 28.94 | 33.28 |
| Why am I getting a Forbidden error when accessing the API? | 26.77 | 33.55 |
| Why is my Ingress returning 503 errors? | 28.94 | 33.08 |
| Why is my PVC stuck in Pending? | 29.63 | 34.22 |
| **Average** | **28.63** | **33.49** |
| **PRD target (>10 tok/s)** | PASS | PASS |

**Technical insight — vocabulary size drives inference speed, not just training speed**: TinyLlama-1.1B is ~17% faster than Qwen2.5-0.5B on T4 despite having over twice the parameters. This is a second independent confirmation of the same underlying effect observed during training: Qwen's much larger vocabulary (~152K tokens vs. TinyLlama's ~32K) means every generation step must compute a softmax over a proportionally larger output layer. That overhead outweighs the compute savings from Qwen's smaller parameter count, producing a model that is both slower to train *and* slower to generate — a non-obvious result that the inference numbers now corroborate directly.

### Semantic Match (LLM-judged, PRD target: >85%)
`semantic.py` is available to score model answers against reference answers on a 1–5 scale via the Groq API.

---

## 8. Deployment (Ollama)

Both quantized models are packaged with an Ollama `Modelfile` using the same Alpaca-style prompt template used during training (`### Instruction:\n...\n\n### Response:\n`), a fixed system prompt scoping the model to the 7 target failure classes, and `temperature=0.3` for consistent, low-variance answers.

```bash
cd models
ollama create k8s-qwen -f Modelfile
ollama create k8s-tinyllama -f Modelfile_tinyllama

ollama run k8s-qwen
ollama run k8s-tinyllama
```

---

## 9. End-to-End Pipeline Summary

| Stage | Script / Notebook | Command |
|---|---|---|
| Data Ingestion | `services/ingestion/ingest_raw.py` | `py services/ingestion/ingest_raw.py` |
| Dataset Generation | `services/dataset/generate.py` | `py services/dataset/generate.py` |
| Dataset Validation | `services/dataset/validate.py` | `py services/dataset/validate.py` |
| Fine-Tuning | `notebooks/*.ipynb` | Run on Kaggle (T4 GPU) |
| Patch Tokenizer | `services/quantization/patch_tokenizer.py` | `py services/quantization/patch_tokenizer.py` |
| Merge LoRA Adapter | `services/quantization/merge_model.py` | `py services/quantization/merge_model.py` |
| GGUF Conversion | `services/quantization/gguf_conversion.py` | `py services/quantization/gguf_conversion.py` |
| Evaluate TinyLlama | `services/evaluation/evaluate_tinyllama.py` | `py services/evaluation/evaluate_tinyllama.py` |
| Evaluate Qwen | `services/evaluation/evaluate.py` | `py services/evaluation/evaluate.py` |
| Semantic Scoring | `semantic.py` | `py semantic.py` |
| Deploy | `models/Modelfile*` | `ollama create` / `ollama run` |
