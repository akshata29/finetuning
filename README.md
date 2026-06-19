---
title: 1-Hour Azure Fine-Tuning Demo
description: Run-of-show, prerequisites, and per-act commands for the end-to-end Azure fine-tuning demo (synthetic data, serverless + GPU LoRA, evaluation, hosting).
author: Playground demo team
ms.date: 2026-06-17
ms.topic: how-to
---

## Overview

An end-to-end, **100% synthetic, PII-free** demo of fine-tuning on Azure AI
Foundry for a sales-call intent / lead-propensity classification use case. It
runs in one hour by pre-baking the slow parts (real SFT + GPU LoRA jobs, the
GPU managed online endpoint) and running the impressive parts live (synthetic
generation sample, serverless upload/job-create/poll, the 3-run evaluation, a
Developer-tier deploy + one inference).

Four acts:

1. **Synthetic data factory** — a teacher LLM generates schema-locked, label-balanced, PII-free transcripts + labels, then dedup + PII scan.
2. **Two ways to fine-tune** — serverless Azure OpenAI SFT/DPO/RFT (live) and GPU managed-compute LoRA (pre-baked).
3. **Rigorous evaluation** — base vs fine-tuned vs optimized-prompt on one Foundry plane, plus offline scikit-learn aggregate metrics.
4. **Hosting** — cheap Developer-tier deploy + live inference and the "one plane" governance story.

## Run-of-show

| Time | Act | Live vs pre-baked | Subcommand |
|---|---|---|---|
| 0:00-0:08 | Framing | — | — |
| 0:08-0:20 | Act 1 synthetic data | Live sample + dedup/PII; dataset pre-generated | `gen-data` |
| 0:20-0:35 | Act 2A serverless SFT | Live upload + job create + poll; finished job pre-baked | `sft` |
| (opt) | Act 2A serverless DPO | Live preference fine-tuning (serverless differentiator) | `dpo` |
| (opt) | Act 2A serverless RFT | Live reinforcement fine-tuning with grader-scored rows | `rft` |
| (opt) | Act 2A1 quick-eval | Live base vs fine-tuned on validation holdout | `quick-eval` |
| (opt) | Act 3A Foundry eval | Upload base/SFT/DPO/RFT runs to the Foundry Evaluations tab | `foundry-eval` |
| 0:35-0:42 | Act 2B GPU LoRA | Pre-baked | `gpu-lora` |
| 0:42-0:55 | Act 3 evaluation | Live 3-run eval + offline sklearn table | `evaluate` |
| 0:55-1:00 | Act 4 hosting + wrap | Live Developer-tier deploy + inference; GPU endpoint pre-deployed | `host` |
| after | Cleanup | Live | `cleanup` |

## Prerequisites

### A. Azure environment and access

- Azure subscription with billing enabled; **Owner/Contributor** on the resource group and the ability to assign RBAC.
- A new **Foundry project** with an `AIProjectClient` endpoint, plus a deployed **teacher/judge model** (gpt-4.1 or gpt-4o) for generation and AI-assisted graders.
- Roles: `Foundry User` on the project (plus `Cognitive Services OpenAI User` for graders) and `Foundry Owner` (or `.../deployments/write`) to deploy fine-tuned models.
- A **region** supporting the chosen fine-tuning model (for example North Central US, Sweden Central, or East US 2).
- **Quota**: serverless fine-tuning capacity for gpt-4.1-mini, and pre-approved GPU VM quota (A100/H100 SKU) for the GPU path — request one to two weeks early.

### B. Data and assets

- The synthetic generation script and taxonomy/seed list in this package.
- Pre-generated **train JSONL** (UTF-8 with BOM, balanced), **validation JSONL**, and **eval JSONL** with `ground_truth` at a realistic ~1-2% buy prevalence, disjoint from train.
- Pre-baked: a completed serverless SFT job, a completed GPU LoRA job, and a deployed AML managed online endpoint.

### C. Tooling

- Python 3.11+ with the packages in [requirements.txt](requirements.txt); `az login` working.
- Azure CLI plus the `ml` extension for the GPU/managed-endpoint path.

### D. People and alignment

- Data scientists / ML engineers who own the current pipeline (hands-on), plus a data/compliance stakeholder for the residency conversation.
- The ROI decision-maker briefed and joining the framing and wrap blocks.

## Pre-flight checklist (T-2 days)

- [ ] Foundry project + teacher/judge model deployed; graders run on a 2-row smoke test.
- [ ] Serverless fine-tuning capacity + GPU VM quota confirmed.
- [ ] Train/val/eval JSONL validated (schema, BOM, prevalence, train/eval disjoint).
- [ ] Pre-baked SFT + LoRA jobs finished; AML endpoint deployed and `invoke` works.
- [ ] 3-run eval + offline sklearn table runs clean against pre-baked predictions.
- [ ] Developer-tier deploy tested (and re-deployable live within the 24h window).
- [ ] Cleanup script ready (delete Developer-tier deployment + AML endpoint).

Run the automated portion:

```bash
python finetuning_demo/preflight.py
```

It prints a structured pass/fail report and exits non-zero when a required
check fails, without raising.

## Setup

```bash
python -m pip install -r finetuning_demo/requirements.txt
```

Configuration is sourced entirely from environment variables (no secrets in
source). Set at least: `AZURE_OPENAI_ENDPOINT`, `AZURE_SUBSCRIPTION_ID`,
`AZURE_RESOURCE_GROUP`, `AOAI_ACCOUNT_NAME`, `AZURE_REGION`,
`AZURE_AI_PROJECT_ENDPOINT`, `TEACHER_MODEL`, `GRADER_MODEL`, and
`SFT_DEPLOYMENT_NAME`.

## Per-act commands

Every command maps 1:1 to a `run_of_show` subcommand. Add `--prebaked` to use
pre-baked artifacts instead of live Azure calls; add `-v` for debug logging.

```bash
# Act 1 — synthetic data factory (live sample, then write train/val/eval JSONL)
python run_of_show.py gen-data --count 200 --eval-count 200
python run_of_show.py gen-data --count 200 --eval-count 200 --dpo  # also write DPO preference pairs
python run_of_show.py gen-data --count 200 --eval-count 200 --rft  # also write RFT graded rows
python run_of_show.py gen-data --prebaked        # verify the pre-baked dataset

# Act 2A — serverless Azure OpenAI SFT (upload -> create -> poll)
python run_of_show.py sft
python run_of_show.py sft --prebaked --results-csv path/to/results.csv

# Act 2A — serverless DPO (Direct Preference Optimization; preferred vs non_preferred)
# Serverless support for DPO (not just SFT) is the headline Azure differentiator.
# Requires the preference files from `gen-data --dpo`.
python run_of_show.py dpo                          # beta=0.1, 2 epochs by default
python run_of_show.py dpo --beta 0.2 --n-epochs 3

# Act 2A — serverless RFT (Reinforcement Fine-Tuning; grader-scored outputs)
# Requires graded files from `gen-data --rft`. RFT trains on an o-series
# reasoning base model (o4-mini), NOT gpt-4.1-mini — needs o4-mini RFT quota.
# Grader choices: `string_match` (deterministic label-match via Azure
# string_check graders) or `model` (Azure score_model LLM judge).
python run_of_show.py rft
python run_of_show.py rft --grader string_match --n-epochs 3
python run_of_show.py rft --grader model --n-epochs 2

# Act 2A deploy — control-plane (ARM) PUT of the fine-tuned model
# By default this deploys the best-validation checkpoint (guards against
# over-training); add --final-model to deploy the job's final model instead.
# 1. Easiest — no id needed (reads data/sft_state.json, picks best checkpoint)
python run_of_show.py deploy --deployment-name sales-intent --sku developer

# 1b. Deploy a DPO or RFT model — reads data/dpo_state.json or data/rft_state.json
python run_of_show.py deploy --method dpo --deployment-name sales-intent-dpo --sku developer
python run_of_show.py deploy --method rft --deployment-name sales-intent-rft --sku developer

# 2. By job id — resolves the best checkpoint live from Azure (any past job)
python run_of_show.py deploy --job-id ftjob-0bc373617c104c049cb77d2565ad1d00 --deployment-name sales-intent --sku developer

# 3. Explicit model id (unchanged); or force the final model with --final-model
python run_of_show.py deploy --model-id <ft-model-id> --deployment-name sales-intent --sku developer
python run_of_show.py deploy --final-model --deployment-name sales-intent --sku developer


# Act 2A1 quick-eval — base vs fine-tuned on the labeled validation holdout
# Runs both deployments over data/validation.jsonl, reports intent accuracy +
# propensity MAE side by side, and writes preds_base.jsonl / preds_finetuned.jsonl.
# Needs both models DEPLOYED (deployment names, not model ids).
#
# Note: fine-tuned models may emit near-synonyms or spelling variants outside the
# canonical intent taxonomy (e.g., "complaint" vs "support_escalation", "escalation"
# vs "support_escalation"). A fuzzy-match normalizer maps these back to canonical
# labels for fair accuracy measurement.
python run_of_show.py quick-eval --base-deployment chat41mini --ft-deployment sales-intent
python run_of_show.py quick-eval --val path/to/validation.jsonl   # defaults to BASE/SFT_DEPLOYMENT_NAME


# Act 3A — Foundry portal evaluations across base/SFT/DPO/RFT
# Replays the validation holdout through each deployment, scores it with custom
# (intent accuracy + propensity MAE) and built-in (F1) evaluators, and uploads
# each run to Build > Evaluations in the Foundry portal. See docs/foundry-eval-showcase.md.
python run_of_show.py foundry-eval --limit 20 --no-upload   # rehearse locally (no portal write)
python run_of_show.py foundry-eval --limit 50               # full showcase -> portal
python run_of_show.py foundry-eval --models sft rft         # only specific arms


# Act 2B — GPU managed-compute LoRA (always pre-baked)
python finetuning_demo/run_of_show.py gpu-lora --prebaked

# Act 3 — base vs fine-tuned vs optimized-prompt + offline metrics
python finetuning_demo/run_of_show.py evaluate --kind propensity
python finetuning_demo/run_of_show.py evaluate --kind intent --prebaked

# Act 4 — Developer-tier live deploy + one inference (GPU endpoint pre-baked)
# Like deploy, --model-id is optional (best checkpoint from saved/job id by default)
# and --method {sft,dpo,rft} selects which saved state file to host from.
python finetuning_demo/run_of_show.py host --deployment-name sales-intent
python finetuning_demo/run_of_show.py host --method rft --deployment-name sales-intent-rft
python finetuning_demo/run_of_show.py host --model-id <ft-model-id> --deployment-name sales-intent
python finetuning_demo/run_of_show.py host --prebaked --endpoint-name sales-lora-endpoint

# Cleanup — delete the live Developer-tier deployment
python finetuning_demo/run_of_show.py cleanup --deployment-name sales-intent

# All — pre-baked dry run across the live-able acts
python finetuning_demo/run_of_show.py all
```

The orchestrator is also module-runnable: `python -m finetuning_demo.run_of_show --help`.

## Cost and tier notes

- **Developer tier** — no hourly hosting fee (per-token only), all regions, near-instant deploy; **no SLA, no data-residency guarantee, fixed 24-hour auto-delete**. Use it for the live throwaway demo deployment only. The ARM deployment `sku.name` is `DeveloperTier` (the portal's "Deployment type"); `--sku developer` maps to it automatically.
- **Standard (Regional) tier** — ~$1.70/hr hosting, in-region data residency, 15-day inactivity auto-delete. Use for any regulated FSI scenario beyond a throwaway eval.
- **Global Standard tier** — cheaper training throughput, no in-region residency guarantee.
- Deleting a deployment never deletes the underlying fine-tuned model; it can be redeployed later.
- GPU managed compute bills per GPU-hour for the whole endpoint lifetime — delete the endpoint promptly after the demo (`cleanup`).

## Do not overclaim

State these limits verbatim during the demo; do not soften or skip them.

- gpt-oss-120b is NOT natively fine-tunable on Azure (only gpt-oss-20b serverless SFT preview).
- RFT is NOT available on gpt-4.1-mini — it requires an o-series reasoning base model (o4-mini GA; gpt-5 gated/invite-only) and its own fine-tuning quota.
- GRPO is NOT turnkey (Azure native = grader-based RFT + DPO; GRPO is BYO via TRL `GRPOTrainer` or Fireworks Reward Kit).
- MACC drawdown for Fireworks-on-Foundry is UNCONFIRMED (verify badge + account team).
- managed pipeline supports plain LoRA only (no 4-bit QLoRA / `target_modules`).
- true BYO-GPU managed compute is classic-hub portal preview.
- the fine-tuning safety/deployability gate is preview with fixed thresholds.

## Honesty flags worth narrating

- **Deploy is a control-plane (ARM) PUT** (`api-version=2024-10-01`), not the data-plane SDK.
- **Aggregate metrics are offline** — Foundry custom evaluators score per-row only; precision / recall / F1 / AUC / top-decile lift are computed offline with scikit-learn (see `offline_metrics.py`).
- The eval set carries a realistic **~1-2% buy prevalence**, so PR-AUC and top-decile lift are the headline metrics; accuracy is misleading at a ~98% negative base rate.

## Advanced features

### Serverless customization methods

Act 2A now supports three serverless fine-tuning methods:

- **SFT** (`sft`) for supervised chat-label fitting.
- **DPO** (`dpo`) for preference optimization over preferred vs non-preferred outputs.
- **RFT** (`rft`) for reinforcement training over grader-scored outputs.

For RFT, `gen-data --rft` writes Azure-format JSONL files (`rft_train.jsonl` /
`rft_validation.jsonl`). Each row carries a `messages` array whose first turn
uses the `developer` role (o-series models reject the chat-SFT "FinetuneChat"
format produced by a `system` role) and whose final message is a `user` turn,
plus top-level `intent` / `outcome` / `propensity_score` reference fields the
server-side grader reads as `{{ item.<field> }}`. The job also pins a
JSON-schema `response_format` so the grader can read the model output as
`{{ sample.output_json.<field> }}`.

Grader choices map to Azure's grader schema:

- `--grader string_match` — a `multi` grader of per-field `string_check`
  (`operation: eq`) subgraders, averaged. Deterministic label-match.
- `--grader model` — a `score_model` grader that asks `GRADER_MODEL` to rate the
  output in `[0,1]`.

RFT is only supported on o-series reasoning base models. The demo pins
`o4-mini-2025-04-16` for RFT jobs (SFT/DPO still use `gpt-4.1-mini`), so RFT
needs separate o4-mini fine-tuning quota in the region. o-series RFT also does
**not** support the developer training tier, so RFT jobs fall back to
`GlobalStandard` training automatically (a warning is logged). The fine-tuning
service also auto-pauses RFT jobs at $5,000 of combined training + grading cost.

### Throttle resilience

The quick-eval runner uses **Standard (regional) deployments**, which share a
transient backend pool and occasionally emit `429 "Backend error"` even when the
deployment's own TPM/RPM quota is untouched (verified by inspecting
`x-ratelimit-remaining-tokens` and `x-ratelimit-remaining-requests` at ~100% on
429 responses). Azure stamps a misleading `retry-after: 30` header, but such
"blips" clear almost immediately.

The quick-eval scorer (`act2a1_quick_eval.py`) bypasses the OpenAI SDK retry
logic and implements a fast custom backoff:

- **First attempt**: immediate
- **Retries 1–7**: exponential backoff (1.5s, 3s, 4.5s, 6s, 6s, 6s, 6s)
- **Strategy**: 8 attempts total; retry-after headers are ignored in favor of
  immediate probing

This trades the SDK's (sometimes misleading) 30s-per-attempt delay for much
faster recovery on backend transients.

### Label normalization

Fine-tuned intent classifiers may emit near-synonyms or spelling variants outside
the canonical taxonomy — for example, `"complaint"` instead of `"support_escalation"`
or `"escalation"` as a standalone label. The quick-eval scorer normalizes these
back to the canonical set using a fuzzy-match heuristic:

1. **Exact match** (case-insensitive): if the prediction matches a canonical label, use it as-is.
2. **Substring containment**: if a prediction contains a canonical label (or vice versa), use the canonical label.
3. **Word-part containment**: if a prediction shares a longer word component with a canonical label (e.g., `"feature"` from `"feature_request"`), use the canonical label.
4. **No match**: return the prediction unchanged (so the scorer counts it as a miss, surfacing the anomaly).

This ensures accuracy and MAE metrics reflect true model skill rather than label-variance quirks. Use `python -c "from finetuning_demo.act2a1_quick_eval import _normalize_intent; print(_normalize_intent('complaint'))"` to test mapping logic.

Quick-eval also includes a resilient fallback parser: when a deployment returns non-JSON text, or JSON that uses alternate keys such as `caller_intent`, `call_outcome`, or `buy_propensity_score`, the scorer attempts to normalize those fields back to the canonical `intent` / `outcome` / `propensity_score` shape before scoring.

If a deployment returns an empty response payload, quick-eval retries once with a stricter JSON-only system prompt before scoring. This prevents base-model formatting drift from collapsing metrics to `intent_acc=0.000` and `prop_mae=n/a` solely due to parser starvation.
