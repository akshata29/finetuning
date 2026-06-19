# Conversation-Strategy Alignment — End-to-End Pipeline

A **self-contained** fine-tuning pipeline for the customer's actual use case:
*aligning conversational strategies from long, multi-turn transcripts*. Unlike
the four-act classifier demo (which trains a model to emit a single
`{intent, outcome, propensity}` label), this pipeline trains an open/base model
to **generate the next agent turn** so it follows a desired conversational
strategy.

Everything is driven by a single script —
[customer_conversation_alignment.py](customer_conversation_alignment.py) — and
all artifacts live in an **isolated** data namespace
(`data/conversation_alignment/`), so nothing collides with the classifier demo's
`data/*.jsonl`.

> Why this exists: it makes the demo *match the customer's ask honestly*. The
> serverless **SFT → DPO → RFT** loop and the Foundry evaluation plane (the Azure
> differentiator) run here on real generation/alignment data instead of a
> classification proxy.

---

## What it demonstrates (the differentiator story)

| Stage | Command | What it proves |
|---|---|---|
| **Capture real conversations** | `capture` | Drive your *deployed* agent, record the transcripts, and distill them into the training corpus — *data from your real corpus* |
| Synthetic multi-turn data | `gen-data` | Schema-locked conversations; preferred vs. weak agent replies |
| Serverless **SFT** | `sft` | Supervised tuning on full multi-turn conversations — no GPU quota |
| Serverless **DPO** | `dpo` | Preference alignment toward your strategy (the direct lever) |
| Serverless **RFT** | `rft` | Reinforcement tuning graded on free-text strategy adherence |
| Offline evaluation | `evaluate` | Deterministic strategy-adherence score per deployment |
| **Foundry** evaluation | `foundry-eval` | Side-by-side base/SFT/DPO/RFT runs in the Evaluations tab |

The five strategies the model is aligned to: `consultative_discovery`,
`value_framing`, `objection_reframe`, `evidence_backed`, `mutual_next_step`.

---

## Synthetic data: the native-platform story ladder

Data generation is itself a differentiator moment. Tell it as **three rungs**,
from no-code to the customer's exact requirement — *parity → credibility → wedge*:

| Rung | Mechanism | What it shows | Limits |
|---|---|---|---|
| 1. No-code | Foundry portal **Data generation (Preview)** — seed with the customer's own reference transcripts | A first-class, in-portal feature (same "native plane" story as Evaluations) | SFT/Eval only; tasks are *Simple Q&A* / *Tool use*; no DPO/RFT, no multi-turn strategy |
| 2. Documented SDK | Microsoft's published **teacher-distillation notebook** ([sample](https://github.com/Azure/azureml-examples/blob/main/sdk/python/foundation-models/system/finetune/Llama-notebooks/datagen/synthetic-data-generation.ipynb), [concept](https://learn.microsoft.com/en-us/azure/foundry-classic/concepts/concept-synthetic-data)) | The teacher-LLM approach is the platform's *endorsed* method, not bespoke glue | Sample produces SFT `messages` only; single-turn label task |
| 3. This pipeline | `gen-data --use-llm` (below) | The same teacher pattern extended to **multi-turn + DPO + RFT** — what the customer actually needs | Requires the small script in this folder (the supported path until the Preview UI covers these shapes) |

> Honest framing: the portal Preview is the visual hook, the notebook is the
> credibility anchor ("Microsoft documents this technique"), and `--use-llm` here
> is the same technique scaled to the preference + reinforcement data the native
> UI can't yet emit. The teacher is swappable — point it at Llama 405B from the
> catalog exactly as the sample notebook shows if the customer wants their
> open-source model as the teacher.

---

## Prerequisites

1. **Python environment** — use the demo virtual environment (it has
   system-site-packages access to the Azure SDKs):

   ```powershell
   cd finetuning
   # PowerShell: invoke the venv python with the call operator (&)
   & .venv\Scripts\python.exe customer_conversation_alignment.py --help
   ```

2. **Environment variables** (`.env` next to the package, or the shell). The
   pipeline reads the same `DemoConfig` keys as the rest of the demo:

   | Variable | Used by | Example |
   |---|---|---|
   | `AZURE_OPENAI_ENDPOINT` | all live calls | `https://astaieus2.openai.azure.com` |
   | `AZURE_OPENAI_API_KEY` | all live calls | *(secret — keep out of git)* |
   | `AZURE_SUBSCRIPTION_ID` | deploy + Foundry upload | `b14d5e08-...` |
   | `AZURE_RESOURCE_GROUP` | deploy + Foundry upload | `astaipublic` |
   | `AOAI_ACCOUNT_NAME` | control-plane deploy | `astaieus2` |
   | `AZURE_AI_PROJECT_ENDPOINT` | Foundry upload | `https://astaieus2.services.ai.azure.com/api/projects/astaieus2proj` |
   | `BASE_DEPLOYMENT_NAME` | RFT grader + eval arm | `chat41mini` |
   | `SFT_DEPLOYMENT_NAME` | Foundry eval arm | `conv-align-sft` |
   | `DPO_DEPLOYMENT_NAME` | Foundry eval arm | `conv-align-dpo` |
   | `RFT_DEPLOYMENT_NAME` | Foundry eval arm | `conv-align-rft` |
   | `TEACHER_MODEL` *(optional)* | `gen-data --use-llm` teacher | `chat41` |
   | `GRADER_MODEL` *(optional)* | RFT model grader | `chat41mini` |

   > `gen-data` needs **no** Azure config — it runs fully offline. The opt-in
   > `gen-data --use-llm` is the exception: it calls the teacher model and so
   > needs `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_API_KEY`.

---

## Quick start (offline data only)

```powershell
cd finetuning
& .venv\Scripts\python.exe customer_conversation_alignment.py gen-data --count 200 --eval-count 60
```

This writes the isolated corpus:

```
data/conversation_alignment/
  conv_sft_train.jsonl     conv_sft_val.jsonl
  conv_dpo_train.jsonl     conv_dpo_val.jsonl
  conv_rft_train.jsonl     conv_rft_val.jsonl
  conv_eval.jsonl
```

---

## Run the acts end to end

All commands run from `finetuning` with `& .venv\Scripts\python.exe`.

### Act 0 — Bootstrap the corpus by capturing the base agent (TracesDistillation-style)

> **The customer-specific opener — and the loop's entry point.** Instead of
> starting from synthetic rows, you **enter the lifecycle at the `capture` box**
> (see the loop below). On **lap 1 there is no tuned agent yet**, so you capture
> from the **base** agent (`chat41mini`) — a generic, un-aligned stand-in for
> "your deployed agent" — and turn those transcripts into the training corpus.
> On later laps you point `--agent-deployment` at the model you just tuned. This
> is the same shape as Microsoft's **TracesDistillation** recipe (in the
> `microsoft-foundry/fine-tuning` samples — pull traces → distill a better
> student), adapted from tool-call distillation to **free-text strategy
> alignment**.

```powershell
# offline, deterministic — proves the pipeline + the gap-report story, zero cost
& .venv\Scripts\python.exe customer_conversation_alignment.py capture --count 200 --eval-count 60

# live — capture from the BASE agent (lap 1 stand-in) with an LLM customer
# simulator (tokens). --concurrency fans the independent conversations across
# worker threads so a full live capture finishes in minutes instead of an hour.
& .venv\Scripts\python.exe customer_conversation_alignment.py capture `
    --count 200 --eval-count 60 --live --use-llm `
    --agent-deployment base --concurrency 4
```

> Live capture runs each conversation's turns sequentially (turn N+1 depends on
> N) but fans **independent conversations** out across `--concurrency` worker
> threads (default `4`; `1` = sequential). `--max-retries` (default `6`) sets the
> per-call 429/5xx retry budget — the client backs off and honors `Retry-After`,
> so throttling is absorbed, not fatal. The offline path ignores both (no I/O).
> Raise concurrency carefully: higher values hit the deployment's TPM limit
> sooner, and you should run only **one** live capture process at a time.

What it does, per conversation:

1. An **LLM customer simulator** role-plays a prospect across a multi-turn deal.
2. Your **deployed baseline agent** answers (live) — *or* a deterministic
   stand-in answers (offline) so the act runs with zero Azure config.
3. Each captured transcript is **scored** for strategy adherence, then
   **distilled** into the drop-in `conv_*.jsonl` corpus: the strategy-aligned
   reply becomes the SFT target / DPO `preferred` / RFT reference, and the
   **real captured baseline reply** becomes the DPO `non_preferred`.

The headline is the **strategy gap report** — the customer's own number:

```text
Captured-corpus strategy gap
====================================================
mode=live  conversations=200  agent=chat41mini
----------------------------------------------------
baseline (captured)            0.41
aligned target                 0.93
opportunity (lift)             0.52
----------------------------------------------------
  consultative_discovery       0.55 -> 1.00  (n=40)
  value_framing                0.30 -> 0.94  (n=40)
  ...
```

Flags: `--live` drives the real agent + simulator; `--agent-deployment` chooses
which deployment to capture from — a **friendly label** (`base`/`sft`/`dpo`/`rft`,
resolved from `.env`) or a raw deployment name (defaults to
`BASE_DEPLOYMENT_NAME` in `--live` mode); `--use-llm` authors the aligned target
replies with the teacher model (true distillation) instead of templated
exemplars; `--concurrency` (default `4`) runs conversations in parallel and
`--max-retries` (default `6`) sets the 429/5xx retry budget; `--seed` for
reproducibility.

#### Where this fits — how Act 0 maps to the loop

> **The lifecycle is a cycle, so it has no fixed start.** You can enter it at
> `gen-data` (cold start — no agent exists yet) **or** at `capture` (point it at
> an agent that already exists). **Act 0 enters at `capture` against the base
> model** (`chat41mini`), prompted as a generic, *un-aligned* assistant — a
> deliberate **stand-in** for "your deployed agent." That is why the opener is
> `capture` even though the diagram draws it last: on the **first lap** there is
> no tuned agent to capture from, so you capture from the base model to bootstrap
> the corpus. On later laps you capture from the model you just tuned.

```text
  ┌── lap 1 enters here (capture from BASE = Act 0) ──┐
  ▼                                                   │
  capture ──▶ train ──▶ deploy ──▶ run inference ──▶ capture (from TUNED) ─┐
  (base)     sft/dpo    (model)    (real convos)     (lap 2+ harvest)      │
    ▲         /rft                                                         │
    └──────────────────────── next iteration ─────────────────────────────┘

  (cold-start alternative: skip Act 0 and enter the loop at `gen-data` instead)
```

**Run the whole loop end to end (one sitting).** Each command maps to one box:

```powershell
# LAP 1 ───────────────────────────────────────────────────────────────────
# 1. capture FROM BASE (Act 0) — bootstrap the corpus from the base agent.
#    (offline: drop --live --use-llm for a zero-cost deterministic run.)
& .venv\Scripts\python.exe customer_conversation_alignment.py capture `
    --count 200 --eval-count 60 --live --use-llm `
    --agent-deployment base --concurrency 4

# 2. train + deploy each method. --deploy writes the deployment name so the
#    'sft'/'dpo'/'rft' capture labels resolve on the next lap.
& .venv\Scripts\python.exe customer_conversation_alignment.py sft `
    --deploy --deployment-name conv-align-sft --sku developer
& .venv\Scripts\python.exe customer_conversation_alignment.py dpo `
    --deploy --deployment-name conv-align-dpo --sku developer
& .venv\Scripts\python.exe customer_conversation_alignment.py rft `
    --deploy --deployment-name conv-align-rft --sku developer

#az storage account update --name astaistor --resource-group astaipublic --public-network-access Enabled

# 3. evaluate base vs sft vs dpo vs rft on one plane.
& .venv\Scripts\python.exe customer_conversation_alignment.py foundry-eval `
    --models base sft dpo rft --limit 50

# LAP 2 ───────────────────────────────────────────────────────────────────
# 4. set the deployment name in .env (e.g. SFT_DEPLOYMENT_NAME=conv-align-sft),
#    then capture FROM YOUR TUNED AGENT and retrain on the improved corpus.
& .venv\Scripts\python.exe customer_conversation_alignment.py capture `
    --count 200 --eval-count 60 --live --use-llm `
    --agent-deployment sft --concurrency 4
# → back to step 2 to retrain on the captured corpus.
```

The loop steps in detail:

1. **`capture --agent-deployment base`** (Act 0) — bootstrap from the base agent.
   *Or* enter cold at **`gen-data`** for a synthetic corpus.
2. **`sft` / `dpo` / `rft --deploy`** — train and deploy your aligned model
   (recorded as `SFT_DEPLOYMENT_NAME` etc. in `.env`).
3. **Run inference** — let the deployed agent hold real conversations.
4. **`capture --agent-deployment sft`** — point capture at *your own deployed
   fine-tuned agent* (the label resolves to `SFT_DEPLOYMENT_NAME`) and distill
   those real conversations into the next `conv_*.jsonl` corpus.
5. **Back to step 2** — retrain on the improved, real-world corpus.

> If the chosen label has no deployment configured yet (for example `sft` before
> you have deployed it), capture fails fast with a clear message telling you to
> set the matching env var or pass the raw deployment name — so you never
> silently capture from the wrong model.

Provenance is preserved for the demo — every raw captured transcript and the
gap report are written under `data/conversation_alignment/captured/`:

```
captured/
  captured_transcripts.jsonl   # raw agent+simulator turns, baseline vs aligned scores
  capture_manifest.json        # run config (mode, agent, counts, seeds)
  capture_gap_report.json      # the headline lift, overall + per-strategy
```

> Going to production: swap the simulated/baseline capture for the
> **`foundry-traces`** Data Generation recipe — pull *real* App Insights traces
> from the agent already in production, run the same distillation transforms, and
> hand the resulting `conv_*.jsonl` straight to Act 2. The schema is identical, so
> Acts 2–3 run unchanged on captured data.

> Note: `capture` **overwrites** the shared `conv_*.jsonl` corpus (that is the
> point — captured data *is* the training data). Re-run `gen-data` to return to
> the synthetic resting state.

### Act 1 — Synthetic multi-turn data factory

```powershell
& .venv\Scripts\python.exe customer_conversation_alignment.py gen-data --count 200 --eval-count 60
```

- `--count` training conversations, `--eval-count` held-out conversations,
  `--seed` for reproducibility (default `1337`).
- The held-out eval set is generated from a **disjoint seed** so no whole
  conversation leaks between train and eval.
- `--use-llm` authors every turn with a **teacher model** (rung 3 above) instead
  of the deterministic templates. It needs Azure config (`AZURE_OPENAI_ENDPOINT`
  + `AZURE_OPENAI_API_KEY`, and optionally `TEACHER_MODEL`, else
  `BASE_DEPLOYMENT_NAME`) and consumes tokens — one structured-output call per
  conversation. Each row falls back to the template if a call fails, so a single
  bad response never aborts the run. The console prints the active source
  (`deterministic templates` vs `LLM teacher`).

  ```powershell
  # diverse, LLM-authored corpus (matches the documented teacher pattern)
  & .venv\Scripts\python.exe customer_conversation_alignment.py gen-data --count 200 --eval-count 60 --use-llm

  # ...faster: fan independent conversations across worker threads
  & .venv\Scripts\python.exe customer_conversation_alignment.py gen-data `
      --count 200 --eval-count 60 --use-llm --concurrency 4
  ```

  > Conversations are independent, so `--use-llm` runs them in parallel
  > (`--concurrency`, default `4`; `1` = sequential). `--max-retries` (default
  > `6`) sets the SDK's 429/5xx retry budget per call — the client backs off and
  > honors `Retry-After`, so transient throttling is absorbed rather than fatal.
  > Raise concurrency carefully: higher values hit the deployment's TPM limit
  > sooner. The offline (templated) path ignores both flags — it has no network.

  > Default (templated) is offline, deterministic, and zero-cost — ideal for dry
  > runs and tests. Use `--use-llm` for the customer-facing story, where genuine
  > variance makes the SFT/DPO/RFT lift credible.

Open a row to walk the schema live:

- `conv_sft_train.jsonl` → `{"messages": [system, user, assistant, …, user, assistant]}`
- `conv_dpo_train.jsonl` → `{"input": {messages ending on user}, "preferred_output": […], "non_preferred_output": […]}`
- `conv_rft_train.jsonl` → `{"messages": [developer, …, user], "strategy": "<target>"}`

### Act 2A — Serverless Supervised Fine-Tune (SFT)

```powershell
& .venv\Scripts\python.exe customer_conversation_alignment.py sft `
    --deploy --deployment-name conv-align-sft --sku developer
```

- Uploads `conv_sft_*`, creates the job, polls to terminal, picks the
  lowest-validation-loss checkpoint, writes `conv_sft_state.json`.
- `--deploy` (optional) deploys the result via the ARM control-plane.
- `--max-polls N` caps polling for a time-boxed demo.

### Act 2B — Serverless Preference Optimization (DPO)

```powershell
& .venv\Scripts\python.exe customer_conversation_alignment.py dpo `
    --deploy --deployment-name conv-align-dpo --sku developer
```

Trains on preferred vs. weak **final agent replies** — the most direct lever for
"prefer our conversational strategy." Writes `conv_dpo_state.json`.

### Act 2C — Serverless Reinforcement Fine-Tune (RFT)

```powershell
& .venv\Scripts\python.exe customer_conversation_alignment.py rft `
    --grader-model chat41mini `
    --deploy --deployment-name conv-align-rft --sku developer
```

- RFT requires an **o-series** base (`o4-mini`) and runs on the `GlobalStandard`
  training tier (it rejects the developer tier).
- A **model grader** scores the free-text reply's strategy adherence. The grader
  model resolves from `--grader-model`, else `GRADER_MODEL`, else
  `BASE_DEPLOYMENT_NAME`.
- Writes `conv_rft_state.json`.

### Act 3 — Offline strategy-alignment score

```powershell
& .venv\Scripts\python.exe customer_conversation_alignment.py evaluate `
    --deployment conv-align-sft --limit 50
```

Replays the held-out conversations through one deployment, generates the next
agent turn, and scores strategy adherence with a deterministic heuristic
(reference: exemplary replies ≈ 0.93, weak replies ≈ 0.01). Use `--delay` to
pace rate-limited deployments.

### Act 3A — Foundry portal evaluation (side-by-side)

```powershell
& .venv\Scripts\python.exe customer_conversation_alignment.py foundry-eval `
    --models base sft dpo rft --limit 50
```

- Builds one prediction dataset per arm and uploads each run to the Azure AI
  Foundry **Evaluations** tab as `conv-alignment-eval-<arm>`.
- Metrics: `strategy_alignment` (custom) and `f1_score` (built-in token overlap
  vs. the exemplary reply).
- `--no-upload` runs locally without the portal; `--no-builtin` drops the F1
  evaluator; `--delay` paces calls. Arms whose `*_DEPLOYMENT_NAME` is unset are
  skipped automatically.
- Writes a combined `data/conversation_alignment/foundry/conv_foundry_eval_report.json`
  and prints a scoreboard.

---

## One-shot: generate + tune all three

```powershell
& .venv\Scripts\python.exe customer_conversation_alignment.py all `
    --count 200 --eval-count 60 --grader-model chat41mini
```

Runs `gen-data` then `sft`, `dpo`, and `rft` in sequence. Add `--deploy
--deployment-name <name>` to deploy each (the deployment name applies per run —
override per method by running the subcommands individually for distinct names).

---

## Commands quick reference

```text
gen-data       --count --eval-count --seed --use-llm
capture        --count --eval-count --seed --live --agent-deployment --use-llm
sft            --deploy --deployment-name --sku --max-polls
dpo            --deploy --deployment-name --sku --max-polls
rft            --grader-model --deploy --deployment-name --sku --max-polls
evaluate       --deployment (required) --limit --delay
foundry-eval   --models {base,sft,dpo,rft} --limit --delay --no-builtin --no-upload
all            --count --eval-count --seed --use-llm --grader-model --deploy --deployment-name --sku
```

Add `-v` / `--verbose` to any command for INFO-level logging.

---

## How it maps to a real long-context corpus

The synthetic data is deliberately compact so the demo runs in the room. To move
to the customer's real corpus, keep the **same schemas** and:

- **Start with `capture`** (Act 0) — it already emits the real schema from a
  deployed agent. In production, swap its simulated capture for the
  `foundry-traces` recipe to pull genuine App Insights traces.
- Replace `generate_conversations()` output with real transcripts — keep the
  exemplary strategy in the `assistant` turns (SFT) and curate
  preferred-vs-rejected **responses** (DPO).
- For long multi-turn context, extend each `messages[]` array with the full
  dialogue; raise the GPU/serverless sequence-length budget accordingly.
- Swap the deterministic `strategy_alignment_score` for your own rubric, or keep
  the RFT **model grader** (`conv_strategy_grader_config`) and point it at your
  policy.

See [docs/fine-tuning-hour-demo-guide.md](docs/fine-tuning-hour-demo-guide.md)
for the presenter playbook and the parity-vs-wedge narrative.

---

## Notes

- **Data isolation:** every file is under `data/conversation_alignment/` with a
  `conv_` prefix — the classifier demo's `data/*.jsonl` is never touched.
- **Import safety:** the module imports with zero Azure SDKs installed; SDKs are
  resolved lazily only when a live operation runs. Secrets come only from the
  environment (`DemoConfig`) — nothing is hardcoded.
- **PowerShell:** invoke the venv python as `& .venv\Scripts\python.exe …` (a
  bare `.venv\Scripts\python.exe` is misparsed as a module).
