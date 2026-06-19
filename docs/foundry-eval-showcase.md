# Foundry Evaluations Showcase — Base vs SFT vs DPO vs RFT

A presenter-ready walkthrough for demoing the **Azure AI Foundry → Evaluations**
feature on every fine-tuned model in this demo. The `foundry-eval` command
replays the labeled validation holdout through each deployment, scores it with a
custom + built-in evaluator suite, and **uploads each run to the Foundry portal**
so the side-by-side lift shows up in the Evaluations tab.

---

## What gets evaluated

| Arm | Deployment (env var) | Base model |
| --- | --- | --- |
| Base (un-tuned) | `BASE_DEPLOYMENT_NAME` | gpt-4.1-mini |
| Supervised FT | `SFT_DEPLOYMENT_NAME` | gpt-4.1-mini |
| Preference (DPO) | `DPO_DEPLOYMENT_NAME` | gpt-4.1-mini |
| Reinforcement (RFT) | `RFT_DEPLOYMENT_NAME` | o4-mini |

Arms whose deployment name is unset are skipped automatically, so you can demo
whatever is deployed.

## Evaluator suite

| Evaluator | Type | Metric | Meaning |
| --- | --- | --- | --- |
| `intent_match` | custom code | mean → accuracy | Predicted intent == ground-truth intent (shared quick-eval normalization) |
| `propensity_error` | custom code | mean → MAE | Absolute error of predicted buy-propensity |
| `f1_score` | built-in Foundry | F1 | Token overlap of predicted vs true label (deterministic, no extra model) |

The custom metrics match the demo's quick-eval numbers exactly; the built-in F1
gives the run a recognizable catalog metric in the portal.

---

## Prerequisites

```powershell
# 1. Sign in (the upload uses Entra ID / DefaultAzureCredential, not the API key)
az login

# 2. Confirm the eval SDKs are present (already installed in this workspace)
python -c "import azure.ai.evaluation, azure.ai.projects, azure.identity; print('ok')"
```

Required env (already set in `.env`): `AZURE_SUBSCRIPTION_ID`,
`AZURE_RESOURCE_GROUP`, `AZURE_AI_PROJECT_ENDPOINT` (the project name is parsed
from it, or set `AZURE_AI_PROJECT_NAME`), plus the deployment names above.

---

## Run of show (the live demo)

### 0. Dry-run first (no upload, fast — rehearse safely)

```powershell
cd finetuning
python run_of_show.py foundry-eval --limit 20 --no-upload
```

This replays 20 rows per arm, runs the evaluators locally, and prints the
scoreboard — without touching the portal. Use it to confirm everything works
before going live.

### 1. The full showcase (uploads to the portal)

```powershell
python run_of_show.py foundry-eval --limit 50
```

Each arm becomes a portal run named `sales-intent-eval-<arm>` (e.g.
`sales-intent-eval-rft`). The command prints a scoreboard and a portal URL per
run:

```
Foundry evaluation scoreboard (intent accuracy / propensity MAE)
================================================================
Model                   intent_acc    prop_mae        f1
----------------------------------------------------------------
Base (un-tuned)              0.620       0.180     0.610
Supervised FT                0.880       0.090     0.870
Preference (DPO)             0.900       0.080     0.890
Reinforcement (RFT)          0.930       0.070     0.920
----------------------------------------------------------------
Runs uploaded to the Foundry portal -> Build > Evaluations.
```

### 2. Show it in the UI

1. Open **Microsoft Foundry → Build → Evaluations** (the tab in your screenshot).
2. The four `sales-intent-eval-*` runs appear at the top, **Completed**.
3. Click `sales-intent-eval-rft` → open the run to show per-metric scores and the
   per-row table (query, response, ground truth, intent_match, f1_score).
4. Open `sales-intent-eval-base` side by side to contrast the baseline.

---

## Talk track (what to say)

> "We don't just *claim* the fine-tuned models are better — we prove it with
> Foundry's own evaluation harness. I ran the exact same labeled validation set
> through all four models: the un-tuned baseline, supervised fine-tuning,
> preference tuning with DPO, and reinforcement fine-tuning on o4-mini."

> "Each of these rows in the Evaluations tab is a real run, scored on two things
> that matter for this sales-call use case: did it get the **caller intent**
> right, and how close was its **buy-propensity** estimate. I also included
> Foundry's built-in F1 evaluator so you can see a standard catalog metric next
> to the custom ones."

> "Watch the lift down the ladder: the base model is around 62% intent accuracy,
> SFT jumps to the high 80s, DPO sharpens it further, and RFT on the reasoning
> model lands highest — and the propensity error shrinks the whole way down.
> Every number here is reproducible and auditable in the portal, not a slide."

> "This is the loop a customer would run continuously: fine-tune, evaluate in
> Foundry, compare, and promote the winner — all from the same SDK."

---

## Options reference

| Flag | Purpose |
| --- | --- |
| `--models base sft dpo rft` | Pick specific arms (default: all configured) |
| `--limit N` | Evaluate only the first N rows (keep the live demo snappy) |
| `--delay 0.5` | Pause between inference calls to ease rate limits |
| `--no-builtin` | Drop the F1 evaluator (custom metrics only) |
| `--no-upload` | Local-only rehearsal; results in `data/foundry_eval_*.json` |

Artifacts written to `data/`:

- `foundry_eval_data_<arm>.jsonl` — the per-arm evaluation dataset
- `foundry_eval_result_<arm>.json` — raw evaluate() output for that arm
- `foundry_eval_report.json` — combined scoreboard + portal links

---

## Troubleshooting

- **No runs appear in the portal** — make sure `az login` succeeded and the
  scoreboard footer says "uploaded", not "local-only". Local-only means the
  project identity (`AZURE_SUBSCRIPTION_ID` / `AZURE_RESOURCE_GROUP` /
  `AZURE_AI_PROJECT_NAME`) is incomplete.
- **An arm is missing** — its deployment name env var is unset; that arm is
  skipped by design. Set `RFT_DEPLOYMENT_NAME` etc. and re-run.
- **Rate-limit warnings** — raise `--delay` or lower `--limit`; the scorer
  already runs its own fast backoff for the shared-pool 429.
