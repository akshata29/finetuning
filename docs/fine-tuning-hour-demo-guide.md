---
title: One-Hour Fine-Tuning Demo Guide — Open-Weight LoRA for Long-Context Conversation Alignment
description: Presenter guide tailoring the Azure fine-tuning demo to a customer focused on LoRA fine-tuning of open-source foundation models for aligning conversational strategies from long multi-turn transcripts.
author: Playground demo team
ms.date: 2026-06-18
ms.topic: how-to
---

## Who this guide is for

This is the presenter playbook for a one-hour fine-tuning session with a
customer whose stated interests are:

- Best practices for fine-tuning **open-source foundation models** on Azure AI
  Foundry.
- **Training-data schema for long, multi-turn conversations.**
- **LoRA configuration, trainer, and scheduler settings.**
- Primary use case: **aligning conversational strategies** from a corpus of
  **long-context conversation transcripts.**

It re-sequences the existing four-act demo so the open-weight LoRA story leads,
and it adds a deep-dive block that answers their four questions head-on.

> Framing honesty: the live demo corpus is *short* synthetic sales-call
> transcripts (intent + propensity labels). The **mechanics** — data schema,
> LoRA knobs, trainer/scheduler, evaluation — are identical to their long-context
> alignment use case. Show the mechanics live on the demo data; map them to their
> corpus in the deep-dive. Do not imply the demo data is long multi-turn.

---

## The story arc (what you are really selling)

The customer asked about GPU LoRA on open-weight models. That is the right place
to *start* — but it is **not** the differentiator. Managed GPU + PEFT/LoRA is
table-stakes; AWS (SageMaker), GCP (Vertex), and any Kubernetes shop can run the
same `transformers + peft + bitsandbytes` recipe. If the whole conversation is
"we can also rent you A100s," you are competing on GPU price, and you lose.

**The classic technical-sales move: parity, then wedge.**

1. **Parity (earn trust):** "Yes — Foundry runs your exact open-weight LoRA/QLoRA
   recipe on managed GPU. Whatever you do on GCP today, you can lift-and-shift.
   We meet you where you are." Spend *minutes* here, not the hour.
2. **Wedge (the differentiator):** "But here's what you can't get anywhere else —
   **serverless fine-tuning**. No GPU quota, no cluster, no trainer babysitting.
   You upload data, call an API, and get a tuned model billed per training
   token — including **preference (DPO)** and **reinforcement (RFT)** alignment,
   which is exactly your use case. And it's on the **same governed plane** as your
   data generation, evaluation, and deployment."

Three messages to land (in priority order):

1. **Serverless is the moat.** SFT, **DPO**, and **RFT** as a managed,
   pay-per-token API — no GPU quota to chase, no cluster to run, no OOM to debug.
   This is the capability competitors don't match for alignment workloads.
2. **One governed plane.** Data → tune → **evaluate (Foundry Evaluations)** →
   deploy → iterate, all in the customer's tenant with one RBAC/residency story.
   Nobody else closes the loop in a single product.
3. **Open-weight when you want it.** The GPU LoRA path is there for full-recipe
   control and license-bound open models — so you're never boxed in. Parity, not
   the pitch.

> Why this matters for *their* alignment goal: aligning conversational strategy
> is a **preference problem**, and **serverless DPO/RFT** is the most direct,
> lowest-friction lever for it. That is a Foundry-native answer a GPU cluster
> alone cannot give them.

---

## The differentiator cheat-sheet (keep this in your head)

| Capability | Everyone has it (parity) | Azure Foundry differentiator |
|---|---|---|
| GPU LoRA / QLoRA on open weights | ✅ SageMaker, Vertex, any k8s | Runs here too — lift-and-shift the recipe |
| **Serverless SFT** (no GPU quota, per-token) | ⚠️ limited / model-locked elsewhere | ✅ Managed API, broad model support |
| **Serverless DPO** (preference alignment) | ❌ rare as a managed API | ✅ First-class, serverless |
| **Serverless RFT** (rubric-graded RL on reasoning models) | ❌ effectively unique | ✅ Grader-scored, o-series |
| Built-in **evaluation plane** wired to tuning | ⚠️ bolt-on / separate tools | ✅ Foundry Evaluations, same project |
| One tenant: data + tune + eval + deploy + RBAC | ⚠️ stitched across services | ✅ One governed plane |

Use this table to redirect any "but GCP can do X" objection back to the wedge:
the row where the competitor has ❌ or ⚠️ is where you spend your time.

---

## Run-of-show (60 minutes)

| Time | Block | Live vs pre-baked | What you run / show |
|---|---|---|---|
| 0:00–0:05 | Framing + their use case | — | Restate corpus + alignment goal; draw the loop; name parity-vs-wedge |
| 0:05–0:13 | **Bootstrap corpus from base agent (capture)** | Live capture | `capture --live --agent-deployment base`; show the **strategy gap report** — the agent's real number |
| 0:13–0:20 | **Multi-turn schema** | Live sample | open a captured JSONL row; walk `messages[]` + long-context variant |
| 0:20–0:26 | GPU LoRA (parity — earn trust, move on) | Pre-baked job | `gpu-lora`; "your recipe runs here"; walk LoRA/trainer briefly |
| 0:26–0:42 | **Serverless SFT + DPO + RFT (the wedge)** | Live upload/poll | `sft`, then `dpo`/`rft`; "no GPU quota; alignment as an API" |
| 0:42–0:52 | **Evaluation in Foundry (one plane)** | Live | `foundry-eval`; Evaluations tab; base vs SFT vs DPO vs RFT |
| 0:52–1:00 | Hosting + governance wrap | Live | `host`; one inference; "one plane" + residency close |

Adjust live/pre-baked to your quota. The GPU LoRA job is always pre-baked
(provisioning exceeds the hour); the serverless paths run live and are the
centerpiece. **Time budget signals the message: lead with *their* captured data,
~6 min on parity, ~16 on the wedge.** If `--live` capture is risky on the day,
run `capture` offline (deterministic) or fall back to `gen-data`.

---

## Block-by-block presenter notes

### 0:00–0:06 — Framing

- Restate their world back to them: "You have long conversation transcripts and
  you want the model to adopt the *strategies* in your best conversations —
  tone, escalation handling, when to probe vs. close."
- Draw the loop on screen: **capture → fine-tune → deploy → run inference →
  capture → iterate.** Call out that it's a **cycle**: today we enter at
  `capture` against the **base** agent (no tuned model exists yet) to bootstrap
  the corpus; after we tune and deploy, later laps capture from *their own* tuned
  agent. (Cold-start alternative: enter at `gen-data`.)
- Set up parity-vs-wedge out loud: "I'll show your GPU LoRA recipe runs here
  unchanged — that's the easy part. Then I'll show the part you can't get
  elsewhere: serverless preference and reinforcement tuning on the same plane."

### 0:05–0:13 — Bootstrap the corpus from the base agent (the customized opener)

This block makes the demo *theirs*, not a canned sample. It is the loop's
**entry point**: on lap 1 there is no tuned agent yet, so you capture the **base**
agent (`chat41mini`) mid-conversation as a stand-in for "your deployed agent,"
turn the transcripts into the training corpus, and show the gap between what the
agent does today and the strategy they want. (Later laps capture from the model
you just tuned — see the loop diagram below.)

```powershell
# live: capture from the BASE agent (lap 1 stand-in) with an LLM customer simulator.
# --concurrency fans independent conversations across threads (full run in minutes).
& .venv\Scripts\python.exe customer_conversation_alignment.py capture `
    --count 200 --eval-count 60 --live --use-llm `
    --agent-deployment base --concurrency 4
```

The headline is the **strategy gap report** — the customer's own number, live:

```text
baseline (captured)            0.41
aligned target                 0.93
opportunity (lift)             0.52
  consultative_discovery       0.55 -> 1.00
  value_framing                0.30 -> 0.94
```

Talking points:

- "This is *your* agent, today. We recorded real conversations, scored how often
  it follows your playbook, and that 0.41 is the gap we're about to close."
- "Nothing here is synthetic theater — the rejected examples are your agent's
  actual replies; the preferred examples are the strategy you want. That's the
  preference data DPO/RFT trains on."
- **Credibility anchor:** "This is the same pattern Microsoft ships as
  **TracesDistillation** in the `microsoft-foundry/fine-tuning` samples — pull
  real production traces, distill a better student. We've adapted it from tool-call
  distillation to your free-text *strategy* alignment. In production you swap our
  capture for the **`foundry-traces`** recipe and pull genuine App Insights
  traces — same schema, same next steps."
- Point to sibling Microsoft demos as proof the building blocks are documented,
  not bespoke: **ZavaRetailAgent** (multi-turn SFT+RFT with a policy grader),
  **SyntheticDatagen-ToolUse** (native programmatic datagen), and the
  **NL-to-Python distillation** sample (judge-filtered synthetic data).

> If a live agent call is risky on the day, run `capture` **offline** (it
> simulates deterministically and still produces the gap report and corpus), or
> fall back to the synthetic `gen-data` factory below.

> **Honest framing — how Act 0 fits the loop, and what's deployed today.**
> The lifecycle is a **cycle**, so it has no fixed start — you can enter it at
> `gen-data` (cold start, no agent yet) **or** at `capture` (point it at an agent
> that already exists). Act 0 enters at `capture` against the **base** model
> (`chat41mini`), prompted as a generic, un-aligned assistant — a deliberate
> *stand-in* for "your deployed agent." That is why the opener is `capture` even
> though the diagram draws `capture` last: **on the first lap there is no tuned
> agent to capture from, so you capture from the base model to bootstrap the
> corpus.** On later laps you capture from the model you just tuned.
>
> ```text
>   ┌── lap 1 enters here (capture from BASE = Act 0) ──┐
>   ▼                                                   │
>   capture ─▶ train ─▶ deploy ─▶ run inference ─▶ capture (from TUNED) ─┐
>   (base)    sft/dpo   (model)   (real convos)    (lap 2+ harvest)      │
>     ▲        /rft                                                      │
>     └──────────────────────── next iteration ────────────────────────┘
>
>   (cold-start alternative: skip Act 0 and enter at `gen-data` instead)
> ```
>
> **Run the whole loop end to end (one sitting).** Each command maps to one box;
> all run from `finetuning_demo` with `& .venv\Scripts\python.exe`:
>
> ```powershell
> # LAP 1 ─────────────────────────────────────────────────────────────────
> # 1. capture FROM BASE (Act 0) — bootstrap the corpus from the base agent.
> #    (offline: drop --live --use-llm for a zero-cost deterministic run.)
> & .venv\Scripts\python.exe customer_conversation_alignment.py capture `
>     --count 200 --eval-count 60 --live --use-llm `
>     --agent-deployment base --concurrency 4
>
> # 2. train + deploy each method. --deploy registers the deployment + writes
> #    its name so the 'sft'/'dpo'/'rft' capture labels resolve next lap.
> & .venv\Scripts\python.exe customer_conversation_alignment.py sft `
>     --deploy --deployment-name conv-align-sft --sku developer
> & .venv\Scripts\python.exe customer_conversation_alignment.py dpo `
>     --deploy --deployment-name conv-align-dpo --sku developer
> & .venv\Scripts\python.exe customer_conversation_alignment.py rft `
>     --deploy --deployment-name conv-align-rft --sku developer
>
> # 3. evaluate base vs sft vs dpo vs rft on one plane.
> & .venv\Scripts\python.exe customer_conversation_alignment.py foundry-eval `
>     --models base sft dpo rft --limit 50
>
> # LAP 2 ─────────────────────────────────────────────────────────────────
> # 4. set the deployment name in .env so the label resolves, e.g.:
> #      SFT_DEPLOYMENT_NAME=conv-align-sft
> #    then capture FROM YOUR TUNED AGENT (run inference + harvest in one step)
> #    and retrain on the improved, real-world corpus.
> & .venv\Scripts\python.exe customer_conversation_alignment.py capture `
>     --count 200 --eval-count 60 --live --use-llm `
>     --agent-deployment sft --concurrency 4
> # → back to step 2 to retrain on the captured corpus.
> ```
>
> `--agent-deployment` accepts a friendly label (`base`/`sft`/`dpo`/`rft`,
> resolved from `.env`) or a raw deployment name. If a label has no deployment
> configured yet, capture fails fast and tells you which env var to set — so you
> never silently capture from the wrong model. In production, swap step 4 for the
> **`foundry-traces`** recipe and pull real App Insights traces from the agent
> already serving — identical schema, so the train/eval steps run unchanged.

### 0:13–0:20 — The multi-turn schema (their question #1)

Open one captured row and walk the schema (see the deep-dive below). Key talking
point: **for conversation alignment, the unit of training is the whole
conversation, not a single Q/A pair.** Show how the `messages[]` array carries
the full multi-turn context and where the "good strategy" lives (the assistant
turns you want the model to imitate). Note that the *same* JSONL schema feeds
both the GPU and serverless paths — one data investment, two tuning options.

> Fallback / dry-run: the synthetic factory makes schema-locked, PII-free data
> on demand if you can't capture live:
>
> ```powershell
> python run_of_show.py gen-data --count 200 --eval-count 200
> ```

### 0:16–0:24 — GPU LoRA on managed GPU (PARITY — earn trust, then move on)

Keep this tight. The goal is to remove the "can Azure even do what we do today?"
objection, not to dwell. Show the pre-baked job and the recipe:

```powershell
python run_of_show.py gpu-lora --prebaked
```

Talk track: "This is your world today — open-weight base, LoRA/QLoRA, your
trainer and scheduler settings, on managed A100s. It runs here unchanged; you
can lift-and-shift off GCP. Full hyperparameter control is in the deep-dive if
your ML engineers want it. **But notice what this still costs you: GPU quota, a
cluster, and someone watching the trainer.** Hold that thought."

Open [act2b_gpu_lora.py](../act2b_gpu_lora.py) only if they ask for the knobs;
otherwise point to the deep-dive and pivot.

### 0:24–0:42 — Serverless SFT + DPO + RFT (THE WEDGE — spend your time here)

walk the actual configuration the demo encodes:

```powershell
This is the centerpiece. The whole point: **no GPU quota, no cluster, no trainer
to babysit — and alignment (DPO/RFT) as a managed API**, which is the exact lever
their conversational-strategy goal needs.

Start with supervised to set the format and baseline behavior:

```powershell
python run_of_show.py sft        # serverless supervised fine-tune
```

While it uploads/polls, narrate the contrast you just set up: "Remember the GPU
path needed quota and a cluster? Here I uploaded a JSONL and called an API. It's
billed per training token. No infrastructure on my side at all."

Then move to **alignment** — the part that is hard or impossible to get as a
managed service anywhere else:

```powershell
python run_of_show.py dpo        # preference alignment (beta=0.1, 2 epochs)
python run_of_show.py rft        # rubric-graded reinforcement fine-tuning
```

- `dpo` — **Direct Preference Optimization**: train on *preferred vs. rejected*
  conversation responses. This is the most direct lever for "make the model
  prefer our good conversational strategy." Tie it to their corpus: "Your best
  transcripts are the *chosen*; weaker ones are *rejected*."
- `rft` — **Reinforcement fine-tuning**: a grader scores each conversation
  against a rubric (e.g., "did it follow the escalation policy?") and the model
  is rewarded toward the strategy. Encode the policy as a grader, not as labels.

Land the wedge: "SFT teaches the format. **DPO and RFT — serverless — are how you
align to the strategy you reward, with no GPU cluster in the loop.** That's the
Foundry-native answer to your alignment problem, and it's the part a rented A100
can't give you."

### 0:42–0:52 — Evaluation in Foundry (one plane — the second differentiator)

```powershell
python run_of_show.py foundry-eval --limit 50
```

Open **Build → Evaluations** and show the uploaded runs side by side (base vs
SFT vs DPO vs RFT). See the dedicated [Foundry eval showcase](foundry-eval-showcase.md)
for the full talk track. Land two points:

- "Every claim of improvement is a row in this table you can audit."
- "**This is the same project** that generated the data and ran the tuning —
  one plane, one RBAC boundary, one residency story. You're not stitching a
  training service to a separate eval tool to a separate registry."

### 0:52–1:00 — Hosting + governance wrap

```powershell
python run_of_show.py host --deployment-name sales-intent
```

Close on the "one plane" story: data prep, tuning, evaluation, deployment, and
RBAC/residency all in the customer's tenant.

---

## Deep dive: answering the four customer questions

### 1) Training-data schema for long, multi-turn conversations

**Base unit.** Every training example is one full conversation as a `messages[]`
array in chat-completion format. The demo's single-turn rows look like this:

```jsonc
{"messages": [
  {"role": "system",    "content": "You are a sales-call strategist..."},
  {"role": "user",      "content": "<transcript or customer turn>"},
  {"role": "assistant", "content": "<the response/label you want imitated>"}
]}
```

**For their long-context alignment use case**, extend the same schema to carry
the entire multi-turn dialogue, alternating `user`/`assistant` for the full
conversation:

```jsonc
{"messages": [
  {"role": "system",    "content": "<role + strategy guidelines>"},
  {"role": "user",      "content": "<customer turn 1>"},
  {"role": "assistant", "content": "<agent turn 1 — exemplary strategy>"},
  {"role": "user",      "content": "<customer turn 2>"},
  {"role": "assistant", "content": "<agent turn 2 — exemplary strategy>"}
  // ... continue for the whole transcript
]}
```

Schema best practices to recommend:

- **One JSON object per line (JSONL).** UTF-8. The demo writes UTF-8-with-BOM,
  which Azure accepts; plain UTF-8 is also fine.
- **Put the strategy you want learned in the `assistant` turns.** The model is
  trained to reproduce assistant content given the preceding context. Curate
  these to your *best* conversations — bad examples teach bad strategy.
- **System prompt carries the policy.** Keep it consistent across rows so the
  model attaches the behavior to the role, not to prompt noise.
- **Loss masking on multi-turn.** By default the trainer learns from every
  assistant turn in the conversation. That is what you want for alignment
  (learn the strategy at every step). If you only want the final turn learned,
  mask earlier assistant turns — call this out as a recipe choice.
- **Sequence length is the gating constraint for long context.** A long
  transcript must fit in `max_seq_length` (tokens) or it gets truncated. See the
  trainer settings below.
- **Hold out whole conversations, never turns.** Your validation split must be
  *disjoint conversations*, not turns sampled from training conversations, or
  you leak context and overstate accuracy.
- **For preference alignment (DPO):** each row is a `(prompt, chosen, rejected)`
  triple — the same conversation context with a *preferred* and a *dispreferred*
  continuation. This is the cleanest schema for "prefer our strategy."

### 2) LoRA configuration

The demo's confirmed classic-pipeline LoRA defaults
([act2b_gpu_lora.py](../act2b_gpu_lora.py), `PIPELINE_LORA_DEFAULTS`):

| Key | Demo default | Guidance for long-context alignment |
|---|---|---|
| `apply_lora` | `"true"` | Keep LoRA on — adapter-only training is the point |
| `lora_r` (rank) | `8` | Start 8–16; raise to 32–64 if the strategy is complex/underfits |
| `lora_alpha` | `128` | Keep `alpha ≈ 8–16 × r` as a scaling rule of thumb |
| `lora_dropout` | `0.0` | 0.05–0.1 helps regularize on smaller curated corpora |
| `precision` | `"16"` | bf16/fp16 on A100/H100; the classic pipeline has no 4-bit knob |

**QLoRA parity** (4-bit) lives on the separate command-job spec
(`custom_qlora_command_job_spec`) because the managed pipeline exposes plain
LoRA only. Its defaults:

- `load_in_4bit` quantization (bitsandbytes) — fit larger bases on one GPU.
- Explicit `target_modules`: `q_proj, k_proj, v_proj, o_proj, gate_proj,
  up_proj, down_proj` (attention + MLP projections for Llama/Phi-style models).

Guidance: **target attention + MLP projections** (as above) for strategy
alignment — restricting LoRA to only `q/v_proj` underfits behavioral change.
Use QLoRA when the open-weight base is large enough that full-precision LoRA
won't fit the GPU.

### 3) Trainer and scheduler settings

Demo defaults (`PIPELINE_TRAINER_DEFAULTS`):

| Key | Demo default | Guidance for long multi-turn |
|---|---|---|
| `num_train_epochs` | `3` | 1–3; watch eval loss — alignment overfits fast on curated data |
| `per_device_train_batch_size` | `1` | Long sequences force small per-device batch |
| `gradient_accumulation_steps` | `8` | Raise this to keep an effective batch of 8–32 without OOM |
| `learning_rate` | `1e-4` | 1e-4 to 2e-4 is typical for LoRA (higher than full FT) |
| `lr_scheduler_type` | `cosine` | Cosine with warmup is a safe default |
| `warmup_steps` | `50` | ~3–5% of total steps; raise for larger corpora |
| `optim` | `adamw_torch` | Use `paged_adamw_8bit` with QLoRA to save memory |
| `weight_decay` | `0.0` | 0.0–0.1; small decay can help generalization |
| `max_seq_length` | `2048` | **Raise to fit whole transcripts** (4k/8k/16k) — biggest lever for long context |
| `seed` | `1337` | Fix for reproducible eval comparisons |

The single most important change for their use case: **raise `max_seq_length`**
until typical conversations fit without truncation, then **increase
`gradient_accumulation_steps`** to recover effective batch size (since long
sequences force `per_device_train_batch_size=1`). Expect higher GPU memory —
this is where QLoRA earns its place.

Default GPU SKU in the demo: `Standard_NC24ads_A100_v4` (A100 80GB). For 8k–16k
sequence lengths on a 7B+ open-weight base, A100 80GB or H100 is the realistic
floor; QLoRA lets you stay on a single card.

### 4) Other fine-tuning considerations

- **Serverless vs. GPU LoRA — which path, when.** Default to **serverless** for
  alignment: no quota, per-token billing, and DPO/RFT are managed. Reach for the
  **GPU LoRA** path only when you need (a) a specific open-weight base not offered
  serverless, (b) a custom training loop / loss, or (c) the adapter artifact in
  your own registry. Most customers start serverless and keep GPU LoRA as the
  escape hatch — not the other way around.
- **Method selection for alignment.** SFT to set format and baseline behavior;
  **DPO** to push toward preferred strategy; **RFT** when you can express the
  strategy as a graded rubric. Most alignment programs run SFT first, then DPO —
  and on Foundry both are **serverless**, which is the differentiator.
- **Data quality over quantity.** A few hundred to a few thousand *curated*
  exemplary conversations beat a large noisy dump. Garbage assistant turns teach
  garbage strategy.
- **Evaluate on held-out conversations** with the Foundry Evaluations tab; track
  a behavioral metric (e.g., adherence to the strategy rubric), not just loss.
- **Catastrophic forgetting.** Heavy alignment can erode general ability; keep a
  small general-capability eval in the mix and prefer lower epochs / lower rank
  if it regresses.
- **Adapter portability.** The LoRA adapter is small and yours; you can host
  base + adapter, swap adapters per strategy, or merge for deployment.
- **Governance.** Everything runs in the customer tenant: data never leaves,
  RBAC controls who can tune/deploy, and region selection covers residency.

---

## Commands quick reference

```powershell
cd finetuning_demo

# Data from THEIR corpus (capture a deployed agent -> training corpus)
# --agent-deployment accepts a friendly label (base/sft/dpo/rft, from .env) or a
# raw name. Use base for the pre-tuning stand-in; sft/dpo/rft to close the loop.
# --concurrency fans independent conversations across threads.
& .venv\Scripts\python.exe customer_conversation_alignment.py capture `
    --count 200 --eval-count 60 --live --use-llm --agent-deployment base --concurrency 4

# Data + schema (synthetic fallback / dry-run); add --use-llm --concurrency 4 for
# a diverse, parallel teacher-authored corpus.
python run_of_show.py gen-data --count 200 --eval-count 200

# Open-weight GPU LoRA (pre-baked job; live spec walk)
python run_of_show.py gpu-lora --prebaked

# Serverless paths (contrast + alignment methods)
python run_of_show.py sft
python run_of_show.py dpo        # preference alignment
python run_of_show.py rft        # rubric-graded alignment

# Evaluation in the Foundry portal
python run_of_show.py foundry-eval --limit 50

# Host + one inference
python run_of_show.py host --deployment-name sales-intent
```

## Pre-flight (T-2 days)

- [ ] GPU quota approved (A100 80GB / H100) in your region.
- [ ] Pre-baked GPU LoRA job finished and inspectable.
- [ ] SFT/DPO/RFT models tuned and ready to deploy (done).
- [ ] `foundry-eval` dry run (`--no-upload`) clean; `az login` working for the
      live upload.
- [ ] One long-context example row prepared (even synthetic) to show the
      multi-turn schema concretely.

## Anticipated questions

### Competitive / differentiator objections (handle these crisply)

- *"GCP/AWS already runs our GPU LoRA — why move?"* — Don't fight on GPU. Agree:
  "You can keep doing exactly that here." Then pivot: "What you *can't* do on a
  rented cluster is call **serverless DPO/RFT** to align to your strategy with no
  quota and no trainer to run, on the **same plane** as your eval and deployment.
  That's the part that removes infra from your team's plate."
- *"Isn't serverless just a wrapper around the same training?"* — The point isn't
  the GPU under the hood; it's that **you never touch it**. No quota requests, no
  cluster lifecycle, no OOM debugging, per-token economics, and **DPO/RFT exposed
  as an API** — which is operationally unique for alignment workloads.
- *"Can we get DPO/RFT as a managed service elsewhere?"* — Preference (DPO) and
  rubric-graded reinforcement (RFT) as a turnkey managed API is effectively a
  Foundry differentiator today; competitors largely require you to build the RL
  loop yourself on raw compute.
- *"What about lock-in?"* — The GPU LoRA path keeps you portable: open-weight
  base, standard `peft` adapter you own. Use serverless for speed, keep GPU LoRA
  as the exit. That symmetry is itself a selling point.

### Technical questions

- *"Which open-weight models are supported?"* — Any model the managed pipeline /
  your command-job environment can load (Phi, Llama, Qwen, Mistral), subject to
  the model's license. The demo defaults to Phi-4-mini-instruct.
- *"Full fine-tune vs. LoRA?"* — LoRA for cost, speed, and adapter portability;
  full FT only when you need to move base weights substantially. For strategy
  alignment, LoRA is almost always the right call.
- *"How long are jobs?"* — Provisioning + training exceeds the hour, which is why
  the GPU job is pre-baked; show the finished job and the recipe.
- *"Can we keep data in-region?"* — Yes; region selection and tenant-scoped
  compute cover residency.
