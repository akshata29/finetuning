"""Agent corpus capture — turn a deployed agent's real conversations into a
strategy-alignment training corpus.

This is the **"bring your real conversations into Azure"** stage of the
conversation-alignment demo. It adapts Microsoft's *TracesDistillation* recipe
(pull real production traces from a deployed Foundry agent, transform them, then
fine-tune a student on them) to this customer's actual use case: aligning the
*free-text conversational strategy* of multi-turn sales conversations rather than
distilling tool-calling behavior.

The story it tells, in order:

1. **Capture** — drive a deployed *baseline* agent with a realistic customer
   simulator across multi-turn conversations and record the transcripts. This
   stands in for "pull the last N days of real conversations from your agent."
   (In production you swap the live capture for Foundry's ``foundry-traces`` Data
   Generation recipe against App Insights — same downstream.)
2. **Transform** — normalize the captured transcripts into clean, role-alternating
   message arrays (the equivalent of the inline fixups TracesDistillation applies
   to Foundry's trace export).
3. **Score the gap** — judge each captured *baseline* reply for strategy adherence
   and report the mean. This quantifies the opportunity: "your agent today scores
   X on your winning strategy; here is the data to close that gap."
4. **Distill** — pair every captured conversation context with a strategy-aligned
   target reply (the *preferred*) and keep the captured baseline reply as the
   *non-preferred*. This produces drop-in SFT / DPO / RFT / eval datasets in the
   exact same schema as :mod:`finetuning.customer_conversation_alignment`,
   so the existing serverless ``sft`` / ``dpo`` / ``rft`` / ``foundry-eval`` acts
   run unchanged on the captured corpus.

Design choices:

* **Offline by default.** With no ``--live`` flag the capture is simulated
  deterministically (templated customer turns + a generic baseline agent), so the
  demo runs anywhere with zero Azure calls and reproducible numbers. ``--live``
  drives the real deployed agent + an LLM customer simulator.
* **Import / quota safe (OWASP A05/A06).** Azure SDKs are resolved lazily and
  only when ``--live`` is set; every endpoint/secret comes from
  :class:`DemoConfig` via the environment.
* **Data isolation.** Raw captured transcripts, the manifest, and the gap report
  live under ``data/conversation_alignment/captured/``. The training files it
  emits are the same ``conv_*.jsonl`` the rest of the pipeline consumes, so
  capture is a drop-in replacement for ``gen-data``.

This module deliberately does **not** use ``from __future__ import annotations``
to stay consistent with :mod:`customer_conversation_alignment` (which defines a
promptflow custom evaluator that rejects stringized hints).
"""

import argparse
import json
import logging
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

# Allow running both as a module and as a plain script.
if __package__ in (None, ""):  # pragma: no cover - script-launch shim
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "finetuning"

from .config import DemoConfig
from .customer_conversation_alignment import (
    CONV_EVAL_FILE,
    CONV_DPO_TRAIN_FILE,
    CONV_DPO_VAL_FILE,
    CONV_RFT_TRAIN_FILE,
    CONV_RFT_VAL_FILE,
    CONV_SFT_TRAIN_FILE,
    CONV_SFT_VAL_FILE,
    DATA_DIR,
    EXIT_ERROR,
    EXIT_SUCCESS,
    STRATEGIES,
    STRATEGY_GUIDELINE,
    _customer_turns,
    _good_turn,
    _INDUSTRIES,
    _OBJECTIONS,
    _PERSONAS,
    _render_transcript,
    _TOPICS,
    _weak_turn,
    generate_agent_turn,
    strategy_alignment_score,
    write_conv_dpo_jsonl,
    write_conv_eval_jsonl,
    write_conv_rft_jsonl,
    write_conv_sft_jsonl,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Capture artifact layout (provenance lives next to the training corpus)
# ---------------------------------------------------------------------------
#: Folder for raw captured transcripts + provenance (kept separate from the
#: training ``conv_*.jsonl`` files so the corpus origin is auditable).
CAPTURE_DIR: Path = DATA_DIR / "captured"
CAPTURED_TRANSCRIPTS_FILE: str = "captured_transcripts.jsonl"
CAPTURE_MANIFEST_FILE: str = "capture_manifest.json"
CAPTURE_GAP_REPORT_FILE: str = "capture_gap_report.json"

#: Defaults for live capture concurrency / rate-limit resilience. Conversations
#: are independent, so we fan them out across a small thread pool; each worker's
#: own conversation is still strictly sequential (turn N+1 depends on turn N).
#: The OpenAI SDK retries 429/5xx with exponential backoff honoring
#: ``Retry-After``, so a raised ``max_retries`` budget absorbs rate limits.
DEFAULT_CONCURRENCY: int = 4
DEFAULT_MAX_RETRIES: int = 6
DEFAULT_REQUEST_TIMEOUT: float = 120.0

#: Friendly ``--agent-deployment`` labels that resolve to the configured
#: deployment names, so the closed loop reads naturally::
#:
#:     gen-data -> sft/dpo/rft (deploy) -> capture --agent-deployment sft
#:
#: ``base`` captures the un-tuned model (the pre-tuning baseline); ``sft`` /
#: ``dpo`` / ``rft`` capture from *your own* deployed fine-tuned agent so the
#: next iteration distills from the model you actually shipped. Any value not in
#: this map is treated as a raw deployment name and passed through untouched.
AGENT_DEPLOYMENT_LABELS: dict[str, str] = {
    "base": "base_deployment_name",
    "sft": "sft_deployment_name",
    "dpo": "dpo_deployment_name",
    "rft": "rft_deployment_name",
}


def resolve_agent_deployment(config: DemoConfig, agent_deployment: str | None) -> str | None:
    """Resolve a friendly label (``base``/``sft``/``dpo``/``rft``) to a deployment.

    Returns the raw value unchanged when it is not a known label, so callers can
    still pass an explicit deployment name. Raises ``ValueError`` when a known
    label maps to an unset config value (for example ``sft`` before the
    fine-tuned model is deployed) so the user gets a clear, actionable error
    instead of capturing from the wrong model.
    """
    if not agent_deployment:
        return agent_deployment
    attr = AGENT_DEPLOYMENT_LABELS.get(agent_deployment.lower())
    if attr is None:
        return agent_deployment  # raw deployment name — pass through
    resolved = getattr(config, attr, "") or ""
    if not resolved:
        env_var = attr.upper()
        raise ValueError(
            f"--agent-deployment '{agent_deployment}' has no deployment configured "
            f"(set {env_var} in .env after you deploy that model, or pass the raw "
            "deployment name)."
        )
    return resolved


#: The baseline agent is intentionally *un-aligned* — a generic assistant prompt,
#: so the conversations we capture reflect real, pre-tuning behavior that the
#: SFT/DPO/RFT loop then improves. (In live mode this is only used if the target
#: deployment has no system prompt of its own.)
BASELINE_AGENT_SYSTEM: str = (
    "You are a helpful B2B sales assistant. Answer the prospect's questions and "
    "keep the conversation moving."
)

#: Customer-simulator persona used in ``--live`` capture to play the prospect.
CUSTOMER_SIM_SYSTEM: str = (
    "You role-play a B2B buyer evaluating a vendor. Stay in character as the "
    "prospect only — never break character or speak as the seller. Keep each "
    "reply to 1-2 natural sentences. Use placeholders only (no real names, "
    "companies, or numbers)."
)

#: Generic, un-strategic baseline replies used by the offline capture simulator.
#: They are plausible and helpful but do not apply any target strategy, so the
#: gap report shows a realistic low baseline score.
_BASELINE_REPLIES: tuple[str, ...] = (
    "Sure — our platform can definitely help with {topic}. It has a lot of "
    "capabilities that {industry} teams use.",
    "Thanks for reaching out. We work with plenty of {industry} companies on "
    "{topic}, so you're in good company.",
    "Happy to help. Just let me know what questions you have about {topic} and "
    "I'll walk you through what we offer.",
    "Good question. We have several features that cover {topic} — I can send "
    "over a deck if that's useful.",
)


# ---------------------------------------------------------------------------
# Scenario sampling (mirrors the synthetic factory so captured + synthetic data
# share one taxonomy)
# ---------------------------------------------------------------------------
def _sample_scenario(index: int, rng: random.Random) -> dict[str, str]:
    """Sample one scenario dict, cycling strategies for an even distribution."""
    strategy = STRATEGIES[index % len(STRATEGIES)]
    return {
        "industry": rng.choice(_INDUSTRIES),
        "persona": rng.choice(_PERSONAS),
        "topic": rng.choice(_TOPICS),
        "objection": rng.choice(_OBJECTIONS),
        "strategy": strategy,
    }


def _baseline_agent_turn(scenario: dict[str, str], rng: random.Random) -> str:
    """Render a generic, un-strategic baseline agent reply (offline capture)."""
    template = rng.choice(_BASELINE_REPLIES)
    return template.format(industry=scenario["industry"], topic=scenario["topic"])


# ---------------------------------------------------------------------------
# Live simulator / agent helpers (lazy Azure use)
# ---------------------------------------------------------------------------
def _simulate_customer_turn(
    config: DemoConfig,
    scenario: dict[str, str],
    phase: str,
    *,
    client: Any,
) -> str:
    """Generate one customer turn from the LLM simulator (``--live`` only).

    ``phase`` is one of ``opening`` / ``middle`` / ``closing`` and selects the
    instruction the simulated buyer follows. Fails safe to the deterministic
    template turn so a transient error never aborts a capture run.
    """
    model = config.teacher_model or config.base_deployment_name
    instruction = {
        "opening": (
            "Open the conversation: briefly say who you are (use your persona) "
            "and what you're evaluating."
        ),
        "middle": f"Raise this objection naturally: {scenario['objection']}.",
        "closing": "Ask the seller where you should go from here.",
    }[phase]
    persona_line = (
        f"You are a {scenario['persona']} at a {scenario['industry']} company "
        f"evaluating a solution for {scenario['topic']}."
    )
    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=0.8,
            messages=[
                {"role": "system", "content": CUSTOMER_SIM_SYSTEM},
                {"role": "user", "content": f"{persona_line}\n{instruction}"},
            ],
        )
        text = completion.choices[0].message.content
        if isinstance(text, str) and text.strip():
            return text.strip()
    except Exception as exc:  # noqa: BLE001 - fail safe to template
        logger.warning("[capture] customer simulator failed (%s); using template", exc)
    opening, middle, closing = _customer_turns(scenario, random.Random(0))
    return {"opening": opening, "middle": middle, "closing": closing}[phase]


def _aligned_reply(
    config: DemoConfig,
    scenario: dict[str, str],
    context_messages: list[dict[str, str]],
    rng: random.Random,
    *,
    use_llm: bool,
    client: Any | None,
) -> str:
    """Produce the strategy-aligned *target* reply for a captured context.

    With ``use_llm`` the target is authored by the teacher model conditioned on
    the captured conversation and the target strategy (true distillation); on any
    failure (or when ``use_llm`` is False) it falls back to the deterministic
    strategy exemplar so a target reply is always available.
    """
    if use_llm and client is not None:
        strategy = scenario["strategy"]
        # Render the captured turns as a plain-text transcript inside ONE
        # instruction message rather than replaying them as live message roles.
        # Replaying prior turns as roles and then appending a "reply as the agent"
        # override trips Azure's jailbreak Prompt Shield (false positive); a single
        # readable transcript + coaching ask is benign and gets the same result.
        transcript = _render_transcript(context_messages)
        try:
            messages = [
                {"role": "system", "content": STRATEGY_GUIDELINE[strategy]},
                {
                    "role": "user",
                    "content": (
                        "Below is a sales conversation transcript so far.\n\n"
                        f"{transcript}\n\n"
                        "Write the next Agent reply so it clearly applies the "
                        f"'{strategy}' strategy. 1-3 sentences. Use placeholders "
                        "for any specific names, numbers, or facts. Return only "
                        "the reply text."
                    ),
                },
            ]
            text = generate_agent_turn(
                config,
                config.teacher_model or config.base_deployment_name,
                messages,
                client=client,
                temperature=0.7,
            )
            if text.strip():
                return text.strip()
        except Exception as exc:  # noqa: BLE001 - fail safe to template
            logger.warning("[capture] aligned-reply teacher failed (%s); using exemplar", exc)
    return _good_turn(scenario["strategy"], scenario, rng)


# ---------------------------------------------------------------------------
# Capture + transform + distill
# ---------------------------------------------------------------------------
def _normalize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Transform a captured transcript into a clean, role-alternating array.

    Mirrors (in spirit) the inline fixups TracesDistillation applies to Foundry's
    trace export: drop empty turns, coerce content to ``str``, and collapse
    consecutive same-role turns by keeping the last. The result always starts at
    a system turn (if present) and alternates user/assistant.
    """
    cleaned: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in ("system", "user", "assistant"):
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        if cleaned and cleaned[-1]["role"] == role and role != "system":
            cleaned[-1] = {"role": role, "content": content.strip()}
            continue
        cleaned.append({"role": role, "content": content.strip()})
    return cleaned


def capture_one(
    config: DemoConfig,
    index: int,
    rng: random.Random,
    *,
    live: bool,
    use_llm: bool,
    agent_deployment: str | None,
    client: Any | None,
) -> dict[str, Any]:
    """Capture one multi-turn conversation and build a training record.

    Returns a record in the same schema as
    :func:`customer_conversation_alignment.generate_conversations` (so the
    existing writers consume it unchanged), augmented with capture provenance:
    ``raw_captured_messages``, ``baseline_reply``, ``baseline_score`` and
    ``aligned_score``.
    """
    scenario = _sample_scenario(index, rng)
    strategy = scenario["strategy"]

    # --- 1. Capture the customer + baseline-agent turns -------------------
    if live and client is not None and agent_deployment:
        opening = _simulate_customer_turn(config, scenario, "opening", client=client)
        baseline_sys = {"role": "system", "content": BASELINE_AGENT_SYSTEM}
        agent_1 = _safe_agent_turn(
            config, agent_deployment, [baseline_sys, {"role": "user", "content": opening}],
            scenario, rng, client=client,
        )
        middle = _simulate_customer_turn(config, scenario, "middle", client=client)
        agent_2 = _safe_agent_turn(
            config, agent_deployment,
            [
                baseline_sys,
                {"role": "user", "content": opening},
                {"role": "assistant", "content": agent_1},
                {"role": "user", "content": middle},
            ],
            scenario, rng, client=client,
        )
        closing = _simulate_customer_turn(config, scenario, "closing", client=client)
        agent_final = _safe_agent_turn(
            config, agent_deployment,
            [
                baseline_sys,
                {"role": "user", "content": opening},
                {"role": "assistant", "content": agent_1},
                {"role": "user", "content": middle},
                {"role": "assistant", "content": agent_2},
                {"role": "user", "content": closing},
            ],
            scenario, rng, client=client,
        )
    else:
        opening, middle, closing = _customer_turns(scenario, rng)
        agent_1 = _baseline_agent_turn(scenario, rng)
        agent_2 = _baseline_agent_turn(scenario, rng)
        agent_final = _baseline_agent_turn(scenario, rng)

    # Raw captured transcript (baseline system + real turns) — provenance only.
    raw_captured_messages = _normalize_messages(
        [
            {"role": "system", "content": BASELINE_AGENT_SYSTEM},
            {"role": "user", "content": opening},
            {"role": "assistant", "content": agent_1},
            {"role": "user", "content": middle},
            {"role": "assistant", "content": agent_2},
            {"role": "user", "content": closing},
            {"role": "assistant", "content": agent_final},
        ]
    )

    # --- 2. Build the aligned target (distillation) ----------------------
    # The training record re-frames the captured customer context under the
    # target-strategy system prompt and pairs it with strategy-aligned agent
    # turns; the captured baseline final reply becomes the non-preferred sample.
    system = {"role": "system", "content": STRATEGY_GUIDELINE[strategy]}
    context_for_target = [
        system,
        {"role": "user", "content": opening},
        {"role": "assistant", "content": agent_1},
        {"role": "user", "content": middle},
        {"role": "assistant", "content": agent_2},
        {"role": "user", "content": closing},
    ]
    good_1 = _aligned_reply(config, scenario, context_for_target[:2], rng, use_llm=use_llm, client=client)
    good_2 = _aligned_reply(config, scenario, context_for_target[:4], rng, use_llm=use_llm, client=client)
    preferred = _aligned_reply(config, scenario, context_for_target, rng, use_llm=use_llm, client=client)
    non_preferred = agent_final  # the real, captured (un-aligned) reply

    messages = [
        system,
        {"role": "user", "content": opening},
        {"role": "assistant", "content": good_1},
        {"role": "user", "content": middle},
        {"role": "assistant", "content": good_2},
        {"role": "user", "content": closing},
        {"role": "assistant", "content": preferred},
    ]
    context_messages = messages[:-1]

    # --- 3. Score the alignment gap --------------------------------------
    baseline_score = strategy_alignment_score(agent_final, strategy)
    aligned_score = strategy_alignment_score(preferred, strategy)

    return {
        "id": f"capture-{index + 1:06d}",
        "strategy": strategy,
        "scenario": scenario,
        "messages": messages,
        "context_messages": context_messages,
        "preferred": preferred,
        "non_preferred": non_preferred,
        "transcript": _render_transcript(context_messages),
        # provenance / gap
        "raw_captured_messages": raw_captured_messages,
        "baseline_reply": agent_final,
        "baseline_score": baseline_score,
        "aligned_score": aligned_score,
        "captured_live": bool(live and client is not None and agent_deployment),
    }


def _safe_agent_turn(
    config: DemoConfig,
    deployment: str,
    messages: list[dict[str, str]],
    scenario: dict[str, str],
    rng: random.Random,
    *,
    client: Any,
) -> str:
    """Call the deployed baseline agent for one turn, failing safe to a template."""
    try:
        text = generate_agent_turn(config, deployment, messages, client=client, temperature=0.5)
        if text.strip():
            return text.strip()
    except Exception as exc:  # noqa: BLE001 - fail safe per turn
        logger.warning("[capture] baseline agent turn failed (%s); using template", exc)
    return _baseline_agent_turn(scenario, rng)


def build_capture_client(config: DemoConfig, *, max_retries: int = DEFAULT_MAX_RETRIES) -> Any:
    """Build a live Azure client tuned for resilient, parallel capture.

    Reuses the tested data-plane client builder, then raises the SDK retry budget
    and sets a per-request timeout via ``with_options``. The OpenAI SDK retries
    429 (rate limit) and 5xx responses with exponential backoff that honors the
    ``Retry-After`` header, so a higher ``max_retries`` lets several concurrent
    workers ride out throttling instead of failing the conversation.
    """
    from . import act2a_serverless_sft as sft  # noqa: PLC0415

    client = sft.build_client(config)
    try:
        return client.with_options(max_retries=max_retries, timeout=DEFAULT_REQUEST_TIMEOUT)
    except Exception:  # noqa: BLE001 - older SDK / test double without with_options
        return client


def _capture_records(
    config: DemoConfig,
    count: int,
    *,
    seed: int,
    live: bool,
    use_llm: bool,
    resolved_agent: str | None,
    client: Any | None,
    concurrency: int,
    label: str = "",
) -> list[dict[str, Any]]:
    """Capture ``count`` conversations, optionally fanned out across threads.

    Each conversation gets its own ``random.Random(seed * P + index)`` — a
    distinct, collision-free stream across train/val/eval splits (``P`` exceeds
    any plausible index) — so the work is both deterministic and thread-safe (a
    shared ``Random`` is neither). Results are returned in index order regardless
    of completion order.
    """
    def _one(index: int) -> dict[str, Any]:
        rng = random.Random(seed * 1_000_003 + index)
        return capture_one(
            config, index, rng,
            live=live, use_llm=use_llm,
            agent_deployment=resolved_agent, client=client,
        )

    workers = max(1, concurrency)
    # Concurrency only helps the live path (the offline simulator does no I/O).
    if workers == 1 or not live:
        return [_one(index) for index in range(count)]

    tag = f"[capture{(' ' + label) if label else ''}]"
    print(
        f"{tag} starting {count} live conversation(s) across {workers} worker(s)...",
        file=sys.stderr, flush=True,
    )
    records: list[dict[str, Any] | None] = [None] * count
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, index): index for index in range(count)}
        for done, future in enumerate(as_completed(futures), start=1):
            records[futures[future]] = future.result()
            if done == count or done % 10 == 0:
                print(f"{tag} {done}/{count} conversations captured",
                      file=sys.stderr, flush=True)
    return [record for record in records if record is not None]


def capture_corpus(
    config: DemoConfig,
    count: int,
    *,
    seed: int = 4242,
    live: bool = False,
    use_llm: bool = False,
    agent_deployment: str | None = None,
    client: Any | None = None,
    concurrency: int = 1,
    max_retries: int = DEFAULT_MAX_RETRIES,
    label: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Capture ``count`` conversations and return ``(records, gap_report)``.

    Builds an Azure client only when ``live`` is set. In live mode independent
    conversations are fanned out across ``concurrency`` worker threads (each
    conversation stays internally sequential); the gap report summarizes the mean
    captured-baseline vs. aligned strategy score (overall and per strategy) — the
    headline number for the demo.
    """
    active = client
    if live and active is None:
        active = build_capture_client(config, max_retries=max_retries)
    resolved_agent = agent_deployment or (config.base_deployment_name if live else None)

    records = _capture_records(
        config, count,
        seed=seed, live=live, use_llm=use_llm,
        resolved_agent=resolved_agent, client=active,
        concurrency=concurrency, label=label,
    )
    gap_report = _build_gap_report(records, live=live, agent_deployment=resolved_agent)
    logger.info(
        "Captured %d conversation(s); baseline mean=%.3f aligned mean=%.3f",
        len(records), gap_report["baseline_mean"], gap_report["aligned_mean"],
    )
    return records, gap_report


def _build_gap_report(
    records: list[dict[str, Any]],
    *,
    live: bool,
    agent_deployment: str | None,
) -> dict[str, Any]:
    """Summarize the strategy gap between captured baseline and aligned replies."""
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    baseline_scores = [r["baseline_score"] for r in records]
    aligned_scores = [r["aligned_score"] for r in records]
    per_strategy: dict[str, Any] = {}
    for strategy in STRATEGIES:
        b = [r["baseline_score"] for r in records if r["strategy"] == strategy]
        a = [r["aligned_score"] for r in records if r["strategy"] == strategy]
        if b:
            per_strategy[strategy] = {
                "count": len(b),
                "baseline_mean": round(_mean(b), 3),
                "aligned_mean": round(_mean(a), 3),
            }
    baseline_mean = _mean(baseline_scores)
    aligned_mean = _mean(aligned_scores)
    return {
        "created": datetime.now(timezone.utc).isoformat(),
        "mode": "live" if live else "offline-simulated",
        "agent_deployment": agent_deployment,
        "conversations": len(records),
        "baseline_mean": round(baseline_mean, 3),
        "aligned_mean": round(aligned_mean, 3),
        "lift": round(aligned_mean - baseline_mean, 3),
        "per_strategy": per_strategy,
    }


# ---------------------------------------------------------------------------
# Provenance writers
# ---------------------------------------------------------------------------
def write_capture_artifacts(
    records: list[dict[str, Any]],
    gap_report: dict[str, Any],
    *,
    out_dir: Path = CAPTURE_DIR,
) -> dict[str, str]:
    """Write raw captured transcripts, a manifest, and the gap report.

    These artifacts make the corpus origin auditable (which conversations were
    captured, live or simulated, and how far each baseline reply was from the
    target strategy). They are *not* training files — the training corpus is
    written by :func:`capture_datasets`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    transcripts_path = out_dir / CAPTURED_TRANSCRIPTS_FILE
    with transcripts_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    {
                        "id": record["id"],
                        "strategy": record["strategy"],
                        "scenario": record["scenario"],
                        "captured_messages": record["raw_captured_messages"],
                        "baseline_reply": record["baseline_reply"],
                        "baseline_score": record["baseline_score"],
                        "aligned_reply": record["preferred"],
                        "aligned_score": record["aligned_score"],
                        "captured_live": record["captured_live"],
                    },
                    ensure_ascii=False,
                )
            )
            handle.write("\n")

    manifest = {
        "created": gap_report["created"],
        "mode": gap_report["mode"],
        "agent_deployment": gap_report["agent_deployment"],
        "conversations": gap_report["conversations"],
        "transcripts_file": CAPTURED_TRANSCRIPTS_FILE,
        "gap_report_file": CAPTURE_GAP_REPORT_FILE,
        "strategies": list(STRATEGIES),
    }
    (out_dir / CAPTURE_MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (out_dir / CAPTURE_GAP_REPORT_FILE).write_text(
        json.dumps(gap_report, indent=2), encoding="utf-8"
    )
    logger.info("Wrote capture artifacts to %s", out_dir)
    return {
        "transcripts": str(transcripts_path),
        "manifest": str(out_dir / CAPTURE_MANIFEST_FILE),
        "gap_report": str(out_dir / CAPTURE_GAP_REPORT_FILE),
    }


# ---------------------------------------------------------------------------
# Dataset assembly — drop-in conv_*.jsonl from the captured corpus
# ---------------------------------------------------------------------------
def capture_datasets(
    config: DemoConfig,
    *,
    count: int,
    eval_count: int,
    seed: int = 4242,
    live: bool = False,
    use_llm: bool = False,
    agent_deployment: str | None = None,
    out_dir: Path = DATA_DIR,
    client: Any | None = None,
    concurrency: int = 1,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, Any]:
    """Capture a corpus and write the drop-in ``conv_*.jsonl`` training files.

    Produces train / val / eval splits from *disjoint* capture seeds (so no
    captured conversation leaks across the split), writes the same SFT / DPO /
    RFT / eval files the rest of the pipeline consumes, and writes the capture
    provenance artifacts. Returns a summary with per-file row counts and the gap
    report. In live mode the three splits share one resilient client and fan
    each split's conversations out across ``concurrency`` worker threads.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve a friendly label (base/sft/dpo/rft) to the configured deployment
    # once, up front, so all three splits capture from the same model and a
    # missing deployment fails fast with a clear message.
    resolved_agent_deployment = resolve_agent_deployment(config, agent_deployment)

    # Build one shared, retry-tuned client for live capture so we don't reconnect
    # per split and so every worker inherits the same 429 backoff budget.
    active = client
    if live and active is None:
        active = build_capture_client(config, max_retries=max_retries)

    train_records, gap_report = capture_corpus(
        config, count, seed=seed, live=live, use_llm=use_llm,
        agent_deployment=resolved_agent_deployment, client=active,
        concurrency=concurrency, max_retries=max_retries, label="train",
    )
    val_size = max(1, count // 5)
    val_records, _ = capture_corpus(
        config, val_size, seed=seed + 1, live=live, use_llm=use_llm,
        agent_deployment=resolved_agent_deployment, client=active,
        concurrency=concurrency, max_retries=max_retries, label="val",
    )
    eval_records, _ = capture_corpus(
        config, eval_count, seed=seed + 2, live=live, use_llm=use_llm,
        agent_deployment=resolved_agent_deployment, client=active,
        concurrency=concurrency, max_retries=max_retries, label="eval",
    )

    counts = {
        CONV_SFT_TRAIN_FILE: write_conv_sft_jsonl(train_records, out_dir / CONV_SFT_TRAIN_FILE),
        CONV_SFT_VAL_FILE: write_conv_sft_jsonl(val_records, out_dir / CONV_SFT_VAL_FILE),
        CONV_DPO_TRAIN_FILE: write_conv_dpo_jsonl(train_records, out_dir / CONV_DPO_TRAIN_FILE),
        CONV_DPO_VAL_FILE: write_conv_dpo_jsonl(val_records, out_dir / CONV_DPO_VAL_FILE),
        CONV_RFT_TRAIN_FILE: write_conv_rft_jsonl(train_records, out_dir / CONV_RFT_TRAIN_FILE),
        CONV_RFT_VAL_FILE: write_conv_rft_jsonl(val_records, out_dir / CONV_RFT_VAL_FILE),
        CONV_EVAL_FILE: write_conv_eval_jsonl(eval_records, out_dir / CONV_EVAL_FILE),
    }

    artifacts = write_capture_artifacts(train_records, gap_report)
    return {"counts": counts, "gap_report": gap_report, "artifacts": artifacts}


def format_gap_report(gap_report: dict[str, Any]) -> str:
    """Render the alignment-gap scoreboard for the console."""
    lines = ["", "Captured-corpus strategy gap", "=" * 52]
    lines.append(
        f"mode={gap_report['mode']}  conversations={gap_report['conversations']}  "
        f"agent={gap_report.get('agent_deployment') or '(simulated)'}"
    )
    lines.append("-" * 52)
    lines.append(
        f"{'baseline (captured)':<28}{gap_report['baseline_mean']:>8.3f}"
    )
    lines.append(
        f"{'aligned target':<28}{gap_report['aligned_mean']:>8.3f}"
    )
    lines.append(
        f"{'opportunity (lift)':<28}{gap_report['lift']:>8.3f}"
    )
    lines.append("-" * 52)
    for strategy, stats in gap_report.get("per_strategy", {}).items():
        lines.append(
            f"  {strategy:<26}{stats['baseline_mean']:>8.3f} -> "
            f"{stats['aligned_mean']:.3f}  (n={stats['count']})"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run_capture(args: argparse.Namespace, config: DemoConfig) -> int:
    """Capture handler shared by this module's CLI and the pipeline's ``capture``."""
    if getattr(args, "live", False):
        val_size = max(1, args.count // 5)
        total = args.count + val_size + args.eval_count
        workers = getattr(args, "concurrency", DEFAULT_CONCURRENCY)
        print("\nCapturing LIVE from the deployed agent (consumes tokens)...")
        print(
            f"  splits: train={args.count} + val={val_size} + eval={args.eval_count} "
            f"= {total} conversations across {workers} worker(s)."
        )
        print(
            "  Each conversation makes several sequential Azure calls; "
            "progress prints below as splits complete.\n",
            flush=True,
        )
    else:
        print("\nSimulating captured conversations offline (deterministic)...")

    summary = capture_datasets(
        config,
        count=args.count,
        eval_count=args.eval_count,
        seed=args.seed,
        live=getattr(args, "live", False),
        use_llm=getattr(args, "use_llm", False),
        agent_deployment=getattr(args, "agent_deployment", None),
        concurrency=getattr(args, "concurrency", DEFAULT_CONCURRENCY),
        max_retries=getattr(args, "max_retries", DEFAULT_MAX_RETRIES),
    )

    mode = summary["gap_report"]["mode"]
    print(f"\nCaptured conversation corpus ({mode}) written to {DATA_DIR}\n")
    for name, rows in summary["counts"].items():
        print(f"  {name:<24} {rows:>5} rows")
    print(format_gap_report(summary["gap_report"]))
    print(f"Provenance: {summary['artifacts']['transcripts']}")
    print(f"Gap report: {summary['artifacts']['gap_report']}\n")
    return EXIT_SUCCESS


def add_capture_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the capture flags to ``parser`` (shared with the pipeline CLI)."""
    parser.add_argument("--count", type=int, default=200, help="training conversations to capture")
    parser.add_argument("--eval-count", type=int, default=60, help="held-out eval conversations")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument(
        "--live", action="store_true",
        help="drive the real deployed agent + LLM customer simulator (needs Azure config + tokens)",
    )
    parser.add_argument(
        "--agent-deployment", default=None,
        help=(
            "deployment to capture from: a friendly label (base/sft/dpo/rft, "
            "resolved from .env) or a raw deployment name. Use 'sft'/'dpo'/'rft' "
            "to capture from your OWN deployed fine-tuned agent and close the "
            "loop. Defaults to BASE_DEPLOYMENT_NAME in --live mode."
        ),
    )
    parser.add_argument(
        "--use-llm", action="store_true",
        help="author the aligned target replies with the teacher model (distillation)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=(
            "live capture: number of conversations to run in parallel "
            f"(default {DEFAULT_CONCURRENCY}; 1 = sequential). Raise carefully — "
            "higher values hit deployment TPM limits sooner."
        ),
    )
    parser.add_argument(
        "--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
        help=(
            "live capture: SDK retry budget per call for 429/5xx with backoff "
            f"(default {DEFAULT_MAX_RETRIES}). Raise if you see rate-limit failures."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Standalone CLI: ``python agent_corpus_capture.py [flags]``."""
    parser = argparse.ArgumentParser(
        prog="agent_corpus_capture",
        description=(
            "Capture a deployed agent's multi-turn conversations and turn them "
            "into a drop-in strategy-alignment training corpus (SFT/DPO/RFT/eval)."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    add_capture_arguments(parser)
    parser.set_defaults(func=run_capture)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for the standalone capture CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    config = DemoConfig.from_env()
    try:
        return int(run_capture(args, config))
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        logger.error("%s", exc)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
