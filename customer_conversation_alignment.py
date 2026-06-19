"""Customer task — Conversational-Strategy Alignment, end to end.

This is a **self-contained** end-to-end pipeline for the customer's actual use
case: *aligning conversational strategies from long, multi-turn conversation
transcripts*. Unlike the four-act classifier demo (which trains a model to emit a
single ``{intent, outcome, propensity}`` label), this pipeline trains a model to
**generate the next agent turn** so that it follows a desired conversational
strategy (consultative discovery, value framing, objection reframing,
evidence-backed claims, mutual next steps).

It runs the same Azure differentiator loop the demo sells, but on
generation/alignment data instead of classification data:

* **Synthetic data factory** — multi-turn conversations where the assistant
  turns embody an exemplary strategy, plus a weak/off-strategy alternative for
  preference training. Deterministic and offline by default (no teacher LLM
  required), so ``gen-data`` always produces data; a live teacher hook is
  pluggable.
* **Serverless SFT** — supervised fine-tuning on the full multi-turn
  conversations (the assistant turns are what we want imitated).
* **Serverless DPO** — preference optimization over *preferred vs. weak* final
  agent responses (the most direct lever for "prefer our strategy").
* **Serverless RFT** — reinforcement fine-tuning on an o-series model with a
  model grader that scores strategy adherence of the free-text reply.
* **Offline evaluation** — replays held-out conversations through a deployment
  and scores strategy adherence with a deterministic heuristic.
* **Foundry evaluation** — uploads side-by-side strategy-alignment runs to the
  Azure AI Foundry **Evaluations** tab via ``azure.ai.evaluation.evaluate``.

Data isolation: every file this module writes lives under
``data/conversation_alignment/`` with a ``conv_`` prefix, so it never collides
with the existing ``data/*.jsonl`` classifier corpus.

Import safety (OWASP A06): this module imports with **zero Azure SDKs** present.
Azure SDKs are resolved lazily inside the functions that need them (reusing the
tested helpers in :mod:`finetuning.act2a_serverless_sft` and
:mod:`finetuning.act3a_foundry_eval`). Every secret/endpoint comes from
:class:`DemoConfig` via the environment (OWASP A05) — nothing is hardcoded.

Note: this module deliberately does **not** use ``from __future__ import
annotations``. It defines a promptflow custom evaluator
(:class:`StrategyAlignmentEvaluator`); promptflow introspects real parameter
type hints and rejects stringized (PEP 563) annotations as "complex types."
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

# Allow running both as a module (``python -m finetuning.customer_...``) and
# as a plain script: register the package context when launched as a script so
# the ``from . import ...`` calls resolve.
if __package__ in (None, ""):  # pragma: no cover - script-launch shim
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "finetuning"

from .config import DemoConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------
EXIT_SUCCESS: int = 0
EXIT_FAILURE: int = 1
EXIT_ERROR: int = 2

# ---------------------------------------------------------------------------
# LLM teacher concurrency / rate-limit resilience (gen-data --use-llm)
# ---------------------------------------------------------------------------
#: Conversations are independent, so the teacher path fans them out across a
#: small thread pool. The OpenAI SDK retries 429/5xx with backoff honoring
#: ``Retry-After``, so a raised ``max_retries`` budget absorbs throttling.
DEFAULT_TEACHER_CONCURRENCY: int = 4
DEFAULT_TEACHER_MAX_RETRIES: int = 6
DEFAULT_TEACHER_TIMEOUT: float = 120.0

# ---------------------------------------------------------------------------
# Data layout — isolated namespace so nothing collides with the classifier demo
# ---------------------------------------------------------------------------
#: Dedicated subfolder for the conversation-alignment corpus and artifacts.
DATA_DIR: Path = Path(__file__).resolve().parent / "data" / "conversation_alignment"

CONV_SFT_TRAIN_FILE: str = "conv_sft_train.jsonl"
CONV_SFT_VAL_FILE: str = "conv_sft_val.jsonl"
CONV_DPO_TRAIN_FILE: str = "conv_dpo_train.jsonl"
CONV_DPO_VAL_FILE: str = "conv_dpo_val.jsonl"
CONV_RFT_TRAIN_FILE: str = "conv_rft_train.jsonl"
CONV_RFT_VAL_FILE: str = "conv_rft_val.jsonl"
CONV_EVAL_FILE: str = "conv_eval.jsonl"

CONV_SFT_STATE_FILE: str = "conv_sft_state.json"
CONV_DPO_STATE_FILE: str = "conv_dpo_state.json"
CONV_RFT_STATE_FILE: str = "conv_rft_state.json"

#: Sub-folder for Foundry evaluation datasets/results.
FOUNDRY_DIR: Path = DATA_DIR / "foundry"
FOUNDRY_REPORT_FILE: str = "conv_foundry_eval_report.json"

# ---------------------------------------------------------------------------
# Strategy taxonomy — the conversational strategies we want the model to adopt
# ---------------------------------------------------------------------------
#: The desired strategies. Each id maps to a one-line coaching guideline (used as
#: the system prompt) plus the positive/negative signal markers that both the
#: synthetic generator embeds and the alignment scorer rewards/penalizes.
STRATEGIES: tuple[str, ...] = (
    "consultative_discovery",
    "value_framing",
    "objection_reframe",
    "evidence_backed",
    "mutual_next_step",
)

#: One-line strategy coaching used as the conversation's system prompt.
STRATEGY_GUIDELINE: dict[str, str] = {
    "consultative_discovery": (
        "You are a consultative sales strategist. Before pitching, ask focused "
        "diagnostic questions to understand the prospect's situation and goals."
    ),
    "value_framing": (
        "You are a value-led sales strategist. Always connect any capability to "
        "the prospect's stated business outcome — never list features in a vacuum."
    ),
    "objection_reframe": (
        "You are a calm, credible sales strategist. Acknowledge objections, then "
        "reframe them around value without pressure or dismissiveness."
    ),
    "evidence_backed": (
        "You are an evidence-led sales strategist. Support claims with concrete "
        "proof — a metric, a comparable customer, or a quantified result."
    ),
    "mutual_next_step": (
        "You are a momentum-oriented sales strategist. Close every exchange by "
        "proposing a specific, low-friction, mutually agreed next step."
    ),
}

#: Positive marker groups per strategy. A response that touches more groups scores
#: higher. Each inner tuple is a synonym group (any one counts as a hit).
STRATEGY_MARKERS: dict[str, tuple[tuple[str, ...], ...]] = {
    "consultative_discovery": (
        ("what ", "how ", "which ", "where "),
        ("help me understand", "tell me more", "walk me through", "could you share"),
        ("?",),
    ),
    "value_framing": (
        ("so you can", "so that you", "which means", "that translates to"),
        ("outcome", "impact", "result", "roi", "bottom line"),
        ("you mentioned", "your goal", "your team", "your priority"),
    ),
    "objection_reframe": (
        ("i understand", "that's fair", "i hear you", "good point"),
        ("the way i'd look at it", "another way to weigh", "reframe", "consider it against"),
        ("worth weighing", "trade-off", "balance that with"),
    ),
    "evidence_backed": (
        ("for example", "for instance", "case in point", "a customer like"),
        ("reduced", "increased", "improved", "cut", "grew"),
        ("%", "percent", "x faster", "in weeks", "measured"),
    ),
    "mutual_next_step": (
        ("next step", "would you be open to", "shall we", "let's"),
        ("schedule", "set up", "book", "by ", "this week", "friday"),
        ("propose we", "i'll send", "we agree", "does that work"),
    ),
}

#: Anti-pattern markers per strategy — their presence multiplies the score down.
STRATEGY_ANTIMARKERS: dict[str, tuple[str, ...]] = {
    "consultative_discovery": ("buy now", "sign today", "limited-time", "just trust me"),
    "value_framing": ("spec sheet", "technically speaking", "feature list", "as i said"),
    "objection_reframe": ("you're wrong", "that's not true", "you have to", "no excuse"),
    "evidence_backed": ("trust me", "i think", "probably", "everyone knows"),
    "mutual_next_step": ("whenever you want", "no rush", "circle back sometime", "ping me eventually"),
}

# ---------------------------------------------------------------------------
# Scenario fragments for the synthetic conversation factory
# ---------------------------------------------------------------------------
_INDUSTRIES: tuple[str, ...] = (
    "manufacturing", "healthcare", "financial services", "retail",
    "logistics", "B2B SaaS", "energy", "telecom",
)
_PERSONAS: tuple[str, ...] = (
    "VP of Operations", "IT Director", "Head of Customer Success",
    "Procurement Lead", "Plant Manager", "Director of Analytics",
)
_TOPICS: tuple[str, ...] = (
    "reducing manual review time", "improving forecast accuracy",
    "consolidating tooling", "cutting onboarding time",
    "increasing first-contact resolution", "tightening data governance",
)
_OBJECTIONS: tuple[str, ...] = (
    "we already have a tool that mostly works",
    "the budget for this cycle is basically locked",
    "my team is worried about another migration",
    "leadership wants to see proof before committing",
    "we tried something like this before and it stalled",
)

# RFT requires an o-series base; reuse the demo's pinned model + fallback tier.
RFT_BASE_MODEL: str = "o4-mini-2025-04-16"
RFT_TRAINING_TYPE: str = "GlobalStandard"

# Azure's RFT model-grader accepts only these base MODEL NAMES (not deployment
# aliases). Passing a deployment name (e.g. "chat41mini") yields a 400
# invalidPayload. Validated up front so the failure is actionable.
SUPPORTED_RFT_GRADER_MODELS: frozenset[str] = frozenset({
    "gpt-4.1", "gpt-4.1-2025-04-14",
    "gpt-4.1-mini", "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano", "gpt-4.1-nano-2025-04-14",
    "gpt-4o", "gpt-4o-2024-08-06",
    "o3-mini", "o3-mini-2025-01-31",
})

#: Default RFT grader model when GRADER_MODEL is unset (a supported base model).
DEFAULT_RFT_GRADER_MODEL: str = "gpt-4.1-mini"

# Friendly arm labels for the Foundry scoreboard (presentation order).
DEFAULT_MODELS: tuple[str, ...] = ("base", "sft", "dpo", "rft")
MODEL_DISPLAY_NAME: dict[str, str] = {
    "base": "Base (un-tuned)",
    "sft": "Supervised FT",
    "dpo": "Preference (DPO)",
    "rft": "Reinforcement (RFT)",
}


# ---------------------------------------------------------------------------
# Strategy alignment scorer (deterministic, dependency-free)
# ---------------------------------------------------------------------------
def strategy_alignment_score(text: str, strategy: str) -> float:
    """Score how well ``text`` adheres to ``strategy`` in ``[0, 1]``.

    Counts how many positive marker groups the response touches (each group is a
    set of synonyms; any one counts once), divided by the number of groups, then
    multiplies the score down when an anti-pattern marker is present. Purely
    lexical and deterministic, so offline evaluation needs no judge model and is
    reproducible. Fails safe to ``0.0`` on empty input or unknown strategy.
    """
    if not text or strategy not in STRATEGY_MARKERS:
        return 0.0
    lowered = text.lower()
    groups = STRATEGY_MARKERS[strategy]
    hits = sum(1 for group in groups if any(token in lowered for token in group))
    score = hits / len(groups) if groups else 0.0
    if any(anti in lowered for anti in STRATEGY_ANTIMARKERS.get(strategy, ())):
        score *= 0.4
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Synthetic multi-turn conversation factory
# ---------------------------------------------------------------------------
def _good_turn(strategy: str, scenario: dict[str, str], rng: random.Random) -> str:
    """Render a strategy-aligned agent turn that embeds the strategy's markers."""
    industry = scenario["industry"]
    persona = scenario["persona"]
    topic = scenario["topic"]
    if strategy == "consultative_discovery":
        return (
            f"Before I suggest anything, help me understand your situation — "
            f"how are you handling {topic} today, and what's the part that "
            f"frustrates the {persona} most?"
        )
    if strategy == "value_framing":
        return (
            f"You mentioned {topic} is the priority, so let's tie this to that "
            f"outcome: the capability matters because it means your {industry} "
            f"team spends less time on rework — which translates to faster cycles."
        )
    if strategy == "objection_reframe":
        return (
            f"That's fair, and I hear you. The way I'd look at it: weigh the "
            f"effort against the cost of leaving {topic} unsolved — that trade-off "
            f"is usually where the value shows up for a {persona}."
        )
    if strategy == "evidence_backed":
        return (
            f"For example, a {industry} customer like yours reduced manual review "
            f"by 40% in eight weeks after focusing on {topic} — that's a measured "
            f"result, not a promise."
        )
    # mutual_next_step
    return (
        f"Here's a concrete next step: would you be open to a 30-minute working "
        f"session this week where we map {topic} to your numbers? I'll send two "
        f"times by Friday — does that work?"
    )


def _weak_turn(strategy: str, scenario: dict[str, str], rng: random.Random) -> str:
    """Render an off-strategy / generic agent turn that trips the anti-markers."""
    topic = scenario["topic"]
    weak_pool = {
        "consultative_discovery": (
            f"Our platform is the best on the market — you should buy now while "
            f"there's a limited-time discount. Just trust me, it handles {topic}."
        ),
        "value_framing": (
            f"Technically speaking, here's the feature list: SSO, dashboards, "
            f"APIs, and connectors. As I said, the spec sheet covers {topic}."
        ),
        "objection_reframe": (
            f"You're wrong about that — that's not true. You have to move on this "
            f"now; there's no excuse to keep struggling with {topic}."
        ),
        "evidence_backed": (
            f"Trust me, I think it probably works great. Everyone knows tools like "
            f"ours fix {topic}."
        ),
        "mutual_next_step": (
            f"Sounds good, no rush — ping me eventually and we can circle back "
            f"sometime about {topic} whenever you want."
        ),
    }
    return weak_pool[strategy]


def _customer_turns(scenario: dict[str, str], rng: random.Random) -> tuple[str, str, str]:
    """Render the three customer turns that frame a conversation."""
    persona = scenario["persona"]
    industry = scenario["industry"]
    topic = scenario["topic"]
    objection = scenario["objection"]
    opening = (
        f"Hi — I'm the {persona} at a {industry} company. We're looking at "
        f"{topic} and trying to figure out if it's worth prioritizing this quarter."
    )
    middle = (
        f"Honestly, {objection}. So I'm not sure where this fits."
    )
    closing = (
        f"Okay, that's helpful. Where would you suggest we go from here?"
    )
    return opening, middle, closing


def _assemble_conversation_record(
    index: int,
    strategy: str,
    scenario: dict[str, str],
    rng: random.Random,
    override: dict[str, Any],
) -> dict[str, Any]:
    """Build one conversation record from a scenario + optional teacher override.

    Any turn the ``override`` omits falls back to the deterministic template
    (``rng`` drives those fallbacks). Kept separate from :func:`generate_conversations`
    so the sequential and parallel teacher paths share identical assembly.
    """
    tmpl_opening, tmpl_middle, tmpl_closing = _customer_turns(scenario, rng)
    opening = override.get("opening") or tmpl_opening
    middle = override.get("middle") or tmpl_middle
    closing = override.get("closing") or tmpl_closing
    good_1 = override.get("good_1") or _good_turn(strategy, scenario, rng)
    good_2 = override.get("good_2") or _good_turn(strategy, scenario, rng)
    preferred = override.get("preferred") or _good_turn(strategy, scenario, rng)
    non_preferred = override.get("non_preferred") or _weak_turn(strategy, scenario, rng)

    system = {"role": "system", "content": STRATEGY_GUIDELINE[strategy]}
    # Full exemplary conversation (used by SFT).
    messages = [
        system,
        {"role": "user", "content": opening},
        {"role": "assistant", "content": good_1},
        {"role": "user", "content": middle},
        {"role": "assistant", "content": good_2},
        {"role": "user", "content": closing},
        {"role": "assistant", "content": preferred},
    ]
    # Context that ends on the final user turn (used by DPO/RFT/eval).
    context_messages = messages[:-1]
    transcript = _render_transcript(context_messages)

    return {
        "id": f"conv-{index + 1:06d}",
        "strategy": strategy,
        "scenario": scenario,
        "messages": messages,
        "context_messages": context_messages,
        "preferred": preferred,
        "non_preferred": non_preferred,
        "transcript": transcript,
    }


def generate_conversations(
    count: int,
    *,
    seed: int = 1337,
    generate_fn: Optional[Callable[[dict[str, str], random.Random], dict[str, Any]]] = None,
    concurrency: int = 1,
) -> list[dict[str, Any]]:
    """Generate ``count`` synthetic multi-turn conversation records.

    Each record carries the full strategy-aligned conversation plus the pieces
    every training format needs::

        {
          "id": "conv-000001",
          "strategy": "value_framing",
          "scenario": {...},
          "messages":        [system, user, assistant, user, assistant, user, assistant],
          "context_messages":[system, user, assistant, user, assistant, user],  # ends on user
          "preferred":     "<exemplary final agent reply>",
          "non_preferred": "<weak/off-strategy final agent reply>",
          "transcript":    "<rendered context as readable text>",
        }

    Deterministic given ``seed``. Pass ``generate_fn`` to substitute a live
    teacher-LLM generator (it receives the sampled ``scenario`` and an RNG and
    may return any of ``opening``/``middle``/``closing`` (customer turns),
    ``good_1``/``good_2`` (mid agent turns), ``preferred``/``non_preferred``
    (final agent replies); any key it omits falls back to the deterministic
    template). By default the template generator runs fully offline so
    ``gen-data`` never needs network. See :func:`build_llm_teacher`.

    ``concurrency`` > 1 (teacher path only) fans the independent ``generate_fn``
    calls out across a thread pool; each conversation gets its own RNG so the
    work stays thread-safe. The offline template path ignores it (no I/O) and
    runs the original deterministic sequential loop.
    """
    # Offline / sequential path — unchanged, deterministic (tests depend on this).
    if generate_fn is None or concurrency <= 1:
        rng = random.Random(seed)
        records: list[dict[str, Any]] = []
        for index in range(count):
            strategy = STRATEGIES[index % len(STRATEGIES)]
            scenario = {
                "industry": rng.choice(_INDUSTRIES),
                "persona": rng.choice(_PERSONAS),
                "topic": rng.choice(_TOPICS),
                "objection": rng.choice(_OBJECTIONS),
                "strategy": strategy,
            }
            override = generate_fn(scenario, rng) if generate_fn is not None else {}
            records.append(
                _assemble_conversation_record(index, strategy, scenario, rng, override)
            )
        logger.info("Generated %d synthetic conversation(s)", len(records))
        return records

    # Parallel teacher path. Sample scenarios deterministically (cheap, no I/O),
    # give each conversation its own RNG, run the teacher calls concurrently, then
    # assemble in index order.
    master = random.Random(seed)
    work: list[tuple[int, str, dict[str, str], random.Random]] = []
    for index in range(count):
        strategy = STRATEGIES[index % len(STRATEGIES)]
        scenario = {
            "industry": master.choice(_INDUSTRIES),
            "persona": master.choice(_PERSONAS),
            "topic": master.choice(_TOPICS),
            "objection": master.choice(_OBJECTIONS),
            "strategy": strategy,
        }
        work.append((index, strategy, scenario, random.Random(seed * 1_000_003 + index)))

    overrides: list[dict[str, Any]] = [{} for _ in range(count)]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(generate_fn, scenario, conv_rng): index
            for index, _strategy, scenario, conv_rng in work
        }
        for future in as_completed(futures):
            overrides[futures[future]] = future.result() or {}

    records = [
        _assemble_conversation_record(index, strategy, scenario, conv_rng, overrides[index])
        for index, strategy, scenario, conv_rng in work
    ]
    logger.info("Generated %d synthetic conversation(s)", len(records))
    return records


def _render_transcript(messages: list[dict[str, str]]) -> str:
    """Render a messages array as a readable multi-turn transcript string."""
    lines: list[str] = []
    for message in messages:
        role = message.get("role", "")
        if role == "system":
            continue
        speaker = "Customer" if role == "user" else "Agent"
        lines.append(f"{speaker}: {message.get('content', '')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optional LLM teacher (diverse generation via Structured Outputs)
# ---------------------------------------------------------------------------
#: Strict JSON schema the teacher returns for one conversation. Every turn is a
#: separate field so the generator stays controllable and the rows map cleanly
#: onto the SFT / DPO / RFT formats.
_TEACHER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "customer_opening": {"type": "string"},
        "agent_turn_1": {"type": "string"},
        "customer_middle": {"type": "string"},
        "agent_turn_2": {"type": "string"},
        "customer_closing": {"type": "string"},
        "agent_preferred": {"type": "string"},
        "agent_weak": {"type": "string"},
    },
    "required": [
        "customer_opening", "agent_turn_1", "customer_middle", "agent_turn_2",
        "customer_closing", "agent_preferred", "agent_weak",
    ],
    "additionalProperties": False,
}

#: System prompt framing the teacher as a sales-conversation author. PII-free,
#: placeholder-only, to keep the synthetic corpus safe to fine-tune on.
_TEACHER_SYSTEM_PROMPT: str = (
    "You author short, realistic, fully synthetic B2B sales conversations to "
    "train a model on conversational strategy. Use placeholders only ([REP], "
    "[PROSPECT], Acme Corp); never real names, companies, emails, or numbers. "
    "The 'agent_preferred' and both 'agent_turn' replies must clearly apply the "
    "target strategy; 'agent_weak' must be a generic, off-strategy reply a "
    "mediocre rep would give. Vary phrasing naturally across conversations."
)


def _teacher_user_prompt(scenario: dict[str, str]) -> str:
    """Render the per-conversation instruction for the teacher LLM."""
    strategy = scenario["strategy"]
    return (
        "Write ONE sales conversation as JSON.\n"
        f"Industry: {scenario['industry']}. Prospect persona: {scenario['persona']}.\n"
        f"Topic the prospect cares about: {scenario['topic']}.\n"
        f"An objection the prospect raises: {scenario['objection']}.\n"
        f"Target conversational strategy: {strategy} — "
        f"{STRATEGY_GUIDELINE[strategy]}\n"
        "The conversation flows: customer_opening -> agent_turn_1 -> "
        "customer_middle (raises the objection) -> agent_turn_2 -> "
        "customer_closing (asks where to go next) -> agent_preferred (the ideal "
        "final reply) and agent_weak (a poor alternative final reply)."
    )


def _make_teacher_client(
    config: DemoConfig, *, max_retries: int = DEFAULT_TEACHER_MAX_RETRIES
) -> Any:
    """Build a retry-tuned Azure OpenAI client for teacher generation or raise.

    Reuses the tested data-plane client builder; the teacher model name is
    resolved from ``TEACHER_MODEL`` (falling back to ``BASE_DEPLOYMENT_NAME``).
    ``with_options`` raises the SDK's automatic 429/5xx retry budget and request
    timeout so concurrent generation absorbs throttling without aborting a run.
    """
    from . import act2a_serverless_sft as sft  # noqa: PLC0415

    client = sft.build_client(config)
    try:
        return client.with_options(
            max_retries=max_retries, timeout=DEFAULT_TEACHER_TIMEOUT
        )
    except AttributeError:  # pragma: no cover - test doubles lack with_options
        return client


def build_llm_teacher(
    config: DemoConfig,
    *,
    client: Any | None = None,
    temperature: float = 0.9,
    max_retries: int = DEFAULT_TEACHER_MAX_RETRIES,
) -> Callable[[dict[str, str], random.Random], dict[str, Any]]:
    """Return a ``generate_fn`` that authors each conversation with a teacher LLM.

    The returned closure issues one Structured-Outputs call per conversation
    against ``TEACHER_MODEL`` (or ``BASE_DEPLOYMENT_NAME``), producing diverse,
    paraphrased customer + agent turns instead of the deterministic templates.
    On any per-row failure it returns ``{}`` so :func:`generate_conversations`
    falls back to the template for that row (OWASP A04 — fail safe), keeping a
    bulk generation run from aborting on a single transient error.

    Note: this trades determinism and tokens for realism. Prefer it when
    preparing a credible corpus; keep the template default for an offline,
    quota-proof live demo.
    """
    teacher_model = config.teacher_model or config.base_deployment_name
    if not teacher_model:
        raise ValueError(
            "LLM teacher needs a model: set TEACHER_MODEL or BASE_DEPLOYMENT_NAME."
        )
    active = client if client is not None else _make_teacher_client(config, max_retries=max_retries)
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "sales_conversation",
            "schema": _TEACHER_SCHEMA,
            "strict": True,
        },
    }

    def _teacher(scenario: dict[str, str], rng: random.Random) -> dict[str, Any]:
        try:
            completion = active.chat.completions.create(
                model=teacher_model,
                temperature=temperature,
                response_format=response_format,
                messages=[
                    {"role": "system", "content": _TEACHER_SYSTEM_PROMPT},
                    {"role": "user", "content": _teacher_user_prompt(scenario)},
                ],
            )
            data = json.loads(completion.choices[0].message.content)
        except Exception as exc:  # noqa: BLE001 - per-row fail safe
            logger.warning("[teacher] generation failed (%s); using template row", exc)
            return {}
        return {
            "opening": data.get("customer_opening"),
            "middle": data.get("customer_middle"),
            "closing": data.get("customer_closing"),
            "good_1": data.get("agent_turn_1"),
            "good_2": data.get("agent_turn_2"),
            "preferred": data.get("agent_preferred"),
            "non_preferred": data.get("agent_weak"),
        }

    return _teacher


# ---------------------------------------------------------------------------
# JSONL writers (UTF-8 with BOM for training files; plain UTF-8 for eval data)
# ---------------------------------------------------------------------------
def _write_jsonl(rows: list[dict[str, Any]], path: str | Path, *, bom: bool = True) -> int:
    """Write ``rows`` as JSONL; return the count. ``bom`` selects utf-8-sig."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    encoding = "utf-8-sig" if bom else "utf-8"
    with out.open("w", encoding=encoding, newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    logger.info("Wrote %d row(s) to %s", len(rows), out)
    return len(rows)


def write_conv_sft_jsonl(records: list[dict[str, Any]], path: str | Path) -> int:
    """Write SFT rows: the full multi-turn conversation as ``{"messages": [...]}``.

    The assistant turns are the exemplary strategy we want the model to imitate,
    so SFT learns the strategy at every agent turn in the conversation.
    """
    rows = [{"messages": record["messages"]} for record in records]
    return _write_jsonl(rows, path)


def write_conv_dpo_jsonl(records: list[dict[str, Any]], path: str | Path) -> int:
    """Write preference (DPO) rows over the *final agent response*.

    Each row pairs the same conversation context with a ``preferred_output``
    (strategy-aligned reply) and a ``non_preferred_output`` (weak/off-strategy
    reply), in Azure's DPO format. This is the most direct lever for "prefer our
    conversational strategy."
    """
    rows = [
        {
            "input": {"messages": record["context_messages"]},
            "preferred_output": [
                {"role": "assistant", "content": record["preferred"]}
            ],
            "non_preferred_output": [
                {"role": "assistant", "content": record["non_preferred"]}
            ],
        }
        for record in records
    ]
    return _write_jsonl(rows, path)


def write_conv_rft_jsonl(records: list[dict[str, Any]], path: str | Path) -> int:
    """Write Reinforcement Fine-Tuning (RFT) rows for an o-series base model.

    Azure RFT requires the instruction turn to use the ``developer`` role (not
    ``system``: o-series models reject the chat-SFT format) and the **final
    message to have the** ``user`` **role**. The target ``strategy`` is carried
    as a top-level reference field so the server-side grader can read it as
    ``{{item.strategy}}``. The free-text reply is graded directly (no JSON
    ``response_format``), since here we reward generation quality, not a label.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        # Re-role the system turn to developer; keep the rest of the context
        # (which already ends on a user turn).
        context = record["context_messages"]
        messages: list[dict[str, str]] = []
        for message in context:
            if message.get("role") == "system":
                messages.append({"role": "developer", "content": message["content"]})
            else:
                messages.append({"role": message["role"], "content": message["content"]})
        rows.append({"messages": messages, "strategy": record["strategy"]})
    return _write_jsonl(rows, path)


def write_conv_eval_jsonl(records: list[dict[str, Any]], path: str | Path) -> int:
    """Write held-out eval rows: context + ground-truth reply + target strategy."""
    rows = [
        {
            "id": record["id"],
            "strategy": record["strategy"],
            "query": record["transcript"],
            "context_messages": record["context_messages"],
            "ground_truth": record["preferred"],
        }
        for record in records
    ]
    return _write_jsonl(rows, path, bom=False)


# ---------------------------------------------------------------------------
# Generate-all entry point
# ---------------------------------------------------------------------------
def generate_all_datasets(
    *,
    count: int,
    eval_count: int,
    seed: int = 1337,
    out_dir: Path = DATA_DIR,
    generate_fn: Optional[Callable[[dict[str, str], random.Random], dict[str, Any]]] = None,
    concurrency: int = 1,
) -> dict[str, int]:
    """Generate the full conversation-alignment corpus into the isolated folder.

    Writes SFT/DPO/RFT train + validation files, plus a held-out eval set built
    from a disjoint seed so no conversation leaks between train and eval.
    Returns a mapping of artifact name -> row count. Pass ``generate_fn`` (e.g.
    from :func:`build_llm_teacher`) to author rows with a teacher LLM instead of
    the deterministic templates. ``concurrency`` > 1 fans the teacher calls out
    across a thread pool (ignored on the offline template path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Train/val share the strategy taxonomy but are split disjointly; eval uses a
    # different seed so whole conversations never leak across the boundary.
    train_records = generate_conversations(
        count, seed=seed, generate_fn=generate_fn, concurrency=concurrency
    )
    val_size = max(1, count // 5)
    val_records = generate_conversations(
        val_size, seed=seed + 1, generate_fn=generate_fn, concurrency=concurrency
    )
    eval_records = generate_conversations(
        eval_count, seed=seed + 2, generate_fn=generate_fn, concurrency=concurrency
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
    return counts


# ---------------------------------------------------------------------------
# RFT model grader (free-text strategy adherence)
# ---------------------------------------------------------------------------
def conv_strategy_grader_config(grader_model: str) -> dict[str, Any]:
    """Build the Azure RFT ``score_model`` grader for strategy adherence.

    The grader asks ``grader_model`` to rate, in ``[0, 1]``, how well the model's
    free-text reply applies the target strategy. Template variables use only the
    ``sample`` / ``item`` namespaces with no inner whitespace, per Azure's RFT
    template contract.
    """
    if not grader_model:
        raise ValueError("RFT requires a grader_model deployment name.")
    if grader_model not in SUPPORTED_RFT_GRADER_MODELS:
        raise ValueError(
            f"RFT model grader {grader_model!r} is not a supported base model. "
            f"Azure's model grader needs a MODEL NAME (not a deployment alias) "
            f"from: {', '.join(sorted(SUPPORTED_RFT_GRADER_MODELS))}. "
            f"Set GRADER_MODEL in .env (e.g. {DEFAULT_RFT_GRADER_MODEL}) or pass "
            f"--grader-model."
        )
    return {
        "type": "score_model",
        "name": "strategy_alignment",
        "model": grader_model,
        "input": [
            {
                "role": "system",
                "content": (
                    "You grade a sales agent's reply for adherence to a target "
                    "conversational strategy. Return a single number in [0,1]: "
                    "1.0 = clearly applies the strategy, 0.0 = ignores or "
                    "contradicts it. Reward diagnostic questions, value framing, "
                    "calm objection reframes, concrete evidence, or a specific "
                    "mutual next step, as the target requires."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Target strategy: {{item.strategy}}.\n"
                    "Agent reply: {{sample.output_text}}\n"
                    "Score (0-1):"
                ),
            },
        ],
        "range": [0, 1],
        "pass_threshold": 0.5,
    }


# ---------------------------------------------------------------------------
# Fine-tuning lifecycle (reuses the tested act2a serverless helpers)
# ---------------------------------------------------------------------------
def _training_files(method: str, out_dir: Path) -> tuple[Path, Path]:
    """Return the ``(train, val)`` JSONL paths for a serverless method."""
    if method == "supervised":
        return out_dir / CONV_SFT_TRAIN_FILE, out_dir / CONV_SFT_VAL_FILE
    if method == "dpo":
        return out_dir / CONV_DPO_TRAIN_FILE, out_dir / CONV_DPO_VAL_FILE
    if method == "reinforcement":
        return out_dir / CONV_RFT_TRAIN_FILE, out_dir / CONV_RFT_VAL_FILE
    raise ValueError(f"Unknown method {method!r}.")


def _state_file(method: str, out_dir: Path) -> Path:
    """Return the state JSON path a finished finetune writes for a method.

    Known methods map to their canonical file; any other label (e.g. ``resume``)
    falls back to ``conv_<label>_state.json`` so re-attached runs still persist.
    """
    known = {
        "supervised": CONV_SFT_STATE_FILE,
        "dpo": CONV_DPO_STATE_FILE,
        "reinforcement": CONV_RFT_STATE_FILE,
    }
    return out_dir / known.get(method, f"conv_{method}_state.json")


def create_rft_alignment_job(
    client: Any,
    training_file_id: str,
    validation_file_id: str,
    *,
    grader_model: str,
    model: str = RFT_BASE_MODEL,
    suffix: str = "conv-rft",
    n_epochs: int = 2,
    seed: int = 105,
    training_type: str = RFT_TRAINING_TYPE,
    extra_hyperparameters: dict[str, Any] | None = None,
) -> str:
    """Create a generation-aligned RFT job and return its job id.

    Differs from the classifier RFT path in :mod:`act2a_serverless_sft` in that
    it grades the **free-text reply** with a model grader and omits the JSON
    ``response_format`` (there is no structured label to constrain). RFT requires
    an o-series base model and rejects the developer training tier, so this
    defaults to ``o4-mini`` on ``GlobalStandard``.

    ``extra_hyperparameters`` (e.g. ``batch_size``, ``learning_rate_multiplier``,
    ``eval_interval``, ``eval_samples``, ``reasoning_effort``,
    ``compute_multiplier``) are merged into the RFT ``hyperparameters`` block;
    ``None`` values are dropped so the server default applies.
    """
    from .act2a_serverless_sft import _sanitize_suffix  # noqa: PLC0415

    extra = {k: v for k, v in (extra_hyperparameters or {}).items() if v is not None}
    method_payload = {
        "type": "reinforcement",
        "reinforcement": {
            "hyperparameters": {"n_epochs": n_epochs, **extra},
            "grader": conv_strategy_grader_config(grader_model),
        },
    }
    job = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model=model,
        suffix=_sanitize_suffix(suffix),
        seed=seed,
        method=method_payload,
        extra_body={"trainingType": training_type},
    )
    logger.info("Created RFT alignment job %s (model=%s)", job.id, model)
    return job.id


def run_finetune(
    config: DemoConfig,
    method: str,
    *,
    out_dir: Path = DATA_DIR,
    max_polls: int | None = None,
    grader_model: str | None = None,
    n_epochs: int | None = None,
    beta: float | None = None,
    batch_size: int | None = None,
    learning_rate_multiplier: float | None = None,
    eval_interval: int | None = None,
    eval_samples: int | None = None,
    reasoning_effort: str | None = None,
    compute_multiplier: float | None = None,
    deploy: bool = False,
    deployment_name: str | None = None,
    sku: str = "developer",
) -> dict[str, Any]:
    """Run one serverless fine-tune (SFT / DPO / RFT) end to end.

    Uploads the method's train/val JSONL, creates the job, polls to terminal,
    resolves the resulting fine-tuned model id (preferring the lowest-validation-
    loss checkpoint), and writes a state JSON. Optionally deploys the model via
    the ARM control-plane when ``deploy`` is set.

    ``n_epochs`` overrides the per-method epoch default; ``beta`` overrides the
    DPO preference strength (ignored for SFT/RFT). ``batch_size`` and
    ``learning_rate_multiplier`` apply to every method. ``eval_interval``,
    ``eval_samples``, ``reasoning_effort`` (low/medium/high), and
    ``compute_multiplier`` apply to RFT only. Any value left ``None`` falls back
    to the Azure/SDK default (the key is simply omitted from the request).

    Returns a summary dict with ``method``, ``job_id``, ``status``, ``model_id``,
    and (when deployed) ``deployment``.
    """
    from . import act2a_serverless_sft as sft  # noqa: PLC0415

    train_path, val_path = _training_files(method, out_dir)
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Missing {method} data ({train_path.name}/{val_path.name}); run "
            f"`gen-data` first."
        )

    client = sft.build_client(config)
    train_id, val_id = sft.upload_files(client, train_path, val_path)
    sft.wait_for_files(client, (train_id, val_id))

    # Hyperparameters shared by every method (omitted when None -> server default).
    common_extra: dict[str, Any] = {
        "batch_size": batch_size,
        "learning_rate_multiplier": learning_rate_multiplier,
    }

    if method == "supervised":
        sft_kwargs: dict[str, Any] = {
            "method": sft.METHOD_SUPERVISED,
            "suffix": "conv-align",
            "training_type": sft.training_type_for_tier(config.deployment_tier),
            "extra_hyperparameters": common_extra,
        }
        if n_epochs is not None:
            sft_kwargs["n_epochs"] = n_epochs
        job_id = sft.create_job(client, train_id, val_id, **sft_kwargs)
    elif method == "dpo":
        dpo_kwargs: dict[str, Any] = {
            "method": sft.METHOD_DPO,
            "suffix": "conv-dpo",
            "training_type": sft.training_type_for_tier(config.deployment_tier),
            "extra_hyperparameters": common_extra,
        }
        if n_epochs is not None:
            dpo_kwargs["n_epochs"] = n_epochs
        if beta is not None:
            dpo_kwargs["beta"] = beta
        job_id = sft.create_job(client, train_id, val_id, **dpo_kwargs)
    elif method == "reinforcement":
        resolved_grader = grader_model or config.grader_model or DEFAULT_RFT_GRADER_MODEL
        if not resolved_grader:
            raise ValueError(
                "RFT needs a grader model; set GRADER_MODEL to a supported base "
                "model name (e.g. gpt-4.1-mini), or pass --grader-model."
            )
        rft_kwargs: dict[str, Any] = {
            "grader_model": resolved_grader,
            "extra_hyperparameters": {
                **common_extra,
                "eval_interval": eval_interval,
                "eval_samples": eval_samples,
                "reasoning_effort": reasoning_effort,
                "compute_multiplier": compute_multiplier,
            },
        }
        if n_epochs is not None:
            rft_kwargs["n_epochs"] = n_epochs
        job_id = create_rft_alignment_job(client, train_id, val_id, **rft_kwargs)
    else:
        raise ValueError(f"Unknown method {method!r}.")

    job = sft.poll_job(client, job_id, max_polls=max_polls, heartbeat=True)
    status = getattr(job, "status", None)
    model_id = None
    if status == "succeeded":
        model_id = sft.best_checkpoint_model_id(client, job_id) or sft.final_model_id(
            client, job_id
        )

    summary: dict[str, Any] = {
        "method": method,
        "job_id": job_id,
        "status": status,
        "model_id": model_id,
        "created": datetime.now(timezone.utc).isoformat(),
    }

    if deploy and model_id and deployment_name:
        response = sft.deploy_finetuned(config, model_id, deployment_name, sku)
        summary["deployment"] = deployment_name
        summary["deploy_response_status"] = response.get("properties", {}).get(
            "provisioningState"
        ) if isinstance(response, dict) else None

    state_path = _state_file(method, out_dir)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s job state -> %s", method, state_path)
    return summary


def list_finetune_jobs(config: DemoConfig, *, limit: int = 10) -> list[Any]:
    """List recent fine-tuning jobs so a user can recover a job id.

    Useful after a dropped connection: the job keeps training server-side, but
    the original CLI process no longer holds its id. This surfaces the recent
    jobs (id / status / fine-tuned model) to pass to :func:`run_finetune_resume`.
    """
    from . import act2a_serverless_sft as sft  # noqa: PLC0415

    client = sft.build_client(config)
    response = client.fine_tuning.jobs.list()
    data = getattr(response, "data", response)
    return list(data)[:limit]


def run_finetune_resume(
    config: DemoConfig,
    job_id: str,
    *,
    method: str = "resume",
    out_dir: Path = DATA_DIR,
    max_polls: int | None = None,
    deploy: bool = False,
    deployment_name: str | None = None,
    sku: str = "developer",
) -> dict[str, Any]:
    """Re-attach to an already-submitted fine-tune, poll it, and optionally deploy.

    Recovers from a dropped connection: a fine-tune keeps training in Azure even
    after the original CLI process dies, so re-attaching by ``job_id`` lets the
    demo finish (wait for terminal state, resolve the best checkpoint, deploy)
    without re-submitting the job or re-consuming training tokens.

    Returns a summary dict shaped like :func:`run_finetune` (``method``,
    ``job_id``, ``status``, ``model_id``, and ``deployment`` when deployed).
    """
    from . import act2a_serverless_sft as sft  # noqa: PLC0415

    client = sft.build_client(config)
    job = sft.poll_job(client, job_id, max_polls=max_polls, heartbeat=True)
    status = getattr(job, "status", None)
    model_id = None
    if status == "succeeded":
        model_id = sft.best_checkpoint_model_id(client, job_id) or sft.final_model_id(
            client, job_id
        )

    summary: dict[str, Any] = {
        "method": method,
        "job_id": job_id,
        "status": status,
        "model_id": model_id,
        "created": datetime.now(timezone.utc).isoformat(),
    }

    if deploy and model_id and deployment_name:
        response = sft.deploy_finetuned(config, model_id, deployment_name, sku)
        summary["deployment"] = deployment_name
        summary["deploy_response_status"] = response.get("properties", {}).get(
            "provisioningState"
        ) if isinstance(response, dict) else None

    state_path = _state_file(method, out_dir)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s job state -> %s", method, state_path)
    return summary


# ---------------------------------------------------------------------------
# Inference — generate the next agent turn from a full multi-turn context
# ---------------------------------------------------------------------------
def _message_text(message: Any) -> str:
    """Coerce a chat-completion message's content into a plain string."""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                text_attr = getattr(item, "text", None)
                if isinstance(text_attr, str):
                    parts.append(text_attr)
        if parts:
            return "\n".join(parts)
    refusal = getattr(message, "refusal", None)
    return refusal if isinstance(refusal, str) else ""


def generate_agent_turn(
    config: DemoConfig,
    deployment_name: str,
    context_messages: list[dict[str, str]],
    *,
    client: Any | None = None,
    temperature: float = 0.3,
) -> str:
    """Generate the next agent turn for ``context_messages`` from a deployment.

    Sends the **full multi-turn context** (unlike the classifier ``infer``, which
    sends only system+user), so the model continues the conversation in the
    desired strategy. Returns the assistant text (empty string on a refusal).

    o-series reasoning deployments (e.g. an RFT-tuned ``o4-mini``) reject any
    ``temperature`` other than the default ``1`` with a 400 ``unsupported_value``.
    Deployment names don't reliably reveal the base model, so on that specific
    error we transparently retry once without ``temperature`` rather than fail
    the arm.
    """
    from . import act2a_serverless_sft as sft  # noqa: PLC0415

    active = client if client is not None else sft.build_client(config)
    try:
        completion = active.chat.completions.create(
            model=deployment_name,
            messages=context_messages,
            temperature=temperature,
        )
    except Exception as exc:  # noqa: BLE001 - narrow to the temperature rejection
        if "temperature" not in str(exc).lower():
            raise
        logger.info(
            "[gen] '%s' rejected temperature=%s; retrying with model default",
            deployment_name, temperature,
        )
        completion = active.chat.completions.create(
            model=deployment_name,
            messages=context_messages,
        )
    return _message_text(completion.choices[0].message)


# ---------------------------------------------------------------------------
# Offline evaluation
# ---------------------------------------------------------------------------
def load_eval_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load the held-out conversation eval rows written by ``gen-data``."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate_alignment(
    config: DemoConfig,
    deployment_name: str,
    eval_path: str | Path,
    *,
    limit: int | None = None,
    client: Any | None = None,
    request_delay: float = 0.0,
) -> dict[str, Any]:
    """Replay held-out conversations through a deployment and score adherence.

    For each eval row, generates the next agent turn and scores it with
    :func:`strategy_alignment_score`. Returns mean alignment, per-row detail, and
    an error count. A failed generation degrades to score ``0.0`` rather than
    aborting (OWASP A04 — fail safe).
    """
    import time  # noqa: PLC0415

    from . import act2a_serverless_sft as sft  # noqa: PLC0415

    rows = load_eval_rows(eval_path)
    if limit is not None and limit > 0:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No eval rows in {eval_path}.")

    active = client if client is not None else sft.build_client(config)
    scored_rows: list[dict[str, Any]] = []
    errors = 0
    for row in rows:
        strategy = row.get("strategy", "")
        try:
            reply = generate_agent_turn(
                config, deployment_name, row["context_messages"], client=active
            )
        except Exception as exc:  # noqa: BLE001 - fail safe per row
            logger.warning("[eval] generation failed for %s: %s", row.get("id"), exc)
            reply = ""
            errors += 1
        scored_rows.append(
            {
                "id": row.get("id"),
                "strategy": strategy,
                "response": reply,
                "alignment": strategy_alignment_score(reply, strategy),
            }
        )
        if request_delay:
            time.sleep(request_delay)

    mean_alignment = (
        sum(item["alignment"] for item in scored_rows) / len(scored_rows)
        if scored_rows
        else 0.0
    )
    return {
        "deployment": deployment_name,
        "rows": len(scored_rows),
        "errors": errors,
        "mean_alignment": mean_alignment,
        "detail": scored_rows,
    }


# ---------------------------------------------------------------------------
# Foundry evaluation — custom promptflow-safe evaluator + upload
# ---------------------------------------------------------------------------
class StrategyAlignmentEvaluator:
    """Per-row strategy-adherence evaluator for Foundry portal runs.

    Returns ``{"strategy_alignment": <score>}`` using the same deterministic
    heuristic as the offline path, so portal and offline numbers line up. Follows
    the promptflow custom-evaluator contract: explicit ``__init__``, simple
    parameter annotations, ``**kwargs`` var-keyword, and no return annotation.
    """

    def __init__(self) -> None:
        """No configuration; an explicit init lets promptflow introspect it."""

    def __call__(
        self,
        *,
        response: str = "",
        strategy: str = "",
        **kwargs: Any,
    ):
        return {"strategy_alignment": strategy_alignment_score(response or "", strategy or "")}


def build_conv_eval_dataset(
    config: DemoConfig,
    deployment_name: str,
    eval_path: str | Path,
    out_path: str | Path,
    *,
    limit: int | None = None,
    request_delay: float = 0.0,
    client: Any | None = None,
) -> dict[str, Any]:
    """Replay held-out conversations through a deployment into a Foundry dataset.

    Writes one JSONL row per example with the columns the evaluators consume:
    ``query`` (rendered transcript), ``response`` (generated agent reply),
    ``ground_truth`` (the exemplary reply), and ``strategy`` (the target).
    """
    from . import act2a_serverless_sft as sft  # noqa: PLC0415

    rows = load_eval_rows(eval_path)
    if limit is not None and limit > 0:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No eval rows in {eval_path}.")

    active = client if client is not None else sft.build_client(config)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    errors = 0
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            try:
                reply = generate_agent_turn(
                    config, deployment_name, row["context_messages"], client=active
                )
            except Exception as exc:  # noqa: BLE001 - fail safe per row
                logger.warning("[foundry] generation failed for %s: %s", row.get("id"), exc)
                reply = ""
                errors += 1
            handle.write(
                json.dumps(
                    {
                        "query": row.get("query", ""),
                        "response": reply,
                        "ground_truth": row.get("ground_truth", ""),
                        "strategy": row.get("strategy", ""),
                    },
                    ensure_ascii=False,
                )
            )
            handle.write("\n")
    return {"rows": len(rows), "errors": errors, "path": str(out)}


def build_judge_model_config(config: DemoConfig) -> dict[str, Any] | None:
    """Build the AzureOpenAI judge ``model_config`` for AI-assisted evaluators.

    The AI-assisted evaluators (coherence, fluency, relevance, similarity) call a
    judge model. Returns the ``model_config`` dict the SDK expects, or ``None``
    when the endpoint, key, or judge deployment is unset (so the caller can run
    the deterministic evaluators only). The judge deployment is ``teacher_model``
    (e.g. ``chat41`` -> gpt-4.1), a strong, supported grader.
    """
    if not (config.azure_openai_endpoint and config.azure_openai_api_key and config.teacher_model):
        return None
    return {
        "azure_endpoint": config.azure_openai_endpoint,
        "api_key": config.azure_openai_api_key,
        "azure_deployment": config.teacher_model,
        "api_version": config.data_plane_api_version,
    }


def build_conv_evaluators(
    *,
    include_builtin: bool = True,
    judge_model_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Construct the evaluator suite + column mapping for the alignment eval.

    The custom :class:`StrategyAlignmentEvaluator` is always present. The
    built-in lexical :class:`F1ScoreEvaluator` (token overlap of the reply vs the
    exemplary reply) is added when ``include_builtin`` is True. When
    ``judge_model_config`` is provided, four AI-assisted (LLM-judge) evaluators
    are added for a richer scoreboard:

    * ``coherence`` — logical flow of the reply given the conversation.
    * ``fluency`` — grammatical/linguistic quality of the reply.
    * ``relevance`` — how well the reply addresses the customer's last turn.
    * ``similarity`` — semantic closeness to the exemplary reply (the proper,
      meaning-aware counterpart to the lexical ``f1_score``).

    All AI-assisted scores are on a 1-5 scale; ``strategy_alignment`` and
    ``f1_score`` are 0-1.
    """
    evaluators: dict[str, Any] = {"strategy_alignment": StrategyAlignmentEvaluator()}
    evaluator_config: dict[str, Any] = {
        "strategy_alignment": {
            "column_mapping": {
                "response": "${data.response}",
                "strategy": "${data.strategy}",
            }
        },
    }
    if include_builtin:
        from azure.ai.evaluation import F1ScoreEvaluator  # noqa: PLC0415

        evaluators["f1_score"] = F1ScoreEvaluator()
        evaluator_config["f1_score"] = {
            "column_mapping": {
                "response": "${data.response}",
                "ground_truth": "${data.ground_truth}",
            }
        }
    if judge_model_config is not None:
        from azure.ai.evaluation import (  # noqa: PLC0415
            CoherenceEvaluator,
            FluencyEvaluator,
            RelevanceEvaluator,
            SimilarityEvaluator,
        )

        evaluators["coherence"] = CoherenceEvaluator(judge_model_config)
        evaluator_config["coherence"] = {
            "column_mapping": {
                "query": "${data.query}",
                "response": "${data.response}",
            }
        }
        evaluators["fluency"] = FluencyEvaluator(judge_model_config)
        evaluator_config["fluency"] = {
            "column_mapping": {"response": "${data.response}"}
        }
        evaluators["relevance"] = RelevanceEvaluator(judge_model_config)
        evaluator_config["relevance"] = {
            "column_mapping": {
                "query": "${data.query}",
                "response": "${data.response}",
            }
        }
        evaluators["similarity"] = SimilarityEvaluator(judge_model_config)
        evaluator_config["similarity"] = {
            "column_mapping": {
                "query": "${data.query}",
                "response": "${data.response}",
                "ground_truth": "${data.ground_truth}",
            }
        }
    return evaluators, evaluator_config



def run_foundry_conv_eval(
    config: DemoConfig,
    label: str,
    deployment_name: str,
    eval_path: str | Path,
    out_dir: str | Path,
    *,
    limit: int | None = None,
    request_delay: float = 0.0,
    include_builtin: bool = True,
    judge_model_config: dict[str, Any] | None = None,
    upload: bool = True,
    client: Any | None = None,
) -> dict[str, Any]:
    """Build the dataset for one arm and run (and optionally upload) the eval.

    When ``upload`` is True and an Azure AI project is configured, the run is
    pushed to the Foundry **Evaluations** tab as ``conv-alignment-eval-<label>``.
    Reuses :func:`act3a_foundry_eval.build_azure_ai_project` for project resolution.
    """
    from azure.ai.evaluation import evaluate  # noqa: PLC0415

    from . import act3a_foundry_eval as foundry  # noqa: PLC0415

    out_dir = Path(out_dir)
    dataset_path = out_dir / f"conv_foundry_data_{label}.jsonl"
    result_path = out_dir / f"conv_foundry_result_{label}.json"

    dataset = build_conv_eval_dataset(
        config, deployment_name, eval_path, dataset_path,
        limit=limit, request_delay=request_delay, client=client,
    )
    evaluators, evaluator_config = build_conv_evaluators(
        include_builtin=include_builtin, judge_model_config=judge_model_config,
    )

    azure_ai_project = foundry.build_azure_ai_project(config) if upload else None
    if upload and azure_ai_project is None:
        logger.warning(
            "[foundry] No Azure AI project configured; running '%s' locally "
            "without portal upload.", label,
        )

    evaluation_name = f"conv-alignment-eval-{label}"
    display = MODEL_DISPLAY_NAME.get(label, label)
    logger.info(
        "[foundry] Evaluating %s arm '%s' (%d row(s)) as '%s'%s",
        display, deployment_name, dataset["rows"], evaluation_name,
        " -> Foundry portal" if azure_ai_project else " (local only)",
    )

    result = evaluate(
        data=str(dataset_path),
        evaluators=evaluators,
        evaluator_config=evaluator_config,
        evaluation_name=evaluation_name,
        azure_ai_project=azure_ai_project,
        output_path=str(result_path),
    )

    metrics = dict(result.get("metrics", {})) if isinstance(result, dict) else {}
    studio_url = result.get("studio_url") if isinstance(result, dict) else None
    return {
        "label": label,
        "display_name": display,
        "deployment": deployment_name,
        "evaluation_name": evaluation_name,
        "rows": dataset["rows"],
        "errors": dataset["errors"],
        "metrics": metrics,
        "studio_url": studio_url,
        "result_path": str(result_path),
        "uploaded": azure_ai_project is not None,
    }


def run_all_foundry_conv_evals(
    config: DemoConfig,
    eval_path: str | Path,
    out_dir: str | Path,
    *,
    models: list[str] | tuple[str, ...] = DEFAULT_MODELS,
    limit: int | None = None,
    request_delay: float = 0.0,
    include_builtin: bool = True,
    judge_model_config: dict[str, Any] | None = None,
    upload: bool = True,
) -> dict[str, Any]:
    """Evaluate every configured model arm and write a combined report.

    Resolves deployment names from config (reusing
    :func:`act3a_foundry_eval.resolve_deployments`), skips arms whose deployment
    is unset, and records per-arm failures rather than aborting the showcase.
    """
    from . import act3a_foundry_eval as foundry  # noqa: PLC0415

    pairs = foundry.resolve_deployments(config, models)
    if not pairs:
        raise ValueError(
            "No evaluatable deployments: set BASE/SFT/DPO/RFT_DEPLOYMENT_NAME."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for label, deployment in pairs:
        try:
            runs.append(
                run_foundry_conv_eval(
                    config, label, deployment, eval_path, out_dir,
                    limit=limit, request_delay=request_delay,
                    include_builtin=include_builtin,
                    judge_model_config=judge_model_config, upload=upload,
                )
            )
        except Exception as exc:  # noqa: BLE001 - record and continue
            logger.error("[foundry] arm '%s' failed: %s", label, exc)
            failures.append({"label": label, "error": str(exc)})

    report = {
        "created": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
        "failures": failures,
    }
    (out_dir / FOUNDRY_REPORT_FILE).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def format_foundry_report(report: dict[str, Any]) -> str:
    """Render a compact scoreboard from a combined Foundry report.

    Columns are shown only when at least one arm reports that metric, so the
    deterministic-only (``--no-ai-assisted``) and AI-assisted runs both render
    cleanly. ``strategy``/``f1`` are 0-1; ``coher``/``fluency``/``relev``/``simil``
    are 1-5 LLM-judge scores.
    """
    runs = report.get("runs", [])

    # Ordered column spec: (header, aggregated-metric-key).
    spec: tuple[tuple[str, str], ...] = (
        ("strategy", "strategy_alignment.strategy_alignment"),
        ("coher", "coherence.coherence"),
        ("fluency", "fluency.fluency"),
        ("relev", "relevance.relevance"),
        ("simil", "similarity.similarity"),
        ("f1", "f1_score.f1_score"),
    )

    def _has(metrics: dict[str, Any], key: str) -> bool:
        return key in metrics and metrics[key] is not None

    columns = [
        (header, key)
        for header, key in spec
        if any(_has(run.get("metrics", {}), key) for run in runs)
    ]

    def _fmt(metrics: dict[str, Any], key: str) -> str:
        if _has(metrics, key):
            try:
                return f"{float(metrics[key]):.3f}"
            except (TypeError, ValueError):
                return str(metrics[key])
        return "  -  "

    width = 22 + 9 * len(columns) + 8
    lines = ["", "Conversation-Alignment Foundry Eval", "=" * width]
    header = f"{'Model':<22}" + "".join(f"{h:>9}" for h, _ in columns) + f"{'rows':>8}"
    lines.append(header)
    lines.append("-" * width)
    for run in runs:
        metrics = run.get("metrics", {})
        cells = "".join(f"{_fmt(metrics, key):>9}" for _, key in columns)
        lines.append(
            f"{run.get('display_name', run.get('label', '?')):<22}"
            f"{cells}{run.get('rows', 0):>8}"
        )
    for failure in report.get("failures", []):
        lines.append(f"{failure.get('label', '?'):<22}{'FAILED':>9}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_config() -> DemoConfig:
    return DemoConfig.from_env()


def cmd_gen_data(args: argparse.Namespace, config: DemoConfig) -> int:
    teacher = None
    concurrency = 1
    if getattr(args, "use_llm", False):
        max_retries = getattr(args, "max_retries", DEFAULT_TEACHER_MAX_RETRIES)
        concurrency = max(1, getattr(args, "concurrency", DEFAULT_TEACHER_CONCURRENCY))
        teacher = build_llm_teacher(config, max_retries=max_retries)
        print(
            f"\nUsing LLM teacher for generation (concurrency={concurrency}; "
            "consumes tokens)..."
        )
    counts = generate_all_datasets(
        count=args.count, eval_count=args.eval_count, seed=args.seed,
        generate_fn=teacher, concurrency=concurrency,
    )
    source = "LLM teacher" if teacher is not None else "deterministic templates"
    print(f"\nConversation-alignment corpus ({source}) written to {DATA_DIR}\n")
    for name, rows in counts.items():
        print(f"  {name:<24} {rows:>5} rows")
    print()
    return EXIT_SUCCESS


def _cmd_finetune(method: str, args: argparse.Namespace, config: DemoConfig) -> int:
    summary = run_finetune(
        config, method,
        max_polls=args.max_polls,
        grader_model=getattr(args, "grader_model", None),
        n_epochs=getattr(args, "n_epochs", None),
        beta=getattr(args, "beta", None),
        batch_size=getattr(args, "batch_size", None),
        learning_rate_multiplier=getattr(args, "learning_rate_multiplier", None),
        eval_interval=getattr(args, "eval_interval", None),
        eval_samples=getattr(args, "eval_samples", None),
        reasoning_effort=getattr(args, "reasoning_effort", None),
        compute_multiplier=getattr(args, "compute_multiplier", None),
        deploy=args.deploy,
        deployment_name=args.deployment_name,
        sku=args.sku,
    )
    print(json.dumps(summary, indent=2))
    return EXIT_SUCCESS if summary.get("status") == "succeeded" else EXIT_FAILURE


def cmd_sft(args: argparse.Namespace, config: DemoConfig) -> int:
    return _cmd_finetune("supervised", args, config)


def cmd_dpo(args: argparse.Namespace, config: DemoConfig) -> int:
    return _cmd_finetune("dpo", args, config)


def cmd_rft(args: argparse.Namespace, config: DemoConfig) -> int:
    return _cmd_finetune("reinforcement", args, config)


def cmd_resume(args: argparse.Namespace, config: DemoConfig) -> int:
    """Re-attach to an already-submitted fine-tune (recover a dropped run)."""
    if not getattr(args, "job_id", None):
        jobs = list_finetune_jobs(config, limit=getattr(args, "limit", 10))
        print("\nRecent fine-tuning jobs:\n")
        for job in jobs:
            print(
                f"  {getattr(job, 'id', '?'):<42} "
                f"{str(getattr(job, 'status', '?')):<12} "
                f"{getattr(job, 'fine_tuned_model', None) or ''}"
            )
        print(
            "\nRe-attach with: resume --job-id <id> "
            "[--deploy --deployment-name <name> --sku developer]\n"
        )
        return EXIT_SUCCESS

    summary = run_finetune_resume(
        config, args.job_id,
        method=getattr(args, "method", "resume"),
        max_polls=args.max_polls,
        deploy=args.deploy,
        deployment_name=args.deployment_name,
        sku=args.sku,
    )
    print(json.dumps(summary, indent=2))
    return EXIT_SUCCESS if summary.get("status") == "succeeded" else EXIT_FAILURE


def cmd_evaluate(args: argparse.Namespace, config: DemoConfig) -> int:
    result = evaluate_alignment(
        config, args.deployment, DATA_DIR / CONV_EVAL_FILE,
        limit=args.limit, request_delay=args.delay,
    )
    print(
        f"\n{args.deployment}: mean strategy alignment "
        f"{result['mean_alignment']:.3f} over {result['rows']} row(s) "
        f"({result['errors']} error(s))\n"
    )
    return EXIT_SUCCESS


def cmd_foundry_eval(args: argparse.Namespace, config: DemoConfig) -> int:
    judge_model_config = None
    if not args.no_ai_assisted:
        judge_model_config = build_judge_model_config(config)
        if judge_model_config is None:
            logger.warning(
                "[foundry] AI-assisted evaluators skipped: set AZURE_OPENAI_ENDPOINT, "
                "AZURE_OPENAI_API_KEY, and TEACHER_MODEL (judge) to enable "
                "coherence/fluency/relevance/similarity."
            )
        else:
            print(
                f"AI-assisted evaluators ON (judge={config.teacher_model}): "
                "coherence, fluency, relevance, similarity."
            )
    report = run_all_foundry_conv_evals(
        config, DATA_DIR / CONV_EVAL_FILE, FOUNDRY_DIR,
        models=tuple(args.models), limit=args.limit, request_delay=args.delay,
        include_builtin=not args.no_builtin, judge_model_config=judge_model_config,
        upload=not args.no_upload,
    )
    print(format_foundry_report(report))
    return EXIT_FAILURE if report.get("failures") else EXIT_SUCCESS


def cmd_capture(args: argparse.Namespace, config: DemoConfig) -> int:
    """Capture a deployed agent's conversations into the training corpus.

    Delegates to :mod:`finetuning.agent_corpus_capture` (lazy import keeps
    this module free of any capture-time dependency at load).
    """
    from . import agent_corpus_capture as capture  # noqa: PLC0415

    return capture.run_capture(args, config)


def cmd_agent_create(args: argparse.Namespace, config: DemoConfig) -> int:
    """Create a managed Foundry Agent that wraps a fine-tuned deployment.

    Delegates to :mod:`finetuning.foundry_agent_service` (lazy import keeps
    this module free of the Agents SDK at load).
    """
    from . import foundry_agent_service as agentsvc  # noqa: PLC0415

    return agentsvc.run_agent_create(args, config)


def cmd_agent_list(args: argparse.Namespace, config: DemoConfig) -> int:
    """List the managed Foundry Agents in the project."""
    from . import foundry_agent_service as agentsvc  # noqa: PLC0415

    return agentsvc.run_agent_list(args, config)


def cmd_agent_delete(args: argparse.Namespace, config: DemoConfig) -> int:
    """Delete a managed Foundry Agent by id."""
    from . import foundry_agent_service as agentsvc  # noqa: PLC0415

    return agentsvc.run_agent_delete(args, config)


def cmd_agent_test(args: argparse.Namespace, config: DemoConfig) -> int:
    """Replay held-out conversations through the Agent and score the replies."""
    from . import foundry_agent_service as agentsvc  # noqa: PLC0415

    return agentsvc.run_agent_test(args, config)


def cmd_distill(args: argparse.Namespace, config: DemoConfig) -> int:
    """Distill captured Agent transcripts into a retrain corpus (close the loop)."""
    from . import distill as distillmod  # noqa: PLC0415

    return distillmod.run_distill(args, config)


def cmd_all(args: argparse.Namespace, config: DemoConfig) -> int:
    cmd_gen_data(args, config)
    for method, handler in (("supervised", cmd_sft), ("dpo", cmd_dpo), ("reinforcement", cmd_rft)):
        logger.info("[all] running %s fine-tune", method)
        handler(args, config)
    return EXIT_SUCCESS


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI for the conversation-alignment pipeline."""
    parser = argparse.ArgumentParser(
        prog="customer_conversation_alignment",
        description=(
            "End-to-end conversational-strategy alignment: synthetic multi-turn "
            "data, serverless SFT/DPO/RFT, offline + Foundry evaluation."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("gen-data", help="generate the conversation-alignment corpus")
    p_gen.add_argument("--count", type=int, default=200, help="training conversations")
    p_gen.add_argument("--eval-count", type=int, default=60, help="held-out eval conversations")
    p_gen.add_argument("--seed", type=int, default=1337)
    p_gen.add_argument("--use-llm", action="store_true",
                       help="author rows with a teacher LLM (diverse; needs Azure config + tokens)")
    p_gen.add_argument("--concurrency", type=int, default=DEFAULT_TEACHER_CONCURRENCY,
                       help="parallel teacher calls when --use-llm (independent conversations)")
    p_gen.add_argument("--max-retries", type=int, default=DEFAULT_TEACHER_MAX_RETRIES,
                       help="SDK 429/5xx retry budget per teacher call when --use-llm")
    p_gen.set_defaults(func=cmd_gen_data)

    p_cap = sub.add_parser(
        "capture",
        help="capture a deployed agent's conversations into the training corpus",
    )
    from . import agent_corpus_capture as _capture  # noqa: PLC0415
    _capture.add_capture_arguments(p_cap)
    p_cap.set_defaults(func=cmd_capture)

    def _add_finetune_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--max-polls", type=int, default=None, help="cap status polls")
        p.add_argument("--n-epochs", type=int, default=None, help="training epochs (overrides default)")
        p.add_argument("--beta", type=float, default=None, help="DPO preference strength (DPO only)")
        p.add_argument("--batch-size", type=int, default=None, help="training batch size (any method)")
        p.add_argument("--learning-rate-multiplier", type=float, default=None,
                       help="learning-rate multiplier (any method)")
        p.add_argument("--eval-interval", type=int, default=None,
                       help="steps between evaluations (RFT only)")
        p.add_argument("--eval-samples", type=int, default=None,
                       help="samples per evaluation (RFT only)")
        p.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default=None,
                       help="reasoning effort for the o-series base (RFT only)")
        p.add_argument("--compute-multiplier", type=float, default=None,
                       help="compute multiplier (RFT only)")
        p.add_argument("--deploy", action="store_true", help="deploy on success")
        p.add_argument("--deployment-name", default=None, help="deployment name when --deploy")
        p.add_argument("--sku", default="developer", help="deployment SKU")

    p_sft = sub.add_parser("sft", help="serverless supervised fine-tune")
    _add_finetune_args(p_sft)
    p_sft.set_defaults(func=cmd_sft)

    p_dpo = sub.add_parser("dpo", help="serverless preference (DPO) fine-tune")
    _add_finetune_args(p_dpo)
    p_dpo.set_defaults(func=cmd_dpo)

    p_rft = sub.add_parser("rft", help="serverless reinforcement (RFT) fine-tune")
    _add_finetune_args(p_rft)
    p_rft.add_argument("--grader-model", default=None, help="grader deployment for RFT")
    p_rft.set_defaults(func=cmd_rft)

    p_res = sub.add_parser(
        "resume",
        help="re-attach to an already-submitted fine-tune (recover a dropped run) and optionally deploy",
    )
    p_res.add_argument("--job-id", default=None,
                       help="job id to re-attach to; omit to list recent jobs")
    p_res.add_argument("--method", default="resume",
                       help="label for the written state file")
    p_res.add_argument("--limit", type=int, default=10,
                       help="number of recent jobs to list when --job-id is omitted")
    _add_finetune_args(p_res)
    p_res.set_defaults(func=cmd_resume)

    p_eval = sub.add_parser("evaluate", help="offline strategy-alignment score")
    p_eval.add_argument("--deployment", required=True, help="deployment name to score")
    p_eval.add_argument("--limit", type=int, default=None)
    p_eval.add_argument("--delay", type=float, default=0.0)
    p_eval.set_defaults(func=cmd_evaluate)

    p_fnd = sub.add_parser("foundry-eval", help="upload alignment runs to Foundry")
    p_fnd.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS),
                       choices=list(DEFAULT_MODELS))
    p_fnd.add_argument("--limit", type=int, default=None)
    p_fnd.add_argument("--delay", type=float, default=0.5)
    p_fnd.add_argument("--no-builtin", action="store_true", help="omit F1ScoreEvaluator")
    p_fnd.add_argument("--no-ai-assisted", action="store_true",
                       help="omit LLM-judge evaluators (coherence/fluency/relevance/similarity)")
    p_fnd.add_argument("--no-upload", action="store_true", help="run locally, no portal upload")
    p_fnd.set_defaults(func=cmd_foundry_eval)

    from . import foundry_agent_service as _agentsvc  # noqa: PLC0415

    p_agc = sub.add_parser("agent-create", help="create a managed Foundry Agent for a deployment")
    _agentsvc.add_agent_create_arguments(p_agc)
    p_agc.set_defaults(func=cmd_agent_create)

    p_agl = sub.add_parser("agent-list", help="list managed Foundry Agents in the project")
    p_agl.set_defaults(func=cmd_agent_list)

    p_agd = sub.add_parser("agent-delete", help="delete a managed Foundry Agent by id")
    p_agd.add_argument("--id", required=True, help="agent id to delete")
    p_agd.set_defaults(func=cmd_agent_delete)

    p_agt = sub.add_parser("agent-test", help="replay test cases through the Agent and score them")
    _agentsvc.add_agent_test_arguments(p_agt)
    p_agt.set_defaults(func=cmd_agent_test)

    from . import distill as _distill  # noqa: PLC0415

    p_dst = sub.add_parser("distill", help="distill Agent transcripts into a retrain corpus")
    _distill.add_distill_arguments(p_dst)
    p_dst.set_defaults(func=cmd_distill)

    p_all = sub.add_parser("all", help="gen-data + sft + dpo + rft")
    p_all.add_argument("--count", type=int, default=200)
    p_all.add_argument("--eval-count", type=int, default=60)
    p_all.add_argument("--seed", type=int, default=1337)
    p_all.add_argument("--use-llm", action="store_true",
                       help="author rows with a teacher LLM (diverse; needs Azure config + tokens)")
    p_all.add_argument("--concurrency", type=int, default=DEFAULT_TEACHER_CONCURRENCY,
                       help="parallel teacher calls when --use-llm (independent conversations)")
    p_all.add_argument("--max-retries", type=int, default=DEFAULT_TEACHER_MAX_RETRIES,
                       help="SDK 429/5xx retry budget per teacher call when --use-llm")
    p_all.add_argument("--max-polls", type=int, default=None)
    p_all.add_argument("--deploy", action="store_true")
    p_all.add_argument("--deployment-name", default=None)
    p_all.add_argument("--sku", default="developer")
    p_all.add_argument("--grader-model", default=None)
    p_all.set_defaults(func=cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the conversation-alignment pipeline."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    config = _load_config()
    try:
        return int(args.func(args, config))
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        logger.error("%s", exc)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
