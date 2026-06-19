"""Act 2A1 — quick base-vs-fine-tuned comparison on the labeled validation set.

A lightweight bridge between Act 2A (deploy) and Act 2B (GPU LoRA): run the
SAME labeled holdout (``validation.jsonl``) through the **base** model deployment
and the **fine-tuned** deployment, parse each structured prediction, and report a
side-by-side scorecard so the fine-tuning lift is visible before moving on.

Design notes
------------
* The validation file is the canonical labeled holdout written by Act 1 in the
  OpenAI chat-SFT shape: each row is ``{"messages": [system, user, assistant]}``
  where the ``assistant`` content is the ground-truth JSON
  (``{"intent", "outcome", "propensity_score"}``).
* Inference reuses :func:`finetuning.act2a_serverless_sft.infer` (a
  data-plane chat completion against a deployment name), so no new Azure surface
  is introduced.
* Metrics are intentionally dependency-free (exact-match intent accuracy and
  propensity mean-absolute-error) so this runs without ``scikit-learn``. The
  per-candidate predictions are also written in the bring-your-own-predictions
  JSONL shape so the richer Act 3 offline metrics can consume them later.
* Every prediction is parsed defensively: a malformed / non-JSON model response
  degrades to an empty label rather than raising (OWASP A04 — fail safe).
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .act2a_serverless_sft import build_client, infer
from .config import DemoConfig
from .taxonomy import INTENT_LABELS

logger = logging.getLogger(__name__)

# Filename for the persisted aggregate scorecard summary (written to out_dir).
RESULT_FILE: str = "quick_eval_result.json"

# A candidate whose failed-call share meets or exceeds this fraction is flagged
# as unreliable: its accuracy reflects quota/transport failures, not model skill.
UNRELIABLE_ERROR_FRACTION: float = 0.5

# Standard / GlobalStandard deployments share a regional pool and intermittently
# return a transient ``429 "Backend error"`` even when the deployment's own
# TPM/RPM quota is essentially untouched (verified: remaining-tokens and
# remaining-requests both ~100% on the 429). Azure stamps a misleading
# ``retry-after: 30`` on that 429, but the blip clears almost immediately, so we
# DISABLE the SDK's retry (which honors that 30s) and run our own short backoff.
CLIENT_MAX_RETRIES: int = 0

# Our own fast retry for the transient backend 429: many attempts, short waits.
TRANSIENT_RETRY_ATTEMPTS: int = 8
TRANSIENT_BACKOFF_SECONDS: float = 1.5
TRANSIENT_BACKOFF_CAP_SECONDS: float = 6.0

# Fallback system prompt when a base model returns empty/non-structured output.
STRICT_JSON_SYSTEM_PROMPT: str = (
    "You are a sales-call classifier. Return only strict JSON with keys: "
    "intent, outcome, propensity_score. intent must be one of: "
    "pricing_inquiry, feature_request, competitor_comparison, "
    "objection_handling, scheduling_followup, support_escalation, "
    "general_interest. outcome can be any short label. "
    "propensity_score must be a float between 0 and 1."
)


def _is_rate_limit(exc: Exception) -> bool:
    """Return True when ``exc`` is an HTTP 429 (rate-limit / backend) error."""
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 429


def _infer_with_backoff(
    config: DemoConfig,
    deployment_name: str,
    transcript: str,
    *,
    client: Any | None,
    attempts: int = TRANSIENT_RETRY_ATTEMPTS,
    sleep: Any = time.sleep,
) -> str | None:
    """Call :func:`infer`, retrying transient 429s with a SHORT custom backoff.

    Azure's shared-pool ``429 "Backend error"`` is transient and unrelated to
    the deployment quota, so it almost always clears within a second or two.
    Rather than obey the misleading ``retry-after: 30`` (which the SDK honors),
    this retries up to ``attempts`` times with a capped linear backoff and logs
    each retry visibly. Non-429 errors propagate immediately.
    """
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            raw = infer(config, deployment_name, transcript, client=client)
            if isinstance(raw, str) and raw.strip():
                return raw

            # Some base deployments return empty content for the generic prompt.
            # Retry once with a stricter JSON-only instruction so parsing and
            # metrics stay comparable across base and fine-tuned candidates.
            forced = infer(
                config,
                deployment_name,
                transcript,
                client=client,
                system_prompt=STRICT_JSON_SYSTEM_PROMPT,
            )
            if isinstance(forced, str) and forced.strip():
                return forced
            return raw
        except Exception as exc:  # noqa: BLE001 - re-raise non-429 below
            if not _is_rate_limit(exc):
                raise
            last_exc = exc
            if attempt == attempts:
                break
            wait = min(TRANSIENT_BACKOFF_SECONDS * attempt, TRANSIENT_BACKOFF_CAP_SECONDS)
            logger.info(
                "[%s] transient 429 (backend blip) on attempt %d/%d; retrying in %.1fs",
                deployment_name,
                attempt,
                attempts,
                wait,
            )
            sleep(wait)
    assert last_exc is not None  # only reached after a 429
    raise last_exc



# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _normalize_intent(predicted: str) -> str:
    """Normalize a predicted intent string to the canonical taxonomy.

    The fine-tuned model sometimes emits near-synonyms or variants outside the
    canonical 7-label set (e.g., ``"complaint"`` vs ``"support_escalation"``,
    ``"discovery"`` vs ``"general_interest"``). This function attempts a
    fuzzy match: if the predicted string is close to a canonical label, return
    the canonical one; otherwise return the predicted string unchanged (so the
    exact-match scorer counts it as a miss, flagging the issue).

    Matching strategy: (1) exact match (case-insensitive); (2) longest substring
    overlap (word by word or as a whole); (3) no match.
    """
    if not predicted or not isinstance(predicted, str):
        return ""
    pred_lower = predicted.lower().strip()

    synonym_map = {
        "complaint": "support_escalation",
        "escalation": "support_escalation",
        "follow_up": "scheduling_followup",
        "follow-up": "scheduling_followup",
        "discovery": "general_interest",
        "feature_inquiry": "feature_request",
    }
    if pred_lower in synonym_map:
        return synonym_map[pred_lower]

    if pred_lower in [l.lower() for l in INTENT_LABELS]:
        return pred_lower  # Already canonical

    # Some base-model outputs return descriptive intent phrases rather than the
    # canonical label. Classify those phrases with keyword heuristics so the
    # quick eval can still compare base-vs-finetuned on the same taxonomy.
    phrase = pred_lower.replace("-", "_")
    if any(token in phrase for token in ("security", "technical issue", "glitch", "support", "dissatisfaction", "discontinue the service", "report issue", "data privacy")):
        return "support_escalation"
    if any(token in phrase for token in ("competitor", "switch", "current provider", "compare", "comparison", "alternative solution", "another provider")):
        return "competitor_comparison"
    if any(token in phrase for token in ("pricing", "price", "cost", "roi", "discount", "rates", "quote")):
        return "pricing_inquiry"
    if any(token in phrase for token in ("feature", "integration", "analytics", "api", "hl7", "fhir", "erp", "connector", "automation", "reporting", "compliance", "capabilit")):
        return "feature_request"
    if any(token in phrase for token in ("schedule", "follow up", "follow_up", "proposal", "next step", "demo", "meeting")):
        return "scheduling_followup"
    if any(token in phrase for token in ("objection", "hesitation", "timing", "budget constraint", "not ready", "capacity", "bandwidth", "resource constraint")):
        return "objection_handling"
    if any(token in phrase for token in ("explore", "interested", "evaluate", "survey", "learn more", "potential solution")):
        return "general_interest"

    # Find the canonical label with the longest substring overlap.
    # Try matching whole strings first, then individual words.
    best_match: str | None = None
    best_len = 0

    for canonical in INTENT_LABELS:
        canonical_lower = canonical.lower()

        # Check whole-string containment.
        if pred_lower in canonical_lower:
            candidate_len = len(pred_lower)
        elif canonical_lower in pred_lower:
            candidate_len = len(canonical_lower)
        else:
            # Try word-by-word: find the longest canonical word in the predicted.
            candidate_len = 0
            for word in canonical_lower.split("_"):
                if word in pred_lower:
                    candidate_len = max(candidate_len, len(word))

        if candidate_len > best_len:
            best_len = candidate_len
            best_match = canonical_lower

    if best_match and best_len > 0:
        return best_match
    return pred_lower


def _extract_intent_from_text(text: str) -> str:
    """Best-effort intent extraction when the model does not return JSON."""
    lowered = text.lower()

    # Prefer explicit intent fields when present.
    key_match = re.search(r"intent\s*[:=]\s*['\"]?([a-z_\- ]+)", lowered)
    if key_match:
        candidate = key_match.group(1).strip().replace("-", "_").replace(" ", "_")
        return _normalize_intent(candidate)

    # Fallback: look for known labels or common synonyms anywhere in text.
    synonyms = [
        "complaint",
        "discovery",
        "follow_up",
        "follow-up",
        "feature_inquiry",
        "escalation",
    ]
    candidates = list(INTENT_LABELS) + synonyms
    best = ""
    for item in candidates:
        token = item.replace("-", "_")
        pattern = r"\b" + re.escape(token.replace("_", " ")) + r"\b"
        if re.search(pattern, lowered):
            if len(token) > len(best):
                best = token
            continue
        if re.search(r"\b" + re.escape(token) + r"\b", lowered):
            if len(token) > len(best):
                best = token
    return _normalize_intent(best) if best else ""


def _extract_propensity_from_text(text: str) -> float | None:
    """Best-effort propensity extraction from plain text."""
    lowered = text.lower()

    # Match decimal score near propensity field names.
    dec = re.search(
        r"propensity(?:_score)?(?:\s+is|\s*[:=])?\s*(0(?:\.\d+)?|1(?:\.0+)?)",
        lowered,
    )
    if dec:
        try:
            return float(dec.group(1))
        except ValueError:
            return None

    # Match percentage near propensity mentions.
    pct = re.search(r"propensity(?:_score)?[^\d]*(\d{1,3})%", lowered)
    if pct:
        try:
            value = float(pct.group(1)) / 100.0
            return max(0.0, min(1.0, value))
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def parse_label(content: str | None) -> dict[str, Any]:
    """Parse a model/ground-truth label JSON into a normalized dict.

    Tolerates ``None``, surrounding prose, and code fences by extracting the
    first ``{...}`` block. Returns ``{"intent": "", "outcome": "",
    "propensity_score": None}`` on any failure rather than raising.
    """
    empty: dict[str, Any] = {"intent": "", "outcome": "", "propensity_score": None}
    if not content:
        return empty
    text = content.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        try:
            parsed = json.loads(text[start : end + 1])
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            intent_value = (
                parsed.get("intent")
                or parsed.get("caller_intent")
                or parsed.get("predicted_intent")
                or ""
            )
            outcome_value = parsed.get("outcome") or parsed.get("call_outcome") or ""
            score = (
                parsed.get("propensity_score")
                if parsed.get("propensity_score") is not None
                else parsed.get("buy_propensity_score")
            )
            try:
                score = float(score) if score is not None else None
            except (TypeError, ValueError):
                score = None
            return {
                "intent": _normalize_intent(str(intent_value).strip()),
                "outcome": str(outcome_value).strip(),
                "propensity_score": score,
            }

    # Fallback for non-JSON responses (common on untuned/base deployments).
    fallback_intent = _extract_intent_from_text(text)
    fallback_score = _extract_propensity_from_text(text)
    if fallback_intent or fallback_score is not None:
        return {
            "intent": fallback_intent,
            "outcome": "",
            "propensity_score": fallback_score,
        }
    return empty


def load_labeled_validation(path: str | Path) -> list[dict[str, Any]]:
    """Load the chat-format validation JSONL into labeled examples.

    Returns one dict per row with ``transcript`` (the user message) and the
    parsed ground-truth ``intent`` / ``outcome`` / ``propensity_score``. Rows
    without a user+assistant message pair are skipped.
    """
    examples: list[dict[str, Any]] = []
    text = Path(path).read_text(encoding="utf-8-sig")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        row = json.loads(stripped)
        messages = row.get("messages", [])
        user = next((m for m in messages if m.get("role") == "user"), None)
        assistant = next((m for m in messages if m.get("role") == "assistant"), None)
        if user is None or assistant is None:
            continue
        truth = parse_label(assistant.get("content"))
        examples.append(
            {
                "transcript": user.get("content", ""),
                "intent": truth["intent"],
                "outcome": truth["outcome"],
                "propensity_score": truth["propensity_score"],
            }
        )
    logger.info("Loaded %d labeled validation example(s) from %s", len(examples), path)
    return examples


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------
def score_candidate(
    config: DemoConfig,
    deployment_name: str,
    examples: list[dict[str, Any]],
    *,
    client: Any | None = None,
    request_delay: float = 0.0,
) -> dict[str, Any]:
    """Run one deployment over the examples and compute a scorecard.

    Returns a dict with the candidate ``name`` (deployment), the per-row
    ``predictions`` (bring-your-own-predictions shape), and aggregate
    ``intent_accuracy`` / ``propensity_mae`` / ``count`` / ``errors`` metrics.

    A failed inference call (for example an exhausted 429 retry) degrades that
    row to an empty prediction and is counted in ``errors`` rather than
    aborting the whole run (OWASP A04 — fail safe, partial results preserved).
    ``request_delay`` paces calls to ease rate-limited deployments.
    """
    active = client if client is not None else build_client(config)
    predictions: list[dict[str, Any]] = []
    correct = 0
    errors = 0
    abs_errors: list[float] = []

    total = len(examples)
    logger.info("Scoring '%s' over %d example(s)...", deployment_name, total)
    for index, example in enumerate(examples, start=1):
        transcript = example["transcript"]
        if request_delay > 0 and index > 1:
            time.sleep(request_delay)
        logger.info(
            "[%s] example %d/%d - calling deployment...",
            deployment_name,
            index,
            total,
        )
        try:
            raw = _infer_with_backoff(config, deployment_name, transcript, client=active)
        except Exception as exc:  # noqa: BLE0001 - keep the eval going on any call failure
            errors += 1
            logger.warning(
                "[%s] example %d/%d FAILED (%s): recording empty prediction",
                deployment_name,
                index,
                total,
                exc.__class__.__name__,
            )
            raw = None
        pred = parse_label(raw)

        is_correct = bool(pred["intent"]) and pred["intent"] == example["intent"]
        if is_correct:
            correct += 1
        if pred["propensity_score"] is not None and example["propensity_score"] is not None:
            abs_errors.append(abs(pred["propensity_score"] - example["propensity_score"]))

        logger.info(
            "[%s] example %d/%d done: pred_intent=%s truth=%s (%s)",
            deployment_name,
            index,
            total,
            pred["intent"] or "<none>",
            example["intent"],
            "hit" if is_correct else "miss",
        )

        predictions.append(
            {
                "query": transcript,
                "response": pred["intent"],
                "raw_response": raw if isinstance(raw, str) else "",
                "ground_truth": example["intent"],
                "propensity_score": pred["propensity_score"]
                if pred["propensity_score"] is not None
                else 0.0,
            }
        )

    count = len(examples)
    return {
        "name": deployment_name,
        "predictions": predictions,
        "count": count,
        "errors": errors,
        "intent_accuracy": (correct / count) if count else 0.0,
        "propensity_mae": (sum(abs_errors) / len(abs_errors)) if abs_errors else None,
    }


def write_predictions(predictions: list[dict[str, Any]], path: str | Path) -> Path:
    """Write bring-your-own-predictions rows as UTF-8-with-BOM JSONL."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="\n") as fh:
        for row in predictions:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
    logger.info("Wrote %d prediction(s) to %s", len(predictions), out)
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _resilient_client(config: DemoConfig) -> Any:
    """Build the data-plane client with the SDK's own retry DISABLED.

    The quick-eval loop runs its own fast backoff (:func:`_infer_with_backoff`)
    for the transient shared-pool ``429 "Backend error"``, so the SDK's retry —
    which obeys Azure's misleading ``retry-after: 30`` — is turned off here
    (``max_retries=0``) to avoid 30-second-per-attempt stalls.

    Falls back to the plain client when the SDK lacks ``with_options`` (older
    versions or test doubles).
    """
    client = build_client(config)
    with_options = getattr(client, "with_options", None)
    if callable(with_options):
        return with_options(max_retries=CLIENT_MAX_RETRIES)
    return client


def compare_models(
    config: DemoConfig,
    val_path: str | Path,
    base_deployment: str,
    finetuned_deployment: str,
    *,
    out_dir: str | Path | None = None,
    request_delay: float = 0.0,
) -> dict[str, Any]:
    """Score the base and fine-tuned deployments on the validation holdout.

    Returns ``{"base": <scorecard>, "finetuned": <scorecard>,
    "intent_accuracy_delta": float, "examples": int}``. When ``out_dir`` is
    given, each candidate's predictions are written to
    ``preds_base.jsonl`` / ``preds_finetuned.jsonl`` for downstream offline
    metrics. ``request_delay`` paces inference calls to ease rate-limited
    deployments.
    """
    examples = load_labeled_validation(val_path)
    if not examples:
        raise ValueError(f"No labeled validation examples found in {val_path}.")

    client = _resilient_client(config)
    base = score_candidate(
        config, base_deployment, examples, client=client, request_delay=request_delay
    )
    finetuned = score_candidate(
        config, finetuned_deployment, examples, client=client, request_delay=request_delay
    )

    if out_dir is not None:
        out = Path(out_dir)
        write_predictions(base["predictions"], out / "preds_base.jsonl")
        write_predictions(finetuned["predictions"], out / "preds_finetuned.jsonl")

    unreliable = [
        candidate["name"]
        for candidate in (base, finetuned)
        if _is_unreliable(candidate)
    ]
    for name in unreliable:
        logger.warning(
            "[Act 2A1] Candidate '%s' failed >= %.0f%% of calls; its scores are "
            "unreliable (likely rate-limit/quota, not model quality). Re-run with "
            "--delay or raise the deployment's TPM quota.",
            name,
            UNRELIABLE_ERROR_FRACTION * 100,
        )

    result = {
        "base": base,
        "finetuned": finetuned,
        "intent_accuracy_delta": finetuned["intent_accuracy"] - base["intent_accuracy"],
        "examples": len(examples),
        "unreliable": unreliable,
    }

    if out_dir is not None:
        write_result(result, Path(out_dir) / RESULT_FILE)

    return result


def write_result(result: dict[str, Any], path: str | Path) -> Path:
    """Persist the aggregate comparison as JSON (per-row predictions excluded).

    The bulky ``predictions`` arrays already live in the ``preds_*.jsonl``
    files, so the summary stores only the scorecard metrics plus metadata and
    a pointer to the prediction files.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _summarize(candidate: dict[str, Any], predictions_file: str) -> dict[str, Any]:
        return {
            "name": candidate["name"],
            "count": candidate.get("count", 0),
            "errors": candidate.get("errors", 0),
            "intent_accuracy": candidate["intent_accuracy"],
            "propensity_mae": candidate["propensity_mae"],
            "predictions_file": predictions_file,
        }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "examples": result["examples"],
        "intent_accuracy_delta": result["intent_accuracy_delta"],
        "unreliable": result.get("unreliable", []),
        "base": _summarize(result["base"], "preds_base.jsonl"),
        "finetuned": _summarize(result["finetuned"], "preds_finetuned.jsonl"),
    }
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Wrote quick-eval result summary to %s", out)
    return out


def _is_unreliable(candidate: dict[str, Any]) -> bool:
    """Return True when a candidate's failed-call share meets the threshold."""
    count = candidate.get("count", 0)
    if not count:
        return False
    return (candidate.get("errors", 0) / count) >= UNRELIABLE_ERROR_FRACTION


def format_scorecard(result: dict[str, Any]) -> str:
    """Render a compact human-readable side-by-side scorecard."""
    base = result["base"]
    ft = result["finetuned"]

    def _mae(value: float | None) -> str:
        return f"{value:.3f}" if value is not None else "n/a"

    lines = [
        f"Quick eval on {result['examples']} validation example(s):",
        f"  {'candidate':<28} {'intent_acc':>10} {'prop_mae':>10} {'errors':>7}",
        f"  {base['name']:<28} {base['intent_accuracy']:>10.3f} {_mae(base['propensity_mae']):>10} {base.get('errors', 0):>7}",
        f"  {ft['name']:<28} {ft['intent_accuracy']:>10.3f} {_mae(ft['propensity_mae']):>10} {ft.get('errors', 0):>7}",
        f"  intent accuracy delta (finetuned - base): {result['intent_accuracy_delta']:+.3f}",
    ]
    unreliable = result.get("unreliable") or []
    if unreliable:
        lines.append(
            f"  WARNING: unreliable (>= {UNRELIABLE_ERROR_FRACTION * 100:.0f}% calls failed): "
            f"{', '.join(unreliable)}"
        )
    return "\n".join(lines)
