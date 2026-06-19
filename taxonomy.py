"""Label taxonomy and seed cross-product for synthetic sales-call generation.

Pure-Python, dependency-free, and fully unit-testable. Encodes the
classification targets (multiclass ``intent``, ``outcome`` enum, and the binary
``buy``/``not_buy`` propensity label) plus the generation cross-product used to
seed the teacher model.

Two generation modes are supported by :func:`iter_seeds`:

* ``"train"`` — intent classes are balanced (round-robin) and the binary
  propensity label is split ~50/50, producing a learnable training set.
* ``"eval"`` — ``buy`` is rare (~1-2%, controlled by ``eval_prevalence``) so the
  evaluation set mirrors the real-world class imbalance for AUC / PR-AUC / lift.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Classification label sets
# ---------------------------------------------------------------------------
INTENT_LABELS: list[str] = [
    "pricing_inquiry",
    "feature_request",
    "competitor_comparison",
    "objection_handling",
    "scheduling_followup",
    "support_escalation",
    "general_interest",
]

OUTCOME_LABELS: list[str] = [
    "closed_won",
    "closed_lost",
    "follow_up_scheduled",
    "no_decision",
    "disqualified",
]

PROPENSITY_LABELS: list[str] = ["buy", "not_buy"]

# ---------------------------------------------------------------------------
# Generation cross-product (intent × outcome × industry × persona ×
# deal-size × objection)
# ---------------------------------------------------------------------------
SEED_DIMENSIONS: dict[str, list[str]] = {
    "intent": INTENT_LABELS,
    "outcome": OUTCOME_LABELS,
    "industry": [
        "saas",
        "manufacturing",
        "healthcare",
        "financial_services",
        "retail",
        "energy",
    ],
    "persona": [
        "economic_buyer",
        "technical_evaluator",
        "end_user",
        "procurement",
        "executive_sponsor",
    ],
    "deal_size": ["smb", "mid_market", "enterprise"],
    "objection": [
        "price",
        "timing",
        "competitor",
        "integration",
        "security",
        "none",
    ],
}

_TRAIN_MODE = "train"
_EVAL_MODE = "eval"


# ---------------------------------------------------------------------------
# Seed generation
# ---------------------------------------------------------------------------
def iter_seeds(
    n: int,
    *,
    mode: str = _TRAIN_MODE,
    eval_prevalence: float = 0.015,
    seed: int = 1337,
) -> Iterator[dict[str, Any]]:
    """Yield ``n`` seed dicts for the teacher model with quota balancing.

    Parameters
    ----------
    n:
        Number of seed records to yield.
    mode:
        ``"train"`` for balanced intent classes and a ~50/50 propensity split;
        ``"eval"`` for rare ``buy`` prevalence (see ``eval_prevalence``).
    eval_prevalence:
        Target share of ``buy`` records in ``"eval"`` mode (default ~1.5%).
        Ignored in ``"train"`` mode.
    seed:
        RNG seed for deterministic, reproducible generation.

    Yields
    ------
    dict[str, Any]
        A seed dict with one value per :data:`SEED_DIMENSIONS` key plus
        ``propensity`` (``"buy"``/``"not_buy"``) and ``mode``.

    Raises
    ------
    ValueError
        If ``n`` is negative, ``mode`` is unknown, or ``eval_prevalence`` is
        outside the ``[0, 1]`` interval.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if mode not in (_TRAIN_MODE, _EVAL_MODE):
        raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")
    if not 0.0 <= eval_prevalence <= 1.0:
        raise ValueError(
            f"eval_prevalence must be in [0, 1], got {eval_prevalence}"
        )

    rng = random.Random(seed)
    intents = SEED_DIMENSIONS["intent"]

    if mode == _EVAL_MODE:
        positive_quota = round(n * eval_prevalence)
    else:  # balanced 50/50 for training
        positive_quota = n // 2

    for i in range(n):
        # Balance intent classes deterministically via round-robin.
        intent = intents[i % len(intents)]
        # Assign the positive label to the first ``positive_quota`` records,
        # then shuffle positions implicitly by interleaving in eval mode.
        if mode == _EVAL_MODE:
            propensity = "buy" if i < positive_quota else "not_buy"
        else:
            propensity = "buy" if (i % 2 == 0 and i // 2 < positive_quota) else "not_buy"

        yield {
            "intent": intent,
            "outcome": rng.choice(SEED_DIMENSIONS["outcome"]),
            "industry": rng.choice(SEED_DIMENSIONS["industry"]),
            "persona": rng.choice(SEED_DIMENSIONS["persona"]),
            "deal_size": rng.choice(SEED_DIMENSIONS["deal_size"]),
            "objection": rng.choice(SEED_DIMENSIONS["objection"]),
            "propensity": propensity,
            "mode": mode,
        }


def positive_rate(records: list[dict[str, Any]]) -> float:
    """Return the share of ``records`` labeled positive (``buy``).

    A record counts as positive when its ``propensity`` field equals ``"buy"``;
    if that field is absent, a ``propensity_score`` ``>= 0.5`` is used as a
    fallback. Returns ``0.0`` for an empty input.
    """
    if not records:
        return 0.0

    positives = 0
    for record in records:
        propensity = record.get("propensity")
        if propensity is not None:
            if propensity == "buy":
                positives += 1
            continue
        score = record.get("propensity_score")
        if isinstance(score, (int, float)) and score >= 0.5:
            positives += 1

    return positives / len(records)
