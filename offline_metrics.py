"""Offline aggregate classification metrics for the Act 3 3-way comparison.

The Azure Evals run only surfaces per-criterion ``pass_rate`` plus a portal
``report_url`` (DR-06); the headline aggregate metrics are computed here, OFFLINE,
from the prediction JSONL — the single source of truth shared with
:mod:`finetuning_demo.act3_evaluation`.

Two metric families are provided, never interchanged (DR-09):

* :func:`propensity_metrics` — BINARY ``buy`` vs ``not_buy`` at ~1-2% prevalence:
  precision/recall/F1/accuracy, confusion matrix oriented ``[[TP,FN],[FP,TN]]``,
  ROC-AUC, and **PR-AUC + top-decile lift** as the headline metrics for the rare
  positive class (accuracy is ~98% for a trivial all-negative model and misleads).
* :func:`intent_metrics` — MULTICLASS intent: **macro-F1** (``average="macro"``),
  per-class precision/recall via ``classification_report``, and a multiclass
  confusion matrix labeled in :data:`finetuning_demo.taxonomy.INTENT_LABELS`
  order.

``scikit-learn`` (and ``numpy``) are optional. This module imports cleanly when
they are absent; the metric functions then raise an actionable :class:`ImportError`
when invoked.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from finetuning_demo.taxonomy import INTENT_LABELS, PROPENSITY_LABELS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional scientific-stack imports (graceful degradation)
# ---------------------------------------------------------------------------
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without numpy
    np = None  # type: ignore[assignment]
    NUMPY_AVAILABLE = False
    logger.warning(
        "numpy is not installed. offline_metrics math will raise ImportError. "
        "Install with: pip install numpy"
    )

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without sklearn
    SKLEARN_AVAILABLE = False
    logger.warning(
        "scikit-learn is not installed. offline_metrics functions will raise "
        "ImportError. Install with: pip install scikit-learn"
    )

#: Positive class for the binary propensity arm; first entry of PROPENSITY_LABELS
#: so the confusion matrix is oriented [[TP, FN], [FP, TN]].
POSITIVE_LABEL: str = PROPENSITY_LABELS[0]


def _require_sklearn() -> None:
    """Raise an actionable :class:`ImportError` if the science stack is absent."""
    missing = []
    if not NUMPY_AVAILABLE:
        missing.append("numpy")
    if not SKLEARN_AVAILABLE:
        missing.append("scikit-learn")
    if missing:
        joined = " ".join(missing)
        raise ImportError(
            f"offline_metrics requires {', '.join(missing)} but it is not "
            f"installed. Install with: pip install {joined}"
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load(path: str | Path) -> tuple[list[str], list[str], list[float]]:
    """Read a prediction JSONL into ``(y_true, y_pred, y_score)``.

    Each line is a bring-your-own-predictions row with ``ground_truth`` (true
    label), ``response`` (predicted label), and an optional ``propensity_score``
    float (defaults to ``0.0`` when absent — e.g. the intent arm). The file is
    read as UTF-8-with-BOM (``utf-8-sig``) so the BOM written by the Phase 1
    JSONL writers is transparently stripped.

    Parameters
    ----------
    path:
        Path to the prediction JSONL file.

    Returns
    -------
    tuple[list[str], list[str], list[float]]
        Parallel lists of true labels, predicted labels, and scores.
    """
    y_true: list[str] = []
    y_pred: list[str] = []
    y_score: list[float] = []
    with Path(path).open(encoding="utf-8-sig") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            y_true.append(str(row["ground_truth"]).strip())
            y_pred.append(str(row["response"]).strip())
            # ``propensity_score`` may be absent or an explicit JSON ``null``
            # (the intent arm omits it); both coerce to 0.0.
            y_score.append(float(row.get("propensity_score") or 0.0))
    return y_true, y_pred, y_score


# ---------------------------------------------------------------------------
# Binary propensity metrics
# ---------------------------------------------------------------------------
def propensity_metrics(path: str | Path, name: str) -> dict[str, Any]:
    """Compute BINARY ``buy``/``not_buy`` metrics for one candidate.

    PR-AUC (average precision) and top-decile lift are the headline metrics for
    the ~1-2% positive prevalence; ROC-AUC and PR-AUC are guarded to ``nan`` when
    the uploaded slice is single-class. The confusion matrix is oriented
    ``[[TP, FN], [FP, TN]]`` (positive class ``"buy"`` first).

    Parameters
    ----------
    path:
        Prediction JSONL path.
    name:
        Candidate name recorded in the result dict.

    Raises
    ------
    ImportError
        If ``numpy``/``scikit-learn`` are not installed.
    """
    _require_sklearn()
    y_true, y_pred, y_score = load(path)
    yt = np.array([1 if label == POSITIVE_LABEL else 0 for label in y_true])
    scores = np.asarray(y_score, dtype=float)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=POSITIVE_LABEL, zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)
    # labels=[buy, not_buy] -> [[TP, FN], [FP, TN]]
    cm = confusion_matrix(y_true, y_pred, labels=list(PROPENSITY_LABELS))

    single_class = len(set(yt.tolist())) < 2
    roc_auc = float("nan") if single_class else float(roc_auc_score(yt, scores))
    pr_auc = float("nan") if single_class else float(average_precision_score(yt, scores))

    # Top-decile lift: positive rate in the top 10% by score / overall base rate.
    base_rate = float(yt.mean()) if yt.size else 0.0
    if yt.size and base_rate > 0:
        order = np.argsort(-scores)
        k = max(1, yt.size // 10)
        top_rate = float(yt[order[:k]].mean())
        lift = top_rate / base_rate
    else:
        lift = float("nan")

    return {
        "candidate": name,
        "kind": "propensity",
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "top_decile_lift": lift,
        "confusion_matrix[[TP,FN],[FP,TN]]": cm.tolist(),
        "positive_rate": base_rate,
    }


# ---------------------------------------------------------------------------
# Multiclass intent metrics
# ---------------------------------------------------------------------------
def intent_metrics(path: str | Path, name: str) -> dict[str, Any]:
    """Compute MULTICLASS intent metrics for one candidate.

    Produces macro-F1 (``average="macro"``), a per-class precision/recall
    ``classification_report`` (as a dict), and a multiclass confusion matrix —
    all labeled in :data:`finetuning_demo.taxonomy.INTENT_LABELS` order.

    Parameters
    ----------
    path:
        Prediction JSONL path.
    name:
        Candidate name recorded in the result dict.

    Raises
    ------
    ImportError
        If ``numpy``/``scikit-learn`` are not installed.
    """
    _require_sklearn()
    y_true, y_pred, _ = load(path)
    labels = list(INTENT_LABELS)

    macro_f1 = float(
        f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "candidate": name,
        "kind": "intent",
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Multi-candidate comparison
# ---------------------------------------------------------------------------
def compare(
    paths: dict[str, str | Path] | list[str | Path],
    kind: str,
) -> list[dict[str, Any]]:
    """Compute the chosen metric family for several candidates.

    Parameters
    ----------
    paths:
        Either a mapping of ``candidate_name -> jsonl_path`` or a list of paths
        (candidate names are derived from each file stem).
    kind:
        ``"propensity"`` for binary metrics or ``"intent"`` for multiclass.

    Returns
    -------
    list[dict[str, Any]]
        One metric dict per candidate, in input order.

    Raises
    ------
    ValueError
        If ``kind`` is not ``"propensity"`` or ``"intent"``.
    ImportError
        If ``numpy``/``scikit-learn`` are not installed.
    """
    if kind == "propensity":
        metric_fn = propensity_metrics
    elif kind == "intent":
        metric_fn = intent_metrics
    else:
        raise ValueError(
            f"kind must be 'propensity' or 'intent', got {kind!r}"
        )

    if isinstance(paths, dict):
        items = list(paths.items())
    else:
        items = [(Path(p).stem, p) for p in paths]

    return [metric_fn(path, name) for name, path in items]
