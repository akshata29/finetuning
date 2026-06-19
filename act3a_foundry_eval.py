"""Act 3A — Foundry cloud evaluation showcase across every tuned model.

This module turns the demo's labeled validation holdout into **Foundry portal
evaluation runs** so the side-by-side lift of each fine-tuning method (base vs
SFT vs DPO vs RFT) shows up in the Azure AI Foundry **Evaluations** tab, ready
to present live.

How it works
------------
1. For each selected deployment, the labeled validation set is replayed through
   the live deployment (reusing :mod:`finetuning_demo.act2a1_quick_eval`'s
   resilient scorer) to produce a per-row prediction dataset.
2. That dataset is scored by a small evaluator suite:

   * ``intent_match`` — custom code evaluator: 1.0 when the predicted intent
     equals the ground-truth intent (uses the same normalization as quick-eval
     so the numbers line up).
   * ``propensity_error`` — custom code evaluator: absolute error of the
     predicted buy-propensity vs ground truth.
   * ``f1_score`` — the built-in Foundry :class:`F1ScoreEvaluator` (token-level
     overlap of the predicted vs ground-truth label), included so the run also
     carries a recognizable catalog metric in the portal.

3. When an Azure AI project is configured, :func:`azure.ai.evaluation.evaluate`
   uploads each run to the Foundry portal under a descriptive name
   (``sales-intent-eval-<model>``), where it appears in the Evaluations tab.

Import safety (OWASP A06): this module imports with **zero Azure SDKs** present.
The evaluator classes are dependency-free; ``azure-ai-evaluation`` is imported
lazily only when an evaluation is actually run.

Note: this module deliberately does **not** use ``from __future__ import
annotations``. Promptflow (used by ``azure-ai-evaluation`` to run custom
evaluators) introspects the evaluators' real parameter type hints; stringized
(PEP 563) annotations make it reject ``response``/``ground_truth`` as "complex
types." Concrete annotations work at runtime on Python 3.10+.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DemoConfig

logger = logging.getLogger(__name__)

#: Friendly labels -> the ``DemoConfig`` attribute holding each deployment name.
DEPLOYMENT_ATTR_BY_MODEL: dict[str, str] = {
    "base": "base_deployment_name",
    "sft": "sft_deployment_name",
    "dpo": "dpo_deployment_name",
    "rft": "rft_deployment_name",
}

#: Default model arms to evaluate, in presentation order (baseline first).
DEFAULT_MODELS: tuple[str, ...] = ("base", "sft", "dpo", "rft")

#: Human-readable arm names for logs and the portal run name.
MODEL_DISPLAY_NAME: dict[str, str] = {
    "base": "Base (un-tuned)",
    "sft": "Supervised FT",
    "dpo": "Preference (DPO)",
    "rft": "Reinforcement (RFT)",
}


# ---------------------------------------------------------------------------
# Custom code evaluators (dependency-free callables)
# ---------------------------------------------------------------------------
class IntentMatchEvaluator:
    """Exact-match intent accuracy as a per-row Foundry evaluator.

    Returns ``{"intent_match": 1.0}`` when the predicted intent equals the
    ground-truth intent (after the shared quick-eval normalization), else
    ``0.0``. The framework averages this across rows into an accuracy metric.
    """

    def __init__(self) -> None:
        """No configuration; an explicit init lets promptflow introspect it."""

    def __call__(
        self,
        *,
        response: str = "",
        ground_truth: str = "",
        **kwargs: Any,
    ):
        from .act2a1_quick_eval import _normalize_intent  # noqa: PLC0415

        pred = _normalize_intent((response or "").strip())
        truth = _normalize_intent((ground_truth or "").strip())
        return {"intent_match": 1.0 if pred and pred == truth else 0.0}


class PropensityErrorEvaluator:
    """Absolute error of the predicted buy-propensity vs ground truth.

    Returns ``{"propensity_abs_error": |pred - truth|}``. Non-numeric or missing
    values degrade to ``0.0`` rather than raising (OWASP A04 — fail safe), so a
    single malformed row never aborts the evaluation.
    """

    def __init__(self) -> None:
        """No configuration; an explicit init lets promptflow introspect it."""

    def __call__(
        self,
        *,
        propensity_pred: float = 0.0,
        propensity_truth: float = 0.0,
        **kwargs: Any,
    ):
        try:
            pred = float(propensity_pred)
            truth = float(propensity_truth)
        except (TypeError, ValueError):
            return {"propensity_abs_error": 0.0}
        return {"propensity_abs_error": abs(pred - truth)}


# ---------------------------------------------------------------------------
# Model / project resolution
# ---------------------------------------------------------------------------
def resolve_deployments(
    config: DemoConfig, models: list[str] | tuple[str, ...]
) -> list[tuple[str, str]]:
    """Map requested model labels to ``(label, deployment_name)`` pairs.

    Unknown labels and labels whose deployment name is unset in the config are
    skipped with a warning, so a partially configured environment (for example
    RFT not yet deployed) still evaluates the arms that exist.
    """
    resolved: list[tuple[str, str]] = []
    for label in models:
        attr = DEPLOYMENT_ATTR_BY_MODEL.get(label)
        if attr is None:
            logger.warning("[Act 3A] Unknown model '%s' (skipped)", label)
            continue
        deployment = getattr(config, attr, "")
        if not deployment:
            logger.warning(
                "[Act 3A] No deployment configured for '%s' (set the matching "
                "env var); skipping",
                label,
            )
            continue
        resolved.append((label, deployment))
    return resolved


def build_azure_ai_project(config: DemoConfig) -> str | dict[str, str] | None:
    """Resolve the ``azure_ai_project`` value that triggers a Foundry upload.

    For a hub-less Azure AI **Foundry (FDP) project**, ``azure.ai.evaluation.evaluate``
    (>= 1.4.0) accepts the **project endpoint string**
    (``https://<account>.services.ai.azure.com/api/projects/<project>``) and uploads
    the run to the Foundry **Evaluations** tab. This is the path this demo uses.

    Falls back to the legacy ``{subscription_id, resource_group_name, project_name}``
    dict only when no endpoint is set (that dict path targets an AML
    ``Microsoft.MachineLearningServices/workspaces`` resource and does not resolve
    hub-less Foundry projects). Returns ``None`` when neither is configured.
    """
    endpoint = config.azure_ai_project_endpoint
    if endpoint:
        return endpoint

    subscription = config.azure_subscription_id
    resource_group = config.azure_resource_group
    project_name = config.azure_ai_project_name
    if not (subscription and resource_group and project_name):
        return None
    return {
        "subscription_id": subscription,
        "resource_group_name": resource_group,
        "project_name": project_name,
    }


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------
def build_eval_dataset(
    config: DemoConfig,
    deployment_name: str,
    val_path: str | Path,
    out_path: str | Path,
    *,
    limit: int | None = None,
    request_delay: float = 0.0,
    client: Any | None = None,
) -> dict[str, Any]:
    """Replay the validation holdout through a deployment into an eval dataset.

    Reuses the quick-eval scorer so predictions match the rest of the demo, then
    writes one JSONL row per example with the columns the evaluators consume:
    ``query`` (transcript), ``response`` (predicted intent), ``ground_truth``
    (true intent), ``propensity_pred`` and ``propensity_truth``.

    Returns a small summary dict (``rows``, ``errors``, ``path``). ``limit``
    subsamples the holdout for a fast live demo; ``request_delay`` paces calls
    to ease rate-limited deployments.
    """
    import json  # noqa: PLC0415

    from . import act2a1_quick_eval as quick  # noqa: PLC0415

    examples = quick.load_labeled_validation(val_path)
    if limit is not None and limit > 0:
        examples = examples[:limit]
    if not examples:
        raise ValueError(f"No labeled validation examples found in {val_path}.")

    active = client if client is not None else quick._resilient_client(config)
    scored = quick.score_candidate(
        config, deployment_name, examples, client=active, request_delay=request_delay
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for example, prediction in zip(examples, scored["predictions"]):
            row = {
                "query": example["transcript"],
                "response": prediction["response"],
                "ground_truth": example["intent"],
                "propensity_pred": prediction["propensity_score"],
                "propensity_truth": example["propensity_score"]
                if example["propensity_score"] is not None
                else 0.0,
            }
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    logger.info(
        "[Act 3A] Wrote %d eval row(s) for '%s' to %s (%d call error(s))",
        len(examples),
        deployment_name,
        out,
        scored["errors"],
    )
    return {"rows": len(examples), "errors": scored["errors"], "path": str(out)}


# ---------------------------------------------------------------------------
# Evaluator suite
# ---------------------------------------------------------------------------
def build_evaluators(*, include_builtin: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    """Construct the evaluator suite and its column mapping.

    Returns ``(evaluators, evaluator_config)`` ready to pass to
    :func:`azure.ai.evaluation.evaluate`. The two custom code evaluators are
    always present; the built-in :class:`F1ScoreEvaluator` is added when
    ``include_builtin`` is True (it needs no model deployment, so it is safe to
    include by default).
    """
    evaluators: dict[str, Any] = {
        "intent_match": IntentMatchEvaluator(),
        "propensity_error": PropensityErrorEvaluator(),
    }
    evaluator_config: dict[str, Any] = {
        "intent_match": {
            "column_mapping": {
                "response": "${data.response}",
                "ground_truth": "${data.ground_truth}",
            }
        },
        "propensity_error": {
            "column_mapping": {
                "propensity_pred": "${data.propensity_pred}",
                "propensity_truth": "${data.propensity_truth}",
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

    return evaluators, evaluator_config


# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------
def run_foundry_eval(
    config: DemoConfig,
    label: str,
    deployment_name: str,
    val_path: str | Path,
    out_dir: str | Path,
    *,
    limit: int | None = None,
    request_delay: float = 0.0,
    include_builtin: bool = True,
    upload: bool = True,
    client: Any | None = None,
) -> dict[str, Any]:
    """Build the dataset for one model and run (and optionally upload) the eval.

    When ``upload`` is True and an Azure AI project is configured, the run is
    pushed to the Foundry portal under ``sales-intent-eval-<label>`` and appears
    in the Evaluations tab. Otherwise it runs locally and writes results to
    ``out_dir``. Returns a result summary with metrics and (when uploaded) the
    portal run id / URL.
    """
    from azure.ai.evaluation import evaluate  # noqa: PLC0415

    out_dir = Path(out_dir)
    dataset_path = out_dir / f"foundry_eval_data_{label}.jsonl"
    result_path = out_dir / f"foundry_eval_result_{label}.json"

    dataset = build_eval_dataset(
        config,
        deployment_name,
        val_path,
        dataset_path,
        limit=limit,
        request_delay=request_delay,
        client=client,
    )

    evaluators, evaluator_config = build_evaluators(include_builtin=include_builtin)

    azure_ai_project = build_azure_ai_project(config) if upload else None
    if upload and azure_ai_project is None:
        logger.warning(
            "[Act 3A] No Azure AI project configured (need AZURE_SUBSCRIPTION_ID, "
            "AZURE_RESOURCE_GROUP, and AZURE_AI_PROJECT_NAME/ENDPOINT); running "
            "'%s' locally without portal upload.",
            label,
        )

    evaluation_name = f"sales-intent-eval-{label}"
    display = MODEL_DISPLAY_NAME.get(label, label)
    logger.info(
        "[Act 3A] Evaluating %s arm '%s' (%d row(s)) as '%s'%s",
        display,
        deployment_name,
        dataset["rows"],
        evaluation_name,
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
    summary = {
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
    if studio_url:
        logger.info("[Act 3A] %s -> portal run: %s", display, studio_url)
    return summary


# ---------------------------------------------------------------------------
# Multi-model orchestration
# ---------------------------------------------------------------------------
def run_all_foundry_evals(
    config: DemoConfig,
    val_path: str | Path,
    out_dir: str | Path,
    *,
    models: list[str] | tuple[str, ...] = DEFAULT_MODELS,
    limit: int | None = None,
    request_delay: float = 0.0,
    include_builtin: bool = True,
    upload: bool = True,
) -> dict[str, Any]:
    """Evaluate every configured model arm and return a combined report.

    Skips arms whose deployment name is unset (e.g. RFT before it is deployed),
    reuses a single resilient client across arms, and writes a combined summary
    JSON to ``out_dir``. Each arm that fails to evaluate is recorded rather than
    aborting the whole showcase.
    """
    import json  # noqa: PLC0415

    from . import act2a1_quick_eval as quick  # noqa: PLC0415

    pairs = resolve_deployments(config, models)
    if not pairs:
        raise ValueError(
            "No evaluatable deployments: set BASE/SFT/DPO/RFT_DEPLOYMENT_NAME."
        )

    out_dir = Path(out_dir)
    client = quick._resilient_client(config)
    runs: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for label, deployment in pairs:
        try:
            runs.append(
                run_foundry_eval(
                    config,
                    label,
                    deployment,
                    val_path,
                    out_dir,
                    limit=limit,
                    request_delay=request_delay,
                    include_builtin=include_builtin,
                    upload=upload,
                    client=client,
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep the showcase going per arm
            logger.error(
                "[Act 3A] Evaluation FAILED for '%s' (%s): %s",
                label,
                deployment,
                exc,
            )
            failures.append({"label": label, "deployment": deployment, "error": str(exc)})

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_name": config.azure_ai_project_name,
        "examples": runs[0]["rows"] if runs else 0,
        "uploaded": any(run.get("uploaded") for run in runs),
        "runs": runs,
        "failures": failures,
    }

    report_path = out_dir / "foundry_eval_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("[Act 3A] Wrote combined Foundry eval report to %s", report_path)
    return report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def format_report(report: dict[str, Any]) -> str:
    """Render a compact, presentation-ready scoreboard of all model arms."""
    runs = report.get("runs", [])
    lines: list[str] = []
    lines.append("Foundry evaluation scoreboard (intent accuracy / propensity MAE)")
    lines.append("=" * 64)
    header = f"{'Model':<22}{'intent_acc':>12}{'prop_mae':>12}{'f1':>10}"
    lines.append(header)
    lines.append("-" * 64)
    for run in runs:
        metrics = run.get("metrics", {})
        acc = _metric(metrics, "intent_match")
        mae = _metric(metrics, "propensity_abs_error")
        f1 = _metric(metrics, "f1_score")
        lines.append(
            f"{run.get('display_name', run.get('label', '?')):<22}"
            f"{_fmt(acc):>12}{_fmt(mae):>12}{_fmt(f1):>10}"
        )
    lines.append("-" * 64)
    if report.get("uploaded"):
        lines.append("Runs uploaded to the Foundry portal -> Build > Evaluations.")
        for run in runs:
            if run.get("studio_url"):
                lines.append(f"  {run['display_name']}: {run['studio_url']}")
    else:
        lines.append("Local-only run (no portal upload). Results in out-dir JSON.")
    failures = report.get("failures", [])
    if failures:
        lines.append("")
        lines.append("Failed arms:")
        for failure in failures:
            lines.append(f"  {failure['label']} ({failure['deployment']}): {failure['error']}")
    return "\n".join(lines)


def _metric(metrics: dict[str, Any], base_name: str) -> float | None:
    """Look up an aggregated metric value across evaluate()'s naming variants.

    :func:`azure.ai.evaluation.evaluate` aggregates per-row outputs under keys
    like ``"<evaluator>.<metric>"`` (and sometimes a ``..._mean`` suffix); this
    finds the first matching numeric value.
    """
    candidates = [
        base_name,
        f"intent_match.{base_name}",
        f"propensity_error.{base_name}",
        f"f1_score.{base_name}",
        f"{base_name}.{base_name}",
    ]
    for key in candidates:
        if key in metrics and isinstance(metrics[key], (int, float)):
            return float(metrics[key])
    # Fallback: first key that ends with the metric name.
    for key, value in metrics.items():
        if key.endswith(base_name) and isinstance(value, (int, float)):
            return float(value)
    return None


def _fmt(value: float | None) -> str:
    """Format a metric for the scoreboard (``n/a`` when missing)."""
    return f"{value:.3f}" if isinstance(value, float) else "n/a"
