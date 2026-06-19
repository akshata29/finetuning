"""Act 3 — Evals API 3-way classification runs (bring-your-own-predictions).

Builds two Azure OpenAI Evals definitions that share the SAME three candidates
(``base`` / ``finetuned`` / ``optimized-prompt``) on identical uploaded
prediction JSONL files:

* a **binary propensity** eval (``buy`` vs ``not_buy``) whose ``label_model``
  grader uses ``passing_labels=["buy"]`` to define the single positive class; and
* a **multiclass intent** eval whose ``label_model`` grader spans the full
  :data:`finetuning.taxonomy.INTENT_LABELS` set with no single positive
  class (DR-09).

The Evals API surface is the OpenAI Evals API reached through
``azure-ai-projects`` (``AIProjectClient.get_openai_client()``). Those SDKs are
optional: this module imports cleanly with no Azure SDKs installed. The grader
and data-source payloads are emitted as plain dicts (which the Evals API accepts
freely) so building them never requires a live client.

Aggregate metrics (precision/recall/F1/AUC/PR-AUC/lift/macro-F1) are computed
OFFLINE from the prediction JSONL — see :mod:`finetuning.offline_metrics`.
The Evals run provides only per-criterion ``pass_rate`` plus the portal
``report_url`` (DR-06).
"""

from __future__ import annotations

import logging
from typing import Any

from finetuning.config import DemoConfig, optional_import
from finetuning.taxonomy import INTENT_LABELS, PROPENSITY_LABELS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grader field constants (reference the bring-your-own-predictions row fields)
# ---------------------------------------------------------------------------
_RESPONSE_REF: str = "{{item.response}}"
_GROUND_TRUTH_REF: str = "{{item.ground_truth}}"
_QUERY_REF: str = "{{item.query}}"

#: Positive class for the binary propensity arm.
POSITIVE_LABEL: str = "buy"


# ---------------------------------------------------------------------------
# Data source config (BYO predictions, custom item schema)
# ---------------------------------------------------------------------------
def byo_data_source_config() -> dict[str, Any]:
    """Return the ``custom`` data-source config for bring-your-own predictions.

    Declares the per-row item fields present in the uploaded prediction JSONL:
    ``query`` (transcript), ``response`` (predicted label), ``ground_truth``
    (true label), and an optional ``propensity_score`` float captured during
    batch inference for the offline AUC/lift math. ``include_sample_schema`` is
    intentionally omitted because predictions come from the dataset and are
    referenced as ``{{item.response}}`` (not ``{{sample.output_text}}``).
    """
    return {
        "type": "custom",
        "item_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "response": {"type": "string"},
                "ground_truth": {"type": "string"},
                "propensity_score": {"type": "number"},
            },
            "required": ["query", "response", "ground_truth"],
        },
    }


# ---------------------------------------------------------------------------
# Testing criteria builders
# ---------------------------------------------------------------------------
def _string_check(name: str) -> dict[str, Any]:
    """Deterministic predicted-label == ground-truth grader (pass_rate==accuracy)."""
    return {
        "type": "string_check",
        "name": name,
        "input": _RESPONSE_REF,
        "reference": _GROUND_TRUTH_REF,
        "operation": "eq",
    }


def _f1_score_evaluator() -> dict[str, Any]:
    """Microsoft built-in token-overlap F1 (wrapped as ``azure_ai_evaluator``)."""
    return {
        "type": "azure_ai_evaluator",
        "name": "f1",
        "evaluator_name": "builtin.f1_score",
        "data_mapping": {
            "response": _RESPONSE_REF,
            "ground_truth": _GROUND_TRUTH_REF,
        },
    }


def build_propensity_criteria(grader_model: str) -> list[dict[str, Any]]:
    """Return the BINARY propensity testing criteria (DR-09).

    Three graders attach to one eval definition:

    #. ``string_check`` — exact predicted==truth label match (accuracy spine).
    #. ``label_model`` — LLM grader with ``passing_labels=["buy"]`` defining the
       single positive class for pass/fail.
    #. ``builtin.f1_score`` — token-overlap F1 aggregated to a pass_rate.

    Parameters
    ----------
    grader_model:
        Deployment name of the judge/grader chat model (e.g. ``gpt-4.1``).
    """
    label_grader = {
        "type": "label_model",
        "name": "buy_label_grader",
        "model": grader_model,
        "input": [
            {
                "role": "developer",
                "content": (
                    "Classify the sales call propensity as one of 'buy' or "
                    "'not_buy' based on the transcript."
                ),
            },
            {
                "role": "user",
                "content": f"Transcript: {_QUERY_REF}\nModel said: {_RESPONSE_REF}",
            },
        ],
        "labels": list(PROPENSITY_LABELS),
        "passing_labels": [POSITIVE_LABEL],
    }
    return [_string_check("exact_label_match"), label_grader, _f1_score_evaluator()]


def build_intent_criteria(grader_model: str) -> list[dict[str, Any]]:
    """Return the MULTICLASS intent testing criteria (DR-09).

    Two graders attach to one eval definition:

    #. ``string_check`` — exact predicted-intent == ground-truth-intent match.
    #. ``label_model`` — LLM grader spanning the FULL
       :data:`finetuning.taxonomy.INTENT_LABELS` set, with no single
       positive class (``passing_labels`` covers every intent so the grader
       classifies rather than gating on one positive label).

    Parameters
    ----------
    grader_model:
        Deployment name of the judge/grader chat model (e.g. ``gpt-4.1``).
    """
    intent_contract = ", ".join(f"'{label}'" for label in INTENT_LABELS)
    label_grader = {
        "type": "label_model",
        "name": "intent_label_grader",
        "model": grader_model,
        "input": [
            {
                "role": "developer",
                "content": (
                    "Classify the caller intent as exactly one of the following "
                    f"labels: {intent_contract}."
                ),
            },
            {
                "role": "user",
                "content": f"Transcript: {_QUERY_REF}\nModel said: {_RESPONSE_REF}",
            },
        ],
        "labels": list(INTENT_LABELS),
        # Multiclass: every intent is an accepted label (no single positive class).
        "passing_labels": list(INTENT_LABELS),
    }
    return [_string_check("exact_intent_match"), label_grader]


# ---------------------------------------------------------------------------
# Evals client + eval/run lifecycle
# ---------------------------------------------------------------------------
def build_openai_client(config: DemoConfig | None = None) -> Any:
    """Build the OpenAI Evals client via ``azure-ai-projects`` (optional SDK).

    Resolves ``azure.ai.projects`` and ``azure.identity`` lazily through
    :func:`finetuning.config.optional_import` so this module imports with
    no Azure SDKs present. Raises an actionable error only when invoked without
    the SDKs or without a configured project endpoint.

    Parameters
    ----------
    config:
        Demo configuration; defaults to :meth:`DemoConfig.from_env`.

    Returns
    -------
    Any
        The OpenAI client exposing ``client.evals.*``.

    Raises
    ------
    RuntimeError
        If the required SDKs are missing or the project endpoint is unset.
    """
    cfg = config or DemoConfig.from_env()
    projects, projects_ok = optional_import("azure.ai.projects")
    identity, identity_ok = optional_import("azure.identity")
    if not (projects_ok and identity_ok):
        raise RuntimeError(
            "Live Evals require 'azure-ai-projects' and 'azure-identity'. "
            "Install with: pip install 'azure-ai-projects==1.0.0' azure-identity"
        )
    if not cfg.azure_ai_project_endpoint:
        raise RuntimeError(
            "AZURE_AI_PROJECT_ENDPOINT is not set; cannot reach the Evals API."
        )
    project_client = projects.AIProjectClient(
        endpoint=cfg.azure_ai_project_endpoint,
        credential=identity.DefaultAzureCredential(),
    )
    return project_client.get_openai_client()


def create_eval(
    client: Any,
    name: str,
    testing_criteria: list[dict[str, Any]],
    data_source_config: dict[str, Any] | None = None,
) -> Any:
    """Create an Evals definition with the given criteria.

    Parameters
    ----------
    client:
        OpenAI Evals client (``client.evals.create``).
    name:
        Human-readable eval name, e.g. ``"sales-propensity-3way"``.
    testing_criteria:
        Output of :func:`build_propensity_criteria` or
        :func:`build_intent_criteria`.
    data_source_config:
        Custom item-schema config; defaults to :func:`byo_data_source_config`.
    """
    return client.evals.create(
        name=name,
        data_source_config=data_source_config or byo_data_source_config(),
        testing_criteria=testing_criteria,
    )


def run_candidate(client: Any, eval_id: str, name: str, file_id: str) -> Any:
    """Create one eval run for a single candidate's uploaded prediction file.

    Uses the bring-your-own-predictions ``jsonl`` data source pointing at an
    uploaded ``file_id`` (one file per candidate; identical rows except the
    ``response``/``propensity_score`` columns).

    Parameters
    ----------
    client:
        OpenAI Evals client (``client.evals.runs.create``).
    eval_id:
        Identifier returned by :func:`create_eval`.
    name:
        Candidate name (``"base"``/``"finetuned"``/``"optimized-prompt"``).
    file_id:
        Uploaded dataset/file id of the candidate's prediction JSONL.
    """
    return client.evals.runs.create(
        eval_id=eval_id,
        name=f"{name}-run",
        metadata={"candidate": name},
        data_source={
            "type": "jsonl",
            "source": {"type": "file_id", "id": file_id},
        },
    )


def _criterion_entries(result: Any) -> list[dict[str, Any]]:
    """Normalize ``per_testing_criteria_results`` to a list of plain dicts."""
    raw = _get(result, "per_testing_criteria_results") or []
    entries: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            entries.append(dict(item))
        else:
            entries.append(
                {
                    "name": _get(item, "name"),
                    "passed": _get(item, "passed"),
                    "failed": _get(item, "failed"),
                    "pass_rate": _get(item, "pass_rate"),
                }
            )
    return entries


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict or attribute-style object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def read_results(client: Any, eval_id: str, run_id: str) -> dict[str, Any]:
    """Retrieve a finished run and return its aggregate results.

    Parameters
    ----------
    client:
        OpenAI Evals client (``client.evals.runs.retrieve``).
    eval_id:
        Identifier returned by :func:`create_eval`.
    run_id:
        Identifier returned by :func:`run_candidate`.

    Returns
    -------
    dict[str, Any]
        ``{"status", "result_counts", "per_testing_criteria_results",
        "report_url"}``; ``per_testing_criteria_results`` is normalized to a list
        of plain dicts with at least ``name`` and ``pass_rate``.
    """
    run = client.evals.runs.retrieve(run_id=run_id, eval_id=eval_id)
    return {
        "status": _get(run, "status"),
        "result_counts": _get(run, "result_counts"),
        "per_testing_criteria_results": _criterion_entries(run),
        "report_url": _get(run, "report_url"),
    }
