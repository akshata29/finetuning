"""Act 2A — Serverless Azure OpenAI supervised fine-tuning (SFT) lifecycle.

Implements the full "Path A" loop for the 1-Hour Azure Fine-Tuning Demo:

1. **Data-plane** (Azure OpenAI SDK against ``https://<acct>.services.ai.azure.com``):
   upload JSONL → create SFT job → poll status → list checkpoints → pick a
   checkpoint → run inference against the deployed fine-tuned model.
2. **Control-plane** (ARM, ``management.azure.com``, ``api-version=2024-10-01``):
   deploy the fine-tuned model / checkpoint and delete the deployment for
   cleanup. The deploy is deliberately an ARM ``PUT`` — *not* the data-plane
   SDK (addresses discrepancy DR-03).

Graceful degradation (OWASP A06 — vulnerable/outdated components avoided via
optional imports): this module imports cleanly with **zero Azure SDKs** present.
The ``openai`` and ``azure-identity`` packages are resolved lazily inside the
functions that need them via :func:`finetuning.config.optional_import`, and
``requests`` is resolved at import time but tolerated when absent. Every secret,
endpoint, and resource identifier is sourced from :class:`DemoConfig` — nothing
is hardcoded (OWASP A05 Security Misconfiguration).
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from collections.abc import Callable
from io import StringIO
from pathlib import Path
from typing import Any

from .config import TIER_NAMES, DemoConfig, optional_import
from .schemas import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
#: OAuth scope for ARM control-plane management tokens.
MANAGEMENT_SCOPE: str = "https://management.azure.com/.default"

#: Default serverless base model for the demo (region-pinned for training).
DEFAULT_BASE_MODEL: str = "gpt-4.1-mini-2025-04-14"

#: Reinforcement fine-tuning (RFT) is only supported on o-series reasoning
#: models (o4-mini GA; gpt-5 gated). gpt-4.1-mini is NOT eligible, so RFT jobs
#: pin this base model rather than :data:`DEFAULT_BASE_MODEL`.
RFT_BASE_MODEL: str = "o4-mini-2025-04-16"

#: Deterministic seed used so the live job reproduces the pre-baked run.
DEFAULT_SEED: int = 105

#: Two epochs balances signal vs. overfitting for the small synthetic set.
DEFAULT_N_EPOCHS: int = 2

#: Serverless fine-tuning customization methods (the portal's "Customization
#: method" dropdown: Supervised / Direct Preference Optimization / Reinforcement).
#: Serverless support for all three is the headline Azure differentiator.
METHOD_SUPERVISED: str = "supervised"
METHOD_DPO: str = "dpo"
METHOD_REINFORCEMENT: str = "reinforcement"
SERVERLESS_METHODS: tuple[str, ...] = (
    METHOD_SUPERVISED,
    METHOD_DPO,
    METHOD_REINFORCEMENT,
)

#: DPO preference-strength hyperparameter (higher = stay closer to the
#: reference model). 0.1 is the Azure/OpenAI default and a safe demo value.
DEFAULT_DPO_BETA: float = 0.1

#: Azure caps the fine-tuned model suffix at 18 characters (no dots).
MAX_SUFFIX_LENGTH: int = 18

#: The training-tier flag passed on the *job-create* call (data-plane). This is
#: separate from the *deployment* ``sku.name`` used by :func:`deploy_finetuned`.
#: Maps the :data:`DemoConfig` tier constants to the exact ``trainingType``
#: strings accepted by the fine-tuning job-create surface (research Gap 2.3).
TRAINING_TYPE_BY_TIER: dict[str, str] = {
    "developer": "developerTier",
    "standard": "Standard",
    "globalStandard": "GlobalStandard",
}

#: Default training tier for the demo (cheapest, preemptible, no hourly fee).
DEFAULT_TRAINING_TYPE: str = "GlobalStandard"

#: RFT (o-series base models) does NOT support the developer training tier; the
#: job-create call rejects it. RFT jobs fall back to this training type.
RFT_FALLBACK_TRAINING_TYPE: str = "GlobalStandard"

#: Maps the :data:`DemoConfig` tier constants to the exact deployment ``sku.name``
#: accepted by the ARM control-plane (distinct from the job-create
#: ``trainingType`` above). ARM rejects the lowercase tier names with
#: ``InvalidResourceProperties`` — these are the exact strings it expects (the
#: Developer SKU is ``DeveloperTier``, matching the portal's "Deployment type").
DEPLOYMENT_SKU_BY_TIER: dict[str, str] = {
    "developer": "DeveloperTier",
    "standard": "Standard",
    "globalStandard": "GlobalStandard",
}

#: Terminal fine-tuning job states.
_TERMINAL_STATES: frozenset[str] = frozenset({"succeeded", "failed", "cancelled"})

#: Terminal states for an uploaded file's import/processing.
_FILE_READY_STATE: str = "processed"
_FILE_FAILED_STATES: frozenset[str] = frozenset({"error", "failed", "deleted"})

#: Default polling cadence/ceiling while a file import completes.
_FILE_POLL_INTERVAL_SECONDS: float = 5.0
_FILE_POLL_TIMEOUT_SECONDS: float = 300.0

# ---------------------------------------------------------------------------
# Optional ``requests`` (HTTP for the ARM control-plane calls)
# ---------------------------------------------------------------------------
# Resolved at import time so tests can monkeypatch the module-level ``requests``
# attribute. ``requests`` is not an Azure SDK, so importing it does not violate
# the "imports with zero Azure SDKs" contract; when absent the control-plane
# helpers raise an actionable error instead of failing at import.
requests, REQUESTS_AVAILABLE = optional_import("requests")


# ---------------------------------------------------------------------------
# Client construction (data-plane Azure OpenAI)
# ---------------------------------------------------------------------------
def build_client(config: DemoConfig) -> Any:
    """Build an ``AzureOpenAI`` data-plane client from :class:`DemoConfig`.

    The ``openai`` SDK is imported lazily so this module loads without it.

    Parameters
    ----------
    config:
        Configuration providing the Azure OpenAI endpoint, key, and data-plane
        API version.

    Returns
    -------
    Any
        A configured ``openai.AzureOpenAI`` client.

    Raises
    ------
    RuntimeError
        When the ``openai`` package is not installed.
    ValueError
        When the endpoint or key required for live calls is missing.
    """
    openai_mod, available = optional_import("openai")
    if not available or openai_mod is None:
        raise RuntimeError(
            "The 'openai' package is required for serverless SFT operations. "
            "Install with: pip install openai"
        )
    if not config.azure_openai_endpoint or not config.azure_openai_api_key:
        raise ValueError(
            "DemoConfig is missing azure_openai_endpoint or azure_openai_api_key; "
            "set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY."
        )
    return openai_mod.AzureOpenAI(
        azure_endpoint=config.azure_openai_endpoint,
        api_key=config.azure_openai_api_key,
        api_version=config.data_plane_api_version,
    )


# ---------------------------------------------------------------------------
# Step 3.1 — Serverless SFT job lifecycle (data-plane)
# ---------------------------------------------------------------------------
def upload_files(client: Any, train: str | Path, val: str | Path) -> tuple[str, str]:
    """Upload the training and validation JSONL files for fine-tuning.

    Both files are uploaded with ``purpose="fine-tune"``. The caller is
    responsible for emitting UTF-8-with-BOM JSONL (see
    :func:`finetuning.schemas.write_sft_jsonl`).

    Parameters
    ----------
    client:
        An ``AzureOpenAI`` client (or test double) exposing ``files.create``.
    train, val:
        Paths to the training and validation JSONL files.

    Returns
    -------
    tuple[str, str]
        ``(training_file_id, validation_file_id)``.
    """
    train_path = Path(train)
    val_path = Path(val)

    with train_path.open("rb") as handle:
        training_file = client.files.create(file=handle, purpose="fine-tune")
    with val_path.open("rb") as handle:
        validation_file = client.files.create(file=handle, purpose="fine-tune")

    logger.info(
        "Uploaded training_file=%s validation_file=%s",
        training_file.id,
        validation_file.id,
    )
    return training_file.id, validation_file.id


def wait_for_files(
    client: Any,
    file_ids: tuple[str, ...] | list[str],
    *,
    interval: float = _FILE_POLL_INTERVAL_SECONDS,
    timeout: float = _FILE_POLL_TIMEOUT_SECONDS,
) -> None:
    """Block until every uploaded file finishes processing.

    Azure rejects an SFT job whose ``training_file``/``validation_file`` import
    has not reached the ``processed`` state ("The specified file reference must
    point to a completed file import."). This polls ``files.retrieve`` for each
    id until it is ready, raising on a failed import or timeout.

    Parameters
    ----------
    client:
        An ``AzureOpenAI`` client (or test double) exposing ``files.retrieve``.
    file_ids:
        File ids returned by :func:`upload_files`.
    interval:
        Seconds between polls.
    timeout:
        Maximum seconds to wait per file before raising.

    Raises
    ------
    RuntimeError
        If a file import fails or does not complete within ``timeout``.
    """
    for file_id in file_ids:
        deadline = time.monotonic() + timeout
        while True:
            status = getattr(client.files.retrieve(file_id), "status", None)
            if status == _FILE_READY_STATE:
                break
            if status in _FILE_FAILED_STATES:
                raise RuntimeError(
                    f"File {file_id} import did not complete (status={status!r})."
                )
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Timed out after {timeout:.0f}s waiting for file {file_id} "
                    f"to reach '{_FILE_READY_STATE}' (last status={status!r})."
                )
            logger.info("Waiting for file %s import (status=%s)", file_id, status)
            time.sleep(interval)
    logger.info("All %d uploaded file(s) processed", len(file_ids))


def _sanitize_suffix(suffix: str) -> str:
    """Coerce ``suffix`` to the Azure constraints (≤18 chars, no dots)."""
    cleaned = suffix.replace(".", "-").strip("-")
    if len(cleaned) > MAX_SUFFIX_LENGTH:
        cleaned = cleaned[:MAX_SUFFIX_LENGTH].rstrip("-")
    if not cleaned:
        raise ValueError(f"Suffix {suffix!r} is empty after sanitization.")
    return cleaned


def create_sft_job(
    client: Any,
    training_file_id: str,
    validation_file_id: str,
    *,
    model: str = DEFAULT_BASE_MODEL,
    suffix: str = "sales-intent",
    seed: int = DEFAULT_SEED,
    n_epochs: int = DEFAULT_N_EPOCHS,
    training_type: str = DEFAULT_TRAINING_TYPE,
) -> str:
    """Create a supervised fine-tuning job and return its job id.

    The payload matches research Gap 2.4 exactly: an explicit ``supervised``
    method with ``n_epochs`` and the ``trainingType`` training-tier flag passed
    via ``extra_body``.

    Parameters
    ----------
    client:
        An ``AzureOpenAI`` client (or test double) exposing
        ``fine_tuning.jobs.create``.
    training_file_id, validation_file_id:
        File ids returned by :func:`upload_files`.
    model:
        Base model to fine-tune (region-pinned for serverless training).
    suffix:
        Fine-tuned model suffix; sanitized to ≤18 chars with no dots.
    seed:
        Deterministic seed for reproducibility.
    n_epochs:
        Number of supervised epochs.
    training_type:
        Training tier flag — one of ``"GlobalStandard"``, ``"Standard"``, or
        ``"developerTier"``. Use :func:`training_type_for_tier` to map a
        :class:`DemoConfig` tier constant.

    Returns
    -------
    str
        The created fine-tuning job id.
    """
    return create_job(
        client,
        training_file_id,
        validation_file_id,
        method=METHOD_SUPERVISED,
        model=model,
        suffix=suffix,
        seed=seed,
        n_epochs=n_epochs,
        training_type=training_type,
    )


def _build_method_payload(
    method: str,
    *,
    n_epochs: int,
    beta: float,
    grader_type: str = "string_match",
    grader_model: str | None = None,
    extra_hyperparameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ``method`` block for a serverless fine-tuning job-create call.

    Branches on the serverless customization method so a single job-create path
    serves Supervised (SFT), Direct Preference Optimization (DPO), and
    Reinforcement (RFT). The grader settings are only used for RFT, where the
    block also carries a JSON-schema ``response_format`` so the grader can read
    structured fields via ``{{ sample.output_json.<field> }}``.

    Parameters
    ----------
    method : str
        One of 'supervised', 'dpo', or 'reinforcement'.
    n_epochs : int
        Number of training epochs.
    beta : float
        Beta (preference strength) for DPO; ignored for SFT and RFT.
    grader_type : str
        For RFT, the internal grader name: 'string_match' (default) or 'model'.
    grader_model : str, optional
        For RFT with ``grader_type='model'``, the grader model name.
    extra_hyperparameters : dict, optional
        Additional hyperparameters merged into the method's ``hyperparameters``
        block (e.g. ``batch_size``, ``learning_rate_multiplier`` for any method;
        ``eval_interval``, ``eval_samples``, ``reasoning_effort``,
        ``compute_multiplier`` for RFT). ``None`` values are dropped so the
        server default applies.

    Returns
    -------
    dict
        A method block ready for job-create.

    Raises
    ------
    ValueError
        When ``method`` is not a recognized serverless method.
    """
    extra = {k: v for k, v in (extra_hyperparameters or {}).items() if v is not None}
    if method == METHOD_SUPERVISED:
        return {
            "type": "supervised",
            "supervised": {"hyperparameters": {"n_epochs": n_epochs, **extra}},
        }
    if method == METHOD_DPO:
        return {
            "type": "dpo",
            "dpo": {"hyperparameters": {"n_epochs": n_epochs, "beta": beta, **extra}},
        }
    if method == METHOD_REINFORCEMENT:
        from .graders import build_azure_grader_config  # noqa: PLC0415 — avoid cycle
        from .schemas import rft_response_format_schema  # noqa: PLC0415

        return {
            "type": "reinforcement",
            "reinforcement": {
                "hyperparameters": {"n_epochs": n_epochs, **extra},
                "grader": build_azure_grader_config(
                    grader_type, grader_model=grader_model
                ),
                "response_format": rft_response_format_schema(),
            },
        }
    raise ValueError(
        f"Unknown method {method!r}; expected one of {sorted(SERVERLESS_METHODS)}."
    )


def create_job(
    client: Any,
    training_file_id: str,
    validation_file_id: str,
    *,
    method: str = METHOD_SUPERVISED,
    model: str = DEFAULT_BASE_MODEL,
    suffix: str = "sales-intent",
    seed: int = DEFAULT_SEED,
    n_epochs: int = DEFAULT_N_EPOCHS,
    training_type: str = DEFAULT_TRAINING_TYPE,
    beta: float = DEFAULT_DPO_BETA,
    grader_type: str = "string_match",
    grader_model: str | None = None,
    extra_hyperparameters: dict[str, Any] | None = None,
) -> str:
    """Create a serverless fine-tuning job of any method and return its job id.

    This is the unified job-create path behind :func:`create_sft_job`,
    :func:`create_dpo_job`, and :func:`create_rft_job`. The ``method`` selects
    the customization technique (``"supervised"``, ``"dpo"``, or
    ``"reinforcement"``); the request body otherwise mirrors the SFT call —
    an explicit ``method`` block plus the ``trainingType`` training-tier
    flag passed via ``extra_body``.

    Parameters
    ----------
    client:
        An ``AzureOpenAI`` client (or test double) exposing
        ``fine_tuning.jobs.create``.
    training_file_id, validation_file_id:
        File ids returned by :func:`upload_files`. For DPO these must reference
        preference-formatted JSONL; for RFT, graded-output JSONL.
    method:
        One of :data:`SERVERLESS_METHODS`. Defaults to supervised.
    model:
        Base model to fine-tune (region-pinned for serverless training).
    suffix:
        Fine-tuned model suffix; sanitized to ≤18 chars with no dots.
    seed:
        Deterministic seed for reproducibility.
    n_epochs:
        Number of training epochs.
    training_type:
        Training tier flag — see :func:`training_type_for_tier`.
    beta:
        DPO preference-strength hyperparameter (ignored for supervised and RFT).
    grader_type:
        RFT grader type ('string_match' or 'model'); ignored for SFT/DPO.

    Returns
    -------
    str
        The created fine-tuning job id.
    """
    safe_suffix = _sanitize_suffix(suffix)
    method_payload = _build_method_payload(
        method, n_epochs=n_epochs, beta=beta, grader_type=grader_type, grader_model=grader_model,
        extra_hyperparameters=extra_hyperparameters,
    )
    job = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model=model,
        suffix=safe_suffix,
        seed=seed,
        method=method_payload,
        extra_body={"trainingType": training_type},
    )
    logger.info(
        "Created %s job %s (model=%s, tier=%s)", method, job.id, model, training_type
    )
    return job.id


def create_dpo_job(
    client: Any,
    training_file_id: str,
    validation_file_id: str,
    *,
    model: str = DEFAULT_BASE_MODEL,
    suffix: str = "sales-dpo",
    seed: int = DEFAULT_SEED,
    n_epochs: int = DEFAULT_N_EPOCHS,
    training_type: str = DEFAULT_TRAINING_TYPE,
    beta: float = DEFAULT_DPO_BETA,
) -> str:
    """Create a Direct Preference Optimization (DPO) fine-tuning job.

    DPO trains the model to favor ``preferred_output`` over
    ``non_preferred_output`` for each prompt — useful when "good vs. bad" is
    easier to express as a comparison than as a single gold label. The training
    and validation files must be preference-formatted (see
    :func:`finetuning.schemas.write_dpo_jsonl`).

    Returns
    -------
    str
        The created fine-tuning job id.
    """
    return create_job(
        client,
        training_file_id,
        validation_file_id,
        method=METHOD_DPO,
        model=model,
        suffix=suffix,
        seed=seed,
        n_epochs=n_epochs,
        training_type=training_type,
        beta=beta,
    )


def create_rft_job(
    client: Any,
    training_file_id: str,
    validation_file_id: str,
    *,
    grader_type: str = "string_match",
    grader_model: str | None = None,
    model: str = RFT_BASE_MODEL,
    suffix: str = "sales-intent",
    seed: int = DEFAULT_SEED,
    n_epochs: int = DEFAULT_N_EPOCHS,
    training_type: str = DEFAULT_TRAINING_TYPE,
) -> str:
    """Convenience wrapper for creating an RFT job.

    RFT is only supported on o-series reasoning base models, so ``model``
    defaults to :data:`RFT_BASE_MODEL` (o4-mini) rather than
    :data:`DEFAULT_BASE_MODEL`.

    Parameters
    ----------
    client : Any
        An ``AzureOpenAI`` client exposing ``fine_tuning.jobs.create``.
    training_file_id, validation_file_id : str
        File ids from :func:`upload_files` (RFT format: messages + reference
        fields).
    grader_type : str
        Internal grader name ('string_match' or 'model'). Defaults to
        'string_match'.
    grader_model : str, optional
        Grader model name when ``grader_type='model'``.
    model : str
        Base model to fine-tune (o4-mini by default).
    suffix : str
        Model suffix.
    seed : int
        Deterministic seed.
    n_epochs : int
        Number of training epochs.
    training_type : str
        Training tier.

    Returns
    -------
    str
        The created RFT job id.
    """
    return create_job(
        client,
        training_file_id,
        validation_file_id,
        method=METHOD_REINFORCEMENT,
        model=model,
        suffix=suffix,
        seed=seed,
        n_epochs=n_epochs,
        training_type=training_type,
        grader_type=grader_type,
        grader_model=grader_model,
    )


def training_type_for_tier(tier: str) -> str:
    """Map a :class:`DemoConfig` tier constant to a ``trainingType`` string.

    Raises
    ------
    ValueError
        When ``tier`` is not a recognized deployment tier.
    """
    try:
        return TRAINING_TYPE_BY_TIER[tier]
    except KeyError as exc:
        raise ValueError(
            f"Unknown tier {tier!r}; expected one of {sorted(TRAINING_TYPE_BY_TIER)}."
        ) from exc


def poll_job(
    client: Any,
    job_id: str,
    *,
    poll_seconds: float = 30.0,
    max_polls: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    max_transient_errors: int = 20,
    heartbeat: bool = False,
) -> Any:
    """Poll a fine-tuning job until it reaches a terminal state.

    Parameters
    ----------
    client:
        An ``AzureOpenAI`` client (or test double) exposing
        ``fine_tuning.jobs.retrieve``.
    job_id:
        The job id returned by :func:`create_sft_job`.
    poll_seconds:
        Delay between status checks.
    max_polls:
        Optional cap on the number of polls; ``None`` polls until terminal.
    sleep:
        Injectable sleep function (eases testing).
    max_transient_errors:
        Consecutive ``retrieve`` failures (e.g. ``APIConnectionError`` /
        timeouts) tolerated before giving up. Fine-tunes run for an hour-plus,
        so a single network blip must not kill the poll — the job keeps
        training server-side. Reset to zero on every successful retrieve.
    heartbeat:
        When set, print a per-poll status line to stderr so a long wait shows
        progress without enabling module-wide ``-v`` logging.

    Returns
    -------
    Any
        The final retrieved job object (carrying ``status`` and, on success,
        ``fine_tuned_model``).
    """
    polls = 0
    transient_errors = 0
    started = time.monotonic()

    def _beat(message: str) -> None:
        if heartbeat:
            elapsed = int(time.monotonic() - started)
            print(
                f"[poll {job_id}] +{elapsed // 60:02d}:{elapsed % 60:02d} {message}",
                file=sys.stderr, flush=True,
            )

    while True:
        try:
            job = client.fine_tuning.jobs.retrieve(job_id)
            transient_errors = 0
        except Exception as exc:  # noqa: BLE001 - tolerate transient network blips
            transient_errors += 1
            if transient_errors > max_transient_errors:
                logger.error(
                    "Poll for job %s failed %d times in a row; giving up. The job "
                    "may still be running in Azure — re-attach with its job id.",
                    job_id, transient_errors,
                )
                raise
            logger.warning(
                "Transient error polling job %s (%d/%d): %s; retrying in %.0fs",
                job_id, transient_errors, max_transient_errors, exc, poll_seconds,
            )
            _beat(f"transient error {transient_errors}/{max_transient_errors}: {exc}")
            sleep(poll_seconds)
            continue
        status = getattr(job, "status", None)
        logger.info("Fine-tune job %s status=%s", job_id, status)
        _beat(f"status={status}")
        if status in _TERMINAL_STATES:
            return job
        polls += 1
        if max_polls is not None and polls >= max_polls:
            logger.warning("Stopped polling job %s after %d polls", job_id, polls)
            return job
        sleep(poll_seconds)


def list_checkpoints(client: Any, job_id: str) -> list[Any]:
    """List the deployable checkpoints for a fine-tuning job.

    Azure emits one checkpoint per epoch; the three most recent are deployable
    when the job finishes.

    Returns
    -------
    list[Any]
        The checkpoint objects (each carries an ``id`` like ``ftchkpt-...``).
    """
    response = client.fine_tuning.jobs.checkpoints.list(job_id)
    data = getattr(response, "data", response)
    checkpoints = list(data)
    logger.info("Listed %d checkpoint(s) for job %s", len(checkpoints), job_id)
    return checkpoints


def _checkpoint_valid_loss(checkpoint: Any) -> float | None:
    """Extract a checkpoint's validation loss from its ``metrics`` mapping."""
    metrics = getattr(checkpoint, "metrics", None)
    if metrics is None:
        return None
    if not isinstance(metrics, dict):
        metrics = getattr(metrics, "__dict__", {}) or {}
    for key in ("full_valid_loss", "valid_loss"):
        value = metrics.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def best_checkpoint_model_id(client: Any, job_id: str) -> str | None:
    """Return the deployable model id of the lowest-validation-loss checkpoint.

    Azure exposes only the most recent checkpoints as deployable. This lists
    them, picks the one with the lowest validation loss (guarding against
    over-training, where the final epoch is not the best), and returns its
    ``fine_tuned_model_checkpoint`` id for :func:`deploy_finetuned`.

    Parameters
    ----------
    client:
        An ``AzureOpenAI`` client exposing ``fine_tuning.jobs.checkpoints.list``.
    job_id:
        The fine-tuning job id.

    Returns
    -------
    str | None
        The best checkpoint's deployable model id, or ``None`` when no
        checkpoint carries a usable validation loss.
    """
    scored: list[tuple[float, Any]] = []
    for checkpoint in list_checkpoints(client, job_id):
        loss = _checkpoint_valid_loss(checkpoint)
        if loss is not None:
            scored.append((loss, checkpoint))
    if not scored:
        logger.warning("No checkpoint with a validation loss for job %s", job_id)
        return None
    best_loss, best = min(scored, key=lambda item: item[0])
    model_id = getattr(best, "fine_tuned_model_checkpoint", None)
    logger.info(
        "Best checkpoint %s (step=%s, valid_loss=%.4f) -> %s",
        getattr(best, "id", "?"),
        getattr(best, "step_number", "?"),
        best_loss,
        model_id,
    )
    return model_id



def _parse_results_rows(results_csv: str | Path) -> list[dict[str, str]]:
    """Read ``results.csv`` content (path or raw CSV text) into dict rows."""
    text: str
    candidate = Path(results_csv) if not isinstance(results_csv, Path) else results_csv
    try:
        is_file = candidate.exists() and candidate.is_file()
    except OSError:
        is_file = False
    if is_file:
        text = candidate.read_text(encoding="utf-8-sig")
    else:
        text = str(results_csv)
    return list(csv.DictReader(StringIO(text)))


def _row_valid_loss(row: dict[str, str]) -> float | None:
    """Extract the per-checkpoint validation loss from a results row."""
    for key in ("full_valid_loss", "valid_loss", "validation_loss"):
        value = row.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def pick_checkpoint(results_csv: str | Path) -> dict[str, Any]:
    """Pick the best checkpoint, preferring an earlier one on divergence.

    Parses ``results.csv``, isolates the end-of-epoch rows that carry a
    validation loss, and selects the epoch with the lowest validation loss.
    When that minimum occurs before the final epoch the training/validation
    curves have diverged (overfitting), so the earlier checkpoint is chosen and
    ``diverged`` is flagged.

    Parameters
    ----------
    results_csv:
        Path to ``results.csv`` or the raw CSV text.

    Returns
    -------
    dict[str, Any]
        ``{"epoch": int, "step": int | None, "valid_loss": float | None,
        "diverged": bool}``. ``epoch`` is 1-based among checkpoint rows;
        ``epoch == 0`` when no validation-loss rows are present.

    Raises
    ------
    ValueError
        When the CSV contains no rows at all.
    """
    rows = _parse_results_rows(results_csv)
    if not rows:
        raise ValueError("results.csv contained no rows.")

    checkpoints: list[tuple[int, int | None, float]] = []
    for index, row in enumerate(rows, start=1):
        valid_loss = _row_valid_loss(row)
        if valid_loss is None:
            continue
        step_raw = row.get("step")
        try:
            step = int(step_raw) if step_raw not in (None, "") else None
        except (TypeError, ValueError):
            step = None
        checkpoints.append((len(checkpoints) + 1, step, valid_loss))

    if not checkpoints:
        logger.warning("No validation-loss rows found; defaulting to final model.")
        return {"epoch": 0, "step": None, "valid_loss": None, "diverged": False}

    best_epoch, best_step, best_loss = min(checkpoints, key=lambda item: item[2])
    last_epoch = checkpoints[-1][0]
    diverged = best_epoch != last_epoch
    if diverged:
        logger.info(
            "Train/val divergence detected; picking earlier checkpoint epoch=%d "
            "(valid_loss=%.4f) over final epoch=%d.",
            best_epoch,
            best_loss,
            last_epoch,
        )
    return {
        "epoch": best_epoch,
        "step": best_step,
        "valid_loss": best_loss,
        "diverged": diverged,
    }


def final_model_id(client: Any, job_id: str) -> str | None:
    """Return the ``fine_tuned_model`` id from a succeeded job, if present."""
    job = client.fine_tuning.jobs.retrieve(job_id)
    return getattr(job, "fine_tuned_model", None)


def download_results(client: Any, job: Any, dest: str | Path) -> Path | None:
    """Download a succeeded job's result-metrics CSV to ``dest``.

    Azure attaches a training-metrics file (the ``results.csv`` consumed by
    :func:`pick_checkpoint`) to a finished job via ``result_files``. This fetches
    the first such file and writes it to ``dest``.

    Parameters
    ----------
    client:
        An ``AzureOpenAI`` client exposing ``files.content``.
    job:
        The terminal job object returned by :func:`poll_job` (carries
        ``result_files``).
    dest:
        Destination path for the CSV (parent directories are created).

    Returns
    -------
    Path | None
        The written path, or ``None`` when the job exposes no result files.
    """
    result_files = getattr(job, "result_files", None) or []
    if not result_files:
        logger.warning("Job %s exposed no result_files to download", getattr(job, "id", "?"))
        return None
    file_id = result_files[0]
    content = client.files.content(file_id)
    raw = getattr(content, "content", None)
    if raw is None and hasattr(content, "read"):
        raw = content.read()
    if raw is None:
        raw = content
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    out = Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw)
    logger.info("Saved fine-tune results metrics to %s", out)
    return out


# ---------------------------------------------------------------------------
# Step 3.2 — Control-plane (ARM) deployment, inference, and cleanup
# ---------------------------------------------------------------------------
def _require_requests() -> Any:
    """Return the module-level ``requests`` or raise an actionable error."""
    if requests is None:
        raise RuntimeError(
            "The 'requests' package is required for ARM control-plane calls. "
            "Install with: pip install requests"
        )
    return requests


def _acquire_management_token(scope: str = MANAGEMENT_SCOPE) -> str:
    """Acquire an ARM bearer token via ``azure-identity`` (lazy import)."""
    identity_mod, available = optional_import("azure.identity")
    if not available or identity_mod is None:
        raise RuntimeError(
            "The 'azure-identity' package is required to acquire a management "
            "token. Install with: pip install azure-identity, or pass token=..."
        )
    credential = identity_mod.DefaultAzureCredential()
    return credential.get_token(scope).token


def _deployment_url(config: DemoConfig, deployment_name: str) -> str:
    """Build the ARM deployment resource URL for ``deployment_name``."""
    missing = config.missing_required(
        ["azure_subscription_id", "azure_resource_group", "aoai_account_name"]
    )
    if missing:
        raise ValueError(
            "DemoConfig is missing control-plane identifiers: "
            f"{', '.join(missing)} (set the corresponding AZURE_* env vars)."
        )
    return (
        "https://management.azure.com/subscriptions/"
        f"{config.azure_subscription_id}/resourceGroups/"
        f"{config.azure_resource_group}/providers/Microsoft.CognitiveServices/"
        f"accounts/{config.aoai_account_name}/deployments/{deployment_name}"
    )


def deploy_finetuned(
    config: DemoConfig,
    model_id: str,
    deployment_name: str,
    sku: str = "developer",
    *,
    capacity: int = 1,
    version: str = "1",
    token: str | None = None,
) -> dict[str, Any]:
    """Deploy a fine-tuned model (or checkpoint) via the ARM control-plane PUT.

    This is intentionally a control-plane call to ``management.azure.com`` with
    ``api-version=2024-10-01`` — *not* the data-plane SDK (research Gap 2.1).

    Parameters
    ----------
    config:
        Provides the subscription, resource group, account, and control-plane
        API version. Secrets are never hardcoded.
    model_id:
        Fine-tuned model id (``...ft-...``) or a ``ftchkpt-...`` checkpoint id.
    deployment_name:
        The deployment name used at inference time.
    sku:
        Deployment SKU — one of ``developer``, ``standard``, ``globalStandard``.
        ``developer`` auto-deletes after 24h; the others after 15 days of
        inactivity (research Gap 2.3).
    capacity:
        SKU capacity units.
    version:
        Model version string (``"1"`` for the first fine-tuned version).
    token:
        Optional pre-acquired ARM bearer token; acquired via ``azure-identity``
        when omitted.

    Returns
    -------
    dict[str, Any]
        The parsed JSON response from ARM.

    Raises
    ------
    ValueError
        When ``sku`` is invalid or required config identifiers are missing.
    RuntimeError
        When ``requests`` (or ``azure-identity`` for the token) is unavailable.
    """
    if sku not in TIER_NAMES:
        raise ValueError(
            f"Invalid sku {sku!r}; expected one of {', '.join(TIER_NAMES)}."
        )
    http = _require_requests()
    url = _deployment_url(config, deployment_name)
    bearer = token if token is not None else _acquire_management_token()

    sku_name = DEPLOYMENT_SKU_BY_TIER[sku]
    body: dict[str, Any] = {
        "sku": {"name": sku_name, "capacity": capacity},
        "properties": {
            "model": {"format": "OpenAI", "name": model_id, "version": version}
        },
    }
    response = http.put(
        url,
        params={"api-version": config.control_plane_api_version},
        headers={
            "Authorization": f"Bearer {bearer}",
            "Content-Type": "application/json",
        },
        data=json.dumps(body),
    )
    response.raise_for_status()
    logger.info(
        "Deployed model=%s as deployment=%s (sku=%s)", model_id, deployment_name, sku
    )
    return response.json()


def infer(
    config: DemoConfig,
    deployment_name: str,
    transcript: str,
    *,
    client: Any | None = None,
    system_prompt: str | None = None,
    temperature: float = 0.0,
) -> str | None:
    """Run a data-plane chat completion against the deployed fine-tuned model.

    Parameters
    ----------
    config:
        Provides the data-plane endpoint, key, and API version.
    deployment_name:
        The deployment name created by :func:`deploy_finetuned`.
    transcript:
        The sales-call transcript to classify.
    client:
        Optional pre-built ``AzureOpenAI`` client (eases testing); built from
        ``config`` when omitted.
    system_prompt:
        Override for the classification system prompt; defaults to the shared
        :data:`finetuning.schemas.SYSTEM_PROMPT`.
    temperature:
        Sampling temperature (deterministic by default).

    Returns
    -------
    str | None
        The model's response content.
    """
    active_client = client if client is not None else build_client(config)
    messages = [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
        {"role": "user", "content": transcript},
    ]
    completion = active_client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        temperature=temperature,
    )
    message = completion.choices[0].message
    content = getattr(message, "content", None)

    # Newer SDK/model responses may return content as a list of typed parts,
    # or leave content empty while carrying refusal text. Coerce all of these
    # into a plain string so downstream evaluators can parse/log consistently.
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                continue
            text_attr = getattr(item, "text", None)
            if isinstance(text_attr, str):
                parts.append(text_attr)
        if parts:
            return "\n".join(parts)

    refusal = getattr(message, "refusal", None)
    if isinstance(refusal, str) and refusal.strip():
        return refusal
    return None


def delete_deployment(
    config: DemoConfig,
    deployment_name: str,
    *,
    token: str | None = None,
) -> int:
    """Delete a fine-tuned deployment via the ARM control-plane DELETE (cleanup).

    Deleting a deployment never deletes the underlying fine-tuned model, which
    can be redeployed later (research Gap 2.3).

    Parameters
    ----------
    config:
        Provides the control-plane identifiers and API version.
    deployment_name:
        The deployment to delete.
    token:
        Optional pre-acquired ARM bearer token; acquired via ``azure-identity``
        when omitted.

    Returns
    -------
    int
        The HTTP status code of the DELETE response.
    """
    http = _require_requests()
    url = _deployment_url(config, deployment_name)
    bearer = token if token is not None else _acquire_management_token()

    response = http.delete(
        url,
        params={"api-version": config.control_plane_api_version},
        headers={"Authorization": f"Bearer {bearer}"},
    )
    response.raise_for_status()
    logger.info("Deleted deployment=%s", deployment_name)
    return response.status_code


__all__ = [
    "MANAGEMENT_SCOPE",
    "DEFAULT_BASE_MODEL",
    "DEFAULT_SEED",
    "DEFAULT_N_EPOCHS",
    "MAX_SUFFIX_LENGTH",
    "TRAINING_TYPE_BY_TIER",
    "DEFAULT_TRAINING_TYPE",
    "REQUESTS_AVAILABLE",
    "build_client",
    "upload_files",
    "create_sft_job",
    "training_type_for_tier",
    "poll_job",
    "list_checkpoints",
    "pick_checkpoint",
    "final_model_id",
    "deploy_finetuned",
    "infer",
    "delete_deployment",
]
