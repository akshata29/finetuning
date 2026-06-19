"""Run-of-show orchestrator for the 1-Hour Azure Fine-Tuning Demo.

Wires the four demo acts into a single CLI with live-vs-pre-baked toggles that
mirror the 1-hour run-of-show:

* ``gen-data``  — Act 1: synthetic data factory (live sample + dedup/PII).
* ``sft``       — Act 2A: serverless Azure OpenAI supervised fine-tuning.
* ``deploy``    — Act 2A: control-plane deploy of the fine-tuned model.
* ``gpu-lora``  — Act 2B: GPU managed-compute LoRA (always pre-baked).
* ``evaluate``  — Act 3: base vs fine-tuned vs optimized-prompt + offline metrics.
* ``foundry-eval`` — Act 3A: Foundry portal evals across base/SFT/DPO/RFT.
* ``host``      — Act 4: Developer-tier live deploy + one inference.
* ``cleanup``   — tear down the live (Developer-tier) deployment.
* ``all``       — run the live-able acts end to end in pre-baked mode.

Import safety (OWASP A06 — avoid hard dependence on optional components): this
module and ``python finetuning/run_of_show.py --help`` import and run with
**zero Azure SDKs installed**. Every act module is imported lazily inside its
subcommand handler, and those act modules only resolve Azure SDKs lazily via
:func:`finetuning.config.optional_import`. No secrets or endpoints are
hardcoded — configuration is sourced once from :class:`DemoConfig` via the
environment (OWASP A05 Security Misconfiguration).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, never imports SDKs at runtime
    from .config import DemoConfig

# Allow running both as a module (``python -m finetuning.run_of_show``) and
# as a plain script (``python finetuning/run_of_show.py``): when launched as
# a script there is no package context, so register one and add the repo root to
# ``sys.path`` so the lazy ``from . import ...`` calls in the handlers resolve.
if __package__ in (None, ""):  # pragma: no cover - script-launch shim
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "finetuning"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------
EXIT_SUCCESS: int = 0
EXIT_FAILURE: int = 1
EXIT_ERROR: int = 2

# ---------------------------------------------------------------------------
# Data layout (shared with preflight)
# ---------------------------------------------------------------------------
#: Default folder holding the generated/pre-baked JSONL datasets.
DATA_DIR: Path = Path(__file__).resolve().parent / "data"
TRAIN_FILE: str = "train.jsonl"
VAL_FILE: str = "validation.jsonl"
EVAL_FILE: str = "eval.jsonl"

#: Candidate prediction files consumed by the offline-metrics comparison.
CANDIDATE_FILES: tuple[str, ...] = (
    "preds_base.jsonl",
    "preds_finetuned.jsonl",
    "preds_optimized_prompt.jsonl",
)

#: Records the fine-tuned model + job id from the last live ``sft`` run so that
#: ``deploy``/``host`` can resolve the model id without re-pasting it.
SFT_STATE_FILE: str = "sft_state.json"

#: Preference (DPO) dataset files and the state written by the ``dpo`` command.
DPO_TRAIN_FILE: str = "dpo_train.jsonl"
DPO_VAL_FILE: str = "dpo_validation.jsonl"
DPO_STATE_FILE: str = "dpo_state.json"

#: Reinforcement (RFT) dataset files and the state written by the ``rft`` command.
RFT_TRAIN_FILE: str = "rft_train.jsonl"
RFT_VAL_FILE: str = "rft_validation.jsonl"
RFT_STATE_FILE: str = "rft_state.json"


def _save_sft_state(data_dir: Path, *, model_id: str | None, job_id: str | None) -> Path:
    """Persist the latest fine-tuned model + job id to ``data/sft_state.json``."""
    path = data_dir / SFT_STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"model_id": model_id, "job_id": job_id}, indent=2),
        encoding="utf-8",
    )
    return path


def _save_dpo_state(data_dir: Path, *, model_id: str | None, job_id: str | None) -> Path:
    """Persist the latest DPO fine-tuned model + job id to ``data/dpo_state.json``."""
    path = data_dir / DPO_STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"model_id": model_id, "job_id": job_id}, indent=2),
        encoding="utf-8",
    )
    return path


def _save_rft_state(data_dir: Path, *, model_id: str | None, job_id: str | None) -> Path:
    """Persist the latest RFT fine-tuned model + job id to ``data/rft_state.json``."""
    path = data_dir / RFT_STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"model_id": model_id, "job_id": job_id}, indent=2),
        encoding="utf-8",
    )
    return path


#: Maps the ``--method`` flag to its saved-state filename so deploy/host can
#: resolve an SFT, DPO, or RFT fine-tuned model from the matching state file.
STATE_FILE_BY_METHOD: dict[str, str] = {
    "sft": SFT_STATE_FILE,
    "dpo": DPO_STATE_FILE,
    "rft": RFT_STATE_FILE,
}


def _load_sft_state(data_dir: Path, method: str = "sft") -> dict[str, str | None]:
    """Read the saved fine-tune state for ``method``, returning ``{}`` on failure.

    ``method`` selects which state file to read (``sft`` -> ``sft_state.json``,
    ``dpo`` -> ``dpo_state.json``, ``rft`` -> ``rft_state.json``).
    """
    state_file = STATE_FILE_BY_METHOD.get(method, SFT_STATE_FILE)
    path = data_dir / state_file
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _resolve_model_id(
    args: argparse.Namespace, config: DemoConfig, sft: object
) -> str | None:
    """Resolve the fine-tuned model id from CLI args, a job id, or saved state.

    Priority: explicit ``--model-id`` > ``--job-id`` > the ``job_id`` saved by
    the matching ``--method`` run (``sft`` by default; ``dpo`` / ``rft`` read
    their own state files). When a job id is resolved, the lowest-validation-
    loss checkpoint is deployed by default (guards against over-training);
    ``--final-model`` forces the job's final model instead.
    """
    model_id = getattr(args, "model_id", None)
    if model_id:
        return model_id

    method = getattr(args, "method", "sft") or "sft"
    state_file = STATE_FILE_BY_METHOD.get(method, SFT_STATE_FILE)
    data_dir = Path(getattr(args, "data_dir", DATA_DIR))
    saved = _load_sft_state(data_dir, method)
    job_id = getattr(args, "job_id", None) or saved.get("job_id")
    if not job_id:
        fallback = saved.get("model_id")
        if fallback:
            logger.info("Using saved %s model id %s from %s", method, fallback, data_dir / state_file)
        return fallback

    client = sft.build_client(config)  # type: ignore[attr-defined]
    if not getattr(args, "final_model", False):
        checkpoint_id = sft.best_checkpoint_model_id(client, job_id)  # type: ignore[attr-defined]
        if checkpoint_id:
            logger.info("Deploying best checkpoint %s from job %s", checkpoint_id, job_id)
            return checkpoint_id
        logger.info("No checkpoint available; falling back to final model for job %s", job_id)

    resolved = sft.final_model_id(client, job_id)  # type: ignore[attr-defined]
    if resolved:
        logger.info("Resolved final model id %s from job %s", resolved, job_id)
        return resolved
    logger.error("Job %s has no fine_tuned_model id (not finished?)", job_id)
    return saved.get("model_id")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def configure_logging(verbose: bool = False) -> None:
    """Configure root logging for the orchestrator."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _build_config() -> DemoConfig:
    """Construct the demo configuration once from the environment.

    Imported lazily so ``--help`` never touches any optional dependency.
    """
    from .config import DemoConfig  # noqa: PLC0415 — lazy keeps --help SDK-free

    return DemoConfig.from_env()


def _report_data_files(data_dir: Path) -> int:
    """Log which expected dataset files are present and return an exit code."""
    missing: list[str] = []
    for name in (TRAIN_FILE, VAL_FILE, EVAL_FILE):
        path = data_dir / name
        if path.exists():
            logger.info("  present: %s", path)
        else:
            missing.append(name)
            logger.warning("  missing: %s", path)
    return EXIT_SUCCESS if not missing else EXIT_FAILURE


# ---------------------------------------------------------------------------
# Act 1 — Synthetic data factory
# ---------------------------------------------------------------------------
def cmd_gen_data(args: argparse.Namespace, config: DemoConfig) -> int:
    """Act 1: generate a synthetic, PII-free dataset (or report the pre-baked one)."""
    from . import schemas, taxonomy  # noqa: PLC0415

    data_dir = Path(getattr(args, "data_dir", DATA_DIR))
    data_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, "prebaked", False):
        logger.info("[Act 1] Pre-baked dataset - verifying files in %s", data_dir)
        return _report_data_files(data_dir)

    from . import act1_synthetic_data as act1  # noqa: PLC0415

    count = getattr(args, "count", 12)
    eval_count = getattr(args, "eval_count", 200)
    logger.info(
        "[Act 1] Live synthetic generation: %d train seeds + %d eval seeds",
        count,
        eval_count,
    )
    train_seeds = list(taxonomy.iter_seeds(count, mode="train"))
    eval_seeds = list(taxonomy.iter_seeds(eval_count, mode="eval"))

    records = act1.generate_records(config, train_seeds)
    records = act1.dedup(records)
    flagged = act1.pii_scan(records)
    if flagged:
        logger.warning("[Act 1] PII scan flagged %d record(s) for regeneration", len(flagged))
    eval_records = act1.generate_records(config, eval_seeds)

    train, val, eval_split = act1.split(records + eval_records)
    schemas.write_sft_jsonl(train, data_dir / TRAIN_FILE)
    schemas.write_sft_jsonl(val, data_dir / VAL_FILE)
    schemas.write_eval_jsonl(eval_split, data_dir / EVAL_FILE)
    logger.info(
        "[Act 1] Wrote %d train / %d val / %d eval rows to %s",
        len(train),
        len(val),
        len(eval_split),
        data_dir,
    )
    if getattr(args, "dpo", False):
        dpo_train = act1.build_preference_records(train)
        dpo_val = act1.build_preference_records(val)
        schemas.write_dpo_jsonl(dpo_train, data_dir / DPO_TRAIN_FILE)
        schemas.write_dpo_jsonl(dpo_val, data_dir / DPO_VAL_FILE)
        logger.info(
            "[Act 1] Wrote %d DPO train / %d DPO val preference pairs to %s",
            len(dpo_train),
            len(dpo_val),
            data_dir,
        )
    if getattr(args, "rft", False):
        rft_train = act1.build_rft_records(train)
        rft_val = act1.build_rft_records(val)
        schemas.write_rft_jsonl(rft_train, data_dir / RFT_TRAIN_FILE)
        schemas.write_rft_jsonl(rft_val, data_dir / RFT_VAL_FILE)
        logger.info(
            "[Act 1] Wrote %d RFT train / %d RFT val records to %s",
            len(rft_train),
            len(rft_val),
            data_dir,
        )
    return EXIT_SUCCESS


# ---------------------------------------------------------------------------
# Act 2A — Serverless SFT
# ---------------------------------------------------------------------------
def cmd_sft(args: argparse.Namespace, config: DemoConfig) -> int:
    """Act 2A: run (or replay) the serverless Azure OpenAI SFT lifecycle."""
    from . import act2a_serverless_sft as sft  # noqa: PLC0415

    data_dir = Path(getattr(args, "data_dir", DATA_DIR))

    if getattr(args, "prebaked", False):
        logger.info("[Act 2A] Pre-baked SFT job - selecting best checkpoint")
        results_csv = getattr(args, "results_csv", None)
        if results_csv:
            choice = sft.pick_checkpoint(results_csv)
            logger.info("[Act 2A] Best checkpoint: %s", choice)
        else:
            logger.info("[Act 2A] No --results-csv provided; nothing to score offline")
        return EXIT_SUCCESS

    train = getattr(args, "train", None) or str(data_dir / TRAIN_FILE)
    val = getattr(args, "val", None) or str(data_dir / VAL_FILE)
    logger.info("[Act 2A] Live SFT: upload %s / %s, create + poll job", train, val)

    client = sft.build_client(config)
    train_id, val_id = sft.upload_files(client, train, val)
    sft.wait_for_files(client, (train_id, val_id))
    training_type = sft.training_type_for_tier(config.deployment_tier)
    job_id = sft.create_sft_job(client, train_id, val_id, training_type=training_type)
    job = sft.poll_job(client, job_id)
    results_path = sft.download_results(client, job, data_dir / "results.csv")
    if results_path is not None:
        choice = sft.pick_checkpoint(results_path)
        logger.info("[Act 2A] Best checkpoint: %s", choice)
    model_id = sft.final_model_id(client, job_id)
    state_path = _save_sft_state(data_dir, model_id=model_id, job_id=job_id)
    logger.info("[Act 2A] Fine-tuned model id: %s", model_id)
    logger.info("[Act 2A] Saved model/job id to %s (deploy/host can omit --model-id)", state_path)
    return EXIT_SUCCESS


def cmd_dpo(args: argparse.Namespace, config: DemoConfig) -> int:
    """Act 2A (DPO): run the serverless Direct Preference Optimization lifecycle.

    Serverless support for DPO — not just SFT — is a headline Azure
    differentiator. This mirrors :func:`cmd_sft` but submits a preference job
    over ``dpo_train.jsonl`` / ``dpo_validation.jsonl`` (generate them with
    ``gen-data --dpo``).
    """
    from . import act2a_serverless_sft as sft  # noqa: PLC0415

    data_dir = Path(getattr(args, "data_dir", DATA_DIR))

    if getattr(args, "prebaked", False):
        logger.info("[Act 2A/DPO] Pre-baked DPO job - nothing to submit")
        return EXIT_SUCCESS

    train = getattr(args, "train", None) or str(data_dir / DPO_TRAIN_FILE)
    val = getattr(args, "val", None) or str(data_dir / DPO_VAL_FILE)
    if not Path(train).exists() or not Path(val).exists():
        logger.error(
            "[Act 2A/DPO] Preference files not found (%s / %s). "
            "Generate them first with: gen-data --dpo",
            train,
            val,
        )
        return EXIT_FAILURE

    beta = float(getattr(args, "beta", sft.DEFAULT_DPO_BETA) or sft.DEFAULT_DPO_BETA)
    n_epochs = int(getattr(args, "n_epochs", sft.DEFAULT_N_EPOCHS) or sft.DEFAULT_N_EPOCHS)
    logger.info(
        "[Act 2A/DPO] Live DPO: upload %s / %s, create + poll job (beta=%s, epochs=%d)",
        train,
        val,
        beta,
        n_epochs,
    )

    client = sft.build_client(config)
    train_id, val_id = sft.upload_files(client, train, val)
    sft.wait_for_files(client, (train_id, val_id))
    training_type = sft.training_type_for_tier(config.deployment_tier)
    job_id = sft.create_dpo_job(
        client,
        train_id,
        val_id,
        suffix="sales-dpo",
        n_epochs=n_epochs,
        training_type=training_type,
        beta=beta,
    )
    job = sft.poll_job(client, job_id)
    results_path = sft.download_results(client, job, data_dir / "dpo_results.csv")
    if results_path is not None:
        choice = sft.pick_checkpoint(results_path)
        logger.info("[Act 2A/DPO] Best checkpoint: %s", choice)
    model_id = sft.final_model_id(client, job_id)
    state_path = _save_dpo_state(data_dir, model_id=model_id, job_id=job_id)
    logger.info("[Act 2A/DPO] DPO fine-tuned model id: %s", model_id)
    logger.info("[Act 2A/DPO] Saved model/job id to %s", state_path)
    return EXIT_SUCCESS


def cmd_rft(args: argparse.Namespace, config: DemoConfig) -> int:
    """Act 2A (RFT): run the serverless Reinforcement Fine-Tuning lifecycle.

    Serverless RFT is powered by a grader (string_match or model) to score
    outputs, enabling reinforcement learning. This mirrors :func:`cmd_dpo` but
    submits a reinforcement job over ``rft_train.jsonl`` / ``rft_validation.jsonl``
    (generate them with ``gen-data --rft``).
    """
    from . import act2a_serverless_sft as sft  # noqa: PLC0415

    data_dir = Path(getattr(args, "data_dir", DATA_DIR))

    if getattr(args, "prebaked", False):
        logger.info("[Act 2A/RFT] Pre-baked RFT job - nothing to submit")
        return EXIT_SUCCESS

    train = getattr(args, "train", None) or str(data_dir / RFT_TRAIN_FILE)
    val = getattr(args, "val", None) or str(data_dir / RFT_VAL_FILE)
    if not Path(train).exists() or not Path(val).exists():
        logger.error(
            "[Act 2A/RFT] Graded files not found (%s / %s). "
            "Generate them first with: gen-data --rft",
            train,
            val,
        )
        return EXIT_FAILURE

    grader_type = getattr(args, "grader", "string_match")
    n_epochs = int(getattr(args, "n_epochs", sft.DEFAULT_N_EPOCHS) or sft.DEFAULT_N_EPOCHS)
    logger.info(
        "[Act 2A/RFT] Live RFT on base %s: upload %s / %s, create + poll job (grader=%s, epochs=%d)",
        sft.RFT_BASE_MODEL,
        train,
        val,
        grader_type,
        n_epochs,
    )

    client = sft.build_client(config)
    train_id, val_id = sft.upload_files(client, train, val)
    sft.wait_for_files(client, (train_id, val_id))
    training_type = sft.training_type_for_tier(config.deployment_tier)
    if training_type == sft.TRAINING_TYPE_BY_TIER["developer"]:
        logger.warning(
            "[Act 2A/RFT] o-series RFT does not support the developer training "
            "tier; falling back to %s",
            sft.RFT_FALLBACK_TRAINING_TYPE,
        )
        training_type = sft.RFT_FALLBACK_TRAINING_TYPE
    job_id = sft.create_rft_job(
        client,
        train_id,
        val_id,
        grader_type=grader_type,
        grader_model=config.grader_model if grader_type == "model" else None,
        suffix="sales-rft",
        n_epochs=n_epochs,
        training_type=training_type,
    )
    job = sft.poll_job(client, job_id)
    results_path = sft.download_results(client, job, data_dir / "rft_results.csv")
    if results_path is not None:
        choice = sft.pick_checkpoint(results_path)
        logger.info("[Act 2A/RFT] Best checkpoint: %s", choice)
    model_id = sft.final_model_id(client, job_id)
    state_path = _save_rft_state(data_dir, model_id=model_id, job_id=job_id)
    logger.info("[Act 2A/RFT] RFT fine-tuned model id: %s", model_id)
    logger.info("[Act 2A/RFT] Saved model/job id to %s", state_path)
    return EXIT_SUCCESS


def cmd_deploy(args: argparse.Namespace, config: DemoConfig) -> int:
    """Act 2A: deploy the fine-tuned model via the ARM control-plane PUT."""
    from . import act2a_serverless_sft as sft  # noqa: PLC0415

    deployment_name = getattr(args, "deployment_name", None) or config.sft_deployment_name
    if getattr(args, "prebaked", False):
        logger.info("[Act 2A deploy] Pre-baked deployment '%s' assumed live", deployment_name)
        return EXIT_SUCCESS

    model_id = _resolve_model_id(args, config, sft)
    if not model_id:
        logger.error(
            "[Act 2A deploy] No model id: pass --model-id, --job-id, or run live 'sft' first"
        )
        return EXIT_ERROR
    sku = getattr(args, "sku", "developer")
    sft.deploy_finetuned(config, model_id, deployment_name, sku=sku)
    logger.info("[Act 2A deploy] Deployed model=%s as '%s' (sku=%s)", model_id, deployment_name, sku)
    return EXIT_SUCCESS


# ---------------------------------------------------------------------------
# Act 2A1 — quick base-vs-fine-tuned comparison on the validation holdout
# ---------------------------------------------------------------------------
def cmd_quick_eval(args: argparse.Namespace, config: DemoConfig) -> int:
    """Act 2A1: score base vs fine-tuned on the labeled validation set."""
    from . import act2a1_quick_eval as quick  # noqa: PLC0415

    data_dir = Path(getattr(args, "data_dir", DATA_DIR))
    val_path = getattr(args, "val", None) or str(data_dir / VAL_FILE)
    base_deployment = getattr(args, "base_deployment", None) or config.base_deployment_name
    ft_deployment = getattr(args, "ft_deployment", None) or config.sft_deployment_name

    if not base_deployment:
        logger.error("[Act 2A1] No base deployment: set BASE_DEPLOYMENT_NAME or pass --base-deployment")
        return EXIT_ERROR
    if not ft_deployment:
        logger.error("[Act 2A1] No fine-tuned deployment: set SFT_DEPLOYMENT_NAME or pass --ft-deployment")
        return EXIT_ERROR

    logger.info(
        "[Act 2A1] Quick eval on %s: base='%s' vs finetuned='%s'",
        val_path,
        base_deployment,
        ft_deployment,
    )
    delay = float(getattr(args, "delay", 0.0) or 0.0)
    result = quick.compare_models(
        config, val_path, base_deployment, ft_deployment, out_dir=data_dir, request_delay=delay
    )
    for line in quick.format_scorecard(result).splitlines():
        logger.info("[Act 2A1] %s", line)
    return EXIT_SUCCESS


def cmd_foundry_eval(args: argparse.Namespace, config: DemoConfig) -> int:
    """Act 3A: submit Foundry portal evaluation runs for each tuned model."""
    from . import act3a_foundry_eval as foundry  # noqa: PLC0415

    data_dir = Path(getattr(args, "data_dir", DATA_DIR))
    val_path = getattr(args, "val", None) or str(data_dir / VAL_FILE)
    models = getattr(args, "models", None) or list(foundry.DEFAULT_MODELS)
    limit = getattr(args, "limit", None)
    delay = float(getattr(args, "delay", 0.0) or 0.0)
    include_builtin = not getattr(args, "no_builtin", False)
    upload = not getattr(args, "no_upload", False)

    pairs = foundry.resolve_deployments(config, models)
    if not pairs:
        logger.error(
            "[Act 3A] No evaluatable deployments. Set BASE/SFT/DPO/RFT_DEPLOYMENT_NAME "
            "or pass --models with configured arms."
        )
        return EXIT_ERROR

    logger.info(
        "[Act 3A] Foundry eval on %s for arms: %s",
        val_path,
        ", ".join(f"{label}={deployment}" for label, deployment in pairs),
    )
    report = foundry.run_all_foundry_evals(
        config,
        val_path,
        data_dir,
        models=models,
        limit=limit,
        request_delay=delay,
        include_builtin=include_builtin,
        upload=upload,
    )
    for line in foundry.format_report(report).splitlines():
        logger.info("[Act 3A] %s", line)
    return EXIT_FAILURE if report.get("failures") else EXIT_SUCCESS


# ---------------------------------------------------------------------------
# Act 2B — GPU LoRA (always pre-baked)
# ---------------------------------------------------------------------------
def cmd_gpu_lora(args: argparse.Namespace, config: DemoConfig) -> int:
    """Act 2B: show the GPU managed-compute LoRA story (always pre-baked)."""
    from . import act2b_gpu_lora as gpu  # noqa: PLC0415

    logger.info("[Act 2B] GPU LoRA - provisioning exceeds the hour; show the pre-baked job")
    spec = gpu.custom_qlora_command_job_spec()
    logger.info("[Act 2B] QLoRA parity command-job spec fields: %s", sorted(spec))
    if not getattr(args, "prebaked", False):
        logger.warning(
            "[Act 2B] Live GPU LoRA needs azure-ai-ml + dedicated GPU quota; "
            "pre-baking is strongly recommended for the 1-hour slot"
        )
    return EXIT_SUCCESS


# ---------------------------------------------------------------------------
# Act 3 — Evaluation
# ---------------------------------------------------------------------------
def cmd_evaluate(args: argparse.Namespace, config: DemoConfig) -> int:
    """Act 3: offline aggregate metrics, plus an optional live 3-run Foundry eval."""
    from . import offline_metrics  # noqa: PLC0415

    kind = getattr(args, "kind", "propensity")
    data_dir = Path(getattr(args, "data_dir", DATA_DIR))
    predictions = getattr(args, "predictions", None) or [
        str(data_dir / name) for name in CANDIDATE_FILES
    ]
    available = [path for path in predictions if Path(path).exists()]

    if available:
        logger.info("[Act 3] Offline %s metrics over %d candidate file(s)", kind, len(available))
        for row in offline_metrics.compare(available, kind):
            logger.info("[Act 3] %s", row)
    else:
        logger.warning("[Act 3] No prediction files found; skipping offline metrics")

    if getattr(args, "prebaked", False):
        return EXIT_SUCCESS

    from . import act3_evaluation as act3  # noqa: PLC0415

    logger.info("[Act 3] Live Foundry eval: one definition, base/finetuned/optimized-prompt runs")
    client = act3.build_openai_client(config)
    criteria = (
        act3.build_propensity_criteria(config.grader_model)
        if kind == "propensity"
        else act3.build_intent_criteria(config.grader_model)
    )
    evaluation = act3.create_eval(client, f"sales-{kind}-3way", criteria)
    eval_id = getattr(evaluation, "id", None) or evaluation
    for candidate, file_id in _candidate_file_ids(args):
        run = act3.run_candidate(client, eval_id, candidate, file_id)
        run_id = getattr(run, "id", None) or run
        logger.info("[Act 3] %s: %s", candidate, act3.read_results(client, eval_id, run_id))
    return EXIT_SUCCESS


def _candidate_file_ids(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Parse ``--candidate name=file_id`` pairs into ``(name, file_id)`` tuples."""
    pairs: list[tuple[str, str]] = []
    for raw in getattr(args, "candidate", None) or []:
        name, _, file_id = raw.partition("=")
        if not file_id:
            logger.warning("[Act 3] Ignoring malformed --candidate %r (expected name=file_id)", raw)
            continue
        pairs.append((name, file_id))
    return pairs


# ---------------------------------------------------------------------------
# Act 4 — Hosting
# ---------------------------------------------------------------------------
def cmd_host(args: argparse.Namespace, config: DemoConfig) -> int:
    """Act 4: Developer-tier live deploy + one inference (GPU endpoint pre-baked)."""
    from . import act2a_serverless_sft as sft  # noqa: PLC0415

    if getattr(args, "prebaked", False):
        endpoint_name = getattr(args, "endpoint_name", None) or "sales-lora-endpoint"
        logger.info("[Act 4] Pre-baked GPU managed online endpoint '%s'", endpoint_name)
        return EXIT_SUCCESS

    model_id = _resolve_model_id(args, config, sft)
    if not model_id:
        logger.error(
            "[Act 4] No model id: pass --model-id, --job-id, or run live 'sft' first"
        )
        return EXIT_ERROR
    deployment_name = getattr(args, "deployment_name", None) or config.sft_deployment_name
    sft.deploy_finetuned(config, model_id, deployment_name, sku="developer")
    transcript = getattr(args, "transcript", None) or (
        "Caller asked for a quote and wants to start next month."
    )
    answer = sft.infer(config, deployment_name, transcript)
    logger.info("[Act 4] Live inference -> %s", answer)
    return EXIT_SUCCESS


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
def cmd_cleanup(args: argparse.Namespace, config: DemoConfig) -> int:
    """Tear down the live (Developer-tier) deployment to stop billing."""
    from . import act2a_serverless_sft as sft  # noqa: PLC0415

    deployment_name = getattr(args, "deployment_name", None) or config.sft_deployment_name
    if deployment_name:
        status = sft.delete_deployment(config, deployment_name)
        logger.info("[cleanup] Deleted deployment '%s' (status=%s)", deployment_name, status)
    else:
        logger.info("[cleanup] No deployment name provided; nothing to delete")

    endpoint_name = getattr(args, "endpoint_name", None)
    if endpoint_name:
        logger.info(
            "[cleanup] GPU endpoint '%s' deletion needs azure-ai-ml MLClient "
            "(hosting_managed_endpoint.delete_endpoint)",
            endpoint_name,
        )
    return EXIT_SUCCESS


# ---------------------------------------------------------------------------
# All — chain the live-able acts in pre-baked mode
# ---------------------------------------------------------------------------
def cmd_all(args: argparse.Namespace, config: DemoConfig) -> int:
    """Run the demo acts end to end (defaults to pre-baked for a dry run)."""
    args.prebaked = True
    for step in (cmd_gen_data, cmd_gpu_lora, cmd_evaluate):
        code = step(args, config)
        if code != EXIT_SUCCESS:
            logger.error("[all] Step %s returned %d; stopping", step.__name__, code)
            return code
    logger.info("[all] Completed the pre-baked dry run")
    return EXIT_SUCCESS


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def create_parser() -> argparse.ArgumentParser:
    """Create the argparse parser with one subcommand per demo act."""
    parser = argparse.ArgumentParser(
        prog="run_of_show",
        description="1-Hour Azure Fine-Tuning Demo run-of-show orchestrator.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--data-dir", type=Path, default=DATA_DIR, help="Folder for dataset JSONL files."
    )
    common.add_argument(
        "--prebaked",
        action="store_true",
        help="Use pre-baked artifacts instead of live Azure calls.",
    )

    sub = parser.add_subparsers(dest="command", required=True, metavar="ACT")

    gen = sub.add_parser("gen-data", parents=[common], help="Act 1: synthetic data factory.")
    gen.add_argument("--count", type=int, default=12, help="Train seeds to generate live.")
    gen.add_argument("--eval-count", type=int, default=200, help="Eval seeds (rare-buy prevalence).")
    gen.add_argument(
        "--dpo",
        action="store_true",
        help="Also write DPO preference pairs (dpo_train.jsonl / dpo_validation.jsonl).",
    )
    gen.add_argument(
        "--rft",
        action="store_true",
        help="Also write RFT graded records (rft_train.jsonl / rft_validation.jsonl).",
    )
    gen.set_defaults(func=cmd_gen_data)

    sft = sub.add_parser("sft", parents=[common], help="Act 2A: serverless SFT.")
    sft.add_argument("--train", help="Training JSONL path (defaults to data-dir).")
    sft.add_argument("--val", help="Validation JSONL path (defaults to data-dir).")
    sft.add_argument("--results-csv", help="Pre-baked results.csv for checkpoint selection.")
    sft.set_defaults(func=cmd_sft)

    dpo = sub.add_parser(
        "dpo", parents=[common], help="Act 2A: serverless DPO (preference fine-tuning)."
    )
    dpo.add_argument("--train", help="Preference training JSONL (defaults to data-dir).")
    dpo.add_argument("--val", help="Preference validation JSONL (defaults to data-dir).")
    dpo.add_argument("--n-epochs", type=int, default=2, help="Number of DPO epochs.")
    dpo.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO preference strength (higher = stay closer to the reference model).",
    )
    dpo.set_defaults(func=cmd_dpo)

    rft = sub.add_parser(
        "rft", parents=[common], help="Act 2A: serverless RFT (reinforcement fine-tuning)."
    )
    rft.add_argument("--train", help="RFT training JSONL (defaults to data-dir).")
    rft.add_argument("--val", help="RFT validation JSONL (defaults to data-dir).")
    rft.add_argument("--n-epochs", type=int, default=2, help="Number of RFT epochs.")
    rft.add_argument(
        "--grader",
        type=str,
        default="string_match",
        choices=["string_match", "model"],
        help="Grader type for RFT scoring.",
    )
    rft.set_defaults(func=cmd_rft)

    deploy = sub.add_parser("deploy", parents=[common], help="Act 2A: control-plane deploy.")
    deploy.add_argument("--model-id", help="Fine-tuned model id to deploy.")
    deploy.add_argument("--job-id", help="Fine-tune job id to resolve the model id from Azure.")
    deploy.add_argument(
        "--method",
        default="sft",
        choices=["sft", "dpo", "rft"],
        help="Which saved fine-tune state to deploy from (sft/dpo/rft).",
    )
    deploy.add_argument(
        "--final-model",
        action="store_true",
        help="Deploy the job's final model instead of the best-validation checkpoint.",
    )
    deploy.add_argument("--deployment-name", help="Target deployment name.")
    deploy.add_argument(
        "--sku",
        default="developer",
        choices=["developer", "standard", "globalStandard"],
        help="Deployment SKU (developer = cheap, 24h auto-delete).",
    )
    deploy.set_defaults(func=cmd_deploy)

    quick_eval = sub.add_parser(
        "quick-eval",
        parents=[common],
        help="Act 2A1: quick base-vs-fine-tuned comparison on the validation set.",
    )
    quick_eval.add_argument("--val", help="Validation JSONL path (defaults to data-dir).")
    quick_eval.add_argument(
        "--base-deployment", help="Base model deployment name (defaults to BASE_DEPLOYMENT_NAME)."
    )
    quick_eval.add_argument(
        "--ft-deployment",
        help="Fine-tuned deployment name (defaults to SFT_DEPLOYMENT_NAME).",
    )
    quick_eval.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to pause between inference calls (eases rate-limited deployments; default 0.5).",
    )
    quick_eval.set_defaults(func=cmd_quick_eval)

    foundry_eval = sub.add_parser(
        "foundry-eval",
        parents=[common],
        help="Act 3A: submit Foundry portal evaluations for base/SFT/DPO/RFT.",
    )
    foundry_eval.add_argument(
        "--models",
        nargs="+",
        choices=["base", "sft", "dpo", "rft"],
        help="Model arms to evaluate (default: all configured).",
    )
    foundry_eval.add_argument("--val", help="Validation JSONL path (defaults to data-dir).")
    foundry_eval.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N validation rows (fast live demo).",
    )
    foundry_eval.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to pause between inference calls (eases rate limits; default 0.5).",
    )
    foundry_eval.add_argument(
        "--no-builtin",
        action="store_true",
        help="Skip the built-in F1ScoreEvaluator (custom metrics only).",
    )
    foundry_eval.add_argument(
        "--no-upload",
        action="store_true",
        help="Run locally without uploading runs to the Foundry portal.",
    )
    foundry_eval.set_defaults(func=cmd_foundry_eval)

    gpu = sub.add_parser("gpu-lora", parents=[common], help="Act 2B: GPU LoRA (pre-baked).")
    gpu.set_defaults(func=cmd_gpu_lora)

    evaluate = sub.add_parser("evaluate", parents=[common], help="Act 3: 3-way eval + offline metrics.")
    evaluate.add_argument(
        "--kind",
        default="propensity",
        choices=["propensity", "intent"],
        help="Which task arm to score.",
    )
    evaluate.add_argument(
        "--predictions",
        nargs="+",
        help="Candidate prediction JSONL paths (defaults to data-dir files).",
    )
    evaluate.add_argument(
        "--candidate",
        action="append",
        metavar="NAME=FILE_ID",
        help="Live run candidate as name=uploaded_file_id (repeatable).",
    )
    evaluate.set_defaults(func=cmd_evaluate)

    host = sub.add_parser("host", parents=[common], help="Act 4: Developer-tier host + inference.")
    host.add_argument("--model-id", help="Fine-tuned model id to host live.")
    host.add_argument("--job-id", help="Fine-tune job id to resolve the model id from Azure.")
    host.add_argument(
        "--method",
        default="sft",
        choices=["sft", "dpo", "rft"],
        help="Which saved fine-tune state to host from (sft/dpo/rft).",
    )
    host.add_argument(
        "--final-model",
        action="store_true",
        help="Host the job's final model instead of the best-validation checkpoint.",
    )
    host.add_argument("--deployment-name", help="Developer-tier deployment name.")
    host.add_argument("--endpoint-name", help="Pre-baked GPU endpoint name.")
    host.add_argument("--transcript", help="Transcript to classify in the live inference.")
    host.set_defaults(func=cmd_host)

    cleanup = sub.add_parser("cleanup", parents=[common], help="Delete the live deployment.")
    cleanup.add_argument("--deployment-name", help="Developer-tier deployment to delete.")
    cleanup.add_argument("--endpoint-name", help="GPU endpoint to delete (guidance only).")
    cleanup.set_defaults(func=cmd_cleanup)

    run_all = sub.add_parser("all", parents=[common], help="Run the acts end to end (pre-baked).")
    run_all.add_argument("--count", type=int, default=12, help="Train seeds for the gen-data step.")
    run_all.add_argument("--eval-count", type=int, default=200, help="Eval seeds for the gen-data step.")
    run_all.add_argument("--kind", default="propensity", choices=["propensity", "intent"])
    run_all.set_defaults(func=cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, build config once, and dispatch the chosen act."""
    parser = create_parser()
    args = parser.parse_args(argv)
    configure_logging(getattr(args, "verbose", False))
    try:
        config = _build_config()
        return int(args.func(args, config))
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as exc:  # surface actionable errors without a raw traceback
        logger.error("%s", exc)
        return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
