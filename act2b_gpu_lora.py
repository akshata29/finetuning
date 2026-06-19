"""Act 2B — GPU-based LoRA / QLoRA fine-tuning on Azure ML managed compute.

This module builds the Path B (BYO-GPU, open-weight) fine-tuning jobs for the
demo. It is import-safe with **no Azure SDKs installed** — the ``azure-ai-ml``
surface is resolved lazily through :func:`finetuning_demo.config.optional_import`
and every job-builder degrades gracefully with an actionable error when the SDK
is absent.

Three surfaces are provided, and they are deliberately kept distinct because the
LoRA hyperparameter keys are valid on only ONE of them (validator findings
DD-04 / DR-07):

* :func:`create_pipeline_lora_job` — **PRIMARY**. The classic hub-based
  ``text_generation_pipeline`` component from the ``azureml`` registry. This is
  the only surface where the confirmed LoRA keys (``apply_lora`` / ``lora_r`` /
  ``lora_alpha`` / ``lora_dropout`` / ``precision``) are valid (DR-04).
* :func:`create_maas_finetuning_job` — **ALTERNATIVE (flagged)**. The new
  Foundry MaaS ``create_finetuning_job`` surface. Its LoRA / quantization
  hyperparameter keys are *unpublished*; the classic keys MUST NOT be passed
  here and any LoRA dict must be verified against a live API before use (DR-07).
* :func:`custom_qlora_command_job_spec` — **PARITY PATH**. A raw ``command`` job
  spec (transformers + peft + bitsandbytes, 4-bit ``load_in_4bit``,
  ``target_modules``) for true QLoRA parity, which the managed pipeline cannot
  express (DD-02 — the component exposes plain LoRA only, no 4-bit / no
  ``target_modules``).

No secrets or endpoints are hardcoded; resource identifiers come from the
caller-supplied :class:`~finetuning_demo.config.DemoConfig` / ``MLClient``.
"""

from __future__ import annotations

import logging
from typing import Any

from finetuning_demo.config import optional_import

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Confirmed classic-pipeline LoRA defaults (Gap 4.1, authoritative).
#
# These keys are valid ONLY on the classic ``text_generation_pipeline``
# component (see :func:`create_pipeline_lora_job`). ``precision`` accepts only
# "16" or "32"; there is NO 4-bit / 8-bit (QLoRA) knob on this surface — that
# parity lives in :func:`custom_qlora_command_job_spec`.
# ---------------------------------------------------------------------------
PIPELINE_LORA_DEFAULTS: dict[str, Any] = {
    "apply_lora": "true",  # string "true"/"false" per component contract
    "lora_r": 8,
    "lora_alpha": 128,
    "lora_dropout": 0.0,
    "precision": "16",  # only "16" or "32" are accepted by the component
}

# Standard Hugging Face trainer keys exposed by the classic component.
PIPELINE_TRAINER_DEFAULTS: dict[str, Any] = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "learning_rate": 1e-4,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 50,
    "optim": "adamw_torch",
    "weight_decay": 0.0,
    "gradient_accumulation_steps": 8,
    "max_seq_length": 2048,
    "seed": 1337,
}

# Default open-weight base model for Path B and the registry that hosts the
# classic finetune pipeline component.
DEFAULT_BASE_MODEL: str = "microsoft/Phi-4-mini-instruct"
PIPELINE_COMPONENT_REGISTRY: str = "azureml"
PIPELINE_COMPONENT_NAME: str = "text_generation_pipeline"

# Default GPU compute SKU for the managed-compute finetune job (A100 80GB).
DEFAULT_FINETUNE_INSTANCE_TYPE: str = "Standard_NC24ads_A100_v4"


def _require_azure_ml() -> Any:
    """Return the imported ``azure.ai.ml`` module or raise an actionable error.

    Keeps the module import-safe: the SDK is only required at call time, so unit
    tests and SDK-free environments can import this file freely.
    """
    module, available = optional_import("azure.ai.ml")
    if not available:
        raise ImportError(
            "azure-ai-ml is required to build GPU LoRA fine-tuning jobs. "
            "Install with: pip install azure-ai-ml azure-identity"
        )
    return module


def create_pipeline_lora_job(
    ml_client: Any,
    *,
    training_data: str,
    validation_data: str | None = None,
    base_model: str = DEFAULT_BASE_MODEL,
    compute: str | None = None,
    instance_type: str = DEFAULT_FINETUNE_INSTANCE_TYPE,
    lora_overrides: dict[str, Any] | None = None,
    trainer_overrides: dict[str, Any] | None = None,
    submit: bool = False,
) -> Any:
    """Build (and optionally submit) the PRIMARY classic-pipeline LoRA job.

    This targets the ``azureml`` registry ``text_generation_pipeline`` component
    — the classic hub-based managed-compute path (DR-04). **This is the only
    surface where the LoRA keys in :data:`PIPELINE_LORA_DEFAULTS` are valid**;
    they are passed straight through to the component here, and deliberately NOT
    to the MaaS surface in :func:`create_maas_finetuning_job`.

    Parameters
    ----------
    ml_client:
        A live ``azure.ai.ml.MLClient`` used to resolve the registry component
        and (when ``submit`` is true) create the job.
    training_data / validation_data:
        Paths / URIs to the JSONL training (and optional validation) data.
    base_model:
        Open-weight base model id (default Phi-4-mini-instruct).
    compute / instance_type:
        Target named compute cluster, or the serverless ``instance_type`` GPU
        SKU when ``compute`` is omitted.
    lora_overrides / trainer_overrides:
        Optional per-run overrides merged over :data:`PIPELINE_LORA_DEFAULTS`
        and :data:`PIPELINE_TRAINER_DEFAULTS`.
    submit:
        When true, submit the job via ``ml_client.jobs.create_or_update`` and
        return the returned job; otherwise return the unsubmitted pipeline job.

    Returns
    -------
    Any
        The constructed (or submitted) Azure ML pipeline job.

    Raises
    ------
    ImportError
        If ``azure-ai-ml`` is not installed.
    """
    aml = _require_azure_ml()
    # ``Input`` lives on the top-level azure.ai.ml namespace.
    from azure.ai.ml import Input  # noqa: PLC0415 — lazy, SDK-gated import
    from azure.ai.ml.constants import AssetTypes  # noqa: PLC0415

    lora_params = {**PIPELINE_LORA_DEFAULTS, **(lora_overrides or {})}
    trainer_params = {**PIPELINE_TRAINER_DEFAULTS, **(trainer_overrides or {})}

    # Resolve the classic finetune pipeline component from the azureml registry.
    component = ml_client.components.get(
        name=PIPELINE_COMPONENT_NAME, label="latest"
    )

    inputs: dict[str, Any] = {
        "model_name": base_model,
        "train_file_path": Input(type=AssetTypes.URI_FILE, path=training_data),
        # The confirmed LoRA keys are valid HERE (classic pipeline only).
        **lora_params,
        **trainer_params,
    }
    if validation_data is not None:
        inputs["validation_file_path"] = Input(
            type=AssetTypes.URI_FILE, path=validation_data
        )

    pipeline_job = component(**inputs)

    # Attach compute: named cluster takes precedence, else serverless SKU.
    if compute is not None:
        pipeline_job.compute = compute
    else:
        pipeline_job.settings = getattr(pipeline_job, "settings", None)
        pipeline_job.resources = {"instance_type": instance_type}

    logger.info(
        "Built classic-pipeline LoRA job (base=%s, lora_r=%s, lora_alpha=%s, "
        "precision=%s)",
        base_model,
        lora_params.get("lora_r"),
        lora_params.get("lora_alpha"),
        lora_params.get("precision"),
    )

    if submit:
        return ml_client.jobs.create_or_update(pipeline_job)
    # Reference ``aml`` so the import is meaningfully used for callers/tests.
    logger.debug("azure.ai.ml version: %s", getattr(aml, "__version__", "unknown"))
    return pipeline_job


def create_maas_finetuning_job(
    ml_client: Any,
    *,
    training_data: str,
    validation_data: str | None = None,
    base_model: str = DEFAULT_BASE_MODEL,
    hyperparameters: dict[str, Any] | None = None,
    submit: bool = False,
) -> Any:
    """Build the ALTERNATIVE (flagged) MaaS fine-tuning job.

    Uses the new Foundry ``azure.ai.ml.finetuning.create_finetuning_job`` surface
    with ``FineTuningTaskType.CHAT_COMPLETION``.

    .. warning::
        The LoRA / quantization hyperparameter dict keys accepted by this MaaS
        surface are **unpublished** (DR-07). The confirmed classic-pipeline LoRA
        keys in :data:`PIPELINE_LORA_DEFAULTS` are deliberately NOT applied here
        — passing them is unverified and may be silently ignored or rejected.
        Any ``hyperparameters`` you supply MUST be verified against a live API
        run before relying on them.

    Parameters
    ----------
    ml_client:
        A live ``azure.ai.ml.MLClient`` (used when ``submit`` is true).
    training_data / validation_data:
        JSONL data paths / URIs for the chat-completion fine-tune.
    base_model:
        Open-weight base model id.
    hyperparameters:
        Optional, UNVERIFIED hyperparameter dict passed through verbatim. No
        classic LoRA keys are injected by this function.
    submit:
        When true, create the job via the MaaS surface and return it.

    Returns
    -------
    Any
        The constructed (or submitted) MaaS fine-tuning job.

    Raises
    ------
    ImportError
        If ``azure-ai-ml`` (with the ``finetuning`` extra) is not installed.
    """
    _require_azure_ml()
    finetuning, available = optional_import("azure.ai.ml.finetuning")
    if not available:
        raise ImportError(
            "azure-ai-ml finetuning surface is required for the MaaS path. "
            "Install with: pip install azure-ai-ml azure-identity"
        )
    from azure.ai.ml.constants import FineTuningTaskType  # noqa: PLC0415

    if hyperparameters:
        logger.warning(
            "MaaS create_finetuning_job hyperparameter keys are UNPUBLISHED "
            "(DR-07). Verify %s against a live API run before relying on it.",
            sorted(hyperparameters),
        )

    # NOTE: classic LoRA keys (PIPELINE_LORA_DEFAULTS) are intentionally NOT
    # passed to this surface — they are valid only on the classic pipeline.
    job = finetuning.create_finetuning_job(
        task=FineTuningTaskType.CHAT_COMPLETION,
        model=base_model,
        training_data=training_data,
        validation_data=validation_data,
        hyperparameters=hyperparameters or None,
    )

    if submit:
        return ml_client.jobs.create_or_update(job)
    return job


def custom_qlora_command_job_spec(
    *,
    training_data: str = "azureml://datastores/workspaceblobstore/paths/train.jsonl",
    base_model: str = DEFAULT_BASE_MODEL,
    instance_type: str = DEFAULT_FINETUNE_INSTANCE_TYPE,
    target_modules: list[str] | None = None,
) -> dict[str, Any]:
    """Return a ``command`` job spec for true 4-bit QLoRA parity (DD-02).

    The classic managed pipeline exposes plain LoRA only (no 4-bit, no
    ``target_modules``). For exact parity with a customer's GCP QLoRA workflow
    this returns a self-contained ``command`` job description that runs a
    ``transformers`` + ``peft`` + ``bitsandbytes`` recipe with ``load_in_4bit``
    and explicit ``target_modules``. It is returned as a plain dict so it can be
    inspected / serialized with no SDK installed; a caller with ``azure-ai-ml``
    can hydrate it into a ``command`` entity.

    This is the **parity path** — clearly distinct from the managed-pipeline and
    MaaS surfaces above.

    Parameters
    ----------
    training_data:
        Datastore URI for the QLoRA training JSONL.
    base_model:
        Open-weight base model id to quantize to 4-bit.
    instance_type:
        GPU SKU for the command job.
    target_modules:
        LoRA target projection layers (defaults to the common attention/MLP
        projections for Llama/Phi-style architectures).

    Returns
    -------
    dict[str, Any]
        A command-job spec dict (command, environment, inputs, resources).
    """
    modules = target_modules or [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    command = (
        "pip install transformers peft bitsandbytes accelerate datasets && "
        "python qlora_train.py "
        "--base_model ${{inputs.base_model}} "
        "--train_file ${{inputs.training_data}} "
        "--load_in_4bit true "
        "--bnb_4bit_quant_type nf4 "
        "--bnb_4bit_compute_dtype bfloat16 "
        "--lora_r 8 --lora_alpha 128 --lora_dropout 0.0 "
        f"--target_modules {','.join(modules)}"
    )
    return {
        # PARITY PATH: 4-bit QLoRA via a BYO command job (not the managed
        # pipeline, which cannot express load_in_4bit / target_modules).
        "type": "command",
        "command": command,
        "code": "./qlora",
        "environment": (
            "azureml://registries/azureml/environments/acpt-pytorch-2.2-cuda12.1/labels/latest"
        ),
        "inputs": {
            "base_model": base_model,
            "training_data": {"type": "uri_file", "path": training_data},
        },
        "resources": {"instance_type": instance_type, "instance_count": 1},
        "lora": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
            "lora_r": 8,
            "lora_alpha": 128,
            "lora_dropout": 0.0,
            "target_modules": modules,
        },
    }
