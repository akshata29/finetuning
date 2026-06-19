"""Managed-online-endpoint scoring script for base + LoRA adapter inference.

Implements the Azure ML inference-server contract (:func:`init` + :func:`run`)
for a Phi-4-mini base model with a separately-registered ``peft`` LoRA adapter.
At ``init()`` the base model is loaded and the adapter is attached and merged
(``merge_and_unload``) for lower inference latency; ``run()`` builds a chat
prompt per record, generates, and coerces the completion into a demo-taxonomy
label.

Label coercion is bound to the **demo taxonomy** (validator finding DR-10): the
allowed labels are the intent classes plus the binary ``buy`` / ``not_buy``
propensity labels imported from :mod:`finetuning.taxonomy` — NOT the
research placeholder ``hot`` / ``warm`` / ``cold``. Scoring and evaluation thus
share one source of truth.

The heavy ML dependencies (``torch`` / ``transformers`` / ``peft``) are imported
lazily inside :func:`init` so that this file can be imported for unit tests with
those modules mocked via ``sys.modules`` (and so it stays import-safe in
SDK-free environments).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from finetuning.taxonomy import INTENT_LABELS, PROPENSITY_LABELS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allowed label space — bound to the demo taxonomy (DR-10).
#
# Intent classes + binary propensity (buy/not_buy). This is the single source
# of truth shared with evaluation; the research placeholder hot/warm/cold is
# explicitly NOT used.
# ---------------------------------------------------------------------------
ALLOWED_LABELS: list[str] = [*INTENT_LABELS, *PROPENSITY_LABELS]

UNKNOWN_LABEL: str = "unknown"

# Base model id. For a fully Azure-only / air-gapped demo, register the base
# weights as a second model asset and point BASE_MODEL_ID at it instead of a HF
# id. Read from the environment only — never hardcode secrets/endpoints.
BASE_MODEL_ID: str = os.getenv("BASE_MODEL_ID", "microsoft/Phi-4-mini-instruct")

# Subfolder of the registered model asset that holds the LoRA adapter
# (adapter_config.json + adapter_model.safetensors).
ADAPTER_SUBDIR: str = os.getenv("LORA_ADAPTER_SUBDIR", "lora_adapter")

# Module-level globals populated by init().
_model: Any = None
_tokenizer: Any = None
_device: str = "cpu"


def init() -> None:
    """Load the base model + LoRA adapter once at container start.

    Heavy imports happen here (not at module scope) so the file imports cleanly
    with ``torch`` / ``transformers`` / ``peft`` mocked or absent.
    """
    global _model, _tokenizer, _device

    import torch  # noqa: PLC0415 — lazy, runtime-only import
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    model_root = os.getenv("AZUREML_MODEL_DIR", "")
    adapter_dir = os.path.join(model_root, ADAPTER_SUBDIR)

    logger.info("Loading base model %s", BASE_MODEL_ID)
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,  # A100 supports bf16; float16 on older GPUs
        device_map={"": 0} if _device == "cuda" else None,
        trust_remote_code=True,
    )

    logger.info("Attaching LoRA adapter from %s", adapter_dir)
    peft_model = PeftModel.from_pretrained(base, adapter_dir)
    # Merge for lower inference latency. Skip merge to hot-swap adapters.
    _model = peft_model.merge_and_unload()
    _model.eval()
    logger.info("init complete on device=%s", _device)


def _build_prompt(record: dict[str, Any]) -> str:
    """Build a chat prompt for one sales-call record."""
    system = (
        "You are a sales-call classifier. Given a sales call summary, respond "
        f"with EXACTLY one label from {ALLOWED_LABELS} as JSON: "
        '{"label": "<label>"}.'
    )
    user = json.dumps(record, ensure_ascii=False)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def coerce_label(text: str) -> str:
    """Coerce raw model output to an allowed demo-taxonomy label.

    Matching is case-insensitive and prefers an exact token match before
    falling back to a substring scan. Returns :data:`UNKNOWN_LABEL` when no
    allowed label is found. The returned value is always either a member of
    :data:`ALLOWED_LABELS` (intent class or ``buy``/``not_buy``) or
    ``"unknown"`` — never ``hot``/``warm``/``cold``.
    """
    lowered = text.lower()

    # Try to parse a JSON {"label": "..."} envelope first.
    try:
        parsed = json.loads(text)
        candidate = str(parsed.get("label", "")).lower()
        for label in ALLOWED_LABELS:
            if candidate == label.lower():
                return label
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass

    # Exact whole-token match wins over a loose substring match.
    tokens = {tok.strip(" \t\r\n\"'{}:,.") for tok in lowered.split()}
    for label in ALLOWED_LABELS:
        if label.lower() in tokens:
            return label
    for label in ALLOWED_LABELS:
        if label.lower() in lowered:
            return label
    return UNKNOWN_LABEL


def run(raw_data: str) -> dict[str, Any]:
    """Score a batch of records for an endpoint invocation.

    Accepts a JSON body shaped as ``{"records": [...]}`` or
    ``{"input_data": {...}}`` (single record). Returns
    ``{"predictions": [{"raw": str, "label": str}, ...]}`` or ``{"error": str}``.
    """
    import torch  # noqa: PLC0415 — lazy, runtime-only import

    try:
        payload = json.loads(raw_data)
        records = payload.get("records") or [payload.get("input_data", payload)]

        prompts = [_build_prompt(r) for r in records]
        inputs = _tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(_device)

        with torch.no_grad():
            generated = _model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,  # deterministic for a classifier
                pad_token_id=_tokenizer.pad_token_id,
            )

        results: list[dict[str, str]] = []
        prompt_len = inputs["input_ids"].shape[1]
        for i in range(len(prompts)):
            new_tokens = generated[i][prompt_len:]
            text = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            results.append({"raw": text, "label": coerce_label(text)})

        return {"predictions": results}
    except Exception as exc:  # surface errors in deployment logs
        logger.exception("scoring failed")
        return {"error": str(exc)}
