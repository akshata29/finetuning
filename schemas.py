"""Strict synthetic-record schema loader and JSONL writers for the demo.

Loads the strict ``synthetic-sales-call`` JSON schema (used for the teacher
model's Structured Outputs call) and provides JSONL writers for the three
training/eval formats:

* :func:`write_sft_jsonl` — OpenAI chat-message SFT shape.
* :func:`write_dpo_jsonl` — OpenAI preference (DPO) shape.
* :func:`write_eval_jsonl` — bring-your-own-predictions eval rows with
  ``ground_truth`` and ``propensity_score``.

All writers emit UTF-8 **with BOM** (``encoding="utf-8-sig"``) so the first
bytes of every file are ``\\xef\\xbb\\xbf`` — required by the Azure fine-tuning
upload path (addresses discrepancy DR-01).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------
_SCHEMAS_DIR = Path(__file__).resolve().parent / "schemas"
_SYNTHETIC_SCHEMA_PATH = _SCHEMAS_DIR / "synthetic-sales-call-schema.json"


def _load_synthetic_schema() -> dict[str, Any]:
    """Load the strict synthetic-sales-call JSON schema from ``schemas/``."""
    with _SYNTHETIC_SCHEMA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


#: The strict JSON schema for a single synthetic sales-call record.
SYNTHETIC_SCHEMA: dict[str, Any] = _load_synthetic_schema()

#: System prompt used to frame the classification target in SFT/DPO rows.
SYSTEM_PROMPT: str = (
    "You are a sales-call classifier. Given a call transcript, return the "
    "caller intent, the call outcome, and a buy propensity_score in [0,1]."
)


def rft_response_format_schema() -> dict[str, Any]:
    """Return the JSON-schema ``response_format`` for an Azure RFT job.

    RFT graders reference the model output as ``{{ sample.output_json.<field> }}``,
    which only works when the job constrains the model to emit structured JSON.
    This schema locks the output to ``intent`` / ``outcome`` / ``propensity_score``
    so the ``string_check`` graders can read each field. The label enums are
    imported lazily to keep this module import-light.
    """
    from .taxonomy import INTENT_LABELS, OUTCOME_LABELS  # noqa: PLC0415

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "sales_call_label",
            "schema": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string", "enum": list(INTENT_LABELS)},
                    "outcome": {"type": "string", "enum": list(OUTCOME_LABELS)},
                    "propensity_score": {"type": "number"},
                },
                "required": ["intent", "outcome", "propensity_score"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> int:
    """Write ``rows`` as UTF-8-with-BOM JSONL; return the count written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
    logger.info("Wrote %d record(s) to %s", len(rows), out)
    return len(rows)


def _assistant_label(record: dict[str, Any]) -> str:
    """Serialize the label fields of a synthetic record for the assistant turn."""
    return json.dumps(
        {
            "intent": record.get("intent"),
            "outcome": record.get("outcome"),
            "propensity_score": record.get("propensity_score"),
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_sft_jsonl(records: list[dict[str, Any]], path: str | Path) -> int:
    """Write SFT rows in OpenAI chat-message shape (UTF-8 with BOM).

    Each output row is ``{"messages": [system, user, assistant]}``. Records that
    already carry a ``"messages"`` key are written verbatim; otherwise a row is
    built from the record's ``transcript`` (user turn) and label fields
    (assistant turn).
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        if "messages" in record:
            rows.append({"messages": record["messages"]})
            continue
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": record.get("transcript", "")},
                    {"role": "assistant", "content": _assistant_label(record)},
                ]
            }
        )
    return _write_jsonl(rows, path)


def write_dpo_jsonl(records: list[dict[str, Any]], path: str | Path) -> int:
    """Write preference (DPO) rows in OpenAI format (UTF-8 with BOM).

    Each output row is
    ``{"input": {"messages": [system, user]}, "preferred_output": [...],
    "non_preferred_output": [...]}``. Records already containing
    ``input``/``preferred_output``/``non_preferred_output`` are written verbatim;
    otherwise a row is built from ``transcript`` plus ``preferred`` and
    ``non_preferred`` assistant responses.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        if {"input", "preferred_output", "non_preferred_output"} <= record.keys():
            rows.append(
                {
                    "input": record["input"],
                    "preferred_output": record["preferred_output"],
                    "non_preferred_output": record["non_preferred_output"],
                }
            )
            continue
        rows.append(
            {
                "input": {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": record.get("transcript", "")},
                    ]
                },
                "preferred_output": [
                    {"role": "assistant", "content": record.get("preferred", "")}
                ],
                "non_preferred_output": [
                    {"role": "assistant", "content": record.get("non_preferred", "")}
                ],
            }
        )
    return _write_jsonl(rows, path)


def write_eval_jsonl(records: list[dict[str, Any]], path: str | Path) -> int:
    """Write eval rows with ground_truth + propensity_score (UTF-8 with BOM).

    Each output row carries a stable ``id`` plus ``query``, ``response``,
    ``ground_truth``, and ``propensity_score``. When a record omits ``id`` a
    stable index-based identifier (``eval-000000``) is assigned. ``query`` falls
    back to the record's ``transcript`` when not provided.
    """
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        rows.append(
            {
                "id": record.get("id", f"eval-{index:06d}"),
                "query": record.get("query", record.get("transcript", "")),
                "response": record.get("response", ""),
                "ground_truth": record.get("ground_truth", ""),
                "propensity_score": record.get("propensity_score"),
            }
        )
    return _write_jsonl(rows, path)


def write_rft_jsonl(records: list[dict[str, Any]], path: str | Path) -> int:
    """Write Reinforcement Fine-Tuning (RFT) rows in Azure's format (UTF-8 with BOM).

    Azure RFT expects each row to carry a ``messages`` array in chat format whose
    **final message has the** ``user`` **role**, plus any extra reference fields
    at the top level for the server-side grader to read as ``{{ item.<field> }}``
    (see the RFT how-to). The instruction turn uses the ``developer`` role (not
    ``system``): o-series reasoning models such as o4-mini reject the chat-SFT
    ("FinetuneChat") format produced by a ``system`` role. Each output row is
    therefore ``{"messages": [developer, user], "intent": ..., "outcome": ...,
    "propensity_score": ...}``.

    Records already containing a ``messages`` key are written verbatim;
    otherwise a row is built from ``transcript`` plus whichever of ``intent`` /
    ``outcome`` / ``propensity_score`` reference fields are present.
    """
    reference_fields = ("intent", "outcome", "propensity_score")
    rows: list[dict[str, Any]] = []
    for record in records:
        if "messages" in record:
            rows.append(record)
            continue
        row: dict[str, Any] = {
            "messages": [
                {"role": "developer", "content": SYSTEM_PROMPT},
                {"role": "user", "content": record.get("transcript", "")},
            ]
        }
        for field in reference_fields:
            if record.get(field) is not None:
                row[field] = record[field]
        rows.append(row)
    return _write_jsonl(rows, path)
