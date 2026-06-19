"""Distill captured Agent transcripts into a retraining corpus (close the loop).

This is the **"close the loop"** stage of the conversation-alignment demo. The
Foundry Agent Service step (:mod:`finetuning_demo.foundry_agent_service`) replays
held-out conversations through the deployed Agent and records, for each case, the
*Agent's actual reply* alongside the *exemplary* ``ground_truth`` reply. This
module turns those paired transcripts into a fresh training corpus so the next
fine-tune learns from the model's own mistakes:

* **SFT corpus** — every conversation context is paired with the exemplary
  ``ground_truth`` reply as the assistant target, so supervised fine-tuning
  imitates the strategy the Agent failed to follow.
* **DPO corpus** — the same context becomes a preference pair: the exemplary
  ``ground_truth`` is ``preferred_output`` and the Agent's real reply is
  ``non_preferred_output``. This is the most direct "prefer the exemplar over
  what we actually shipped" signal.

By default the corpus is focused on the cases worth correcting (the Agent's
``strategy_score`` below a threshold), which is exactly the data that moves the
model. Pass ``--all`` to distill every case.

Design choices mirror the rest of the demo:

* **Load / import safe (OWASP A05/A06).** No Azure SDK import at module load; the
  distillation is pure local file transformation, so importing this module never
  requires Azure to be configured.
* **Reuses the canonical writers.** The SFT/DPO JSONL is produced by the same
  :func:`write_conv_sft_jsonl` / :func:`write_conv_dpo_jsonl` the synthetic
  generator uses, so the distilled files are byte-compatible with the training
  path and can be retrained simply by pointing ``run_finetune`` at the output
  directory.
* **Data isolation.** The distilled corpus lives under
  ``data/conversation_alignment/distilled/`` so it never clobbers the original
  synthetic corpus until you choose to retrain from it.

This module deliberately does **not** use ``from __future__ import annotations``
to stay consistent with :mod:`customer_conversation_alignment`.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

# Allow running both as a module and as a plain script.
if __package__ in (None, ""):  # pragma: no cover - script-launch shim
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "finetuning_demo"

from .config import DemoConfig
from .customer_conversation_alignment import (
    CONV_DPO_TRAIN_FILE,
    CONV_DPO_VAL_FILE,
    CONV_SFT_TRAIN_FILE,
    CONV_SFT_VAL_FILE,
    DATA_DIR,
    EXIT_ERROR,
    EXIT_SUCCESS,
    load_eval_rows,
    write_conv_dpo_jsonl,
    write_conv_sft_jsonl,
)
from .foundry_agent_service import AGENT_SERVICE_DIR, AGENT_TRANSCRIPTS_FILE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
#: Where the distilled retrain corpus is written (isolated from the synthetic one).
DISTILLED_DIR: Path = DATA_DIR / "distilled"

#: Default source: the transcripts the Agent test step captured.
DEFAULT_TRANSCRIPTS: Path = AGENT_SERVICE_DIR / AGENT_TRANSCRIPTS_FILE

#: Held-out eval rows, used to recover each case's full multi-turn context.
DEFAULT_EVAL_FILE: Path = DATA_DIR / "conv_eval.jsonl"

#: Provenance + counts summary written next to the distilled corpus.
DISTILL_REPORT_FILE: str = "distill_summary.json"

#: Cases scoring at/above this keep the model's behavior; below it is "worth
#: correcting." Score is the 0-1 strategy-alignment metric.
DEFAULT_SCORE_THRESHOLD: float = 0.75

#: Fraction of distilled rows held out as validation for the retrain.
DEFAULT_VAL_FRACTION: float = 0.2


# ---------------------------------------------------------------------------
# Core distillation
# ---------------------------------------------------------------------------
def _load_transcripts(path: str | Path) -> list[dict[str, Any]]:
    """Read the Agent test transcript JSONL (one record per case)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No Agent transcripts at {path}. Run `agent-test` first to capture "
            f"real conversations to distill from."
        )
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _index_eval_contexts(eval_path: str | Path) -> dict[str, list[dict[str, str]]]:
    """Map each eval row id -> its full ``context_messages`` (system + turns)."""
    index: dict[str, list[dict[str, str]]] = {}
    for row in load_eval_rows(eval_path):
        row_id = row.get("id")
        context = row.get("context_messages")
        if row_id is not None and context:
            index[str(row_id)] = context
    return index


def _split_train_val(
    records: list[dict[str, Any]], val_fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically split records into (train, val) by position.

    A positional split (every Nth row to validation) keeps the function pure and
    reproducible without a RNG, and guarantees at least one row in each split
    when there are >= 2 records.
    """
    if len(records) < 2:
        return records, []
    val_fraction = min(max(val_fraction, 0.0), 0.9)
    stride = max(2, round(1 / val_fraction)) if val_fraction > 0 else 0
    if stride == 0:
        return records, []
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if (index + 1) % stride == 0:
            val.append(record)
        else:
            train.append(record)
    if not val:  # tiny corpora — peel one row off the end
        val.append(train.pop())
    return train, val


def distill_corpus(
    *,
    transcripts_path: str | Path = DEFAULT_TRANSCRIPTS,
    eval_path: str | Path = DEFAULT_EVAL_FILE,
    out_dir: str | Path = DISTILLED_DIR,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    include_all: bool = False,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    save: bool = True,
) -> dict[str, Any]:
    """Build a retrain corpus from captured Agent transcripts.

    Pairs every selected case's conversation context with the exemplary
    ``ground_truth`` reply (preferred) and the Agent's actual reply
    (non-preferred), writing both an SFT corpus (imitate the exemplar) and a DPO
    corpus (prefer the exemplar over what we shipped). By default only cases that
    scored below ``score_threshold`` are distilled (the ones worth correcting);
    ``include_all`` distills every completed case.

    Returns a summary with the per-corpus row counts, the selection stats, and
    (when ``save``) the written artifact paths.
    """
    transcripts = _load_transcripts(transcripts_path)
    contexts = _index_eval_contexts(eval_path)

    selected: list[dict[str, Any]] = []
    skipped_no_context = 0
    skipped_incomplete = 0
    skipped_above_threshold = 0

    for row in transcripts:
        if str(row.get("run_status")) != "completed":
            skipped_incomplete += 1
            continue
        agent_reply = (row.get("agent_response") or "").strip()
        ground_truth = (row.get("ground_truth") or "").strip()
        if not agent_reply or not ground_truth:
            skipped_incomplete += 1
            continue
        score = float(row.get("strategy_score") or 0.0)
        if not include_all and score >= score_threshold:
            skipped_above_threshold += 1
            continue
        row_id = str(row.get("id"))
        context = contexts.get(row_id)
        if not context:
            skipped_no_context += 1
            continue
        selected.append(
            {
                "id": row_id,
                "strategy": row.get("strategy", ""),
                "context_messages": context,
                # SFT target = the exemplary reply appended as the assistant turn.
                "messages": context + [{"role": "assistant", "content": ground_truth}],
                # DPO pair.
                "preferred": ground_truth,
                "non_preferred": agent_reply,
                "strategy_score": score,
            }
        )

    summary: dict[str, Any] = {
        "source_transcripts": str(transcripts_path),
        "eval_source": str(eval_path),
        "include_all": include_all,
        "score_threshold": score_threshold,
        "transcripts_total": len(transcripts),
        "selected": len(selected),
        "skipped_incomplete": skipped_incomplete,
        "skipped_above_threshold": skipped_above_threshold,
        "skipped_no_context": skipped_no_context,
        "sft_rows": {"train": 0, "val": 0},
        "dpo_rows": {"train": 0, "val": 0},
    }

    if not selected:
        logger.warning("Nothing to distill (no cases matched the selection).")
        summary["artifacts"] = {}
        return summary

    train, val = _split_train_val(selected, val_fraction)

    if save:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        sft_train = write_conv_sft_jsonl(train, out_path / CONV_SFT_TRAIN_FILE)
        sft_val = write_conv_sft_jsonl(val, out_path / CONV_SFT_VAL_FILE)
        dpo_train = write_conv_dpo_jsonl(train, out_path / CONV_DPO_TRAIN_FILE)
        dpo_val = write_conv_dpo_jsonl(val, out_path / CONV_DPO_VAL_FILE)
        summary["sft_rows"] = {"train": sft_train, "val": sft_val}
        summary["dpo_rows"] = {"train": dpo_train, "val": dpo_val}
        summary["artifacts"] = {
            "dir": str(out_path),
            CONV_SFT_TRAIN_FILE: str(out_path / CONV_SFT_TRAIN_FILE),
            CONV_SFT_VAL_FILE: str(out_path / CONV_SFT_VAL_FILE),
            CONV_DPO_TRAIN_FILE: str(out_path / CONV_DPO_TRAIN_FILE),
            CONV_DPO_VAL_FILE: str(out_path / CONV_DPO_VAL_FILE),
        }
        (out_path / DISTILL_REPORT_FILE).write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        logger.info("Wrote distilled corpus to %s", out_path)
    else:
        summary["sft_rows"] = {"train": len(train), "val": len(val)}
        summary["dpo_rows"] = {"train": len(train), "val": len(val)}

    return summary


def format_distill_report(summary: dict[str, Any]) -> str:
    """Render the distillation scoreboard for the console."""
    lines = ["", "Distillation \u2014 close the loop", "=" * 52]
    lines.append(
        f"transcripts={summary['transcripts_total']}  "
        f"selected={summary['selected']}  "
        f"(threshold<{summary['score_threshold']}, all={summary['include_all']})"
    )
    lines.append("-" * 52)
    lines.append(
        f"  skipped: incomplete={summary['skipped_incomplete']}  "
        f"above-threshold={summary['skipped_above_threshold']}  "
        f"no-context={summary['skipped_no_context']}"
    )
    lines.append("-" * 52)
    sft = summary["sft_rows"]
    dpo = summary["dpo_rows"]
    lines.append(f"  SFT corpus   train={sft['train']:>4}  val={sft['val']:>4}")
    lines.append(f"  DPO corpus   train={dpo['train']:>4}  val={dpo['val']:>4}")
    artifacts = summary.get("artifacts") or {}
    if artifacts.get("dir"):
        lines.append("-" * 52)
        lines.append(f"  -> {artifacts['dir']}")
        lines.append("  Retrain with: sft / dpo  (point training at this dir)")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def add_distill_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the ``distill`` subcommand arguments."""
    parser.add_argument(
        "--transcripts", default=str(DEFAULT_TRANSCRIPTS),
        help="Agent test transcript JSONL to distill (default: last agent-test run).",
    )
    parser.add_argument(
        "--eval", dest="eval_path", default=str(DEFAULT_EVAL_FILE),
        help="Eval file used to recover each case's full conversation context.",
    )
    parser.add_argument(
        "--out-dir", default=str(DISTILLED_DIR),
        help="Where to write the distilled SFT/DPO corpus.",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_SCORE_THRESHOLD,
        help="Distill cases scoring below this strategy score (0-1).",
    )
    parser.add_argument(
        "--all", dest="include_all", action="store_true",
        help="Distill every completed case, not just the weak ones.",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=DEFAULT_VAL_FRACTION,
        help="Fraction of distilled rows held out for validation.",
    )


def run_distill(args: argparse.Namespace, config: DemoConfig) -> int:
    """CLI handler: distill captured transcripts into a retrain corpus."""
    try:
        summary = distill_corpus(
            transcripts_path=getattr(args, "transcripts", DEFAULT_TRANSCRIPTS),
            eval_path=getattr(args, "eval_path", DEFAULT_EVAL_FILE),
            out_dir=getattr(args, "out_dir", DISTILLED_DIR),
            score_threshold=getattr(args, "threshold", DEFAULT_SCORE_THRESHOLD),
            include_all=getattr(args, "include_all", False),
            val_fraction=getattr(args, "val_fraction", DEFAULT_VAL_FRACTION),
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return EXIT_ERROR
    print(format_distill_report(summary))
    return EXIT_SUCCESS if summary["selected"] else EXIT_ERROR


def build_parser() -> argparse.ArgumentParser:
    """Standalone parser so the module is runnable on its own."""
    parser = argparse.ArgumentParser(description="Distill Agent transcripts -> retrain corpus.")
    add_distill_arguments(parser)
    parser.set_defaults(func=run_distill)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Module entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_distill(args, DemoConfig.from_env())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
