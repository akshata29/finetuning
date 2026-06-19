"""Foundry Agent Service — wrap a fine-tuned deployment in a managed Agent and
exercise it with test cases to capture real, scored conversation data.

This is the **"stand up a real Agent in Foundry"** stage of the conversation-
alignment demo. Where :mod:`finetuning_demo.act3a_foundry_eval` evaluates the raw
*model deployment* and :mod:`finetuning_demo.agent_corpus_capture` drives the
deployment directly through chat-completions, this module creates a first-class
**Foundry Agent Service entity** (the ``agents`` blade in the portal): a managed
wrapper of *deployment + instructions* that you can chat with, evaluate, and —
later — attach tools and persistent threads to.

The story it tells, in order:

1. **Create** — register a managed Agent that points at your tuned deployment
   (``conv-align-sft`` by default), with strategy-aligned standing instructions.
   It appears in the Foundry portal and is reusable by id.
2. **Test / capture** — replay the held-out eval conversations (the same
   ``conv_eval.jsonl`` the Foundry scoreboard uses) *through the Agent* via real
   threads + runs, capture each Agent reply, and score it for strategy adherence.
   This is "run test cases through the deployed agent and capture the real data."
3. **Close the loop** — the captured transcripts (Agent reply vs. the exemplary
   ``ground_truth``) are saved with provenance so a later lap can distill them
   into a retraining corpus, exactly like :mod:`agent_corpus_capture`.

Design choices mirror the rest of the demo:

* **Import / quota safe (OWASP A05/A06).** The Azure Agents/Identity SDKs are
  imported lazily *inside* the functions that need them, so importing this module
  (and the whole demo) never requires Azure to be configured or reachable.
* **Credential via ``DefaultAzureCredential``.** Agent Service authenticates with
  your Azure AD token (``az login``), not an API key — the FDP project endpoint
  comes from :class:`DemoConfig`.
* **Data isolation.** Captured Agent transcripts and the test report live under
  ``data/conversation_alignment/agent_service/``.

This module deliberately does **not** use ``from __future__ import annotations``
to stay consistent with :mod:`customer_conversation_alignment`.
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Allow running both as a module and as a plain script.
if __package__ in (None, ""):  # pragma: no cover - script-launch shim
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "finetuning_demo"

from .config import DemoConfig
from .customer_conversation_alignment import (
    DATA_DIR,
    EXIT_ERROR,
    EXIT_SUCCESS,
    STRATEGY_GUIDELINE,
    load_eval_rows,
    strategy_alignment_score,
)
from .agent_corpus_capture import resolve_agent_deployment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
#: Where captured Agent transcripts + the test report are written.
AGENT_SERVICE_DIR: Path = DATA_DIR / "agent_service"
AGENT_TRANSCRIPTS_FILE: str = "agent_test_transcripts.jsonl"
AGENT_TEST_REPORT_FILE: str = "agent_test_report.json"

#: The held-out conversations replayed as test cases (same file the scoreboard uses).
DEFAULT_EVAL_FILE: Path = DATA_DIR / "conv_eval.jsonl"

#: Default name for the managed Agent created against the SFT deployment.
DEFAULT_AGENT_NAME: str = "conv-alignment-sft-agent"

#: ``create_and_process`` poll cadence (seconds) while a run executes server-side.
DEFAULT_POLL_INTERVAL: int = 2

#: Fine-tuned S0 deployments carry a low call-rate limit; how many times to retry a
#: single turn that fails with ``rate_limit_exceeded`` before giving up.
DEFAULT_RATE_LIMIT_RETRIES: int = 6

#: Fallback backoff (seconds) when the rate-limit error omits a retry hint.
DEFAULT_RATE_LIMIT_BACKOFF: float = 30.0

#: The managed Agent's standing instructions (its system prompt). The fine-tuned
#: model already learned the per-strategy behavior in training; this frames the
#: role, and each test case layers its specific target strategy onto the run via
#: ``additional_instructions`` so the Agent is steered exactly as eval expects.
AGENT_INSTRUCTIONS: str = (
    "You are an expert B2B sales strategist. Continue the conversation with one "
    "concise, on-strategy reply that advances the deal. Depending on the guidance "
    "for this exchange: ask focused diagnostic questions before pitching, frame "
    "every capability against the prospect's stated business outcome, acknowledge "
    "objections and reframe them around value without pressure, support claims "
    "with concrete proof (a metric or comparable customer), and close with a "
    "specific, low-friction, mutually agreed next step."
)


# ---------------------------------------------------------------------------
# Project / Agent clients
# ---------------------------------------------------------------------------
def build_project_client(config: DemoConfig) -> Any:
    """Build an :class:`AIProjectClient` for the configured Foundry (FDP) project.

    Uses ``DefaultAzureCredential`` (your ``az login`` token) — Agent Service does
    not accept the data-plane API key. Raises ``ValueError`` with an actionable
    message when the project endpoint is unset. The Azure SDKs are imported lazily
    so this module stays import-safe without Azure configured.
    """
    endpoint = (config.azure_ai_project_endpoint or "").strip()
    if not endpoint:
        raise ValueError(
            "AZURE_AI_PROJECT_ENDPOINT is not set. Point it at your Foundry (FDP) "
            "project endpoint, e.g. "
            "https://<account>.services.ai.azure.com/api/projects/<project>."
        )
    from azure.ai.projects import AIProjectClient  # noqa: PLC0415
    from azure.identity import DefaultAzureCredential  # noqa: PLC0415

    return AIProjectClient(endpoint=endpoint, credential=DefaultAzureCredential())


def create_agent(
    config: DemoConfig,
    *,
    model_label: str = "sft",
    name: str | None = None,
    instructions: str | None = None,
    project: Any | None = None,
) -> dict[str, Any]:
    """Create a managed Foundry Agent pointing at a resolved deployment.

    ``model_label`` is a friendly label (``base``/``sft``/``dpo``/``rft``, resolved
    from ``.env``) or a raw deployment name. Returns the new Agent's ``id``,
    ``name``, and resolved ``model`` (deployment). When ``project`` is omitted a
    client is built and closed here.
    """
    deployment = resolve_agent_deployment(config, model_label)
    if not deployment:
        raise ValueError(
            f"No deployment resolved for model '{model_label}'. Set the matching "
            "*_DEPLOYMENT_NAME in .env or pass a raw deployment name."
        )
    agent_name = name or DEFAULT_AGENT_NAME
    own = project is None
    project = project or build_project_client(config)
    try:
        agent = project.agents.create_agent(
            model=deployment,
            name=agent_name,
            instructions=instructions or AGENT_INSTRUCTIONS,
        )
    finally:
        if own:
            project.close()
    logger.info("Created agent %s (%s) -> %s", agent.name, agent.id, deployment)
    return {"id": agent.id, "name": agent.name, "model": deployment}


def list_agents(config: DemoConfig, *, project: Any | None = None) -> list[dict[str, Any]]:
    """List the managed Agents in the project (id, name, model)."""
    own = project is None
    project = project or build_project_client(config)
    try:
        return [
            {"id": a.id, "name": a.name, "model": getattr(a, "model", "")}
            for a in project.agents.list_agents()
        ]
    finally:
        if own:
            project.close()


def delete_agent(config: DemoConfig, agent_id: str, *, project: Any | None = None) -> bool:
    """Delete a managed Agent by id. Returns ``True`` on success."""
    own = project is None
    project = project or build_project_client(config)
    try:
        project.agents.delete_agent(agent_id)
        logger.info("Deleted agent %s", agent_id)
        return True
    finally:
        if own:
            project.close()


# ---------------------------------------------------------------------------
# Test-case driver (threads + runs)
# ---------------------------------------------------------------------------
def _run_status(run: Any) -> str:
    """Normalize a ``ThreadRun.status`` enum/string to its lowercase value.

    ``RunStatus`` is a string enum whose ``str()`` renders as ``RunStatus.COMPLETED``
    while its ``.value`` is the API token ``"completed"`` — compare on the value.
    """
    return str(getattr(run.status, "value", run.status)).lower()


def _is_rate_limited(error: Any) -> bool:
    """True when a run's ``last_error`` is an Azure OpenAI rate-limit error."""
    code = getattr(error, "code", None)
    if code is None and isinstance(error, dict):
        code = error.get("code")
    return str(code) == "rate_limit_exceeded"


def _retry_after_seconds(error: Any, default: float = DEFAULT_RATE_LIMIT_BACKOFF) -> float:
    """Parse the suggested wait from a rate-limit message ("retry after N seconds")."""
    message = getattr(error, "message", None)
    if message is None and isinstance(error, dict):
        message = error.get("message")
    match = re.search(r"retry after (\d+(?:\.\d+)?)\s*second", str(message or ""), re.IGNORECASE)
    if match:
        return float(match.group(1)) + 1.0
    return default


def run_agent_turn(
    project: Any,
    agent_id: str,
    query: str,
    *,
    strategy: str | None = None,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
    max_rate_limit_retries: int = DEFAULT_RATE_LIMIT_RETRIES,
) -> dict[str, Any]:
    """Run one test case through the Agent and return its reply + run status.

    Creates a fresh thread, posts ``query`` as the user turn, runs the Agent
    (layering the case's target ``strategy`` via ``additional_instructions``),
    reads back the Agent's reply, then deletes the thread to avoid clutter. A
    failed/empty run degrades to an empty reply rather than raising, so one bad
    case never aborts the batch (OWASP A04 — fail safe). Fine-tuned S0
    deployments rate-limit aggressively, so a run that fails with
    ``rate_limit_exceeded`` is retried with the server-suggested backoff.
    """
    from azure.ai.agents.models import MessageRole  # noqa: PLC0415

    agents = project.agents

    for attempt in range(max_rate_limit_retries + 1):
        thread = agents.threads.create()
        try:
            agents.messages.create(thread_id=thread.id, role="user", content=query)
            run_kwargs: dict[str, Any] = {"agent_id": agent_id, "polling_interval": poll_interval}
            if strategy and strategy in STRATEGY_GUIDELINE:
                run_kwargs["additional_instructions"] = STRATEGY_GUIDELINE[strategy]
            run = agents.runs.create_and_process(thread.id, **run_kwargs)

            status = _run_status(run)
            last_error = getattr(run, "last_error", None)

            if status == "failed" and _is_rate_limited(last_error) and attempt < max_rate_limit_retries:
                wait = _retry_after_seconds(last_error)
                logger.info(
                    "rate-limited (attempt %d/%d); backing off %.0fs",
                    attempt + 1, max_rate_limit_retries, wait,
                )
                time.sleep(wait)
                continue

            reply = ""
            if status == "completed":
                message = agents.messages.get_last_message_text_by_role(
                    thread.id, MessageRole.AGENT,
                )
                if message is not None:
                    reply = message.text.value
            return {
                "response": reply,
                "status": status,
                "thread_id": thread.id,
                "error": str(last_error) if last_error else None,
            }
        finally:
            try:
                agents.threads.delete(thread.id)
            except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
                logger.debug("thread %s cleanup failed: %s", thread.id, exc)

    # Exhausted retries while still rate-limited.
    return {
        "response": "",
        "status": "failed",
        "thread_id": None,
        "error": "rate_limit_exceeded (retries exhausted)",
    }


def _summarize(
    results: list[dict[str, Any]],
    *,
    agent_id: str,
    model_label: str,
    eval_path: str | Path,
) -> dict[str, Any]:
    """Aggregate per-case results into a scored test report."""
    scores = [r["strategy_score"] for r in results]
    completed = sum(1 for r in results if r["run_status"] == "completed")
    per: dict[str, list[float]] = {}
    for r in results:
        per.setdefault(r["strategy"], []).append(r["strategy_score"])
    per_strategy = {
        strategy: {"mean": sum(vals) / len(vals), "count": len(vals)}
        for strategy, vals in per.items()
    }
    return {
        "created": datetime.now(timezone.utc).isoformat(),
        "agent_id": agent_id,
        "model_label": model_label,
        "eval_path": str(eval_path),
        "test_cases": len(results),
        "completed": completed,
        "mean_strategy_score": sum(scores) / len(scores) if scores else 0.0,
        "per_strategy": per_strategy,
        "results": results,
    }


def _write_artifacts(
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    out_dir: Path = AGENT_SERVICE_DIR,
) -> dict[str, str]:
    """Write captured Agent transcripts + the test report for retrain provenance."""
    out_dir.mkdir(parents=True, exist_ok=True)
    transcripts_path = out_dir / AGENT_TRANSCRIPTS_FILE
    with transcripts_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in results:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
    report_path = out_dir / AGENT_TEST_REPORT_FILE
    # The report mirrors summary but drops the verbose per-case rows (in the JSONL).
    report = {k: v for k, v in summary.items() if k != "results"}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote agent test artifacts to %s", out_dir)
    return {"transcripts": str(transcripts_path), "report": str(report_path)}


def test_agent(
    config: DemoConfig,
    *,
    agent_id: str | None = None,
    model_label: str = "sft",
    eval_path: str | Path = DEFAULT_EVAL_FILE,
    limit: int | None = None,
    save: bool = True,
    ephemeral: bool = False,
    project: Any | None = None,
) -> dict[str, Any]:
    """Replay held-out conversations through the Agent, scoring each reply.

    When ``agent_id`` is omitted a managed Agent is created against ``model_label``
    and reused for every case. By default that Agent is **kept** (so it shows in
    the portal and can be reused via ``--id``); pass ``ephemeral=True`` to delete
    it after the run. Returns a scored summary and, when ``save``, writes the
    captured transcripts + report under ``data/conversation_alignment/agent_service``.
    """
    rows = load_eval_rows(eval_path)
    if limit is not None and limit > 0:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No eval rows in {eval_path}.")

    own_project = project is None
    project = project or build_project_client(config)
    created_here = False
    try:
        if not agent_id:
            info = create_agent(config, model_label=model_label, project=project)
            agent_id = info["id"]
            created_here = True
            print(
                f"Created Agent '{info['name']}' ({agent_id}) -> deployment "
                f"{info['model']}"
            )

        results: list[dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            strategy = row.get("strategy", "")
            query = row.get("query", "")
            turn = run_agent_turn(project, agent_id, query, strategy=strategy)
            score = strategy_alignment_score(turn["response"], strategy)
            results.append(
                {
                    "id": row.get("id"),
                    "strategy": strategy,
                    "query": query,
                    "ground_truth": row.get("ground_truth", ""),
                    "agent_response": turn["response"],
                    "strategy_score": score,
                    "run_status": turn["status"],
                    "thread_id": turn["thread_id"],
                    "error": turn["error"],
                }
            )
            print(f"  [{index:>3}/{len(rows)}] {strategy:<24} {turn['status']:>9}  score={score:.3f}")

        summary = _summarize(results, agent_id=agent_id, model_label=model_label, eval_path=eval_path)

        if created_here and ephemeral:
            try:
                project.agents.delete_agent(agent_id)
                summary["agent_deleted"] = True
            except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
                logger.warning("could not delete transient agent %s: %s", agent_id, exc)
                summary["agent_deleted"] = False

        if save:
            summary["artifacts"] = _write_artifacts(results, summary)
        return summary
    finally:
        if own_project:
            project.close()


def format_agent_report(summary: dict[str, Any]) -> str:
    """Render the Agent test scoreboard for the console."""
    lines = ["", "Foundry Agent Service — test capture", "=" * 52]
    lines.append(
        f"agent={summary['agent_id']}  model={summary['model_label']}  "
        f"cases={summary['test_cases']}  completed={summary['completed']}"
    )
    lines.append("-" * 52)
    lines.append(f"{'mean strategy score':<28}{summary['mean_strategy_score']:>8.3f}")
    lines.append("-" * 52)
    for strategy, stats in summary.get("per_strategy", {}).items():
        lines.append(f"  {strategy:<26}{stats['mean']:>8.3f}  (n={stats['count']})")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI handlers
# ---------------------------------------------------------------------------
def run_agent_create(args: argparse.Namespace, config: DemoConfig) -> int:
    """Create a managed Agent pointing at the chosen deployment."""
    info = create_agent(
        config,
        model_label=getattr(args, "model", "sft"),
        name=getattr(args, "name", None),
    )
    print(
        f"\nCreated Foundry Agent:\n  id:    {info['id']}\n  name:  {info['name']}\n"
        f"  model: {info['model']}\n\nReuse it with: agent-test --id {info['id']}\n"
    )
    return EXIT_SUCCESS


def run_agent_list(args: argparse.Namespace, config: DemoConfig) -> int:
    """List the managed Agents in the project."""
    agents = list_agents(config)
    if not agents:
        print("\nNo agents in the project.\n")
        return EXIT_SUCCESS
    print(f"\n{len(agents)} agent(s):")
    for a in agents:
        print(f"  {a['id']}  {a['name']:<32} model={a['model']}")
    print()
    return EXIT_SUCCESS


def run_agent_delete(args: argparse.Namespace, config: DemoConfig) -> int:
    """Delete a managed Agent by id."""
    delete_agent(config, args.id)
    print(f"\nDeleted agent {args.id}\n")
    return EXIT_SUCCESS


def run_agent_test(args: argparse.Namespace, config: DemoConfig) -> int:
    """Replay held-out conversations through the Agent and score the replies."""
    print("\nTesting the Foundry Agent against held-out conversations (live)...")
    summary = test_agent(
        config,
        agent_id=getattr(args, "id", None),
        model_label=getattr(args, "model", "sft"),
        limit=getattr(args, "limit", None),
        ephemeral=getattr(args, "ephemeral", False),
    )
    print(format_agent_report(summary))
    if summary.get("artifacts"):
        print(f"Transcripts: {summary['artifacts']['transcripts']}")
        print(f"Report:      {summary['artifacts']['report']}\n")
    return EXIT_SUCCESS


def add_agent_create_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach ``agent-create`` flags (shared with the pipeline CLI)."""
    parser.add_argument(
        "--model", default="sft",
        help="deployment to wrap: friendly label (base/sft/dpo/rft) or raw name (default sft)",
    )
    parser.add_argument(
        "--name", default=None,
        help=f"agent display name (default '{DEFAULT_AGENT_NAME}')",
    )


def add_agent_test_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach ``agent-test`` flags (shared with the pipeline CLI)."""
    parser.add_argument(
        "--id", default=None,
        help="existing agent id to test; if omitted a new agent is created and kept",
    )
    parser.add_argument(
        "--model", default="sft",
        help="deployment to wrap when creating a new agent (default sft)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="cap the number of held-out conversations replayed as test cases",
    )
    parser.add_argument(
        "--ephemeral", action="store_true",
        help="delete the auto-created agent after the run (default: keep it)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Standalone CLI: ``python foundry_agent_service.py <command>``."""
    parser = argparse.ArgumentParser(
        prog="foundry_agent_service",
        description=(
            "Create and exercise a managed Foundry Agent that wraps a fine-tuned "
            "deployment, capturing scored conversation data from test cases."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="create a managed Agent for a deployment")
    add_agent_create_arguments(p_create)
    p_create.set_defaults(func=run_agent_create)

    p_list = sub.add_parser("list", help="list managed Agents in the project")
    p_list.set_defaults(func=run_agent_list)

    p_delete = sub.add_parser("delete", help="delete a managed Agent by id")
    p_delete.add_argument("--id", required=True, help="agent id to delete")
    p_delete.set_defaults(func=run_agent_delete)

    p_test = sub.add_parser("test", help="replay test cases through the Agent and score them")
    add_agent_test_arguments(p_test)
    p_test.set_defaults(func=run_agent_test)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for the standalone Agent Service CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    config = DemoConfig.from_env()
    try:
        return int(args.func(args, config))
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        logger.error("%s", exc)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
