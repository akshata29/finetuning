"""FastAPI control plane for the conversation-alignment fine-tuning demo.

This wraps the **existing** demo functions (the same code the CLI drives) behind
a small HTTP API so a React frontend can run the full loop:

1. **Synthetic data** generation + preview
2. **Fine-tuning** (SFT / DPO / RFT) with hyperparameters
3. **Deployment** of a fine-tuned model
4. **Foundry evaluation** scoreboard
5. **Agent Service** create / list / delete / test (live threads + runs)
6. **Distill** captured transcripts into a retrain corpus
7. **Retrain** from the distilled corpus (closes the loop)

Long-running stages run as background jobs (see :mod:`finetuning.api.jobs`);
the frontend polls ``GET /api/jobs/{id}`` for status + streamed logs. Fast,
read-only operations (config, dataset listing/preview, distill) return inline.

Security posture:

* **No secrets leave the box.** ``GET /api/config`` reports only whether secrets
  are present (booleans), never the values. Deployment names and endpoints are
  surfaced because the UI needs them and they are not credentials.
* **Path-traversal safe (OWASP A01/A03).** Dataset preview resolves the requested
  name against a server-built whitelist of files inside the data directories and
  rejects anything else.
* **CORS** is scoped to local dev origins by default; override with
  ``DEMO_API_CORS_ORIGINS`` (comma-separated).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Allow running both as a module and as a plain script.
if __package__ in (None, ""):  # pragma: no cover - script-launch shim
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    __package__ = "finetuning.api"

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..config import DemoConfig
from ..customer_conversation_alignment import (
    CONV_EVAL_FILE,
    DATA_DIR,
    DEFAULT_MODELS,
    FOUNDRY_DIR,
    FOUNDRY_REPORT_FILE,
)
from ..distill import DISTILLED_DIR, DISTILL_REPORT_FILE
from ..foundry_agent_service import (
    AGENT_SERVICE_DIR,
    AGENT_TEST_REPORT_FILE,
    AGENT_TRANSCRIPTS_FILE,
)
from .jobs import manager

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Conversation-Alignment Fine-Tuning Demo API",
    description="HTTP control plane wrapping the CLI demo functions.",
    version="1.0.0",
)

_default_origins = "http://localhost:5173,http://127.0.0.1:5173"
_origins = [o.strip() for o in os.getenv("DEMO_API_CORS_ORIGINS", _default_origins).split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    manager.install_log_routing()
    logger.info("Demo API ready (CORS origins=%s)", _origins)


def _config() -> DemoConfig:
    """Fresh config each call so .env edits are picked up without a restart."""
    return DemoConfig.from_env()


# ---------------------------------------------------------------------------
# Dataset catalog (path-traversal safe)
# ---------------------------------------------------------------------------
def _data_dirs() -> dict[str, Path]:
    return {
        "corpus": DATA_DIR,
        "distilled": DISTILLED_DIR,
        "foundry": FOUNDRY_DIR,
        "agent_service": AGENT_SERVICE_DIR,
    }


def _catalog() -> dict[str, Path]:
    """Build name -> path whitelist of all .jsonl/.json files in data dirs."""
    catalog: dict[str, Path] = {}
    for group, base in _data_dirs().items():
        if not base.exists():
            continue
        for path in sorted(base.glob("*.json")) + sorted(base.glob("*.jsonl")):
            catalog[f"{group}/{path.name}"] = path
    return catalog


def _count_rows(path: Path) -> Optional[int]:
    if path.suffix != ".jsonl":
        return None
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Health + config
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    """Non-secret config snapshot for the UI (booleans for any secret)."""
    cfg = _config()
    return {
        "project_endpoint": cfg.azure_ai_project_endpoint,
        "openai_endpoint": cfg.azure_openai_endpoint,
        "region": cfg.azure_region,
        "deployment_tier": cfg.deployment_tier,
        "deployments": {
            "base": cfg.base_deployment_name,
            "sft": cfg.sft_deployment_name,
            "dpo": cfg.dpo_deployment_name,
            "rft": cfg.rft_deployment_name,
        },
        "teacher_model": cfg.teacher_model,
        "grader_model": cfg.grader_model,
        "secrets_present": {
            "openai_api_key": bool(cfg.azure_openai_api_key),
            "subscription_id": bool(cfg.azure_subscription_id),
            "resource_group": bool(cfg.azure_resource_group),
        },
        "available_models": list(DEFAULT_MODELS),
    }


# ---------------------------------------------------------------------------
# Stage 1 — Synthetic data
# ---------------------------------------------------------------------------
class GenerateDataRequest(BaseModel):
    count: int = Field(60, ge=1, le=2000)
    eval_count: int = Field(30, ge=1, le=1000)
    seed: int = 1337
    use_llm: bool = False
    concurrency: int = Field(4, ge=1, le=16)


@app.post("/api/data/generate")
def generate_data(req: GenerateDataRequest) -> dict[str, Any]:
    from ..customer_conversation_alignment import build_llm_teacher, generate_all_datasets  # noqa: PLC0415

    cfg = _config()

    def _run() -> dict[str, Any]:
        teacher = None
        concurrency = 1
        if req.use_llm:
            teacher = build_llm_teacher(cfg, max_retries=6)
            concurrency = req.concurrency
            print(f"Using LLM teacher for generation (concurrency={concurrency})...")
        counts = generate_all_datasets(
            count=req.count, eval_count=req.eval_count, seed=req.seed,
            generate_fn=teacher, concurrency=concurrency,
        )
        print("Synthetic corpus written:")
        for name, rows in counts.items():
            print(f"  {name:<26} {rows:>5} rows")
        return {"counts": counts, "source": "llm" if teacher else "templates"}

    job = manager.submit("generate-data", _run, params=req.model_dump())
    return {"job_id": job.id}


@app.get("/api/data/datasets")
def list_datasets() -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for name, path in _catalog().items():
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        items.append({
            "name": name,
            "group": name.split("/", 1)[0],
            "filename": path.name,
            "bytes": size,
            "rows": _count_rows(path),
        })
    return {"datasets": items}


@app.get("/api/data/preview")
def preview_dataset(
    name: str = Query(..., description="catalog name, e.g. corpus/conv_sft_train.jsonl"),
    limit: int = Query(20, ge=1, le=500),
) -> dict[str, Any]:
    catalog = _catalog()
    path = catalog.get(name)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Unknown dataset '{name}'.")
    rows: list[Any] = []
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    rows.append({"_raw": line})
                if len(rows) >= limit:
                    break
    else:  # whole json doc
        try:
            rows = [json.loads(path.read_text(encoding="utf-8-sig"))]
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=500, detail=f"Bad JSON: {exc}") from exc
    return {"name": name, "rows": rows, "returned": len(rows)}


# ---------------------------------------------------------------------------
# Stage 2 — Fine-tuning
# ---------------------------------------------------------------------------
class FineTuneRequest(BaseModel):
    method: str = Field("supervised", pattern="^(supervised|dpo|reinforcement)$")
    n_epochs: Optional[int] = Field(None, ge=1, le=20)
    beta: Optional[float] = Field(None, gt=0, le=5)
    batch_size: Optional[int] = Field(None, ge=1, le=256)
    learning_rate_multiplier: Optional[float] = Field(None, gt=0, le=100)
    eval_interval: Optional[int] = Field(None, ge=1)
    eval_samples: Optional[int] = Field(None, ge=1)
    reasoning_effort: Optional[str] = Field(None, pattern="^(low|medium|high)$")
    compute_multiplier: Optional[float] = Field(None, gt=0, le=10)
    grader_model: Optional[str] = None
    max_polls: Optional[int] = Field(None, ge=1)
    deploy: bool = False
    deployment_name: Optional[str] = None
    sku: str = "developer"


@app.post("/api/finetune")
def start_finetune(req: FineTuneRequest) -> dict[str, Any]:
    from ..customer_conversation_alignment import run_finetune  # noqa: PLC0415

    cfg = _config()

    def _run() -> dict[str, Any]:
        return run_finetune(
            cfg, req.method,
            max_polls=req.max_polls, grader_model=req.grader_model,
            n_epochs=req.n_epochs, beta=req.beta,
            batch_size=req.batch_size,
            learning_rate_multiplier=req.learning_rate_multiplier,
            eval_interval=req.eval_interval, eval_samples=req.eval_samples,
            reasoning_effort=req.reasoning_effort,
            compute_multiplier=req.compute_multiplier,
            deploy=req.deploy, deployment_name=req.deployment_name, sku=req.sku,
        )

    job = manager.submit(f"finetune-{req.method}", _run, params=req.model_dump())
    return {"job_id": job.id}


@app.get("/api/finetune/states")
def finetune_states() -> dict[str, Any]:
    """Read the per-method job state JSON written by the last fine-tune."""
    from ..customer_conversation_alignment import (  # noqa: PLC0415
        CONV_DPO_STATE_FILE, CONV_RFT_STATE_FILE, CONV_SFT_STATE_FILE,
    )

    states: dict[str, Any] = {}
    for label, fname in (
        ("supervised", CONV_SFT_STATE_FILE),
        ("dpo", CONV_DPO_STATE_FILE),
        ("reinforcement", CONV_RFT_STATE_FILE),
    ):
        path = DATA_DIR / fname
        if path.exists():
            try:
                states[label] = json.loads(path.read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError:
                states[label] = None
    return {"states": states}


@app.get("/api/finetune/jobs")
def finetune_jobs(limit: int = Query(10, ge=1, le=50)) -> dict[str, Any]:
    """List recent Azure fine-tuning jobs (live call; needs Azure config)."""
    from ..customer_conversation_alignment import list_finetune_jobs  # noqa: PLC0415

    try:
        jobs = list_finetune_jobs(_config(), limit=limit)
    except Exception as exc:  # noqa: BLE001 - surface as a clean error
        raise HTTPException(status_code=502, detail=f"Azure list failed: {exc}") from exc
    return {
        "jobs": [
            {
                "id": getattr(j, "id", None),
                "status": str(getattr(j, "status", None)),
                "model": getattr(j, "model", None),
                "fine_tuned_model": getattr(j, "fine_tuned_model", None),
            }
            for j in jobs
        ]
    }


# ---------------------------------------------------------------------------
# Stage 3 — Deployment
# ---------------------------------------------------------------------------
class DeployRequest(BaseModel):
    model_id: str
    deployment_name: str
    sku: str = "developer"


@app.post("/api/deploy")
def deploy_model(req: DeployRequest) -> dict[str, Any]:
    from .. import act2a_serverless_sft as sft  # noqa: PLC0415

    cfg = _config()

    def _run() -> dict[str, Any]:
        print(f"Deploying {req.model_id} -> {req.deployment_name} (sku={req.sku})...")
        response = sft.deploy_finetuned(cfg, req.model_id, req.deployment_name, req.sku)
        state = None
        if isinstance(response, dict):
            state = response.get("properties", {}).get("provisioningState")
        print(f"Deployment provisioning state: {state}")
        return {"deployment_name": req.deployment_name, "provisioning_state": state}

    job = manager.submit("deploy", _run, params=req.model_dump())
    return {"job_id": job.id}


# ---------------------------------------------------------------------------
# Stage 4 — Foundry evaluation
# ---------------------------------------------------------------------------
class FoundryEvalRequest(BaseModel):
    models: list[str] = Field(default_factory=lambda: ["base", "sft", "rft"])
    limit: Optional[int] = Field(None, ge=1, le=500)
    delay: float = 0.0
    no_builtin: bool = False
    no_ai_assisted: bool = False
    no_upload: bool = False


@app.post("/api/foundry-eval")
def start_foundry_eval(req: FoundryEvalRequest) -> dict[str, Any]:
    from ..customer_conversation_alignment import (  # noqa: PLC0415
        build_judge_model_config, format_foundry_report, run_all_foundry_conv_evals,
    )

    cfg = _config()

    def _run() -> dict[str, Any]:
        judge = None
        if not req.no_ai_assisted:
            judge = build_judge_model_config(cfg)
            if judge is None:
                print("AI-assisted evaluators skipped (judge config incomplete).")
            else:
                print(f"AI-assisted evaluators ON (judge={cfg.teacher_model}).")
        report = run_all_foundry_conv_evals(
            cfg, DATA_DIR / CONV_EVAL_FILE, FOUNDRY_DIR,
            models=tuple(req.models), limit=req.limit, request_delay=req.delay,
            include_builtin=not req.no_builtin, judge_model_config=judge,
            upload=not req.no_upload,
        )
        print(format_foundry_report(report))
        return report

    job = manager.submit("foundry-eval", _run, params=req.model_dump())
    return {"job_id": job.id}


@app.get("/api/foundry-eval/report")
def foundry_report() -> dict[str, Any]:
    path = FOUNDRY_DIR / FOUNDRY_REPORT_FILE
    if not path.exists():
        return {"report": None}
    return {"report": json.loads(path.read_text(encoding="utf-8-sig"))}


# ---------------------------------------------------------------------------
# Stage 5 — Agent Service
# ---------------------------------------------------------------------------
class AgentCreateRequest(BaseModel):
    model: str = "sft"
    name: Optional[str] = None


class AgentTestRequest(BaseModel):
    id: Optional[str] = None
    model: str = "sft"
    limit: Optional[int] = Field(None, ge=1, le=200)
    ephemeral: bool = False


@app.post("/api/agents")
def create_agent_endpoint(req: AgentCreateRequest) -> dict[str, Any]:
    from ..foundry_agent_service import create_agent  # noqa: PLC0415

    try:
        return create_agent(_config(), model_label=req.model, name=req.name)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/agents")
def list_agents_endpoint() -> dict[str, Any]:
    from ..foundry_agent_service import list_agents  # noqa: PLC0415

    try:
        return {"agents": list_agents(_config())}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.delete("/api/agents/{agent_id}")
def delete_agent_endpoint(agent_id: str) -> dict[str, Any]:
    from ..foundry_agent_service import delete_agent  # noqa: PLC0415

    try:
        ok = delete_agent(_config(), agent_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"deleted": ok, "id": agent_id}


@app.post("/api/agents/test")
def test_agent_endpoint(req: AgentTestRequest) -> dict[str, Any]:
    from ..foundry_agent_service import test_agent  # noqa: PLC0415

    cfg = _config()

    def _run() -> dict[str, Any]:
        return test_agent(
            cfg, agent_id=req.id, model_label=req.model,
            limit=req.limit, ephemeral=req.ephemeral,
        )

    job = manager.submit("agent-test", _run, params=req.model_dump())
    return {"job_id": job.id}


@app.get("/api/agents/transcripts")
def agent_transcripts() -> dict[str, Any]:
    report_path = AGENT_SERVICE_DIR / AGENT_TEST_REPORT_FILE
    transcripts_path = AGENT_SERVICE_DIR / AGENT_TRANSCRIPTS_FILE
    report = None
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    rows: list[Any] = []
    if transcripts_path.exists():
        with transcripts_path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return {"report": report, "transcripts": rows}


# ---------------------------------------------------------------------------
# Stage 6 — Distill
# ---------------------------------------------------------------------------
class DistillRequest(BaseModel):
    threshold: float = Field(0.75, ge=0, le=1)
    include_all: bool = False
    val_fraction: float = Field(0.2, ge=0, le=0.9)


@app.post("/api/distill")
def distill_endpoint(req: DistillRequest) -> dict[str, Any]:
    from ..distill import distill_corpus  # noqa: PLC0415

    try:
        return distill_corpus(
            score_threshold=req.threshold,
            include_all=req.include_all,
            val_fraction=req.val_fraction,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/distill/summary")
def distill_summary() -> dict[str, Any]:
    path = DISTILLED_DIR / DISTILL_REPORT_FILE
    if not path.exists():
        return {"summary": None}
    return {"summary": json.loads(path.read_text(encoding="utf-8-sig"))}


# ---------------------------------------------------------------------------
# Stage 7 — Retrain from the distilled corpus
# ---------------------------------------------------------------------------
class RetrainRequest(BaseModel):
    method: str = Field("supervised", pattern="^(supervised|dpo|reinforcement)$")
    n_epochs: Optional[int] = Field(None, ge=1, le=20)
    beta: Optional[float] = Field(None, gt=0, le=5)
    batch_size: Optional[int] = Field(None, ge=1, le=256)
    learning_rate_multiplier: Optional[float] = Field(None, gt=0, le=100)
    eval_interval: Optional[int] = Field(None, ge=1)
    eval_samples: Optional[int] = Field(None, ge=1)
    reasoning_effort: Optional[str] = Field(None, pattern="^(low|medium|high)$")
    compute_multiplier: Optional[float] = Field(None, gt=0, le=10)
    grader_model: Optional[str] = None
    max_polls: Optional[int] = Field(None, ge=1)
    deploy: bool = False
    deployment_name: Optional[str] = None
    sku: str = "developer"


@app.post("/api/distill/retrain")
def retrain_endpoint(req: RetrainRequest) -> dict[str, Any]:
    from ..customer_conversation_alignment import run_finetune  # noqa: PLC0415

    cfg = _config()

    def _run() -> dict[str, Any]:
        print(f"Retraining ({req.method}) from distilled corpus at {DISTILLED_DIR}...")
        return run_finetune(
            cfg, req.method, out_dir=DISTILLED_DIR,
            max_polls=req.max_polls, grader_model=req.grader_model,
            n_epochs=req.n_epochs, beta=req.beta,
            batch_size=req.batch_size,
            learning_rate_multiplier=req.learning_rate_multiplier,
            eval_interval=req.eval_interval, eval_samples=req.eval_samples,
            reasoning_effort=req.reasoning_effort,
            compute_multiplier=req.compute_multiplier,
            deploy=req.deploy, deployment_name=req.deployment_name, sku=req.sku,
        )

    job = manager.submit(f"retrain-{req.method}", _run, params=req.model_dump())
    return {"job_id": job.id}


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------
@app.get("/api/jobs")
def list_jobs() -> dict[str, Any]:
    return {"jobs": manager.list()}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, log_offset: int = Query(0, ge=0)) -> dict[str, Any]:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job '{job_id}'.")
    return job.snapshot(log_offset=log_offset)
