// Typed fetch client for the demo FastAPI backend. All calls go through the Vite
// dev proxy at /api -> http://127.0.0.1:8000, so they are same-origin in dev.

import type {
  AgentInfo,
  AgentTranscript,
  ConfigInfo,
  DatasetInfo,
  DistillSummary,
  FoundryReport,
  JobAccepted,
  JobSnapshot,
  JobSummary,
  PreviewResponse,
} from "./types";

const BASE = "/api";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = (body && (body.detail ?? body.message)) || detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, String(detail));
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

function postJson<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, { method: "POST", body: JSON.stringify(body) });
}

// ---- Config ---------------------------------------------------------------
export const getHealth = () => request<{ status: string }>("/health");
export const getConfig = () => request<ConfigInfo>("/config");

// ---- Stage 1: Synthetic data ----------------------------------------------
export interface GenerateDataParams {
  count: number;
  eval_count: number;
  seed: number;
  use_llm: boolean;
  concurrency: number;
}
export const generateData = (p: GenerateDataParams) =>
  postJson<JobAccepted>("/data/generate", p);

export const listDatasets = () =>
  request<{ datasets: DatasetInfo[] }>("/data/datasets");

export const previewDataset = (name: string, limit = 20) =>
  request<PreviewResponse>(
    `/data/preview?name=${encodeURIComponent(name)}&limit=${limit}`
  );

// ---- Stage 2: Fine-tuning -------------------------------------------------
export interface FineTuneParams {
  method: "supervised" | "dpo" | "reinforcement";
  n_epochs?: number | null;
  beta?: number | null;
  batch_size?: number | null;
  learning_rate_multiplier?: number | null;
  eval_interval?: number | null;
  eval_samples?: number | null;
  reasoning_effort?: "low" | "medium" | "high" | null;
  compute_multiplier?: number | null;
  grader_model?: string | null;
  max_polls?: number | null;
  deploy: boolean;
  deployment_name?: string | null;
  sku: string;
}
export const startFinetune = (p: FineTuneParams) =>
  postJson<JobAccepted>("/finetune", p);

export const getFinetuneStates = () =>
  request<{ states: Record<string, unknown> }>("/finetune/states");

export const listFinetuneJobs = (limit = 10) =>
  request<{ jobs: Record<string, unknown>[] }>(`/finetune/jobs?limit=${limit}`);

// ---- Stage 3: Deployment --------------------------------------------------
export interface DeployParams {
  model_id: string;
  deployment_name: string;
  sku: string;
}
export const deployModel = (p: DeployParams) =>
  postJson<JobAccepted>("/deploy", p);

// ---- Stage 4: Foundry evaluation ------------------------------------------
export interface FoundryEvalParams {
  models: string[];
  limit?: number | null;
  delay: number;
  no_builtin: boolean;
  no_ai_assisted: boolean;
  no_upload: boolean;
}
export const startFoundryEval = (p: FoundryEvalParams) =>
  postJson<JobAccepted>("/foundry-eval", p);

export const getFoundryReport = () =>
  request<{ report: FoundryReport | null }>("/foundry-eval/report");

// ---- Stage 5: Agent Service -----------------------------------------------
export const createAgent = (model: string, name?: string) =>
  postJson<AgentInfo>("/agents", { model, name: name ?? null });

export const listAgents = () => request<{ agents: AgentInfo[] }>("/agents");

export const deleteAgent = (id: string) =>
  request<{ deleted: boolean; id: string }>(
    `/agents/${encodeURIComponent(id)}`,
    { method: "DELETE" }
  );

export interface AgentTestParams {
  id?: string | null;
  model: string;
  limit?: number | null;
  ephemeral: boolean;
}
export const testAgent = (p: AgentTestParams) =>
  postJson<JobAccepted>("/agents/test", p);

export const getTranscripts = () =>
  request<{ report: Record<string, unknown> | null; transcripts: AgentTranscript[] }>(
    "/agents/transcripts"
  );

// ---- Stage 6: Distill -----------------------------------------------------
export interface DistillParams {
  threshold: number;
  include_all: boolean;
  val_fraction: number;
}
export const runDistill = (p: DistillParams) =>
  postJson<DistillSummary>("/distill", p);

export const getDistillSummary = () =>
  request<{ summary: DistillSummary | null }>("/distill/summary");

// ---- Stage 7: Retrain -----------------------------------------------------
export const retrain = (p: FineTuneParams) =>
  postJson<JobAccepted>("/distill/retrain", p);

// ---- Jobs -----------------------------------------------------------------
export const listJobs = () => request<{ jobs: JobSummary[] }>("/jobs");

export const getJob = (id: string, logOffset = 0) =>
  request<JobSnapshot>(`/jobs/${encodeURIComponent(id)}?log_offset=${logOffset}`);
