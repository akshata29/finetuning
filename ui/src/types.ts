// Type definitions mirroring the FastAPI backend responses.

export type JobStatus = "pending" | "running" | "succeeded" | "failed";

export interface JobSnapshot {
  id: string;
  kind: string;
  status: JobStatus;
  params: Record<string, unknown>;
  created: string | null;
  started: string | null;
  finished: string | null;
  result: unknown;
  error: string | null;
  logs: string[];
  log_count: number;
  log_offset: number;
}

export interface JobSummary {
  id: string;
  kind: string;
  status: JobStatus;
  params: Record<string, unknown>;
  created: string | null;
  started: string | null;
  finished: string | null;
  result: unknown;
  error: string | null;
  log_count: number;
}

export interface ConfigInfo {
  project_endpoint: string | null;
  openai_endpoint: string | null;
  region: string | null;
  deployment_tier: string | null;
  teacher_model: string | null;
  grader_model: string | null;
  deployments: Record<string, string | null>;
  available_models: string[];
  secrets_present: Record<string, boolean>;
}

export interface DatasetInfo {
  name: string;
  group: string;
  filename: string;
  bytes: number;
  rows: number | null;
}

export interface PreviewResponse {
  name: string;
  rows: Record<string, unknown>[];
  returned: number;
}

export interface AgentInfo {
  id: string;
  name: string | null;
  model: string | null;
}

export interface FoundryReport {
  created: unknown;
  runs: Record<string, unknown>[];
  failures: Record<string, unknown>[];
  [key: string]: unknown;
}

export interface DistillSummary {
  selected: number;
  skipped: number;
  train: number;
  val: number;
  threshold: number;
  include_all: boolean;
  [key: string]: unknown;
}

export interface AgentTranscript {
  id: string;
  strategy: string;
  query: string;
  ground_truth: string;
  agent_response: string;
  strategy_score: number | null;
  run_status: string;
  thread_id: string | null;
  error: string | null;
}

// Job kicked off by a POST that returns { job_id }.
export interface JobAccepted {
  job_id: string;
}

// Stage identifiers used by the sidebar stepper.
export type StageId =
  | "data"
  | "finetune"
  | "deploy"
  | "eval"
  | "agent"
  | "distill";
