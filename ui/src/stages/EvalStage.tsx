import { useEffect, useState } from "react";
import { ExternalLink, FlaskConical, Play, RefreshCw } from "lucide-react";
import {
  getFoundryReport,
  startFoundryEval,
  type FoundryEvalParams,
} from "../api";
import { useJobContext } from "../context/JobContext";
import {
  Button,
  Card,
  EmptyState,
  ErrorBanner,
  Field,
  Toggle,
} from "../components/ui";
import type { FoundryReport } from "../types";

const ALL_MODELS = ["base", "sft", "dpo", "rft"];

// (header, aggregated metric key) — mirrors format_foundry_report's column spec.
const COLUMNS: [string, string][] = [
  ["strategy", "strategy_alignment.strategy_alignment"],
  ["coher", "coherence.coherence"],
  ["fluency", "fluency.fluency"],
  ["relev", "relevance.relevance"],
  ["simil", "similarity.similarity"],
  ["f1", "f1_score.f1_score"],
];

interface FoundryRun {
  label?: string;
  display_name?: string;
  rows?: number;
  metrics?: Record<string, unknown>;
  studio_url?: string | null;
}

export function EvalStage() {
  const { trackJob } = useJobContext();
  const [models, setModels] = useState<string[]>(["base", "sft", "rft"]);
  const [limit, setLimit] = useState<string>("");
  const [noAiAssisted, setNoAiAssisted] = useState(false);
  const [noUpload, setNoUpload] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [report, setReport] = useState<FoundryReport | null>(null);
  const loadReport = async () => {
    try {
      const { report } = await getFoundryReport();
      setReport(report);
    } catch (e) {
      setError((e as Error).message);
    }
  };
  useEffect(() => {
    loadReport();
  }, []);

  const toggleModel = (m: string) =>
    setModels((prev) =>
      prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m]
    );

  const submit = async () => {
    setSubmitting(true);
    setError(null);
    try {
      const params: FoundryEvalParams = {
        models,
        limit: limit ? +limit : null,
        delay: 0,
        no_builtin: false,
        no_ai_assisted: noAiAssisted,
        no_upload: noUpload,
      };
      const { job_id } = await startFoundryEval(params);
      trackJob(job_id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  const runs = (report?.runs ?? []) as FoundryRun[];
  const activeColumns = COLUMNS.filter(([, key]) =>
    runs.some((r) => r.metrics && r.metrics[key] != null)
  );

  return (
    <div className="grid gap-5 xl:grid-cols-[380px_1fr]">
      <Card
        title="Run Foundry evaluation"
        description="Score each model arm with built-in + AI-assisted evaluators."
      >
        {error && (
          <div className="mb-4">
            <ErrorBanner message={error} />
          </div>
        )}
        <Field label="Model arms">
          <div className="flex flex-wrap gap-2">
            {ALL_MODELS.map((m) => (
              <button
                key={m}
                onClick={() => toggleModel(m)}
                className={`rounded-lg border px-3 py-1.5 text-sm font-medium uppercase transition ${
                  models.includes(m)
                    ? "border-accent/50 bg-accent/15 text-slate-100"
                    : "border-white/10 bg-ink-850 text-slate-500 hover:bg-ink-800"
                }`}
              >
                {m}
              </button>
            ))}
          </div>
        </Field>
        <div className="mt-4">
          <Field label="Row limit" hint="blank = full eval set">
            <input
              type="number"
              className="input"
              placeholder="all"
              value={limit}
              min={1}
              onChange={(e) => setLimit(e.target.value)}
            />
          </Field>
        </div>
        <div className="mt-4 space-y-3">
          <Toggle
            checked={noAiAssisted}
            onChange={setNoAiAssisted}
            label="Skip AI-assisted evaluators (deterministic only, faster)"
          />
          <Toggle
            checked={noUpload}
            onChange={setNoUpload}
            label="Local only (don't upload runs to the Foundry portal)"
          />
        </div>
        <div className="mt-5">
          <Button
            onClick={submit}
            loading={submitting}
            disabled={models.length === 0}
            icon={<Play className="h-4 w-4" />}
          >
            Run evaluation
          </Button>
        </div>
      </Card>

      <Card
        title="Scoreboard"
        description="strategy/f1 are 0-1 · coher/fluency/relev/simil are 1-5 LLM-judge scores."
        actions={
          <Button variant="ghost" onClick={loadReport} icon={<RefreshCw className="h-4 w-4" />}>
            Refresh
          </Button>
        }
      >
        {runs.length === 0 ? (
          <EmptyState>
            <FlaskConical className="mx-auto mb-2 h-6 w-6 opacity-50" />
            No evaluation report yet. Run an evaluation to populate the scoreboard.
          </EmptyState>
        ) : (
          <div className="overflow-x-auto rounded-lg border border-white/5">
            <table className="w-full text-sm">
              <thead className="bg-ink-850 text-left text-xs uppercase tracking-wide text-slate-500">
                <tr>
                  <th className="px-3 py-2 font-medium">Model</th>
                  {activeColumns.map(([h]) => (
                    <th key={h} className="px-3 py-2 text-right font-medium">
                      {h}
                    </th>
                  ))}
                  <th className="px-3 py-2 text-right font-medium">rows</th>
                  <th className="px-3 py-2" />
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {runs.map((run, i) => (
                  <tr key={i} className="hover:bg-white/5">
                    <td className="px-3 py-2 font-medium text-slate-200">
                      {run.display_name ?? run.label ?? "?"}
                    </td>
                    {activeColumns.map(([h, key]) => (
                      <td key={h} className="px-3 py-2 text-right font-mono text-slate-300">
                        {fmtMetric(run.metrics?.[key])}
                      </td>
                    ))}
                    <td className="px-3 py-2 text-right text-slate-500">{run.rows ?? 0}</td>
                    <td className="px-3 py-2 text-right">
                      {run.studio_url && (
                        <a
                          href={run.studio_url}
                          target="_blank"
                          rel="noreferrer"
                          className="inline-flex items-center gap-1 text-xs text-accent-soft hover:text-accent-glow"
                        >
                          Portal <ExternalLink className="h-3 w-3" />
                        </a>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {report?.failures && (report.failures as unknown[]).length > 0 && (
          <div className="mt-3 text-xs text-rose-300">
            {(report.failures as { label?: string; error?: string }[]).map((f, i) => (
              <div key={i}>
                {f.label}: {f.error}
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}

function fmtMetric(value: unknown): string {
  if (value == null) return "—";
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(3) : String(value);
}
