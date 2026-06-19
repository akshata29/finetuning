import { useEffect, useState } from "react";
import { RefreshCw, Rocket } from "lucide-react";
import { deployModel, listFinetuneJobs } from "../api";
import { useJobContext } from "../context/JobContext";
import { Button, Card, EmptyState, ErrorBanner, Field } from "../components/ui";

export function DeployStage() {
  const { trackJob } = useJobContext();
  const [modelId, setModelId] = useState("");
  const [deploymentName, setDeploymentName] = useState("");
  const [sku, setSku] = useState("developer");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [jobs, setJobs] = useState<Record<string, unknown>[]>([]);
  const [loadingJobs, setLoadingJobs] = useState(false);

  const loadJobs = async () => {
    setLoadingJobs(true);
    setError(null);
    try {
      const { jobs } = await listFinetuneJobs(15);
      setJobs(jobs);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoadingJobs(false);
    }
  };
  useEffect(() => {
    loadJobs();
  }, []);

  const submit = async () => {
    setSubmitting(true);
    setError(null);
    try {
      const { job_id } = await deployModel({
        model_id: modelId,
        deployment_name: deploymentName,
        sku,
      });
      trackJob(job_id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="grid gap-5 xl:grid-cols-[420px_1fr]">
      <Card
        title="Deploy a fine-tuned model"
        description="Publish a completed fine-tuned model id to a managed deployment."
      >
        {error && (
          <div className="mb-4">
            <ErrorBanner message={error} />
          </div>
        )}
        <Field label="Fine-tuned model id" hint="e.g. gpt-4.1-mini-2025…:conv-align:…">
          <input
            className="input font-mono"
            placeholder="ft:…"
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
          />
        </Field>
        <div className="mt-4 grid grid-cols-2 gap-4">
          <Field label="Deployment name">
            <input
              className="input"
              placeholder="conv-align-sft"
              value={deploymentName}
              onChange={(e) => setDeploymentName(e.target.value)}
            />
          </Field>
          <Field label="SKU">
            <input className="input" value={sku} onChange={(e) => setSku(e.target.value)} />
          </Field>
        </div>
        <div className="mt-5">
          <Button
            onClick={submit}
            loading={submitting}
            disabled={!modelId || !deploymentName}
            icon={<Rocket className="h-4 w-4" />}
          >
            Deploy model
          </Button>
        </div>
      </Card>

      <Card
        title="Recent fine-tuning jobs"
        description="Live from Azure. Click a finished job to pre-fill the model id."
        actions={
          <Button
            variant="ghost"
            onClick={loadJobs}
            icon={<RefreshCw className={`h-4 w-4 ${loadingJobs ? "animate-spin" : ""}`} />}
          >
            Refresh
          </Button>
        }
      >
        {jobs.length === 0 ? (
          <EmptyState>No fine-tuning jobs found (or Azure unreachable).</EmptyState>
        ) : (
          <div className="space-y-2">
            {jobs.map((j, i) => {
              const ft = j.fine_tuned_model as string | null;
              return (
                <button
                  key={(j.id as string) ?? i}
                  disabled={!ft}
                  onClick={() => ft && setModelId(ft)}
                  className="flex w-full items-center justify-between gap-3 rounded-lg border border-white/5 bg-ink-850/40 px-3 py-2.5 text-left transition hover:bg-ink-800 disabled:cursor-default disabled:opacity-60"
                >
                  <div className="min-w-0">
                    <div className="truncate font-mono text-xs text-slate-300">
                      {(j.id as string) ?? "—"}
                    </div>
                    <div className="truncate text-[11px] text-slate-500">
                      {ft ?? "no model yet"}
                    </div>
                  </div>
                  <span className="shrink-0 rounded bg-ink-800 px-2 py-0.5 text-xs text-slate-400">
                    {String(j.status)}
                  </span>
                </button>
              );
            })}
          </div>
        )}
      </Card>
    </div>
  );
}
