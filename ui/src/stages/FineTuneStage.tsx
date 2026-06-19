import { useEffect, useState } from "react";
import { Play, RefreshCw } from "lucide-react";
import {
  getFinetuneStates,
  startFinetune,
  type FineTuneParams,
} from "../api";
import { useJobContext } from "../context/JobContext";
import { Button, Card, EmptyState, ErrorBanner, Field, Toggle } from "../components/ui";

const METHODS: { id: FineTuneParams["method"]; label: string; blurb: string }[] = [
  { id: "supervised", label: "SFT", blurb: "Supervised fine-tuning" },
  { id: "dpo", label: "DPO", blurb: "Preference optimization" },
  { id: "reinforcement", label: "RFT", blurb: "Reinforcement (graded)" },
];

export function FineTuneStage() {
  const { trackJob } = useJobContext();
  const [method, setMethod] = useState<FineTuneParams["method"]>("supervised");
  const [nEpochs, setNEpochs] = useState<string>("");
  const [beta, setBeta] = useState<string>("");
  const [batchSize, setBatchSize] = useState<string>("");
  const [lrMultiplier, setLrMultiplier] = useState<string>("");
  const [evalInterval, setEvalInterval] = useState<string>("");
  const [evalSamples, setEvalSamples] = useState<string>("");
  const [reasoningEffort, setReasoningEffort] = useState<string>("");
  const [computeMultiplier, setComputeMultiplier] = useState<string>("");
  const [graderModel, setGraderModel] = useState<string>("");
  const [maxPolls, setMaxPolls] = useState<string>("");
  const [deploy, setDeploy] = useState(false);
  const [deploymentName, setDeploymentName] = useState<string>("");
  const [sku, setSku] = useState("developer");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isRft = method === "reinforcement";

  const [states, setStates] = useState<Record<string, unknown>>({});
  const loadStates = async () => {
    try {
      const { states } = await getFinetuneStates();
      setStates(states);
    } catch {
      /* ignore */
    }
  };
  useEffect(() => {
    loadStates();
  }, []);

  const submit = async () => {
    setSubmitting(true);
    setError(null);
    try {
      const params: FineTuneParams = {
        method,
        n_epochs: nEpochs ? +nEpochs : null,
        beta: beta ? +beta : null,
        batch_size: batchSize ? +batchSize : null,
        learning_rate_multiplier: lrMultiplier ? +lrMultiplier : null,
        eval_interval: evalInterval ? +evalInterval : null,
        eval_samples: evalSamples ? +evalSamples : null,
        reasoning_effort:
          (reasoningEffort as FineTuneParams["reasoning_effort"]) || null,
        compute_multiplier: computeMultiplier ? +computeMultiplier : null,
        grader_model: graderModel || null,
        max_polls: maxPolls ? +maxPolls : null,
        deploy,
        deployment_name: deploymentName || null,
        sku,
      };
      const { job_id } = await startFinetune(params);
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
        title="Start a fine-tuning job"
        description="Runs live against Azure AI Foundry. Long-running — progress streams to the console."
      >
        {error && (
          <div className="mb-4">
            <ErrorBanner message={error} />
          </div>
        )}

        <Field label="Method">
          <div className="grid grid-cols-3 gap-2">
            {METHODS.map((m) => (
              <button
                key={m.id}
                onClick={() => setMethod(m.id)}
                className={`rounded-lg border px-3 py-2.5 text-left transition ${
                  method === m.id
                    ? "border-accent/50 bg-accent/15"
                    : "border-white/10 bg-ink-850 hover:bg-ink-800"
                }`}
              >
                <div className="text-sm font-semibold text-slate-100">{m.label}</div>
                <div className="text-[11px] text-slate-500">{m.blurb}</div>
              </button>
            ))}
          </div>
        </Field>

        <div className="mt-4 grid grid-cols-2 gap-4">
          <Field label="Epochs" hint="blank = default">
            <input
              type="number"
              className="input"
              placeholder="auto"
              value={nEpochs}
              min={1}
              onChange={(e) => setNEpochs(e.target.value)}
            />
          </Field>
          <Field label="Beta (DPO)" hint="DPO only">
            <input
              type="number"
              step="0.1"
              className="input"
              placeholder="auto"
              value={beta}
              disabled={method !== "dpo"}
              onChange={(e) => setBeta(e.target.value)}
            />
          </Field>
          <Field label="Batch size" hint="any method">
            <input
              type="number"
              className="input"
              placeholder="auto"
              value={batchSize}
              min={1}
              onChange={(e) => setBatchSize(e.target.value)}
            />
          </Field>
          <Field label="LR multiplier" hint="any method">
            <input
              type="number"
              step="0.1"
              className="input"
              placeholder="auto"
              value={lrMultiplier}
              min={0}
              onChange={(e) => setLrMultiplier(e.target.value)}
            />
          </Field>
          <Field label="Grader model" hint="RFT only">
            <input
              className="input"
              placeholder="default grader"
              value={graderModel}
              disabled={!isRft}
              onChange={(e) => setGraderModel(e.target.value)}
            />
          </Field>
          <Field label="Max polls" hint="blank = wait fully">
            <input
              type="number"
              className="input"
              placeholder="unbounded"
              value={maxPolls}
              min={1}
              onChange={(e) => setMaxPolls(e.target.value)}
            />
          </Field>
        </div>

        <div
          className={`mt-4 rounded-lg border border-white/5 bg-ink-850/50 p-4 transition ${
            isRft ? "" : "opacity-50"
          }`}
        >
          <div className="mb-3 text-xs font-semibold uppercase tracking-wide text-accent-soft">
            Reinforcement (RFT) only
          </div>
          <div className="grid grid-cols-2 gap-4">
            <Field label="Eval interval" hint="steps between evals">
              <input
                type="number"
                className="input"
                placeholder="auto"
                value={evalInterval}
                min={1}
                disabled={!isRft}
                onChange={(e) => setEvalInterval(e.target.value)}
              />
            </Field>
            <Field label="Eval samples" hint="samples per eval">
              <input
                type="number"
                className="input"
                placeholder="auto"
                value={evalSamples}
                min={1}
                disabled={!isRft}
                onChange={(e) => setEvalSamples(e.target.value)}
              />
            </Field>
            <Field label="Reasoning effort">
              <select
                className="input"
                value={reasoningEffort}
                disabled={!isRft}
                onChange={(e) => setReasoningEffort(e.target.value)}
              >
                <option value="">auto</option>
                <option value="low">low</option>
                <option value="medium">medium</option>
                <option value="high">high</option>
              </select>
            </Field>
            <Field label="Compute multiplier">
              <input
                type="number"
                step="0.1"
                className="input"
                placeholder="auto"
                value={computeMultiplier}
                min={0}
                disabled={!isRft}
                onChange={(e) => setComputeMultiplier(e.target.value)}
              />
            </Field>
          </div>
        </div>

        <div className="mt-4 rounded-lg border border-white/5 bg-ink-850/50 p-4">
          <Toggle
            checked={deploy}
            onChange={setDeploy}
            label="Auto-deploy when the job completes"
          />
          {deploy && (
            <div className="mt-3 grid grid-cols-2 gap-4">
              <Field label="Deployment name" hint="blank = auto">
                <input
                  className="input"
                  placeholder="auto"
                  value={deploymentName}
                  onChange={(e) => setDeploymentName(e.target.value)}
                />
              </Field>
              <Field label="SKU">
                <input
                  className="input"
                  value={sku}
                  onChange={(e) => setSku(e.target.value)}
                />
              </Field>
            </div>
          )}
        </div>

        <div className="mt-5">
          <Button onClick={submit} loading={submitting} icon={<Play className="h-4 w-4" />}>
            Launch {method.toUpperCase()} job
          </Button>
        </div>
      </Card>

      <Card
        title="Last job states"
        description="Persisted state from the most recent fine-tune of each method."
        actions={
          <Button variant="ghost" onClick={loadStates} icon={<RefreshCw className="h-4 w-4" />}>
            Refresh
          </Button>
        }
      >
        {Object.keys(states).length === 0 ? (
          <EmptyState>No fine-tune state recorded yet.</EmptyState>
        ) : (
          <div className="space-y-3">
            {Object.entries(states).map(([label, data]) => (
              <div key={label} className="rounded-lg border border-white/5 bg-ink-850/40 p-3">
                <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-accent-soft">
                  {label}
                </div>
                <pre className="overflow-x-auto text-xs text-slate-300">
                  {JSON.stringify(data, null, 2)}
                </pre>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
