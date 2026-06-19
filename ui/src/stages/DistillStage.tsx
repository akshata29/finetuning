import { useEffect, useState } from "react";
import { Recycle, RefreshCw, Sparkles } from "lucide-react";
import {
  getDistillSummary,
  retrain,
  runDistill,
  type DistillParams,
  type FineTuneParams,
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
import type { DistillSummary } from "../types";

export function DistillStage() {
  const { trackJob } = useJobContext();
  const [error, setError] = useState<string | null>(null);

  const [threshold, setThreshold] = useState("0.75");
  const [includeAll, setIncludeAll] = useState(false);
  const [valFraction, setValFraction] = useState("0.2");
  const [distilling, setDistilling] = useState(false);
  const [summary, setSummary] = useState<DistillSummary | null>(null);

  const [method, setMethod] = useState<FineTuneParams["method"]>("supervised");
  const [nEpochs, setNEpochs] = useState("");
  const [retraining, setRetraining] = useState(false);

  const loadSummary = async () => {
    try {
      const { summary } = await getDistillSummary();
      setSummary(summary);
    } catch {
      /* ignore */
    }
  };
  useEffect(() => {
    loadSummary();
  }, []);

  const doDistill = async () => {
    setDistilling(true);
    setError(null);
    try {
      const params: DistillParams = {
        threshold: +threshold,
        include_all: includeAll,
        val_fraction: +valFraction,
      };
      const result = await runDistill(params);
      setSummary(result);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setDistilling(false);
    }
  };

  const doRetrain = async () => {
    setRetraining(true);
    setError(null);
    try {
      const params: FineTuneParams = {
        method,
        n_epochs: nEpochs ? +nEpochs : null,
        beta: null,
        grader_model: null,
        max_polls: null,
        deploy: false,
        deployment_name: null,
        sku: "developer",
      };
      const { job_id } = await retrain(params);
      trackJob(job_id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setRetraining(false);
    }
  };

  return (
    <div className="space-y-5">
      {error && <ErrorBanner message={error} />}

      <div className="grid gap-5 xl:grid-cols-2">
        <Card
          title="Distill transcripts"
          description="Turn captured agent transcripts into a fresh SFT + DPO retrain corpus."
        >
          <div className="grid grid-cols-2 gap-4">
            <Field label="Score threshold" hint="select cases below this score">
              <input
                type="number"
                step="0.05"
                min={0}
                max={1}
                className="input"
                value={threshold}
                onChange={(e) => setThreshold(e.target.value)}
              />
            </Field>
            <Field label="Val fraction">
              <input
                type="number"
                step="0.05"
                min={0}
                max={0.9}
                className="input"
                value={valFraction}
                onChange={(e) => setValFraction(e.target.value)}
              />
            </Field>
          </div>
          <div className="mt-4">
            <Toggle
              checked={includeAll}
              onChange={setIncludeAll}
              label="Include all completed cases (ignore threshold)"
            />
          </div>
          <div className="mt-5">
            <Button
              onClick={doDistill}
              loading={distilling}
              icon={<Recycle className="h-4 w-4" />}
            >
              Distill corpus
            </Button>
          </div>
        </Card>

        <Card
          title="Retrain from distilled corpus"
          description="Closes the loop — fine-tune again on the distilled data. Runs live."
        >
          <div className="grid grid-cols-2 gap-4">
            <Field label="Method">
              <select
                className="input"
                value={method}
                onChange={(e) => setMethod(e.target.value as FineTuneParams["method"])}
              >
                <option value="supervised">SFT</option>
                <option value="dpo">DPO</option>
                <option value="reinforcement">RFT</option>
              </select>
            </Field>
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
          </div>
          <div className="mt-5">
            <Button
              onClick={doRetrain}
              loading={retraining}
              disabled={!summary || summary.train === 0}
              icon={<Sparkles className="h-4 w-4" />}
            >
              Retrain {method.toUpperCase()}
            </Button>
          </div>
          {(!summary || summary.train === 0) && (
            <p className="mt-2 text-xs text-slate-500">
              Distill a corpus first to enable retraining.
            </p>
          )}
        </Card>
      </div>

      <Card
        title="Distill summary"
        actions={
          <Button variant="ghost" onClick={loadSummary} icon={<RefreshCw className="h-4 w-4" />}>
            Refresh
          </Button>
        }
      >
        {!summary ? (
          <EmptyState>No distilled corpus yet.</EmptyState>
        ) : (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Stat label="Selected" value={summary.selected} />
            <Stat label="Skipped" value={summary.skipped} />
            <Stat label="Train rows" value={summary.train} accent />
            <Stat label="Val rows" value={summary.val} accent />
          </div>
        )}
      </Card>
    </div>
  );
}

function Stat({
  label,
  value,
  accent = false,
}: {
  label: string;
  value: number;
  accent?: boolean;
}) {
  return (
    <div className="rounded-lg border border-white/5 bg-ink-850/40 p-4">
      <div className="text-xs uppercase tracking-wide text-slate-500">{label}</div>
      <div
        className={`mt-1 text-2xl font-semibold ${
          accent ? "text-accent-soft" : "text-slate-100"
        }`}
      >
        {value}
      </div>
    </div>
  );
}
