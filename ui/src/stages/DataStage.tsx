import { useEffect, useState } from "react";
import { Eye, Play, RefreshCw } from "lucide-react";
import {
  generateData,
  listDatasets,
  previewDataset,
  type GenerateDataParams,
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
import type { DatasetInfo, PreviewResponse } from "../types";

export function DataStage() {
  const { trackJob } = useJobContext();
  const [form, setForm] = useState<GenerateDataParams>({
    count: 60,
    eval_count: 30,
    seed: 1337,
    use_llm: false,
    concurrency: 4,
  });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [loadingDs, setLoadingDs] = useState(false);
  const [preview, setPreview] = useState<PreviewResponse | null>(null);
  const [previewName, setPreviewName] = useState<string | null>(null);

  const loadDatasets = async () => {
    setLoadingDs(true);
    try {
      const { datasets } = await listDatasets();
      setDatasets(datasets);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoadingDs(false);
    }
  };

  useEffect(() => {
    loadDatasets();
  }, []);

  const submit = async () => {
    setSubmitting(true);
    setError(null);
    try {
      const { job_id } = await generateData(form);
      trackJob(job_id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  const openPreview = async (name: string) => {
    setPreviewName(name);
    setPreview(null);
    try {
      setPreview(await previewDataset(name, 15));
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <div className="grid gap-5 xl:grid-cols-[380px_1fr]">
      <Card
        title="Generate synthetic corpus"
        description="Build the SFT / DPO / RFT training and evaluation datasets."
      >
        {error && (
          <div className="mb-4">
            <ErrorBanner message={error} />
          </div>
        )}
        <div className="grid grid-cols-2 gap-4">
          <Field label="Train count">
            <input
              type="number"
              className="input"
              value={form.count}
              min={1}
              onChange={(e) => setForm({ ...form, count: +e.target.value })}
            />
          </Field>
          <Field label="Eval count">
            <input
              type="number"
              className="input"
              value={form.eval_count}
              min={1}
              onChange={(e) => setForm({ ...form, eval_count: +e.target.value })}
            />
          </Field>
          <Field label="Seed">
            <input
              type="number"
              className="input"
              value={form.seed}
              onChange={(e) => setForm({ ...form, seed: +e.target.value })}
            />
          </Field>
          <Field label="Concurrency" hint="LLM teacher only">
            <input
              type="number"
              className="input"
              value={form.concurrency}
              min={1}
              max={16}
              disabled={!form.use_llm}
              onChange={(e) => setForm({ ...form, concurrency: +e.target.value })}
            />
          </Field>
        </div>
        <div className="mt-4">
          <Toggle
            checked={form.use_llm}
            onChange={(v) => setForm({ ...form, use_llm: v })}
            label="Use LLM teacher (richer, slower) instead of templates"
          />
        </div>
        <div className="mt-5">
          <Button onClick={submit} loading={submitting} icon={<Play className="h-4 w-4" />}>
            Generate datasets
          </Button>
        </div>
      </Card>

      <Card
        title="Datasets"
        description="Files on disk in the data directories."
        actions={
          <Button
            variant="ghost"
            onClick={loadDatasets}
            icon={<RefreshCw className={`h-4 w-4 ${loadingDs ? "animate-spin" : ""}`} />}
          >
            Refresh
          </Button>
        }
      >
        {datasets.length === 0 ? (
          <EmptyState>No datasets yet. Generate a corpus to get started.</EmptyState>
        ) : (
          <div className="overflow-hidden rounded-lg border border-white/5">
            <table className="w-full text-sm">
              <thead className="bg-ink-850 text-left text-xs uppercase tracking-wide text-slate-500">
                <tr>
                  <th className="px-3 py-2 font-medium">Group</th>
                  <th className="px-3 py-2 font-medium">File</th>
                  <th className="px-3 py-2 text-right font-medium">Rows</th>
                  <th className="px-3 py-2 text-right font-medium">Size</th>
                  <th className="px-3 py-2" />
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {datasets.map((d) => (
                  <tr key={d.name} className="hover:bg-white/5">
                    <td className="px-3 py-2">
                      <span className="rounded bg-ink-800 px-1.5 py-0.5 text-xs text-slate-400">
                        {d.group}
                      </span>
                    </td>
                    <td className="px-3 py-2 font-mono text-xs text-slate-300">
                      {d.filename}
                    </td>
                    <td className="px-3 py-2 text-right text-slate-400">
                      {d.rows ?? "—"}
                    </td>
                    <td className="px-3 py-2 text-right text-slate-500">
                      {formatBytes(d.bytes)}
                    </td>
                    <td className="px-3 py-2 text-right">
                      <button
                        onClick={() => openPreview(d.name)}
                        className="inline-flex items-center gap-1 text-xs text-accent-soft hover:text-accent-glow"
                      >
                        <Eye className="h-3.5 w-3.5" /> Preview
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {previewName && (
          <div className="mt-5">
            <div className="mb-2 flex items-center justify-between">
              <h4 className="font-mono text-xs text-slate-400">{previewName}</h4>
              <button
                onClick={() => {
                  setPreviewName(null);
                  setPreview(null);
                }}
                className="text-xs text-slate-500 hover:text-slate-300"
              >
                Close
              </button>
            </div>
            <pre className="max-h-80 overflow-auto rounded-lg border border-white/5 bg-ink-950/80 p-3 text-xs leading-relaxed text-slate-300">
              {preview
                ? preview.rows.map((r) => JSON.stringify(r, null, 2)).join("\n\n")
                : "Loading…"}
            </pre>
          </div>
        )}
      </Card>
    </div>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
