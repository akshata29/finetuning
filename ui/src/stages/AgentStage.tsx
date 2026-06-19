import { useEffect, useState } from "react";
import { Bot, Play, Plus, RefreshCw, Trash2 } from "lucide-react";
import {
  createAgent,
  deleteAgent,
  getTranscripts,
  listAgents,
  testAgent,
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
import type { AgentInfo, AgentTranscript } from "../types";

export function AgentStage() {
  const { trackJob } = useJobContext();
  const [error, setError] = useState<string | null>(null);

  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [loadingAgents, setLoadingAgents] = useState(false);
  const [createModel, setCreateModel] = useState("sft");
  const [createName, setCreateName] = useState("");
  const [creating, setCreating] = useState(false);

  const [testId, setTestId] = useState("");
  const [testModel, setTestModel] = useState("sft");
  const [testLimit, setTestLimit] = useState("");
  const [ephemeral, setEphemeral] = useState(false);
  const [testing, setTesting] = useState(false);

  const [transcripts, setTranscripts] = useState<AgentTranscript[]>([]);

  const loadAgents = async () => {
    setLoadingAgents(true);
    setError(null);
    try {
      const { agents } = await listAgents();
      setAgents(agents);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoadingAgents(false);
    }
  };
  const loadTranscripts = async () => {
    try {
      const { transcripts } = await getTranscripts();
      setTranscripts(transcripts);
    } catch {
      /* ignore */
    }
  };
  useEffect(() => {
    loadAgents();
    loadTranscripts();
  }, []);

  const doCreate = async () => {
    setCreating(true);
    setError(null);
    try {
      const agent = await createAgent(createModel, createName || undefined);
      setCreateName("");
      setTestId(agent.id);
      await loadAgents();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setCreating(false);
    }
  };

  const doDelete = async (id: string) => {
    setError(null);
    try {
      await deleteAgent(id);
      await loadAgents();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const doTest = async () => {
    setTesting(true);
    setError(null);
    try {
      const { job_id } = await testAgent({
        id: testId || null,
        model: testModel,
        limit: testLimit ? +testLimit : null,
        ephemeral,
      });
      trackJob(job_id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setTesting(false);
    }
  };

  const meanScore =
    transcripts.length > 0
      ? transcripts
          .map((t) => t.strategy_score ?? 0)
          .reduce((a, b) => a + b, 0) / transcripts.length
      : null;

  return (
    <div className="space-y-5">
      {error && <ErrorBanner message={error} />}

      <div className="grid gap-5 xl:grid-cols-2">
        <Card
          title="Create agent"
          description="Wrap a fine-tuned deployment in a managed Agent Service entity."
        >
          <div className="grid grid-cols-2 gap-4">
            <Field label="Model arm">
              <select
                className="input"
                value={createModel}
                onChange={(e) => setCreateModel(e.target.value)}
              >
                {["base", "sft", "dpo", "rft"].map((m) => (
                  <option key={m} value={m}>
                    {m.toUpperCase()}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Name" hint="blank = auto">
              <input
                className="input"
                placeholder="conv-align-agent"
                value={createName}
                onChange={(e) => setCreateName(e.target.value)}
              />
            </Field>
          </div>
          <div className="mt-5">
            <Button onClick={doCreate} loading={creating} icon={<Plus className="h-4 w-4" />}>
              Create agent
            </Button>
          </div>
        </Card>

        <Card
          title="Test agent"
          description="Run the eval set through live threads + runs and capture transcripts."
        >
          <Field label="Agent id" hint="blank = create an ephemeral agent for this run">
            <input
              className="input font-mono"
              placeholder="asst_…"
              value={testId}
              onChange={(e) => setTestId(e.target.value)}
            />
          </Field>
          <div className="mt-4 grid grid-cols-2 gap-4">
            <Field label="Model arm" hint="used when no id given">
              <select
                className="input"
                value={testModel}
                onChange={(e) => setTestModel(e.target.value)}
              >
                {["base", "sft", "dpo", "rft"].map((m) => (
                  <option key={m} value={m}>
                    {m.toUpperCase()}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Limit" hint="blank = full set">
              <input
                type="number"
                className="input"
                placeholder="all"
                value={testLimit}
                min={1}
                onChange={(e) => setTestLimit(e.target.value)}
              />
            </Field>
          </div>
          <div className="mt-4">
            <Toggle
              checked={ephemeral}
              onChange={setEphemeral}
              label="Ephemeral (delete the agent after testing)"
            />
          </div>
          <div className="mt-5">
            <Button onClick={doTest} loading={testing} icon={<Play className="h-4 w-4" />}>
              Run agent test
            </Button>
          </div>
        </Card>
      </div>

      <Card
        title="Agents"
        actions={
          <Button
            variant="ghost"
            onClick={loadAgents}
            icon={<RefreshCw className={`h-4 w-4 ${loadingAgents ? "animate-spin" : ""}`} />}
          >
            Refresh
          </Button>
        }
      >
        {agents.length === 0 ? (
          <EmptyState>No agents yet.</EmptyState>
        ) : (
          <div className="space-y-2">
            {agents.map((a) => (
              <div
                key={a.id}
                className="flex items-center justify-between gap-3 rounded-lg border border-white/5 bg-ink-850/40 px-3 py-2.5"
              >
                <div className="flex min-w-0 items-center gap-2.5">
                  <Bot className="h-4 w-4 shrink-0 text-accent-soft" />
                  <div className="min-w-0">
                    <div className="truncate text-sm text-slate-200">{a.name ?? "—"}</div>
                    <div className="truncate font-mono text-xs text-slate-500">
                      {a.id} · {a.model ?? "?"}
                    </div>
                  </div>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <button
                    onClick={() => setTestId(a.id)}
                    className="text-xs text-accent-soft hover:text-accent-glow"
                  >
                    Use
                  </button>
                  <button
                    onClick={() => doDelete(a.id)}
                    className="rounded p-1 text-slate-500 hover:bg-rose-500/10 hover:text-rose-300"
                    title="Delete"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>

      <Card
        title="Captured transcripts"
        description={
          meanScore != null
            ? `${transcripts.length} cases · mean strategy ${meanScore.toFixed(3)}`
            : "Latest agent test transcripts."
        }
        actions={
          <Button variant="ghost" onClick={loadTranscripts} icon={<RefreshCw className="h-4 w-4" />}>
            Refresh
          </Button>
        }
      >
        {transcripts.length === 0 ? (
          <EmptyState>No transcripts captured yet.</EmptyState>
        ) : (
          <div className="space-y-2">
            {transcripts.map((t) => (
              <details
                key={t.id}
                className="group rounded-lg border border-white/5 bg-ink-850/40 px-3 py-2.5"
              >
                <summary className="flex cursor-pointer list-none items-center justify-between gap-3">
                  <div className="flex min-w-0 items-center gap-2">
                    <span className="rounded bg-ink-800 px-1.5 py-0.5 text-[11px] text-slate-400">
                      {t.strategy}
                    </span>
                    <span className="truncate text-sm text-slate-300">{t.query}</span>
                  </div>
                  <span
                    className={`shrink-0 font-mono text-xs ${scoreColor(t.strategy_score)}`}
                  >
                    {t.strategy_score != null ? t.strategy_score.toFixed(3) : "—"}
                  </span>
                </summary>
                <div className="mt-3 space-y-2 text-sm">
                  <Excerpt label="Ground truth" text={t.ground_truth} />
                  <Excerpt label="Agent response" text={t.agent_response} />
                  {t.error && <div className="text-xs text-rose-300">error: {t.error}</div>}
                </div>
              </details>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}

function Excerpt({ label, text }: { label: string; text: string }) {
  return (
    <div>
      <div className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</div>
      <p className="mt-0.5 whitespace-pre-wrap text-slate-300">{text}</p>
    </div>
  );
}

function scoreColor(score: number | null): string {
  if (score == null) return "text-slate-500";
  if (score >= 0.66) return "text-emerald-300";
  if (score >= 0.33) return "text-amber-300";
  return "text-rose-300";
}
