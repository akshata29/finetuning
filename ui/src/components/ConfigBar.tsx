import { useEffect, useState } from "react";
import { CheckCircle2, RefreshCw, ShieldCheck, ShieldX } from "lucide-react";
import { getConfig } from "../api";
import type { ConfigInfo } from "../types";

export function ConfigBar() {
  const [config, setConfig] = useState<ConfigInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      setConfig(await getConfig());
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const secretsOk =
    config && Object.values(config.secrets_present).every(Boolean);

  return (
    <div className="flex flex-wrap items-center gap-x-6 gap-y-2 border-b border-white/5 bg-ink-900/40 px-6 py-3 text-xs">
      {error ? (
        <span className="text-rose-300">API offline: {error}</span>
      ) : config ? (
        <>
          <Meta label="Project" value={hostOf(config.project_endpoint)} />
          <Meta label="Region" value={config.region ?? "—"} />
          <Meta label="Tier" value={config.deployment_tier ?? "—"} />
          <Meta label="Teacher" value={config.teacher_model ?? "—"} />
          <div className="flex items-center gap-1.5">
            <span className="uppercase tracking-wide text-slate-500">Deployments</span>
            <span className="flex items-center gap-1.5">
              {Object.entries(config.deployments).map(([k, v]) => (
                <span
                  key={k}
                  className={`rounded px-1.5 py-0.5 font-mono ${
                    v ? "bg-ink-800 text-slate-300" : "bg-ink-850 text-slate-600 line-through"
                  }`}
                  title={v ?? "not configured"}
                >
                  {k}
                </span>
              ))}
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            {secretsOk ? (
              <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
            ) : (
              <ShieldX className="h-3.5 w-3.5 text-amber-400" />
            )}
            <span className={secretsOk ? "text-emerald-300" : "text-amber-300"}>
              {secretsOk ? "secrets present" : "missing secrets"}
            </span>
          </div>
        </>
      ) : (
        <span className="text-slate-500">Loading config…</span>
      )}

      <button
        onClick={load}
        className="ml-auto inline-flex items-center gap-1.5 rounded px-2 py-1 text-slate-400 hover:bg-white/5 hover:text-slate-200"
      >
        <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
        Refresh
      </button>
    </div>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="uppercase tracking-wide text-slate-500">{label}</span>
      <span className="font-mono text-slate-300">{value}</span>
      <CheckCircle2 className="h-3 w-3 text-emerald-500/70" />
    </span>
  );
}

function hostOf(url: string | null): string {
  if (!url) return "—";
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}
