import { useEffect, useRef } from "react";
import { ChevronDown, Terminal, X } from "lucide-react";
import { useJobContext } from "../context/JobContext";
import { StatusPill } from "./ui";

export function JobConsole() {
  const { job, logs, consoleOpen, setConsoleOpen, clear } = useJobContext();
  const scrollRef = useRef<HTMLDivElement>(null);
  const stickRef = useRef(true);

  // Auto-scroll to bottom while the user hasn't scrolled up.
  useEffect(() => {
    const el = scrollRef.current;
    if (el && stickRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [logs, consoleOpen]);

  if (!job && logs.length === 0) return null;

  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-0 z-30 flex justify-center px-4 pb-4">
      <div className="pointer-events-auto w-full max-w-5xl overflow-hidden rounded-xl border border-white/10 bg-ink-900/95 shadow-2xl backdrop-blur">
        <header className="flex items-center justify-between gap-3 border-b border-white/5 px-4 py-2.5">
          <div className="flex items-center gap-2.5">
            <Terminal className="h-4 w-4 text-accent-soft" />
            <span className="text-sm font-medium text-slate-200">
              {job ? job.kind : "Job console"}
            </span>
            {job && <StatusPill status={job.status} />}
            {job && (
              <span className="font-mono text-xs text-slate-500">{job.id}</span>
            )}
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={() => setConsoleOpen(!consoleOpen)}
              className="rounded p-1 text-slate-400 hover:bg-white/5 hover:text-slate-200"
              title={consoleOpen ? "Collapse" : "Expand"}
            >
              <ChevronDown
                className={`h-4 w-4 transition-transform ${consoleOpen ? "" : "rotate-180"}`}
              />
            </button>
            <button
              onClick={clear}
              className="rounded p-1 text-slate-400 hover:bg-white/5 hover:text-slate-200"
              title="Dismiss"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </header>

        {consoleOpen && (
          <div
            ref={scrollRef}
            onScroll={(e) => {
              const el = e.currentTarget;
              stickRef.current =
                el.scrollHeight - el.scrollTop - el.clientHeight < 40;
            }}
            className="max-h-72 overflow-y-auto bg-ink-950/80 px-4 py-3 font-mono text-xs leading-relaxed text-slate-300"
          >
            {logs.length === 0 ? (
              <p className="text-slate-500">Waiting for output…</p>
            ) : (
              logs.map((line, i) => (
                <div key={i} className="whitespace-pre-wrap break-words">
                  {line || "\u00a0"}
                </div>
              ))
            )}
            {job?.status === "failed" && job.error && (
              <div className="mt-2 whitespace-pre-wrap break-words text-rose-300">
                {job.error}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
