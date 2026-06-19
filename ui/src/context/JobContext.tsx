import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { getJob } from "../api";
import type { JobSnapshot } from "../types";

const POLL_MS = 1200;

interface JobContextValue {
  /** Snapshot of the currently-tracked job (null when none). */
  job: JobSnapshot | null;
  /** Accumulated log lines for the tracked job. */
  logs: string[];
  /** Whether the console panel is expanded. */
  consoleOpen: boolean;
  setConsoleOpen: (open: boolean) => void;
  /** Start tracking a job id (resets logs, opens the console). */
  trackJob: (jobId: string) => void;
  /** Stop tracking and clear the console. */
  clear: () => void;
}

const JobContext = createContext<JobContextValue | null>(null);

function isTerminal(status: string | undefined): boolean {
  return status === "succeeded" || status === "failed";
}

export function JobProvider({ children }: { children: ReactNode }) {
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<JobSnapshot | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [consoleOpen, setConsoleOpen] = useState(false);
  const offsetRef = useRef(0);

  const trackJob = useCallback((id: string) => {
    setJobId(id);
    setJob(null);
    setLogs([]);
    offsetRef.current = 0;
    setConsoleOpen(true);
  }, []);

  const clear = useCallback(() => {
    setJobId(null);
    setJob(null);
    setLogs([]);
    offsetRef.current = 0;
  }, []);

  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    let timer: number | undefined;

    const poll = async () => {
      try {
        const snap = await getJob(jobId, offsetRef.current);
        if (cancelled) return;
        setJob(snap);
        if (snap.logs.length > 0) {
          setLogs((prev) => [...prev, ...snap.logs]);
          offsetRef.current = snap.log_count;
        }
        if (isTerminal(snap.status)) return; // stop polling on terminal state
      } catch {
        // transient error; keep polling
      }
      if (!cancelled) timer = window.setTimeout(poll, POLL_MS);
    };

    poll();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [jobId]);

  return (
    <JobContext.Provider
      value={{ job, logs, consoleOpen, setConsoleOpen, trackJob, clear }}
    >
      {children}
    </JobContext.Provider>
  );
}

export function useJobContext(): JobContextValue {
  const ctx = useContext(JobContext);
  if (!ctx) throw new Error("useJobContext must be used within a JobProvider");
  return ctx;
}
