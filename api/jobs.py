"""In-process background job manager for the demo API.

Long-running demo stages (synthetic generation with a teacher LLM, fine-tuning,
deployment, Foundry evaluation, agent test capture, retraining) take minutes to
hours. The HTTP API must return immediately and let the frontend poll for
progress, so each long stage runs on a background thread tracked by this
manager.

Two things make the live logs work:

* **Thread-routed stdout.** :class:`_ThreadRoutedStream` wraps the real stdout
  and, based on the *current thread*, appends every ``print`` to the matching
  job's log buffer while still echoing to the server console. This is what
  surfaces the demo functions' rich ``print`` output (scoreboards, per-case
  lines) in the UI without modifying any of them.
* **Thread-routed logging.** :class:`_JobLogHandler` does the same for the
  ``logging`` records the demo emits, routed by ``record.thread``.

The manager is deliberately in-memory and single-process: this is a local demo
control plane, not a multi-worker production service. Run the API with a single
uvicorn worker.
"""

from __future__ import annotations

import io
import logging
import sys
import threading
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Optional

logger = logging.getLogger(__name__)

#: Cap a single job's retained log lines so a runaway poller can't exhaust RAM.
MAX_LOG_LINES: int = 5000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    """A single tracked background task."""

    id: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending | running | succeeded | failed
    created: str = field(default_factory=_now)
    started: Optional[str] = None
    finished: Optional[str] = None
    result: Any = None
    error: Optional[str] = None
    logs: Deque[str] = field(default_factory=lambda: deque(maxlen=MAX_LOG_LINES))
    _buffer: str = field(default="", repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def append_log(self, text: str) -> None:
        # Buffer until a newline so a separate write of '\n' (how print emits the
        # line terminator) doesn't create a spurious empty log entry.
        with self._lock:
            self._buffer += text
            if "\n" not in self._buffer:
                return
            *complete, self._buffer = self._buffer.split("\n")
            for line in complete:
                self.logs.append(line)

    def flush_log(self) -> None:
        """Emit any buffered partial line (call when the job finishes)."""
        with self._lock:
            if self._buffer:
                self.logs.append(self._buffer)
                self._buffer = ""

    def snapshot(self, *, log_offset: int = 0, include_logs: bool = True) -> dict[str, Any]:
        """Serialize the job; ``log_offset`` returns only newer log lines."""
        with self._lock:
            all_logs = list(self.logs)
        payload: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "params": self.params,
            "status": self.status,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "result": self.result,
            "error": self.error,
            "log_count": len(all_logs),
        }
        if include_logs:
            offset = max(0, min(log_offset, len(all_logs)))
            payload["logs"] = all_logs[offset:]
            payload["log_offset"] = offset
        return payload


class _ThreadRoutedStream(io.TextIOBase):
    """A stdout/stderr proxy that tees writes to the current thread's job."""

    def __init__(self, original: Any, manager: "JobManager") -> None:
        self._original = original
        self._manager = manager

    def write(self, text: str) -> int:  # type: ignore[override]
        job = self._manager.job_for_thread(threading.get_ident())
        if job is not None and text:
            job.append_log(text)
        if self._original is not None:
            try:
                self._original.write(text)
            except Exception:  # noqa: BLE001 - never let logging break a write
                pass
        return len(text)

    def flush(self) -> None:  # type: ignore[override]
        if self._original is not None:
            try:
                self._original.flush()
            except Exception:  # noqa: BLE001
                pass


class _JobLogHandler(logging.Handler):
    """A logging handler that routes records to the originating thread's job."""

    def __init__(self, manager: "JobManager") -> None:
        super().__init__()
        self._manager = manager
        self.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        job = self._manager.job_for_thread(record.thread)
        if job is None:
            return
        try:
            job.append_log(self.format(record))
        except Exception:  # noqa: BLE001 - logging must never raise
            pass


class JobManager:
    """Tracks background jobs and routes their stdout/logging into log buffers."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._thread_to_job: dict[int, Job] = {}
        self._lock = threading.Lock()
        self._installed = False

    # -- log routing install -------------------------------------------------
    def install_log_routing(self) -> None:
        """Install the thread-routed stdout/stderr/logging hooks (idempotent)."""
        if self._installed:
            return
        sys.stdout = _ThreadRoutedStream(sys.__stdout__, self)
        sys.stderr = _ThreadRoutedStream(sys.__stderr__, self)
        handler = _JobLogHandler(self)
        handler.setLevel(logging.INFO)
        root = logging.getLogger()
        if root.level > logging.INFO or root.level == logging.NOTSET:
            root.setLevel(logging.INFO)
        root.addHandler(handler)
        self._installed = True

    def job_for_thread(self, ident: int) -> Optional[Job]:
        return self._thread_to_job.get(ident)

    # -- job lifecycle -------------------------------------------------------
    def submit(
        self,
        kind: str,
        target: Callable[..., Any],
        *args: Any,
        params: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Job:
        """Create a job and run ``target(*args, **kwargs)`` on a worker thread."""
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, params=params or {})
        with self._lock:
            self._jobs[job.id] = job

        def _worker() -> None:
            ident = threading.get_ident()
            self._thread_to_job[ident] = job
            job.status = "running"
            job.started = _now()
            try:
                job.result = target(*args, **kwargs)
                job.status = "succeeded"
            except Exception as exc:  # noqa: BLE001 - capture for the UI
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.append_log("".join(traceback.format_exc()))
                logger.exception("job %s (%s) failed", job.id, kind)
            finally:
                job.flush_log()
                job.finished = _now()
                self._thread_to_job.pop(ident, None)

        thread = threading.Thread(target=_worker, name=f"job-{job.id}", daemon=True)
        thread.start()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        return [j.snapshot(include_logs=False) for j in sorted(jobs, key=lambda j: j.created, reverse=True)]


#: Process-wide singleton used by the FastAPI app.
manager = JobManager()
