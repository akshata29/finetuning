"""Run the demo API: ``python -m finetuning_demo.api``.

Starts uvicorn with a single worker (the job manager is in-process state, so
multiple workers would not share job status). Override host/port via
``DEMO_API_HOST`` / ``DEMO_API_PORT``.
"""

from __future__ import annotations

import os


def main() -> None:
    import uvicorn  # noqa: PLC0415 - optional dependency, imported on demand

    host = os.getenv("DEMO_API_HOST", "127.0.0.1")
    port = int(os.getenv("DEMO_API_PORT", "8000"))
    uvicorn.run("finetuning_demo.api.app:app", host=host, port=port, workers=1, reload=False)


if __name__ == "__main__":
    main()
