"""Throttle diagnostic — capture the RAW Azure OpenAI 429 (TPM vs RPM vs backend).

Answers the question "is this a TPM limit, an RPM limit, or shared-pool backend
capacity?" by reading the actual response headers and body instead of guessing.

It makes ONE chat call to the target deployment with retries DISABLED so the SDK
does not silently swallow the 429. Whatever comes back — 200 or 429 — it prints:

* HTTP status and the request id (``apim-request-id`` / ``x-ms-request-id``).
* ``retry-after`` / ``retry-after-ms`` (how long Azure asks you to wait).
* Every ``x-ratelimit-*`` header (remaining tokens vs remaining requests — the
  smoking gun: if BOTH remain high yet you still got 429, it is NOT your quota).
* The error ``type`` and ``message`` from the body (``Backend error`` => shared
  pool load-shedding; ``token rate limit`` => TPM; ``call rate limit`` => RPM).

Run from the repo root so the venv with the SDK is used:

    cd c:\\repos\\playground
    & .\\finetuning_demo\\.venv\\Scripts\\python.exe -m finetuning_demo.diagnose_throttle --deployment chat41mini

Or from the package dir with its venv active:

    python diagnose_throttle.py --deployment chat41mini --burst 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # pragma: no cover - script-launch shim
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "finetuning_demo"

from .act2a_serverless_sft import build_client
from .config import DemoConfig

#: Headers worth printing for a throttle diagnosis (lower-cased match).
_INTERESTING_PREFIXES: tuple[str, ...] = ("x-ratelimit", "retry-after", "x-ms-", "apim-")


def _print_headers(headers: Any) -> None:
    """Print the throttle-relevant headers from an httpx-style header mapping."""
    if headers is None:
        print("  (no headers available)")
        return
    items = sorted(headers.items())
    shown = False
    for key, value in items:
        if key.lower().startswith(_INTERESTING_PREFIXES):
            print(f"  {key}: {value}")
            shown = True
    if not shown:
        print("  (no x-ratelimit / retry-after headers present)")


def _one_call(client: Any, deployment: str, index: int) -> bool:
    """Make a single no-retry chat call; print the raw outcome. Return True on 200."""
    print(f"\n=== call {index} -> {deployment} (max_retries=0) ===")
    no_retry = client.with_options(max_retries=0)
    try:
        raw = no_retry.chat.completions.with_raw_response.create(
            model=deployment,
            temperature=0.0,
            max_tokens=16,
            messages=[
                {"role": "system", "content": "Reply with the single word: ok."},
                {"role": "user", "content": "ping"},
            ],
        )
    except Exception as exc:  # noqa: BLE001 - we want to inspect ANY failure
        status = getattr(getattr(exc, "response", None), "status_code", "?")
        print(f"  STATUS: {status} ({type(exc).__name__})")
        response = getattr(exc, "response", None)
        _print_headers(getattr(response, "headers", None))
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            err = body.get("error", body)
            print(f"  error.type: {err.get('type')!r}")
            print(f"  error.code: {err.get('code')!r}")
            print(f"  error.message: {err.get('message')!r}")
        else:
            text = getattr(response, "text", None)
            print(f"  body: {text or body!r}")
        return False

    print(f"  STATUS: {raw.status_code} (200 OK)")
    _print_headers(raw.headers)
    return True


def main(argv: list[str] | None = None) -> int:
    """Run the throttle diagnostic and return a process exit code."""
    parser = argparse.ArgumentParser(description="Azure OpenAI throttle diagnostic.")
    parser.add_argument("--deployment", required=True, help="Deployment name to probe.")
    parser.add_argument(
        "--burst",
        type=int,
        default=1,
        help="Number of back-to-back calls (probes RPM/concurrency). Default 1.",
    )
    args = parser.parse_args(argv)

    config = DemoConfig.from_env()
    print(f"endpoint: {config.azure_openai_endpoint}")
    print(f"api-version: {config.data_plane_api_version}")
    client = build_client(config)

    successes = 0
    for i in range(1, args.burst + 1):
        if _one_call(client, args.deployment, i):
            successes += 1

    print(f"\nsummary: {successes}/{args.burst} succeeded on the first try (no retries)")
    print(
        "interpretation:\n"
        "  - 'Backend error' / type=invalid_request_error  -> shared-pool capacity\n"
        "    (NOT your TPM/RPM quota; only PTU reserves capacity).\n"
        "  - 'token rate limit'  -> TPM; check x-ratelimit-remaining-tokens.\n"
        "  - 'call rate limit'   -> RPM; check x-ratelimit-remaining-requests.\n"
        "  - high remaining-tokens AND remaining-requests with a 429 proves it is\n"
        "    not your deployment quota."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
