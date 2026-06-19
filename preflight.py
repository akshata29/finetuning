"""Pre-flight checker for the 1-Hour Azure Fine-Tuning Demo.

Validates demo prerequisites and reports a structured pass/fail summary
*without ever raising*: missing required environment variables, optional Azure
SDK availability, the deployment region, presence of the expected dataset
files, and deployment-tier sanity. ``main()`` prints the report and exits
non-zero when a required check fails (OWASP A05 — fail loudly on
misconfiguration, but never with an unhandled traceback).

This module imports and runs with **zero Azure SDKs installed**: optional SDKs
are probed through :func:`finetuning.config.optional_import`, which returns
availability flags instead of raising.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Allow running both as a module (``python -m finetuning.preflight``) and as
# a plain script (``python finetuning/preflight.py``): when launched as a
# script there is no package context, so register one and add the repo root to
# ``sys.path`` before resolving the relative imports below.
if __package__ in (None, ""):  # pragma: no cover - script-launch shim
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "finetuning"

from .config import TIER_NAMES, DemoConfig, optional_import
from .run_of_show import DATA_DIR, EVAL_FILE, TRAIN_FILE, VAL_FILE

logger = logging.getLogger(__name__)

EXIT_SUCCESS: int = 0
EXIT_FAILURE: int = 1

#: Optional Azure / ML SDKs probed during preflight (none are import-time deps).
OPTIONAL_SDKS: tuple[str, ...] = (
    "openai",
    "azure.ai.ml",
    "azure.ai.projects",
    "azure.identity",
    "sklearn",
)

SEVERITY_REQUIRED: str = "required"
SEVERITY_RECOMMENDED: str = "recommended"


def _record(
    name: str,
    ok: bool,
    severity: str,
    detail: str,
) -> dict[str, Any]:
    """Build a single structured check record."""
    return {"name": name, "ok": ok, "severity": severity, "detail": detail}


def check(
    config: DemoConfig | None = None,
    *,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Run all preflight checks and return a structured report.

    Never raises. The returned ``passed`` flag reflects only ``required``
    checks; ``recommended`` checks (optional SDKs, dataset presence) are
    reported but do not fail preflight on their own.

    Parameters
    ----------
    config:
        Demo configuration; defaults to :meth:`DemoConfig.from_env`.
    data_dir:
        Folder expected to hold the dataset JSONL files; defaults to
        :data:`finetuning.run_of_show.DATA_DIR`.

    Returns
    -------
    dict[str, Any]
        ``{"passed": bool, "checks": list[dict]}``.
    """
    cfg = config or DemoConfig.from_env()
    folder = data_dir or DATA_DIR
    checks: list[dict[str, Any]] = []

    # Required environment variables (core endpoint / identity fields).
    missing_env = cfg.missing_required()
    checks.append(
        _record(
            "required_env_vars",
            ok=not missing_env,
            severity=SEVERITY_REQUIRED,
            detail=(
                "all required env vars set"
                if not missing_env
                else f"missing: {', '.join(missing_env)}"
            ),
        )
    )

    # Region must be set for model/quota selection.
    checks.append(
        _record(
            "region",
            ok=bool(cfg.azure_region),
            severity=SEVERITY_REQUIRED,
            detail=(
                f"AZURE_REGION={cfg.azure_region}"
                if cfg.azure_region
                else "AZURE_REGION is not set"
            ),
        )
    )

    # Deployment tier sanity.
    tier_ok = cfg.deployment_tier in TIER_NAMES
    checks.append(
        _record(
            "deployment_tier",
            ok=tier_ok,
            severity=SEVERITY_REQUIRED,
            detail=(
                f"tier '{cfg.deployment_tier}' is valid"
                if tier_ok
                else f"tier '{cfg.deployment_tier}' not in {', '.join(TIER_NAMES)}"
            ),
        )
    )

    # Optional SDK availability (recommended; not import-time dependencies).
    for sdk in OPTIONAL_SDKS:
        _, available = optional_import(sdk)
        checks.append(
            _record(
                f"sdk:{sdk}",
                ok=available,
                severity=SEVERITY_RECOMMENDED,
                detail="importable" if available else "not installed",
            )
        )

    # Expected dataset files (recommended; can be generated live in Act 1).
    for name in (TRAIN_FILE, VAL_FILE, EVAL_FILE):
        path = folder / name
        checks.append(
            _record(
                f"data:{name}",
                ok=path.exists(),
                severity=SEVERITY_RECOMMENDED,
                detail=str(path) + (" present" if path.exists() else " missing"),
            )
        )

    passed = all(item["ok"] for item in checks if item["severity"] == SEVERITY_REQUIRED)
    return {"passed": passed, "checks": checks}


def configure_logging(verbose: bool = False) -> None:
    """Configure root logging for the preflight script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def format_report(report: dict[str, Any]) -> str:
    """Render the structured report as aligned, human-readable lines."""
    lines = ["Pre-flight checklist:"]
    for item in report["checks"]:
        mark = "PASS" if item["ok"] else "FAIL"
        flag = "" if item["severity"] == SEVERITY_REQUIRED else " (recommended)"
        lines.append(f"  [{mark}] {item['name']}{flag}: {item['detail']}")
    overall = "READY" if report["passed"] else "NOT READY"
    lines.append(f"Overall: {overall} (required checks {'all pass' if report['passed'] else 'failing'})")
    return "\n".join(lines)


def create_parser() -> argparse.ArgumentParser:
    """Create the preflight argument parser."""
    parser = argparse.ArgumentParser(
        prog="preflight",
        description="Validate prerequisites for the 1-Hour Azure Fine-Tuning Demo.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR, help="Folder expected to hold dataset JSONL files."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Print the preflight report and exit non-zero when required checks fail."""
    parser = create_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    report = check(data_dir=args.data_dir)
    print(format_report(report))
    return EXIT_SUCCESS if report["passed"] else EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
