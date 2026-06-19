"""Centralized configuration for the 1-Hour Azure Fine-Tuning Demo.

All secrets, endpoints, and resource identifiers are read from environment
variables only — nothing is hardcoded (OWASP A05 Security Misconfiguration /
A07 Identification and Authentication Failures: no credentials in source).

This module imports cleanly with no Azure SDKs installed. Optional SDKs are
resolved lazily via :func:`optional_import`, which never raises on a missing
dependency; live Azure calls (made elsewhere) fail only with actionable errors.
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier names (deployment SKUs)
# ---------------------------------------------------------------------------
TIER_DEVELOPER: str = "developer"
TIER_STANDARD: str = "standard"
TIER_GLOBAL_STANDARD: str = "globalStandard"
TIER_NAMES: tuple[str, ...] = (TIER_DEVELOPER, TIER_STANDARD, TIER_GLOBAL_STANDARD)

# ---------------------------------------------------------------------------
# API versions
# ---------------------------------------------------------------------------
DATA_PLANE_API_VERSION: str = "2025-04-01-preview"
CONTROL_PLANE_API_VERSION: str = "2024-10-01"

#: Default location of the optional ``.env`` file (next to this package).
DEFAULT_ENV_FILE = os.path.join(os.path.dirname(__file__), ".env")


# ---------------------------------------------------------------------------
# .env loader (dependency-free; existing process env always wins)
# ---------------------------------------------------------------------------
def load_env_file(path: str | None = None, *, override: bool = False) -> int:
    """Load ``KEY=VALUE`` pairs from a ``.env`` file into :data:`os.environ`.

    A ``.env`` file is plain text and is **not** read by the process
    automatically; call this to make its values visible to
    :meth:`DemoConfig.from_env`. Lines that are blank or start with ``#`` are
    ignored, surrounding quotes are stripped, and a leading ``export`` is
    tolerated. By default, variables already present in the environment are left
    untouched (the shell wins); pass ``override=True`` to replace them.

    Never raises: a missing or unreadable file simply loads nothing.

    Parameters
    ----------
    path:
        Path to the ``.env`` file. Defaults to :data:`DEFAULT_ENV_FILE`.
    override:
        When ``True``, values from the file replace existing environment
        variables. Defaults to ``False``.

    Returns
    -------
    int
        The number of variables applied to :data:`os.environ`.
    """
    target = path or DEFAULT_ENV_FILE
    applied = 0
    try:
        with open(target, encoding="utf-8-sig") as handle:
            lines = handle.readlines()
    except OSError:
        return 0

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            applied += 1
    return applied



def _project_name_from_endpoint(endpoint: str) -> str:
    """Extract the Foundry project name from an AI project endpoint URL.

    The Foundry project endpoint looks like
    ``https://<account>.services.ai.azure.com/api/projects/<project_name>``;
    this returns the trailing ``<project_name>`` segment (empty string when the
    endpoint is blank or unrecognized). Used to populate the ``project_name``
    field that :func:`azure.ai.evaluation.evaluate` needs to upload cloud
    evaluation runs to the portal.
    """
    if not endpoint:
        return ""
    marker = "/api/projects/"
    if marker in endpoint:
        return endpoint.split(marker, 1)[1].strip("/").split("/", 1)[0]
    return endpoint.rstrip("/").rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------
# Optional-SDK import helper (graceful degradation)
# ---------------------------------------------------------------------------
def optional_import(name: str) -> tuple[ModuleType | None, bool]:
    """Attempt to import ``name``; return ``(module_or_None, available_bool)``.

    Never raises on a missing optional dependency. When the module is absent an
    actionable install hint is logged at warning level and ``(None, False)`` is
    returned, allowing callers to degrade gracefully.

    Parameters
    ----------
    name:
        Importable module path, e.g. ``"openai"`` or ``"azure.ai.projects"``.

    Returns
    -------
    tuple[ModuleType | None, bool]
        The imported module and ``True`` when available; ``(None, False)``
        otherwise.
    """
    try:
        module = importlib.import_module(name)
        return module, True
    except ImportError:
        logger.warning(
            "Optional dependency '%s' is not installed. Live Azure operations "
            "requiring it will be unavailable. Install with: pip install %s",
            name,
            name.split(".")[0],
        )
        return None, False


# ---------------------------------------------------------------------------
# Demo configuration
# ---------------------------------------------------------------------------
@dataclass
class DemoConfig:
    """Environment-sourced configuration for the fine-tuning demo.

    Every field is populated from an environment variable by
    :meth:`from_env`. No defaults carry secrets; endpoint/key fields default to
    empty strings so that the object always constructs without raising, while
    live callers can detect missing values and emit actionable errors.
    """

    # Project / account endpoints and identifiers
    azure_ai_project_endpoint: str = ""
    azure_ai_project_name: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_subscription_id: str = ""
    azure_resource_group: str = ""
    aoai_account_name: str = ""
    azure_region: str = ""

    # Deployment names (per act / base model)
    sft_deployment_name: str = ""
    dpo_deployment_name: str = ""
    rft_deployment_name: str = ""
    base_deployment_name: str = ""

    # Teacher / grader models for synthetic data and evals
    teacher_model: str = ""
    grader_model: str = ""

    # Embedding deployment for dedup / leakage similarity checks
    embedding_model: str = "text-embedding-3-large"

    # Tier + API version selections (overridable via env)
    deployment_tier: str = TIER_DEVELOPER
    data_plane_api_version: str = DATA_PLANE_API_VERSION
    control_plane_api_version: str = CONTROL_PLANE_API_VERSION

    # Raw environment snapshot for diagnostics (no secrets logged)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> DemoConfig:
        """Build a :class:`DemoConfig` from environment variables.

        Parameters
        ----------
        env:
            Optional mapping to read from instead of :data:`os.environ` (useful
            for tests). Defaults to the live process environment.

        Returns
        -------
        DemoConfig
            A populated configuration object. Missing variables become empty
            strings; this method never raises.
        """
        if env is None:
            # Auto-load an optional .env file (shell env still wins); a no-op
            # when the file is absent. Skipped when an explicit mapping is given
            # (e.g. tests), keeping those fully deterministic.
            load_env_file()
        src = os.environ if env is None else env

        def get(key: str, default: str = "") -> str:
            return src.get(key, default)

        project_endpoint = get("AZURE_AI_PROJECT_ENDPOINT")
        return cls(
            azure_ai_project_endpoint=project_endpoint,
            azure_ai_project_name=get("AZURE_AI_PROJECT_NAME")
            or _project_name_from_endpoint(project_endpoint),
            azure_openai_endpoint=get("AZURE_OPENAI_ENDPOINT"),
            azure_openai_api_key=get("AZURE_OPENAI_API_KEY"),
            azure_subscription_id=get("AZURE_SUBSCRIPTION_ID"),
            azure_resource_group=get("AZURE_RESOURCE_GROUP"),
            aoai_account_name=get("AOAI_ACCOUNT_NAME"),
            azure_region=get("AZURE_REGION"),
            sft_deployment_name=get("SFT_DEPLOYMENT_NAME"),
            dpo_deployment_name=get("DPO_DEPLOYMENT_NAME"),
            rft_deployment_name=get("RFT_DEPLOYMENT_NAME"),
            base_deployment_name=get("BASE_DEPLOYMENT_NAME"),
            teacher_model=get("TEACHER_MODEL"),
            grader_model=get("GRADER_MODEL"),
            embedding_model=get("EMBEDDING_MODEL", "text-embedding-3-large"),
            deployment_tier=get("AZURE_DEPLOYMENT_TIER", TIER_DEVELOPER),
            data_plane_api_version=get("AZURE_DATA_PLANE_API_VERSION", DATA_PLANE_API_VERSION),
            control_plane_api_version=get(
                "AZURE_CONTROL_PLANE_API_VERSION", CONTROL_PLANE_API_VERSION
            ),
            extra={
                "AZURE_LANGUAGE_ENDPOINT": get("AZURE_LANGUAGE_ENDPOINT"),
                "AZURE_LANGUAGE_KEY": get("AZURE_LANGUAGE_KEY"),
            },
        )

    def missing_required(self, keys: list[str] | None = None) -> list[str]:
        """Return the names of required fields that are unset (empty).

        Helps live callers produce actionable errors before issuing Azure
        requests. By default checks the core endpoint/identity fields.
        """
        required = keys or [
            "azure_openai_endpoint",
            "azure_subscription_id",
            "azure_resource_group",
            "aoai_account_name",
        ]
        return [name for name in required if not getattr(self, name, "")]
