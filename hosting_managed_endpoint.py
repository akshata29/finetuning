"""Act 4 — Managed online endpoint hosting for the base + LoRA adapter.

Builds the Azure ML managed online endpoint and deployment that serve the
fine-tuned classifier. Two deployment shapes are supported:

* :func:`deploy_endpoint` — a custom-code deployment that pairs the registered
  LoRA adapter model with ``onlinescoring/score.py`` and the custom conda env.
* :func:`deploy_mlflow_nocode` — the no-code path for the finetune job's MLflow
  output model (the MLflow flavor supplies the scoring wrapper, so no
  ``score.py`` / custom env is needed).

Critical settings baked in (validator finding DR-05 and Gap 1.3/1.7):

* ``instance_type="Standard_NC24ads_A100_v4"`` — 1x A100 80GB GPU SKU.
* ``OnlineRequestSettings(request_timeout_ms=90000)`` — LLM generation easily
  exceeds the 5s default; raised well above it (max 180000).
* Liveness/readiness probes with ``initial_delay=600`` — base-model download +
  load is multi-minute; a short probe delay would kill the container mid-init.
* A note on the 20% AzureML quota headroom required for deployment.

This module is import-safe with **no Azure SDKs installed**: ``azure-ai-ml`` is
resolved lazily via :func:`finetuning.config.optional_import` and every
function raises an actionable :class:`ImportError` at call time when absent. No
secrets or endpoints are hardcoded.
"""

from __future__ import annotations

import logging
from typing import Any

from finetuning.config import optional_import

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Critical deployment constants (Gap 1.3 / DR-05).
# ---------------------------------------------------------------------------
GPU_INSTANCE_TYPE: str = "Standard_NC24ads_A100_v4"  # 1x A100 80GB
REQUEST_TIMEOUT_MS: int = 90000  # raise from the 5s default; max is 180000
PROBE_INITIAL_DELAY: int = 600  # model load is slow; avoid early probe kills
PROBE_PERIOD: int = 30
PROBE_TIMEOUT: int = 10
PROBE_FAILURE_THRESHOLD: int = 30

# NOTE (20% quota headroom): "Azure Machine Learning reserves 20% of your
# compute resources for performing upgrades." A single 24-vCPU
# Standard_NC24ads_A100_v4 therefore needs ~1.2x the SKU's vCPU-family quota.

DEFAULT_SCORING_DIR: str = "./finetuning/onlinescoring"
DEFAULT_SCORING_SCRIPT: str = "score.py"


def _require_ml_entities() -> Any:
    """Import the ``azure.ai.ml`` SDK or raise an actionable error.

    Keeps the module import-safe by deferring the SDK requirement to call time.
    """
    module, available = optional_import("azure.ai.ml")
    if not available:
        raise ImportError(
            "azure-ai-ml is required to host the managed online endpoint. "
            "Install with: pip install azure-ai-ml azure-identity"
        )
    return module


def register_adapter_model(
    ml_client: Any,
    *,
    name: str,
    adapter_path: str,
    description: str = "LoRA adapter for sales-call classification",
) -> Any:
    """Register the local LoRA adapter folder as a custom model asset.

    Parameters
    ----------
    ml_client:
        A live ``azure.ai.ml.MLClient``.
    name:
        Registered model asset name.
    adapter_path:
        Local folder containing ``adapter_config.json`` + adapter weights.
    description:
        Human-readable model description.

    Returns
    -------
    Any
        The created/updated model asset.

    Raises
    ------
    ImportError
        If ``azure-ai-ml`` is not installed.
    """
    _require_ml_entities()
    from azure.ai.ml.constants import AssetTypes  # noqa: PLC0415
    from azure.ai.ml.entities import Model  # noqa: PLC0415

    model = Model(
        path=adapter_path,
        type=AssetTypes.CUSTOM_MODEL,
        name=name,
        description=description,
    )
    logger.info("Registering LoRA adapter model %s from %s", name, adapter_path)
    return ml_client.models.create_or_update(model)


def deploy_endpoint(
    ml_client: Any,
    *,
    endpoint_name: str,
    model: Any,
    environment: Any,
    deployment_name: str = "blue",
    scoring_dir: str = DEFAULT_SCORING_DIR,
    scoring_script: str = DEFAULT_SCORING_SCRIPT,
    instance_type: str = GPU_INSTANCE_TYPE,
    instance_count: int = 1,
    wait: bool = True,
) -> Any:
    """Create the endpoint + custom-code GPU deployment and route all traffic.

    Applies the critical settings verbatim: GPU SKU, raised
    ``request_timeout_ms``, and long-``initial_delay`` liveness/readiness probes
    (DR-05). Remember the 20% AzureML quota headroom for the chosen GPU family.

    Parameters
    ----------
    ml_client:
        A live ``azure.ai.ml.MLClient``.
    endpoint_name:
        Name of the managed online endpoint.
    model / environment:
        Registered model asset (or id) and inference Environment (or id).
    deployment_name:
        Deployment name (default ``"blue"``).
    scoring_dir / scoring_script:
        Code folder and scoring script for the custom deployment.
    instance_type / instance_count:
        GPU SKU and instance count.
    wait:
        When true, block on the long-running create operations.

    Returns
    -------
    Any
        The created/updated deployment.

    Raises
    ------
    ImportError
        If ``azure-ai-ml`` is not installed.
    """
    _require_ml_entities()
    from azure.ai.ml.entities import (  # noqa: PLC0415
        CodeConfiguration,
        ManagedOnlineDeployment,
        ManagedOnlineEndpoint,
        OnlineRequestSettings,
        ProbeSettings,
    )

    endpoint = ManagedOnlineEndpoint(name=endpoint_name, auth_mode="key")
    create_ep = ml_client.online_endpoints.begin_create_or_update(endpoint)
    if wait:
        create_ep.result()

    probe = ProbeSettings(
        initial_delay=PROBE_INITIAL_DELAY,
        period=PROBE_PERIOD,
        timeout=PROBE_TIMEOUT,
        success_threshold=1,
        failure_threshold=PROBE_FAILURE_THRESHOLD,
    )
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model,
        environment=environment,
        code_configuration=CodeConfiguration(
            code=scoring_dir, scoring_script=scoring_script
        ),
        instance_type=instance_type,
        instance_count=instance_count,
        request_settings=OnlineRequestSettings(
            request_timeout_ms=REQUEST_TIMEOUT_MS,
            max_concurrent_requests_per_instance=1,
            max_queue_wait_ms=60000,
        ),
        liveness_probe=probe,
        readiness_probe=probe,
    )
    logger.info(
        "Deploying %s/%s on %s (timeout=%dms, probe_delay=%ds)",
        endpoint_name,
        deployment_name,
        instance_type,
        REQUEST_TIMEOUT_MS,
        PROBE_INITIAL_DELAY,
    )
    create_dep = ml_client.online_deployments.begin_create_or_update(deployment)
    result = create_dep.result() if wait else create_dep

    # Route 100% of traffic to this deployment.
    endpoint.traffic = {deployment_name: 100}
    route = ml_client.online_endpoints.begin_create_or_update(endpoint)
    if wait:
        route.result()
    return result


def deploy_mlflow_nocode(
    ml_client: Any,
    *,
    endpoint_name: str,
    model: Any,
    deployment_name: str = "blue",
    instance_type: str = GPU_INSTANCE_TYPE,
    instance_count: int = 1,
    wait: bool = True,
) -> Any:
    """Deploy the finetune job's MLflow model with NO ``score.py`` / custom env.

    The MLflow flavor supplies the scoring wrapper, so ``code_configuration``
    and a custom conda env are intentionally omitted. The raised request timeout
    is still applied because generation is slow.

    Parameters
    ----------
    ml_client:
        A live ``azure.ai.ml.MLClient``.
    endpoint_name:
        Managed online endpoint name.
    model:
        MLflow-flavored model asset (or ``azureml:...@latest`` id) from the
        finetune job output.
    deployment_name / instance_type / instance_count / wait:
        See :func:`deploy_endpoint`.

    Returns
    -------
    Any
        The created/updated MLflow no-code deployment.

    Raises
    ------
    ImportError
        If ``azure-ai-ml`` is not installed.
    """
    _require_ml_entities()
    from azure.ai.ml.entities import (  # noqa: PLC0415
        ManagedOnlineDeployment,
        ManagedOnlineEndpoint,
        OnlineRequestSettings,
    )

    endpoint = ManagedOnlineEndpoint(name=endpoint_name, auth_mode="key")
    create_ep = ml_client.online_endpoints.begin_create_or_update(endpoint)
    if wait:
        create_ep.result()

    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model,  # MLflow model — no code_configuration / custom env
        instance_type=instance_type,
        instance_count=instance_count,
        request_settings=OnlineRequestSettings(
            request_timeout_ms=REQUEST_TIMEOUT_MS
        ),
    )
    logger.info(
        "Deploying MLflow no-code model to %s/%s on %s",
        endpoint_name,
        deployment_name,
        instance_type,
    )
    create_dep = ml_client.online_deployments.begin_create_or_update(deployment)
    result = create_dep.result() if wait else create_dep

    endpoint.traffic = {deployment_name: 100}
    route = ml_client.online_endpoints.begin_create_or_update(endpoint)
    if wait:
        route.result()
    return result


def invoke(
    ml_client: Any,
    *,
    endpoint_name: str,
    request_file: str,
    deployment_name: str | None = None,
) -> Any:
    """Invoke the endpoint with a request payload file.

    Parameters
    ----------
    ml_client:
        A live ``azure.ai.ml.MLClient``.
    endpoint_name:
        Endpoint to invoke.
    request_file:
        Path to a JSON request body (e.g. ``{"records": [...]}``).
    deployment_name:
        Optional specific deployment to target.

    Returns
    -------
    Any
        The raw invocation response.

    Raises
    ------
    ImportError
        If ``azure-ai-ml`` is not installed.
    """
    _require_ml_entities()
    logger.info("Invoking endpoint %s with %s", endpoint_name, request_file)
    return ml_client.online_endpoints.invoke(
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
        request_file=request_file,
    )


def delete_endpoint(
    ml_client: Any, *, endpoint_name: str, wait: bool = True
) -> Any:
    """Delete the managed online endpoint (stops GPU billing).

    Parameters
    ----------
    ml_client:
        A live ``azure.ai.ml.MLClient``.
    endpoint_name:
        Endpoint to delete.
    wait:
        When true, block until deletion completes.

    Returns
    -------
    Any
        The delete poller (or its result when ``wait`` is true).

    Raises
    ------
    ImportError
        If ``azure-ai-ml`` is not installed.
    """
    _require_ml_entities()
    logger.info("Deleting endpoint %s", endpoint_name)
    poller = ml_client.online_endpoints.begin_delete(name=endpoint_name)
    return poller.result() if wait else poller
