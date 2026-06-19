"""Grader types for Reinforcement Fine-Tuning (RFT).

RFT uses a grader to score model outputs, enabling iterative improvement via
reinforcement learning. Two grader types are supported:

1. **String/label-match grader** — Deterministic scoring based on exact
   substring matching. Used for intent/outcome classification where the
   canonical labels are known and high-precision ground truth is available.
2. **Model grader** — LLM-based scoring using a teacher/judge model.
   Useful for complex tasks where human-written rules are insufficient.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grader types
# ---------------------------------------------------------------------------
GRADER_STRING_MATCH: str = "string_match"
GRADER_MODEL: str = "model"
SUPPORTED_GRADERS: tuple[str, ...] = (GRADER_STRING_MATCH, GRADER_MODEL)


# ---------------------------------------------------------------------------
# Grader interface
# ---------------------------------------------------------------------------
class Grader:
    """Abstract base for output graders used in RFT."""

    def grade(self, output: dict[str, Any], *, ground_truth: dict[str, Any]) -> dict[str, Any]:
        """Score a model output against ground truth.

        Returns
        -------
        dict
            A grade dict with keys:
            - ``"result"``: float in [0, 1] or int (score/points)
            - ``"reason"``: str (optional, explanation for the grade)
            - ``"grader_type"``: str (the grader type identifier)
        """
        raise NotImplementedError


class StringMatchGrader(Grader):
    """Deterministic grader: score 1 if output contains ground-truth labels, 0 otherwise.

    Used for intent and outcome classification. The output JSON is parsed and
    checked against canonical labels. Multi-field grading is supported
    (all fields must match for full score).
    """

    def __init__(self, target_fields: list[str] | None = None):
        """Initialize with target fields to grade (e.g., ['intent', 'outcome']).

        If None, defaults to ['intent'].
        """
        self.target_fields = target_fields or ["intent"]

    def grade(self, output: dict[str, Any], *, ground_truth: dict[str, Any]) -> dict[str, Any]:
        """Score output by checking if target fields match ground truth."""
        # Try to parse the assistant response as JSON.
        assistant_content = None
        if isinstance(output, dict):
            # output is already a dict (parsed once).
            assistant_content = output
        elif isinstance(output, str):
            # Try to parse as JSON.
            try:
                start = output.find("{")
                end = output.rfind("}")
                if start != -1 and end != -1 and end > start:
                    assistant_content = json.loads(output[start : end + 1])
            except (ValueError, TypeError):
                pass

        if not assistant_content:
            logger.debug("StringMatchGrader: could not parse output as JSON, score=0")
            return {
                "result": 0,
                "reason": "output is not valid JSON",
                "grader_type": GRADER_STRING_MATCH,
            }

        # Check all target fields.
        matched_count = 0
        for field in self.target_fields:
            predicted = str(assistant_content.get(field, "")).lower().strip()
            expected = str(ground_truth.get(field, "")).lower().strip()
            if predicted == expected:
                matched_count += 1

        score = matched_count / len(self.target_fields) if self.target_fields else 0
        return {
            "result": score,
            "reason": f"{matched_count}/{len(self.target_fields)} field(s) matched",
            "grader_type": GRADER_STRING_MATCH,
        }


class ModelGrader(Grader):
    """LLM-based grader using a teacher/judge model to score outputs.

    Calls an Azure OpenAI grader model to evaluate the output quality.
    Requires the model to be deployed and accessible via the OpenAI SDK.
    """

    def __init__(self, client: Any, deployment: str, system_prompt: str | None = None):
        """Initialize with an OpenAI client and deployment name.

        Parameters
        ----------
        client : Any
            An AzureOpenAI client instance.
        deployment : str
            Deployment name for the grader model.
        system_prompt : str, optional
            System prompt for the grader. If None, uses a default instruction.
        """
        self.client = client
        self.deployment = deployment
        self.system_prompt = system_prompt or (
            "You are a sales-call intent/outcome evaluator. Given a transcript "
            "and a predicted classification, rate the prediction accuracy on a "
            "scale of 0-10. Return ONLY the numeric score."
        )

    def grade(self, output: dict[str, Any], *, ground_truth: dict[str, Any]) -> dict[str, Any]:
        """Call the grader model to score the output."""
        # Prepare the grader prompt.
        ground_truth_str = json.dumps(ground_truth, ensure_ascii=False)
        output_str = (
            json.dumps(output, ensure_ascii=False)
            if isinstance(output, dict)
            else str(output)
        )

        user_message = (
            f"Ground truth:\n{ground_truth_str}\n\n"
            f"Predicted output:\n{output_str}\n\n"
            f"Rate the prediction accuracy (0-10):"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
                max_tokens=10,
            )
            score_str = response.choices[0].message.content.strip()
            # Try to parse as a number.
            try:
                score = float(score_str) / 10.0  # Convert from 0-10 to 0-1.
                score = max(0.0, min(1.0, score))  # Clamp to [0, 1].
            except ValueError:
                logger.debug(
                    "ModelGrader: could not parse score %r, defaulting to 0.5",
                    score_str,
                )
                score = 0.5
            return {
                "result": score,
                "reason": f"grader response: {score_str}",
                "grader_type": GRADER_MODEL,
            }
        except Exception as e:
            logger.exception("ModelGrader: grading call failed, score=0.5")
            return {
                "result": 0.5,
                "reason": f"grader error: {e}",
                "grader_type": GRADER_MODEL,
            }


def build_grader(grader_type: str, **kwargs: Any) -> Grader:
    """Factory function to build a grader by type.

    Parameters
    ----------
    grader_type : str
        Either 'string_match' or 'model'.
    **kwargs
        Grader-specific kwargs (e.g., client, deployment for ModelGrader).

    Returns
    -------
    Grader
        An instance of the requested grader type.

    Raises
    ------
    ValueError
        If grader_type is not recognized.
    """
    if grader_type == GRADER_STRING_MATCH:
        return StringMatchGrader(target_fields=kwargs.get("target_fields"))
    if grader_type == GRADER_MODEL:
        return ModelGrader(
            client=kwargs["client"],
            deployment=kwargs["deployment"],
            system_prompt=kwargs.get("system_prompt"),
        )
    raise ValueError(
        f"Unknown grader type {grader_type!r}; expected one of {sorted(SUPPORTED_GRADERS)}."
    )


# ---------------------------------------------------------------------------
# Azure RFT grader-config builder
# ---------------------------------------------------------------------------
#: Default reference fields graded server-side by Azure RFT.
DEFAULT_GRADER_FIELDS: tuple[str, ...] = ("intent", "outcome")


def build_azure_grader_config(
    grader_type: str,
    *,
    target_fields: tuple[str, ...] = DEFAULT_GRADER_FIELDS,
    grader_model: str | None = None,
) -> dict[str, Any]:
    """Build the server-side grader config Azure RFT expects on job-create.

    The local :class:`StringMatchGrader` / :class:`ModelGrader` classes score
    rows *offline*; Azure RFT instead runs the grader itself during training and
    requires its own grader specification. This maps the demo's internal grader
    names to the Azure grader schema:

    * ``string_match`` -> a ``string_check`` (``operation: eq``) per target
      field, combined with a ``multi`` grader that averages them. The model
      output is referenced as ``{{sample.output_json.<field>}}`` (requires the
      job's JSON-schema ``response_format``) and the ground truth as
      ``{{item.<field>}}`` (the reference fields in the training rows).
    * ``model`` -> a ``score_model`` grader that asks ``grader_model`` to rate
      the output in ``[0, 1]``.

    See https://learn.microsoft.com/azure/ai-foundry/openai/how-to/reinforcement-fine-tuning.

    Raises
    ------
    ValueError
        When ``grader_type`` is unknown, ``target_fields`` is empty, or a model
        grader is requested without ``grader_model``.
    """
    if grader_type == GRADER_STRING_MATCH:
        if not target_fields:
            raise ValueError("string_match grader requires at least one target field")
        subgraders = {
            field: {
                "type": "string_check",
                "name": f"{field}_match",
                "operation": "eq",
                "input": f"{{{{sample.output_json.{field}}}}}",
                "reference": f"{{{{item.{field}}}}}",
            }
            for field in target_fields
        }
        if len(subgraders) == 1:
            return next(iter(subgraders.values()))
        expression = "(" + " + ".join(target_fields) + f") / {len(target_fields)}"
        return {
            "type": "multi",
            "name": "label_match",
            "graders": subgraders,
            "calculate_output": expression,
        }
    if grader_type == GRADER_MODEL:
        if not grader_model:
            raise ValueError("model grader requires a grader_model deployment/name")
        return {
            "type": "score_model",
            "name": "label_quality",
            "model": grader_model,
            "input": [
                {
                    "role": "system",
                    "content": (
                        "You score a sales-call classifier. Compare the model "
                        "output to the ground-truth labels and return a single "
                        "number in [0,1]: 1.0 when intent and outcome both match, "
                        "0.0 when neither matches, 0.5 when exactly one matches."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Ground truth: intent={{item.intent}}, "
                        "outcome={{item.outcome}}.\n"
                        "Model output: {{sample.output_text}}\n"
                        "Score (0-1):"
                    ),
                },
            ],
            "range": [0, 1],
            "pass_threshold": 0.5,
        }
    raise ValueError(
        f"Unknown grader type {grader_type!r}; expected one of {sorted(SUPPORTED_GRADERS)}."
    )
