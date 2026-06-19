"""Engineered few-shot system prompt for the optimized-prompt eval arm.

The 3-way Act 3 comparison evaluates three candidates on identical rows and
graders: ``base`` (raw deployment), ``finetuned`` (SFT/DPO model), and
``optimized-prompt`` (the SAME base deployment driven by a deliberately
engineered system prompt). This module owns the third arm's prompt.

The prompt encodes an explicit label contract plus a small set of in-context
examples (one per class, including a hard negative) so the base model classifies
both the multiclass ``intent`` and the binary ``buy``/``not_buy`` propensity
without fine-tuning. It is pure data — no Azure SDKs are imported here, so this
module loads cleanly in any environment.
"""

from __future__ import annotations

import logging

from finetuning.taxonomy import INTENT_LABELS, PROPENSITY_LABELS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engineered few-shot system prompt (optimized-prompt arm)
# ---------------------------------------------------------------------------
#: Comma-joined label contracts rendered into the prompt so the engineered
#: prompt always mirrors the canonical taxonomy.
_INTENT_CONTRACT: str = ", ".join(INTENT_LABELS)
_PROPENSITY_CONTRACT: str = ", ".join(PROPENSITY_LABELS)

OPTIMIZED_SYSTEM_PROMPT: str = f"""\
You are a precise B2B sales-call classifier. Read the call transcript and return \
ONLY a compact JSON object with three fields and nothing else.

Label contract (use these exact tokens):
- intent: one of [{_INTENT_CONTRACT}]
- propensity: one of [{_PROPENSITY_CONTRACT}] — "buy" ONLY when the caller shows \
concrete purchase intent (explicit pricing/contract negotiation, committed \
timeline, allocated budget, or a request to proceed/sign); otherwise "not_buy".
- propensity_score: a float in [0,1] estimating purchase likelihood (calibrated, \
not just 0/1).

Decision rules:
- Interest, demos, or "maybe next quarter" without budget or commitment => not_buy.
- Pricing/feature questions alone are NOT buy unless paired with a commitment signal.
- When the caller asks to sign, send an invoice, or confirm a start date => buy.

Examples:
Transcript: "Caller is ready to sign, asked us to send the invoice and a start date."
Output: {{"intent": "pricing_inquiry", "propensity": "buy", "propensity_score": 0.91}}
Transcript: "Caller liked the demo but has no budget approved and wants to revisit next quarter."
Output: {{"intent": "scheduling_followup", "propensity": "not_buy", "propensity_score": 0.18}}
Transcript: "Caller compared us to a competitor on price and raised an integration concern."
Output: {{"intent": "competitor_comparison", "propensity": "not_buy", "propensity_score": 0.34}}

Return only the JSON object."""


def build_optimized_messages(transcript: str) -> list[dict[str, str]]:
    """Return chat messages for the optimized-prompt arm for one transcript.

    Pairs :data:`OPTIMIZED_SYSTEM_PROMPT` (system turn) with the call
    ``transcript`` (user turn), ready for a batch-inference call against the
    base deployment that produces the optimized-prompt candidate's predictions.

    Parameters
    ----------
    transcript:
        The raw sales-call transcript to classify.

    Returns
    -------
    list[dict[str, str]]
        ``[{"role": "system", ...}, {"role": "user", ...}]``.
    """
    return [
        {"role": "system", "content": OPTIMIZED_SYSTEM_PROMPT},
        {"role": "user", "content": transcript},
    ]
