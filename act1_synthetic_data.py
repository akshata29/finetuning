"""Act 1 — Synthetic Data Factory for the 1-Hour Azure Fine-Tuning Demo.

Generates diverse, label-balanced, **PII-free** synthetic sales-call transcripts
with intent / outcome / propensity labels and a rationale, then applies the
quality controls that make the corpus safe to fine-tune on:

* :func:`generate_records` — teacher-LLM generation (gpt-4.1) under the strict
  ``synthetic-sales-call`` JSON schema (Azure OpenAI Structured Outputs). The
  system prompt enforces placeholder-only content (``[REP]``, ``[PROSPECT]``,
  ``Acme Corp``).
* :func:`dedup` — embedding-based near-duplicate removal (cosine similarity
  against ``text-embedding-3-large``), with an injectable ``embed_fn`` so unit
  tests run with no network.
* :func:`pii_scan` — regex baseline for emails / phones / SSNs / account
  numbers, with optional Azure AI Language PII detection when available.
* :func:`split` — leakage-free train / val / eval partition with eval ``buy``
  prevalence held within the production-realistic ``[0.01, 0.02]`` band.

The module imports cleanly with **no** Azure SDKs installed. Optional SDKs are
resolved lazily via :func:`finetuning_demo.config.optional_import`; live Azure
operations fail only with an actionable error when invoked without the SDK or
required configuration (OWASP A05: no credentials or endpoints in source).
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
from collections.abc import Callable, Sequence
from typing import Any

from finetuning_demo.config import DemoConfig, optional_import
from finetuning_demo.schemas import SYNTHETIC_SCHEMA
from finetuning_demo.taxonomy import INTENT_LABELS, OUTCOME_LABELS, positive_rate

logger = logging.getLogger(__name__)

# Type alias for an embedding function: maps texts -> one vector per text.
EmbedFn = Callable[[Sequence[str]], list[list[float]]]

# ---------------------------------------------------------------------------
# Generation constants
# ---------------------------------------------------------------------------
#: Default teacher generation temperature — high for transcript diversity.
TEACHER_TEMPERATURE: float = 0.9

#: Embedding deployment used for dedup / leakage similarity checks.
EMBEDDING_MODEL: str = "text-embedding-3-large"

#: Schema name registered with the Structured Outputs call.
_SCHEMA_NAME: str = "sales_call"

#: Verbatim PII-free system prompt (research: "Act 1 — teacher generation,
#: schema-locked + PII-free (primary path)").
SYSTEM_PROMPT: str = (
    "Generate diverse, fully synthetic, PII-FREE B2B sales-call transcripts + "
    "labels. Use placeholders only ([REP],[PROSPECT],Acme Corp); never real "
    "names/companies/emails/account numbers."
)

# ---------------------------------------------------------------------------
# PII detection patterns (regex baseline)
# ---------------------------------------------------------------------------
_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(
        r"\b(?:\+?\d{1,2}[\s.\-])?\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}\b"
    ),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "account_number": re.compile(
        r"\b(?:acct|account|a/c)\s*#?\s*\d{6,}\b|\b\d{10,}\b",
        re.IGNORECASE,
    ),
}


# ---------------------------------------------------------------------------
# Step 2.1 — Teacher-LLM generation with strict Structured Outputs
# ---------------------------------------------------------------------------
def _build_user_prompt(seed: dict[str, Any]) -> str:
    """Render a single-call generation instruction from a taxonomy ``seed``."""
    return (
        "Generate ONE call. Seed: "
        f"intent={seed.get('intent')}, "
        f"outcome={seed.get('outcome')}, "
        f"industry={seed.get('industry')}, "
        f"persona={seed.get('persona')}, "
        f"deal_size={seed.get('deal_size')}, "
        f"objection={seed.get('objection')}. "
        "Emit propensity_score in [0,1] and a short rationale. "
        "Use placeholders only — no real names, companies, emails, phones, or "
        "account numbers."
    )


def _make_azure_openai_client(config: DemoConfig) -> Any:
    """Construct an ``AzureOpenAI`` client or raise an actionable error.

    Raises
    ------
    RuntimeError
        When the ``openai`` SDK is unavailable or required configuration
        (endpoint / key) is missing. The module still imports without the SDK;
        only invocation surfaces this error.
    """
    openai_module, available = optional_import("openai")
    if not available or openai_module is None:
        raise RuntimeError(
            "The 'openai' SDK is required for live synthetic-data generation but "
            "is not installed. Install it with: pip install openai"
        )

    missing = [
        name
        for name, value in (
            ("AZURE_OPENAI_ENDPOINT", config.azure_openai_endpoint),
            ("AZURE_OPENAI_API_KEY", config.azure_openai_api_key),
            ("TEACHER_MODEL", config.teacher_model),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing required configuration for synthetic-data generation: "
            f"{', '.join(missing)}. Set the corresponding environment "
            "variable(s) before calling generate_records()."
        )

    return openai_module.AzureOpenAI(
        azure_endpoint=config.azure_openai_endpoint,
        api_key=config.azure_openai_api_key,
        api_version=config.data_plane_api_version,
    )


def generate_records(
    config: DemoConfig,
    seeds: list[dict[str, Any]],
    *,
    client: Any | None = None,
) -> list[dict[str, Any]]:
    """Generate synthetic sales-call records from ``seeds`` via the teacher LLM.

    Each seed drives one Structured-Outputs call against ``config.teacher_model``
    using the strict :data:`~finetuning_demo.schemas.SYNTHETIC_SCHEMA`. The
    returned records carry the schema fields (``transcript``, ``intent``,
    ``outcome``, ``propensity_score``, ``rationale``) plus generation metadata
    (``mode``, ``propensity``, ``seed_id``) copied from the seed for downstream
    leakage-free splitting.

    Parameters
    ----------
    config:
        Environment-sourced configuration (endpoint, key, teacher model).
    seeds:
        Taxonomy seed dicts from
        :func:`finetuning_demo.taxonomy.iter_seeds`.
    client:
        Optional pre-built Azure OpenAI client (primarily for testing). When
        ``None`` a client is constructed from ``config``.

    Returns
    -------
    list[dict[str, Any]]
        One synthetic record per seed.

    Raises
    ------
    RuntimeError
        When invoked without the ``openai`` SDK or required configuration.
    """
    if client is None:
        client = _make_azure_openai_client(config)

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": _SCHEMA_NAME,
            "schema": SYNTHETIC_SCHEMA,
            "strict": True,
        },
    }

    records: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        completion = client.chat.completions.create(
            model=config.teacher_model,
            temperature=TEACHER_TEMPERATURE,
            response_format=response_format,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(seed)},
            ],
        )
        content = completion.choices[0].message.content
        record: dict[str, Any] = json.loads(content)
        # Attach generation metadata for leakage-free splitting (not part of the
        # strict schema, so kept separate from the model-emitted fields).
        record["mode"] = seed.get("mode")
        record["propensity"] = seed.get("propensity")
        record["seed_id"] = seed.get("seed_id", index)
        records.append(record)

    logger.info("Generated %d synthetic record(s) from %d seed(s)", len(records), len(seeds))
    return records


# ---------------------------------------------------------------------------
# Step 2.1b — Preference (DPO) pair derivation
# ---------------------------------------------------------------------------
def _other_label(labels: list[str], current: Any) -> str:
    """Return a deterministically-chosen label different from ``current``.

    Picks the next label in ``labels`` (wrapping around) so the corruption is
    reproducible; falls back to the first label when ``current`` is absent or
    unrecognized.
    """
    if current in labels:
        return labels[(labels.index(current) + 1) % len(labels)]
    return labels[0]


def build_preference_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Derive DPO preference pairs from labeled synthetic records.

    For each record the *preferred* response is the gold label
    (``intent`` / ``outcome`` / ``propensity_score``) and the *non_preferred*
    response is a deterministically corrupted label — a different intent and
    outcome plus a propensity miscalibrated to the opposite side of 0.5. This
    teaches DPO to favor the correct structured answer over a plausible but
    wrong one, with **no extra teacher calls**, so the preference set is fully
    reproducible from an existing generation run.

    Parameters
    ----------
    records:
        Labeled synthetic records from :func:`generate_records` (each carrying
        ``transcript``, ``intent``, ``outcome``, ``propensity_score``).

    Returns
    -------
    list[dict[str, Any]]
        Rows shaped ``{"transcript", "preferred", "non_preferred"}`` ready for
        :func:`finetuning_demo.schemas.write_dpo_jsonl`.
    """
    preference_rows: list[dict[str, Any]] = []
    for record in records:
        gold = {
            "intent": record.get("intent"),
            "outcome": record.get("outcome"),
            "propensity_score": record.get("propensity_score"),
        }
        score = record.get("propensity_score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            wrong_score: Any = round(1.0 - float(score), 3)
        else:
            wrong_score = 0.5
        corrupt = {
            "intent": _other_label(INTENT_LABELS, record.get("intent")),
            "outcome": _other_label(OUTCOME_LABELS, record.get("outcome")),
            "propensity_score": wrong_score,
        }
        preference_rows.append(
            {
                "transcript": record.get("transcript", ""),
                "preferred": json.dumps(gold, ensure_ascii=False),
                "non_preferred": json.dumps(corrupt, ensure_ascii=False),
            }
        )
    logger.info("Built %d DPO preference pair(s)", len(preference_rows))
    return preference_rows


def build_rft_records(
    records: list[dict[str, Any]], grader_type: str = "string_match"
) -> list[dict[str, Any]]:
    """Derive Reinforcement Fine-Tuning (RFT) records for Azure's job format.

    Azure RFT grades model outputs **server-side** during training, so the data
    rows only need the prompt plus the ground-truth reference fields the grader
    reads as ``{{ item.<field> }}``. Each output row therefore carries the
    ``transcript`` and the gold ``intent`` / ``outcome`` / ``propensity_score``;
    :func:`finetuning_demo.schemas.write_rft_jsonl` turns these into the
    ``messages`` + reference-field JSONL Azure expects. No teacher calls or
    offline grading happen here.

    Parameters
    ----------
    records:
        Labeled synthetic records from :func:`generate_records` (each carrying
        ``transcript``, ``intent``, ``outcome``, ``propensity_score``).
    grader_type:
        Retained for call-site compatibility; the grader runs server-side and
        does not change the data rows. Both 'string_match' and 'model' produce
        the same reference-field rows.

    Returns
    -------
    list[dict[str, Any]]
        Rows shaped ``{"transcript", "intent", "outcome", "propensity_score"}``
        ready for :func:`finetuning_demo.schemas.write_rft_jsonl`.
    """
    rft_rows: list[dict[str, Any]] = [
        {
            "transcript": record.get("transcript", ""),
            "intent": record.get("intent"),
            "outcome": record.get("outcome"),
            "propensity_score": record.get("propensity_score"),
        }
        for record in records
    ]
    logger.info(
        "Built %d RFT record(s) for grader type '%s' (graded server-side)",
        len(rft_rows),
        grader_type,
    )
    return rft_rows


# ---------------------------------------------------------------------------
# Step 2.2a — Embedding-based dedup
# ---------------------------------------------------------------------------
def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the cosine similarity of two equal-length vectors.

    Returns ``0.0`` when either vector has zero magnitude.
    """
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _default_embed_fn(config: DemoConfig | None = None) -> EmbedFn:
    """Build an Azure ``text-embedding-3-large`` embedding function.

    The returned callable performs a live Azure OpenAI call and is used only
    when no ``embed_fn`` is injected into :func:`dedup`. Tests inject a stub to
    avoid any network access.
    """
    resolved = config or DemoConfig.from_env()
    client = _make_azure_openai_client(resolved)
    embedding_model = resolved.embedding_model or EMBEDDING_MODEL

    def _embed(texts: Sequence[str]) -> list[list[float]]:
        response = client.embeddings.create(model=embedding_model, input=list(texts))
        return [item.embedding for item in response.data]

    return _embed


def dedup(
    records: list[dict[str, Any]],
    threshold: float = 0.92,
    *,
    embed_fn: EmbedFn | None = None,
) -> list[dict[str, Any]]:
    """Remove near-duplicate records by transcript embedding cosine similarity.

    A greedy pass keeps each record whose maximum cosine similarity to every
    previously kept record is **below** ``threshold``; records at or above the
    threshold are dropped as near-duplicates.

    Parameters
    ----------
    records:
        Records carrying a ``transcript`` field.
    threshold:
        Cosine-similarity cutoff in ``[0, 1]`` (default ``0.92``). Higher keeps
        more (only drops very close duplicates); lower is more aggressive.
    embed_fn:
        Injectable embedding function mapping a sequence of texts to one vector
        per text. Defaults to a live Azure ``text-embedding-3-large`` call;
        inject a stub for offline tests.

    Returns
    -------
    list[dict[str, Any]]
        The deduplicated records, preserving input order.

    Raises
    ------
    ValueError
        If ``threshold`` is outside ``[0, 1]``.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    if not records:
        return []

    resolved_embed = embed_fn or _default_embed_fn()
    vectors = resolved_embed([record.get("transcript", "") for record in records])
    if len(vectors) != len(records):
        raise ValueError(
            f"embed_fn returned {len(vectors)} vectors for {len(records)} records"
        )

    kept: list[dict[str, Any]] = []
    kept_vectors: list[Sequence[float]] = []
    dropped = 0
    for record, vector in zip(records, vectors, strict=True):
        is_duplicate = any(
            _cosine_similarity(vector, kept_vector) >= threshold
            for kept_vector in kept_vectors
        )
        if is_duplicate:
            dropped += 1
            continue
        kept.append(record)
        kept_vectors.append(vector)

    logger.info("Dedup kept %d of %d record(s) (%d near-duplicate(s) removed)", len(kept), len(records), dropped)
    return kept


# ---------------------------------------------------------------------------
# Step 2.2b — PII post-scan
# ---------------------------------------------------------------------------
def _regex_pii_flags(text: str) -> list[str]:
    """Return the names of PII pattern categories that match ``text``."""
    return [name for name, pattern in _PII_PATTERNS.items() if pattern.search(text)]


def _azure_language_pii_flags(text: str) -> list[str]:
    """Return PII category tags from Azure AI Language, or ``[]`` when absent.

    Optional path: requires ``azure-ai-textanalytics`` and configuration. Any
    failure (missing SDK, missing config, service error) degrades silently to an
    empty list so the regex baseline remains the source of truth offline.
    """
    module, available = optional_import("azure.ai.textanalytics")
    if not available or module is None:
        return []

    config = DemoConfig.from_env()
    endpoint = config.extra.get("AZURE_LANGUAGE_ENDPOINT", "")
    key = config.extra.get("AZURE_LANGUAGE_KEY", "")
    if not endpoint or not key:
        return []

    credentials_module, cred_available = optional_import("azure.core.credentials")
    if not cred_available or credentials_module is None:
        return []

    try:
        client = module.TextAnalyticsClient(
            endpoint=endpoint,
            credential=credentials_module.AzureKeyCredential(key),
        )
        result = client.recognize_pii_entities([text])
        flags: list[str] = []
        for document in result:
            if getattr(document, "is_error", False):
                continue
            flags.extend(entity.category for entity in document.entities)
        return flags
    except Exception:  # pragma: no cover - defensive: optional live path
        logger.warning("Azure AI Language PII scan failed; using regex baseline only.")
        return []


def pii_scan(
    records: list[dict[str, Any]],
    *,
    use_azure_language: bool = False,
) -> list[dict[str, Any]]:
    """Flag records whose transcript contains detectable PII.

    Applies the regex baseline (emails / phones / SSNs / account numbers) to
    every record's ``transcript``, optionally augmenting with Azure AI Language
    PII detection when ``use_azure_language`` is set and the SDK/config are
    available.

    Parameters
    ----------
    records:
        Records carrying a ``transcript`` field.
    use_azure_language:
        When ``True``, also query Azure AI Language. Defaults to ``False`` so the
        scan is fully offline and deterministic.

    Returns
    -------
    list[dict[str, Any]]
        Shallow copies of the flagged records, each augmented with a
        ``_pii_flags`` list naming the matched categories. The regenerate-on-fail
        caller drops/regenerates any record returned here.
    """
    flagged: list[dict[str, Any]] = []
    for record in records:
        transcript = record.get("transcript", "")
        flags = _regex_pii_flags(transcript)
        if use_azure_language:
            flags.extend(_azure_language_pii_flags(transcript))
        if flags:
            flagged_record = dict(record)
            flagged_record["_pii_flags"] = sorted(set(flags))
            flagged.append(flagged_record)

    if flagged:
        logger.warning("PII scan flagged %d of %d record(s)", len(flagged), len(records))
    return flagged


# ---------------------------------------------------------------------------
# Step 2.2c — Leakage-free train / val / eval split
# ---------------------------------------------------------------------------
EVAL_PREVALENCE_MIN: float = 0.01
EVAL_PREVALENCE_MAX: float = 0.02

_SIGNATURE_DIMS: tuple[str, ...] = (
    "intent",
    "outcome",
    "industry",
    "persona",
    "deal_size",
    "objection",
)


def _is_buy(record: dict[str, Any]) -> bool:
    """Return whether ``record`` carries the positive (``buy``) label."""
    propensity = record.get("propensity")
    if propensity is not None:
        return propensity == "buy"
    score = record.get("propensity_score")
    return isinstance(score, (int, float)) and score >= 0.5


def _leakage_key(record: dict[str, Any]) -> str:
    """Return a stable leakage key (normalized transcript) for ``record``."""
    return " ".join(record.get("transcript", "").split()).lower()


def _carve_eval(
    pool: list[dict[str, Any]],
    eval_prevalence: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Construct an eval set at ``eval_prevalence`` from an unlabeled-mode pool.

    Used only when no records carry ``mode == "eval"``. Selects a small integer
    number of positives plus enough negatives to hit the target prevalence,
    returning ``(eval, remaining_pool)``.
    """
    positives = [r for r in pool if _is_buy(r)]
    negatives = [r for r in pool if not _is_buy(r)]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    # Choose the positive count first, then size the eval set so that
    # positives / eval_size lands inside the prevalence band.
    target_positives = max(1, round(len(pool) * eval_prevalence))
    target_positives = min(target_positives, len(positives))
    if target_positives == 0:
        return [], pool

    eval_size = round(target_positives / eval_prevalence)
    needed_negatives = min(max(eval_size - target_positives, 0), len(negatives))

    eval_records = positives[:target_positives] + negatives[:needed_negatives]
    rng.shuffle(eval_records)

    chosen = {id(r) for r in eval_records}
    remaining = [r for r in pool if id(r) not in chosen]
    return eval_records, remaining


def split(
    records: list[dict[str, Any]],
    *,
    val_fraction: float = 0.1,
    eval_prevalence: float = 0.015,
    seed: int = 1337,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition ``records`` into leakage-free ``(train, val, eval)`` splits.

    Records tagged ``mode == "eval"`` form the eval set (production-realistic
    rare ``buy`` prevalence); the remainder forms the train pool from which a
    deterministic ``val_fraction`` is carved. When no eval-mode records exist
    (e.g. a flat dict corpus), an eval set is synthesized at ``eval_prevalence``.

    Leakage control removes any eval record whose normalized transcript also
    appears in the train pool, guaranteeing the splits are disjoint. The eval
    ``buy`` prevalence is asserted to fall within ``[0.01, 0.02]``.

    Parameters
    ----------
    records:
        Synthetic records, ideally tagged with ``mode`` and ``propensity``.
    val_fraction:
        Share of the train pool held out for validation (default ``0.1``).
    eval_prevalence:
        Target eval ``buy`` prevalence when synthesizing an eval set
        (default ``0.015``).
    seed:
        RNG seed for deterministic shuffling.

    Returns
    -------
    tuple[list, list, list]
        ``(train, val, eval)``.

    Raises
    ------
    ValueError
        If ``val_fraction`` is outside ``[0, 1)``.
    AssertionError
        If the eval ``buy`` prevalence falls outside ``[0.01, 0.02]`` or the
        splits are not leakage-free.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    rng = random.Random(seed)

    eval_records = [r for r in records if r.get("mode") == "eval"]
    train_pool = [r for r in records if r.get("mode") != "eval"]

    if not eval_records:
        eval_records, train_pool = _carve_eval(train_pool, eval_prevalence, rng)

    # Cross-split leakage removal: drop eval rows whose transcript also appears
    # in the train pool (research: "remove any eval row too close to a train
    # row"). Train rows are preserved.
    train_keys = {_leakage_key(r) for r in train_pool}
    eval_records = [r for r in eval_records if _leakage_key(r) not in train_keys]

    # Disjointness guarantee.
    eval_keys = {_leakage_key(r) for r in eval_records}
    assert train_keys.isdisjoint(eval_keys), "train/eval splits share transcripts (leakage)"

    # Carve validation from the train pool.
    shuffled = list(train_pool)
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * val_fraction)
    val_records = shuffled[:n_val]
    train_records = shuffled[n_val:]

    prevalence = positive_rate(eval_records)
    assert EVAL_PREVALENCE_MIN <= prevalence <= EVAL_PREVALENCE_MAX, (
        f"eval buy prevalence {prevalence:.4f} outside "
        f"[{EVAL_PREVALENCE_MIN}, {EVAL_PREVALENCE_MAX}]"
    )

    logger.info(
        "Split: train=%d, val=%d, eval=%d (eval prevalence=%.4f)",
        len(train_records),
        len(val_records),
        len(eval_records),
        prevalence,
    )
    return train_records, val_records, eval_records
