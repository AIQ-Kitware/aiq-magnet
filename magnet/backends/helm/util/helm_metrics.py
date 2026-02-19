"""magnet.backends.helm.helm_metrics

Shared metric registries + categorization helpers.

This is intentionally small and stable: both single-run analysis and run diffs
use the same classification rules.

Design preference
-----------------
Constants are encapsulated in a class so notebooks can monkeypatch / extend
without import-time side effects.
"""

from __future__ import annotations

from typing import Optional


class METRIC_PREFIXES:
    """Registry of metric prefixes we care about."""

    CORE_PREFIXES: tuple[str, ...] = (
        'exact_match',
        'quasi_exact_match',
        'prefix_exact_match',
        'quasi_prefix_exact_match',
        'classification_micro_f1',
        'classification_macro_f1',
        'f1_score',
        'rouge_l',
        'bleu_',
        'ifeval_strict_accuracy',
        'wildbench_score',
        'wildbench_score_rescaled',
        'omni_math_accuracy',
        'chain_of_thought_correctness',
        'math_equiv',
        'math_equiv_chain_of_thought',
        'safety_score',
        'safety_gpt_score',
        'safety_llama_score',
        'air_score',
        'air_category_',
    )

    BOOKKEEPING_PREFIXES: tuple[str, ...] = (
        # token/size/runtime / resource accounting
        'num_',
        'training_',
        'inference_',
        'batch_size',
        'max_prob',
        'logprob',
        'num_perplexity_tokens',
        'num_bytes',
        'perplexity',
        'bits_per_byte',
        'logprob_per_byte',
        # decoding / stopping bookkeeping
        'finish_reason_',
        'prompt_truncated',
        # calibration / fitting plumbing
        'ece_',
        'platt_',
        'selective_',
        # meta / dataset sizing
        'num_instances',
        'num_train_',
        'num_references',
    )


def classify_metric(metric_name: Optional[str]) -> tuple[str, str | None]:
    """Return (metric_class, matched_prefix).

    metric_class ∈ {'core', 'bookkeeping', 'untracked'}
    """
    if not metric_name:
        return ('untracked', None)
    for p in METRIC_PREFIXES.CORE_PREFIXES:
        if metric_name.startswith(p):
            return ('core', p)
    for p in METRIC_PREFIXES.BOOKKEEPING_PREFIXES:
        if metric_name.startswith(p):
            return ('bookkeeping', p)
    return ('untracked', None)


def metric_family(metric_name: Optional[str]) -> str:
    """A lightweight family heuristic used for summaries."""
    if not metric_name:
        return '?'
    # hierarchical families
    if metric_name.startswith('air_'):
        return 'air'
    if metric_name.startswith('bias_metric:'):
        return 'bias_metric'
    if metric_name.startswith('safety_'):
        return 'safety'
    if metric_name.startswith('bbq_'):
        return 'bbq'
    if '@' in metric_name:
        return metric_name.split('@', 1)[0]
    return metric_name.split('_', 1)[0].split(':', 1)[0]
