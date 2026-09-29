"""MAGNET evidence projection over an ``magnet_evals`` run (plan M4/M5).

An ``aiq-evals`` run is a record of what a native engine did, and it may hold
many result records: several Inspect logs, an OLMo suite, HELM
split/perturbation variants. MAGNET turns one run into **one** evidence view
and one flat kwdagger row:

* a *selector* names exactly one claim-facing metric; unset keys are
  wildcards. Zero or several matches are reported, never averaged;
* the *eligibility policy* is MAGNET's judgment and never alters the run;
* ``score`` is exposed only when the selection is unambiguous and eligible,
  so a claim cannot silently vote with the wrong or an ineligible number.

Nothing here imports a native engine; bundles are read with ``magnet_evals``'s
engine-free readers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

SELECTOR_KEYS = ('task', 'model_role', 'metric', 'scorer', 'score', 'group', 'reducer')
COVERAGE_POLICIES = ('complete', 'any')
EVIDENCE_SCHEMA = 'magnet-aiq-evals-evidence/1'


@dataclass(frozen=True)
class Selection:
    metric: Any | None
    candidates: int
    error: str | None = None


def normalize_selector(selector: Mapping[str, Any] | None) -> dict[str, str | None]:
    """Omitted keys are wildcards; an explicit ``None`` requires the field to be unset
    (e.g. HELM's unperturbed statistic has ``score=None``)."""
    selector = dict(selector or {})
    unknown = set(selector) - set(SELECTOR_KEYS)
    if unknown:
        raise ValueError(f'unknown evidence selector keys {sorted(unknown)}; allowed: {list(SELECTOR_KEYS)}')
    return {key: None if value is None else str(value) for key, value in selector.items()}


def _matches(metric: Any, wanted: Mapping[str, str | None]) -> bool:
    for key, value in wanted.items():
        actual = getattr(metric, key)
        if value is None:
            if actual is not None:
                return False
        elif actual is None or str(actual) != value:
            return False
    return True


def select_metric(result: Any, selector: Mapping[str, Any] | None) -> Selection:
    """Choose exactly one ``MetricRecord`` from an ``EvaluationResult``.

    With no selector, the only metric in the run is selected. A run with
    several metrics needs an explicit selector: a HELM run alone carries split
    and perturbation variants of every statistic.
    """
    wanted = normalize_selector(selector)
    candidates = [
        metric
        for record in result.records
        for metric in record.metrics
        if _matches(metric, wanted)
    ]
    if len(candidates) == 1:
        return Selection(candidates[0], 1)
    if not candidates:
        return Selection(None, 0, f'selector {wanted} matched no metric')
    return Selection(
        None,
        len(candidates),
        f'selector {wanted} is ambiguous: {len(candidates)} metrics match; '
        'name task/scorer/score/metric/group/reducer until exactly one matches',
    )


@dataclass
class EvidenceView:
    """One claim-facing view of one aiq-evals run (not stored in the run)."""

    engine: str
    run_status: str
    measurement_identity: str
    identity_reusable: bool
    normalized_artifact_identity: str | None
    run_path: str
    selector: dict[str, str | None]
    candidates: int
    selection_error: str | None
    selected: dict[str, Any] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)
    coverage_policy: str = 'complete'
    eligible: bool = False
    ineligible_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {'schema': EVIDENCE_SCHEMA, **self.__dict__}


def evidence_view(bundle: Any, selector: Mapping[str, Any] | None = None, coverage_policy: str = 'complete') -> EvidenceView:
    """Project a loaded ``magnet_evals`` RunBundle into MAGNET evidence."""
    if coverage_policy not in COVERAGE_POLICIES:
        raise ValueError(f'coverage_policy must be one of {COVERAGE_POLICIES}')
    result = bundle.result
    selection = select_metric(result, selector)
    view = EvidenceView(
        engine=result.engine,
        run_status=result.status,
        measurement_identity=bundle.resolved.identity.digest,
        identity_reusable=bool(bundle.resolved.identity.reusable),
        normalized_artifact_identity=bundle.manifest.get('normalized_artifact_identity'),
        run_path=str(bundle.path),
        selector=normalize_selector(selector),
        candidates=selection.candidates,
        selection_error=selection.error,
        coverage_policy=coverage_policy,
    )
    reasons = []
    if result.status != 'succeeded':
        reasons.append(f'run status is {result.status}')
    if selection.metric is None:
        reasons.append(selection.error or 'no metric selected')
    else:
        metric = selection.metric
        view.selected = {
            key: getattr(metric, key) for key in (*SELECTOR_KEYS, 'value', 'denominator')
        }
        record = next(
            r for r in result.records if r.task == metric.task and r.model_role == metric.model_role
        )
        view.coverage = record.coverage.to_dict()
        if coverage_policy == 'complete' and record.coverage.status != 'complete':
            reasons.append(f'coverage is {record.coverage.status}, policy requires complete')
        if not math.isfinite(float(metric.value)):
            reasons.append('selected value is not finite')
    view.ineligible_reasons = reasons
    view.eligible = not reasons
    return view


def flat_metrics(view: Mapping[str, Any]) -> dict[str, Any]:
    """Scalar ``metrics.<node>.*`` leaves for one kwdagger row.

    ``score`` appears only for an unambiguous, eligible selection; everything
    else is claim-facing provenance so a verdict can explain itself.
    """
    selected = view.get('selected') or {}
    coverage = view.get('coverage') or {}
    flat: dict[str, Any] = {
        'eligible': bool(view['eligible']),
        'ineligible_reasons': '; '.join(view.get('ineligible_reasons') or []),
        'run_status': view['run_status'],
        'engine': view['engine'],
        'measurement_identity': view['measurement_identity'],
        'normalized_artifact_identity': view.get('normalized_artifact_identity'),
        'run_path': view['run_path'],
        'candidates': view['candidates'],
        'coverage_policy': view['coverage_policy'],
    }
    for key in SELECTOR_KEYS:
        flat[f'selected.{key}'] = selected.get(key)
    flat['selected.value'] = selected.get('value')
    flat['denominator'] = selected.get('denominator')
    for key in ('status', 'expected', 'processed', 'failed'):
        flat[f'coverage.{key}'] = coverage.get(key)
    if view['eligible']:
        flat['score'] = selected['value']
    return flat
