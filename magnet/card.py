"""
The evaluation result card: what a recipe produces.

MAGNET has used "evaluation card" for two different things — the YAML program
that specifies an evaluation, and the result of running it. This module names
the second one :class:`EvaluationResultCard`; the first is an
:class:`~magnet.recipe.EvaluationRecipe`.

Neither name is taken. :class:`magnet.evaluation.EvaluationCard` remains what
it has always been — the loaded YAML program and the runner that executes it —
and nothing here changes, wraps, or deprecates it. A team that never writes a
recipe never meets these classes.

The split is the point. Today one object is the program, the run, and the
result at once, so "the card said VERIFIED" and "the card sweeps three
thresholds" are sentences about the same thing. Separating them means a result
can be written, diffed, and archived without dragging the runner along, and it
gives the result somewhere to record what it was standing on.

A result card has two components:

    empirical    — the verdict, its per-sweep outcomes, metrics, provenance
    theoretical  — reserved; see :attr:`EvaluationResultCard.theoretical`

Either may be absent. A card with only an empirical component is every card
MAGNET produces today, and is everything this module builds. The second slot is
a placeholder here on purpose: it is filled by separate work, and the shape of
the recipe-to-result-card pipeline can be judged without it.

Example:
    >>> from magnet.card import EvaluationResultCard, EmpiricalResult, Verdict
    >>> card = EvaluationResultCard(
    ...     title='Addition commutes',
    ...     empirical=EmpiricalResult(verdict=Verdict.VERIFIED),
    ... )
    >>> print(card.render())
    Addition commutes
    ================================
    RESULT:  VERIFIED
    BASIS:   not recorded
"""
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

__all__ = [
    'Verdict',
    'ClaimOutcome',
    'EmpiricalResult',
    'EvaluationResultCard',
]


class Verdict(StrEnum):
    """
    The outcome of evaluating a claim.

    These are the strings MAGNET has always used, kept verbatim so that verdicts
    from a recipe and from the YAML runner are comparable without translation.
    """

    VERIFIED = 'VERIFIED'
    FALSIFIED = 'FALSIFIED'
    INCONCLUSIVE = 'INCONCLUSIVE'
    UNEVALUATED = 'UNEVALUATED'


@dataclass
class ClaimOutcome:
    """One claim evaluated at one point of the sweep."""

    claim: str
    verdict: Verdict = Verdict.UNEVALUATED
    message: str = ''
    symbols: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'claim': self.claim,
            'verdict': str(self.verdict),
            'message': self.message,
            'symbols': _simple_view(self.symbols),
        }


@dataclass
class EmpiricalResult:
    """The measured half of a card."""

    verdict: Verdict = Verdict.UNEVALUATED
    outcomes: tuple[ClaimOutcome, ...] = ()
    aggregation: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def default_provenance() -> dict[str, Any]:
        from magnet import __version__

        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'host': platform.node(),
            'python': platform.python_version(),
            'magnet': __version__,
        }

    @property
    def counts(self) -> dict[str, int]:
        counts = {str(v): 0 for v in Verdict}
        for outcome in self.outcomes:
            counts[str(outcome.verdict)] += 1
        return {k: v for k, v in counts.items() if v}

    def to_dict(self) -> dict:
        return {
            'verdict': str(self.verdict),
            'counts': self.counts,
            'aggregation': self.aggregation,
            'metrics': self.metrics,
            'provenance': self.provenance,
            'outcomes': [o.to_dict() for o in self.outcomes],
        }


@dataclass
class EvaluationResultCard:
    """
    A produced evaluation artifact: what was claimed, what was measured, and
    what the measurement is standing on.
    """

    title: str = ''
    description: str = ''
    version: str = ''
    organizations: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    recipe: str | None = None
    empirical: EmpiricalResult | None = None

    #: Reserved for what the verdict is standing on: the statements the claim is
    #: grounded on and every way the experiment departs from their hypotheses.
    #: Separate work fills this in; nothing in this module interprets it, which
    #: is why it is typed loosely and only asked for a ``to_dict``. Keeping the
    #: slot visible is deliberate — a result card that cannot say what it
    #: assumed is the gap this structure exists to close, and leaving the field
    #: out would hide that it is still open.
    theoretical: Any = None

    @property
    def verdict(self) -> Verdict:
        return self.empirical.verdict if self.empirical else Verdict.UNEVALUATED

    def to_dict(self) -> dict:
        return {
            'title': self.title,
            'description': self.description,
            'version': self.version,
            'organizations': list(self.organizations),
            'tags': list(self.tags),
            'recipe': self.recipe,
            'empirical': self.empirical.to_dict() if self.empirical else None,
            # Whatever occupies the slot is asked for its own dict rather than
            # inspected here, so filling it in does not touch this method.
            'theoretical': self.theoretical.to_dict() if self.theoretical else None,
        }

    def write(self, path) -> None:
        """Write the card as JSON."""
        import json

        import ubelt as ub

        path = ub.Path(path)
        path.parent.ensuredir()
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str) + '\n')

    def render(self, verbose: bool = False) -> str:
        """
        Human-readable summary.

        The basis line sits directly under the verdict on purpose. A verdict
        read without knowing what it assumes is the thing this whole mechanism
        exists to prevent.
        """
        lines = [self.title, '================================']
        lines.append(f'RESULT:  {self.verdict}')

        if self.empirical and len(self.empirical.outcomes) > 1:
            counts = ', '.join(f'{v} {k}' for k, v in self.empirical.counts.items())
            lines.append(f'         ({counts})')

        lines.extend(self._basis_lines(verbose=verbose))
        return '\n'.join(lines)

    def _basis_lines(self, verbose: bool = False) -> list[str]:
        """
        What the verdict is standing on.

        The seam for :attr:`theoretical`. Today a result card knows nothing
        about its own basis and says so; the work that fills the slot in
        replaces this method and nothing else in the module.
        """
        if self.theoretical is None:
            return ['BASIS:   not recorded']
        return [f'BASIS:   {self.theoretical}']

    def summarize(self, verbose: bool = False) -> None:
        print(self.render(verbose=verbose))


def _simple_view(symbols: dict[str, Any]) -> dict[str, Any]:
    """
    Keep only symbol values that serialize cleanly.

    Symbol values are arbitrary — dataframes, fitted models, HELM suite handles.
    A card has to be writable to JSON, so anything that is not obviously
    representable is summarized by its type rather than dropped silently.
    """
    out = {}
    for key, value in symbols.items():
        if key.startswith('__'):
            continue
        if isinstance(value, (int, float, str, bool, type(None))):
            out[key] = value
        elif isinstance(value, (list, tuple, dict)):
            try:
                import json

                json.dumps(value)
                out[key] = value
            except (TypeError, ValueError):
                out[key] = f'<{type(value).__name__}>'
        else:
            out[key] = f'<{type(value).__name__}>'
    return out
