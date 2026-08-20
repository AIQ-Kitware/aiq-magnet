"""
Assembling edges into a card's theoretical component, and accounting for them.

Naming a grounding statement names its hypothesis list, so every hypothesis is
then one of:

    discharged   an edge says the experiment establishes it
    a gap        an edge says how the experiment departs from it
    unaccounted  no edge at all; nobody has looked

Unaccounted hypotheses are reported rather than rejected. The count is what
separates a result that departed from its theorem in four places from one that
departed in none.
"""
from dataclasses import dataclass
from typing import Iterable, Sequence

from magnet.theory.model import (
    Freshness,
    Hypothesis,
    ProofStatus,
    Relation,
    Severity,
    Theorem,
    max_severity,
    severity_rank,
)

__all__ = ['AssumptionCoverage', 'CoverageReport', 'TheoreticalBasis']


@dataclass
class AssumptionCoverage:
    """Per-statement accounting."""

    theorem: Theorem
    discharged: tuple = ()
    gaps: tuple = ()
    unaccounted: tuple[Hypothesis, ...] = ()
    orphaned: tuple = ()

    @property
    def enumerated(self) -> bool:
        return self.theorem.hypotheses_enumerated

    @property
    def is_complete(self) -> bool:
        """
        Every hypothesis accounted for and none orphaned.

        False when the hypotheses were never enumerated: a statement whose
        binders nobody exported is not a covered one.
        """
        return self.enumerated and not self.unaccounted and not self.orphaned

    @property
    def max_severity(self) -> Severity:
        return max_severity([e.severity for e in self.gaps])

    def to_dict(self) -> dict:
        return {
            'theorem': self.theorem.declaration,
            'informal': self.theorem.informal,
            'proof': str(self.theorem.proof),
            'review': str(self.theorem.review),
            'extra_axioms': list(self.theorem.extra_axioms),
            'hypotheses_enumerated': self.enumerated,
            'complete': self.is_complete,
            'max_severity': str(self.max_severity),
            'discharged': [e.to_dict() for e in self.discharged],
            'gaps': [e.to_dict() for e in self.gaps],
            'unaccounted': [
                {'name': h.name, 'informal': h.informal, 'structural': h.structural}
                for h in self.unaccounted
            ],
            'orphaned': [e.to_dict() for e in self.orphaned],
        }


@dataclass
class CoverageReport:
    """Accounting across every statement a card is grounded on."""

    per_theorem: tuple[AssumptionCoverage, ...] = ()
    #: Edges naming a statement nothing is grounded on. Without this they would
    #: be silently dropped -- an author writes an edge, it never appears, and
    #: nothing says why. Usually it means a grounding is missing, or the edge
    #: belongs on a different statement.
    dangling: tuple = ()

    @property
    def is_complete(self) -> bool:
        return (
            bool(self.per_theorem)
            and not self.dangling
            and all(c.is_complete for c in self.per_theorem)
        )

    @property
    def max_severity(self) -> Severity:
        return max_severity([c.max_severity for c in self.per_theorem])

    @property
    def unaccounted(self) -> tuple[Hypothesis, ...]:
        return tuple(h for c in self.per_theorem for h in c.unaccounted)

    @property
    def unproved(self) -> tuple[Theorem, ...]:
        """Grounding statements not closed under the kernel axioms."""
        return tuple(c.theorem for c in self.per_theorem if c.theorem.proof is not ProofStatus.PROVED)

    @property
    def stale(self) -> tuple:
        """Edges whose reference no longer resolves at the pinned commit."""
        return tuple(
            e
            for c in self.per_theorem
            for e in (c.gaps + c.discharged + c.orphaned)
            if e.freshness in (Freshness.STALE, Freshness.BROKEN)
        )

    def summary(self) -> str:
        """
        One line, for printing next to a result.

        Example:
            >>> CoverageReport().summary()
            'not grounded on any statement'
        """
        if not self.per_theorem:
            return 'not grounded on any statement'

        n_discharged = sum(len(c.discharged) for c in self.per_theorem)
        n_gaps = sum(len(c.gaps) for c in self.per_theorem)
        n_unaccounted = sum(len(c.unaccounted) for c in self.per_theorem)
        n_unknown = sum(1 for c in self.per_theorem if not c.enumerated)
        total = n_discharged + n_gaps + n_unaccounted

        parts = []
        if n_discharged:
            parts.append(f'{n_discharged} discharged')
        if n_gaps:
            parts.append(f'{n_gaps} relaxed (max severity {self.max_severity})')
        if n_unaccounted:
            parts.append(f'{n_unaccounted} unaccounted')

        if not total:
            body = 'no assumptions recorded'
        else:
            noun = 'assumption' if total == 1 else 'assumptions'
            body = f'standing on {total} {noun}: ' + ', '.join(parts)

        if n_unknown:
            plural = '' if n_unknown == 1 else 's'
            body += f' ({n_unknown} statement{plural} with hypotheses not enumerated)'
        if self.dangling:
            body += f'; {len(self.dangling)} dangling'
        return body

    def to_dict(self) -> dict:
        return {
            'complete': self.is_complete,
            'max_severity': str(self.max_severity),
            'summary': self.summary(),
            'statements': [c.to_dict() for c in self.per_theorem],
            'dangling': [e.to_dict() for e in self.dangling],
        }


@dataclass
class TheoreticalBasis:
    """
    The theoretical component of a card.

    What the statements give you, what the experiment gave up to be runnable,
    and whether anything fell between the two.
    """

    groundings: tuple = ()
    edges: tuple = ()
    formalizations: tuple = ()

    @classmethod
    def from_ledger(cls, ledger, formalizations: Iterable = ()) -> 'TheoreticalBasis':
        """
        Build a basis from statically extracted annotations.

        Parse a repository, turn what it declares into
        :class:`~magnet.theory.predicates.Edge` and
        :class:`~magnet.theory.predicates.Grounding` objects, and report on
        them. Nothing is imported or executed.

        Unresolvable annotations are skipped here and reported by
        :func:`magnet.theory.static.lint`, so an unreadable reference surfaces
        as a lint error instead of a silent hole in the ledger.
        """
        from magnet.theory.predicates import Edge, Grounding

        groundings = []
        edges = []
        for annotation in ledger.annotations:
            if not annotation.resolved:
                continue
            options = dict(annotation.options)
            site = options.pop('site', None) or annotation.site
            if annotation.predicate == 'grounds':
                groundings.append(Grounding(annotation.ref, site=site, **options))
            else:
                severity = options.pop('severity', None)
                edges.append(
                    Edge(
                        annotation.ref,
                        relation=Relation(annotation.predicate),
                        severity=severity,
                        site=site,
                        **options,
                    )
                )
        basis = cls(
            groundings=tuple(groundings),
            edges=tuple(edges),
            formalizations=tuple(formalizations),
        )
        return basis.resolve() if formalizations else basis

    # ------------------------------------------------------------- resolution

    def resolve(self, *formalizations) -> 'TheoreticalBasis':
        """
        Attach real statements to the references, and set freshness.

        Until this runs, a basis is a bag of strings: it knows what the code
        *claims* about itself and nothing about whether those claims name
        anything real. Resolution is where a reference to a renamed binder
        becomes visible as :attr:`Freshness.BROKEN` rather than passing silently.
        """
        sources = list(formalizations) or list(self.formalizations)
        for annotation in list(self.groundings) + list(self.edges):
            annotation.freshness = Freshness.UNKNOWN
            for formalization in sources:
                try:
                    formalization.resolve(annotation.ref)
                except KeyError:
                    if annotation.ref.declaration in formalization:
                        annotation.freshness = Freshness.BROKEN
                    continue
                annotation.freshness = Freshness.CURRENT
                break
        return TheoreticalBasis(
            groundings=self.groundings,
            edges=self.edges,
            formalizations=tuple(sources),
        )

    def statements(self) -> tuple[Theorem, ...]:
        """
        The grounding statements, resolved against the formalizations if any.

        A grounding whose declaration is not in any loaded formalization still
        yields a placeholder, so an unresolvable reference shows up in the
        report instead of vanishing from it.
        """
        out: dict[str, Theorem] = {}
        for grounding in self.groundings:
            declaration = grounding.ref.declaration
            if declaration in out:
                continue
            theorem = None
            for formalization in self.formalizations:
                if declaration in formalization:
                    theorem = formalization[declaration]
                    break
            if theorem is None:
                theorem = Theorem(
                    declaration=declaration,
                    informal=grounding.informal,
                    note='not found in any loaded formalization',
                )
            out[declaration] = theorem
        return tuple(out.values())

    def coverage(self) -> CoverageReport:
        """Account for every hypothesis of every grounding statement."""
        per_theorem = []
        for theorem in self.statements():
            edges = [e for e in self.edges if e.ref.declaration == theorem.declaration]
            names = {h.name for h in theorem.hypotheses}

            if theorem.hypotheses_enumerated:
                orphaned = tuple(e for e in edges if e.ref.hypothesis not in names)
                live = [e for e in edges if e.ref.hypothesis in names]
            else:
                orphaned = ()
                live = edges

            discharged = tuple(e for e in live if not e.is_gap)
            gaps = tuple(
                sorted(
                    (e for e in live if e.is_gap),
                    key=lambda e: -severity_rank(e.severity),
                )
            )
            covered = {e.ref.hypothesis for e in live}
            unaccounted = tuple(h for h in theorem.hypotheses if h.name not in covered)

            per_theorem.append(
                AssumptionCoverage(
                    theorem=theorem,
                    discharged=discharged,
                    gaps=gaps,
                    unaccounted=unaccounted,
                    orphaned=orphaned,
                )
            )
        grounded = {t.declaration for t in self.statements()}
        dangling = tuple(e for e in self.edges if e.ref.declaration not in grounded)
        return CoverageReport(per_theorem=tuple(per_theorem), dangling=dangling)

    # -------------------------------------------------------------- reporting

    def to_dict(self) -> dict:
        return {
            'formalizations': [
                {
                    'name': f.name,
                    'repository': f.repository,
                    'commit': f.commit,
                    'review': str(f.review),
                }
                for f in self.formalizations
            ],
            'groundings': [g.to_dict() for g in self.groundings],
            'edges': [e.to_dict() for e in self.edges],
            'coverage': self.coverage().to_dict(),
        }

    def render(self) -> str:
        """A human-readable edge table, ordered by severity."""
        lines: list[str] = []
        for cov in self.coverage().per_theorem:
            theorem = cov.theorem
            lines.append(f'{theorem.declaration}  [{theorem.proof}, {theorem.review}]')
            if theorem.informal:
                lines.append(f'    {theorem.informal}')
            if theorem.extra_axioms:
                lines.append(f'    extra axioms: {", ".join(theorem.extra_axioms)}')

            for edge in cov.gaps + cov.discharged:
                marker = f'[{edge.severity:6}]'
                lines.append(f'    {marker} {edge.ref.hypothesis}: {edge.relation}')
                hypothesis = _find_hypothesis(theorem, edge.ref.hypothesis)
                if hypothesis is not None and hypothesis.informal:
                    lines.append(f'             ideal:  {hypothesis.informal}')
                if edge.informal:
                    lines.append(f'             actual: {edge.informal}')
                if edge.evidence:
                    lines.append(f'             proof:  {edge.evidence}')
                if edge.site:
                    lines.append(f'             site:   {edge.site}')
            for hypothesis in cov.unaccounted:
                tag = 'UNACCOUNTED' + (' (structural)' if hypothesis.structural else '')
                lines.append(f'    [{tag}] {hypothesis.name}: {hypothesis.informal}')
            for edge in cov.orphaned:
                lines.append(
                    f'    [ORPHANED] {edge.ref.hypothesis} is not a hypothesis of this statement'
                )

            if cov.is_complete:
                lines.append('    coverage: complete')
            elif not cov.enumerated:
                lines.append('    coverage: UNKNOWN (hypotheses not enumerated)')
            else:
                unmet = len(cov.unaccounted) + len(cov.orphaned)
                lines.append(f'    coverage: INCOMPLETE ({unmet} unaccounted or orphaned)')
            lines.append('')

        report = self.coverage()
        if report.dangling:
            lines.append('dangling -- these name a statement nothing is grounded on:')
            for edge in report.dangling:
                lines.append(f'    {edge.relation} {edge.ref.key}')
                if edge.site:
                    lines.append(f'             site:   {edge.site}')
            lines.append('')

        if not lines:
            return 'no theoretical basis recorded'
        return '\n'.join(lines).rstrip()


def _find_hypothesis(theorem: Theorem, name) -> Hypothesis | None:
    for hypothesis in theorem.hypotheses:
        if hypothesis.name == name:
            return hypothesis
    return None
