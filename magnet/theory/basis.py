"""
Assembling edges into a card's theoretical component, and accounting for them.

The accounting is the product. Naming a grounding statement names its
hypothesis list, so every hypothesis then has to be one of:

    discharged   -- an edge says the experiment establishes it
    a gap        -- an edge says how the experiment departs from it
    unaccounted  -- no edge at all; nobody has looked

Unaccounted is not an error and never withholds a verdict. It is context that
belongs next to the verdict, because VERIFIED standing on two assumptions
nobody has examined is a materially weaker statement than VERIFIED standing on
none, and without this accounting nothing in the system can tell them apart.
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
from magnet.theory.registry import REGISTRY

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
        return max_severity([e.resolved_severity for e in self.gaps])

    @property
    def failed_checks(self) -> tuple:
        """Checkers that ran and came back false -- measured contradictions."""
        return tuple(e for e in self.gaps if e.check_outcome == 'failed')

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
    def failed_checks(self) -> tuple:
        return tuple(e for c in self.per_theorem for e in c.failed_checks)

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
        One line, for printing next to a verdict.

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
            'failed_checks': [e.to_dict() for e in self.failed_checks],
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
    def collect(
        cls,
        *objects,
        registry=REGISTRY,
        extra_edges: Sequence = (),
        extra_groundings: Sequence = (),
        formalizations: Iterable = (),
    ) -> 'TheoreticalBasis':
        """
        Gather annotations from decorated objects and the registry.

        Groundings come from the objects passed in and from ``extra_groundings``
        -- a card is grounded on what *it* claims, not on whatever else happens
        to be imported. Edges are gathered more widely, including from the
        registry for any hypothesis of a grounding statement, because the
        relaxations are usually in code the card calls rather than code it
        contains.

        A grounding that names its ``formalization=`` contributes it here, so a
        card that declares where its theorem came from does not also have to be
        handed the same body a second time. Explicitly passed formalizations
        come first, and win when both name the same declaration.
        """
        groundings = list(extra_groundings)
        edges = []
        seen: set[int] = set()

        def push(edge) -> None:
            if id(edge) not in seen:
                seen.add(id(edge))
                edges.append(edge)

        for obj in objects:
            groundings.extend(getattr(obj, '__magnet_groundings__', ()))
            for edge in getattr(obj, '__magnet_edges__', ()):
                push(edge)
        for edge in extra_edges:
            push(edge)

        if registry is not None:
            grounded = {g.ref.declaration for g in groundings}
            for edge in registry.edges:
                if edge.ref.declaration in grounded:
                    push(edge)

        bodies = list(formalizations)
        known = {id(f) for f in bodies}
        for grounding in groundings:
            body = getattr(grounding, 'formalization', None)
            if body is not None and id(body) not in known:
                known.add(id(body))
                bodies.append(body)

        return cls(
            groundings=tuple(groundings),
            edges=tuple(edges),
            formalizations=tuple(bodies),
        )

    @classmethod
    def from_ledger(cls, ledger, formalizations: Iterable = ()) -> 'TheoreticalBasis':
        """
        Build a basis from statically extracted annotations.

        This is the path that does not execute anything: parse a repository,
        turn what it declares into the same :class:`~magnet.theory.predicates.Edge`
        and :class:`~magnet.theory.predicates.Grounding` objects the runtime
        produces, and report on them identically. The only thing missing is
        evidence -- no observations, no witnessed values, no check outcomes,
        since nothing ran.

        Unresolvable annotations are skipped here and surfaced by
        :func:`magnet.theory.static.lint` instead; a reference nobody can read
        is a lint error, not a silent hole in the ledger.
        """
        from magnet.theory.predicates import Edge, Grounding
        from magnet.theory.registry import TheoryRegistry

        scratch = TheoryRegistry()
        groundings = []
        edges = []
        for annotation in ledger.annotations:
            if not annotation.resolved:
                continue
            options = dict(annotation.options)
            options.pop('registry', None)
            options.pop('observe', None)
            options.pop('witness_params', None)
            site = options.pop('site', None) or annotation.site
            if annotation.predicate == 'grounds':
                groundings.append(
                    Grounding(annotation.ref, site=site, registry=scratch, **options)
                )
            else:
                relation = (
                    Relation.CHECKS
                    if annotation.predicate == 'checks'
                    else Relation(annotation.predicate)
                )
                severity = options.pop('severity', None)
                edges.append(
                    Edge(
                        annotation.ref,
                        relation=relation,
                        severity=severity,
                        site=site,
                        registry=scratch,
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
                    key=lambda e: -severity_rank(e.resolved_severity),
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
                marker = f'[{edge.resolved_severity:6}]'
                relation = str(edge.relation)
                if edge.check_outcome:
                    relation = f'{relation} ({edge.check_outcome})'
                lines.append(f'    {marker} {edge.ref.hypothesis}: {relation}')
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
