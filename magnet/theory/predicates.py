"""
The predicates: how code stands with respect to an idealized hypothesis.

Six verbs, one object. Each reads as a true sentence at the site it annotates::

    satisfies     the experiment establishes it
    approximates  the same object, in a finite or numerical version
    substitutes   a *different* object stands in
    assumes       relied on; nothing establishes or checks it
    ignores       a side condition delimiting the regime, dropped
    violates      known to fail -- ideally with proof, via ``evidence=``

:func:`grounds` covers the other axis, claim to theorem.

:mod:`magnet.theory.static` reads annotations out of source, so calling one has
no runtime effect beyond building the object. Both forms are parsed:

.. code:: python

    assumes('Paper.main::hlipschitz', informal='never tested')

    @approximates('Paper.main::hcover', informal='fixed pool')
    def __init__(self, num_example_runs=64): ...

The decorator returns the function unchanged and records where it was
attached. Annotated code runs identically whether or not anyone is auditing it.
"""
from magnet.theory.model import (
    RELATION_DEFAULT_SEVERITY,
    CodeSite,
    Freshness,
    HypothesisRef,
    Relation,
    ReviewStatus,
    Severity,
    parse_ref,
)

__all__ = [
    'Edge',
    'Grounding',
    'satisfies',
    'approximates',
    'substitutes',
    'assumes',
    'ignores',
    'violates',
    'grounds',
    'PREDICATE_NAMES',
]

#: Names the static extractor looks for. Kept next to the definitions so the
#: two cannot fall out of step.
PREDICATE_NAMES = frozenset(
    {
        'satisfies',
        'approximates',
        'substitutes',
        'assumes',
        'ignores',
        'violates',
        'grounds',
    }
)


class _Annotation:
    """Shared decorator form: record where it was attached, change nothing."""

    site: CodeSite | None

    def __call__(self, obj):
        self.site = CodeSite.from_object(obj)
        attr = '__magnet_groundings__' if isinstance(self, Grounding) else '__magnet_edges__'
        try:
            setattr(obj, attr, getattr(obj, attr, ()) + (self,))
        except AttributeError:
            pass  # builtins and slotted objects
        return obj


class Edge(_Annotation):
    """
    One hypothesis-to-code correspondence.

    Constructed through the verbs rather than directly, so the relation is
    always set by the one that reads correctly at the site.
    """

    def __init__(
        self,
        ref,
        relation,
        severity=None,
        informal: str = '',
        note: str = '',
        evidence: str | None = None,
        id: str | None = None,
        review=ReviewStatus.DRAFT,
        kind: str | None = None,
        site=None,
        anchor: str | None = None,
    ) -> None:
        self.ref: HypothesisRef = parse_ref(ref)
        self.relation = Relation(relation)
        self.severity = Severity(
            severity if severity is not None else RELATION_DEFAULT_SEVERITY[self.relation]
        )
        self.informal = informal
        self.note = note
        #: For ``violates``: the declaration of a counterexample that proves it.
        self.evidence = evidence
        self.id = id
        self.review = ReviewStatus(review)
        #: Optional finer tag within a relation (``metric-swap`` vs
        #: ``estimator-swap`` under ``substitutes``). Never the relation itself.
        self.kind = kind
        self.site = CodeSite.parse(site) if isinstance(site, str) else site
        #: A literal expected on the referenced line. Only meaningful for an
        #: edge declared away from its code site, where nothing else keeps the
        #: line number true. See :func:`magnet.theory.static.check_sites`.
        self.anchor = anchor
        self.freshness = Freshness.UNKNOWN

    @property
    def is_gap(self) -> bool:
        return self.relation is not Relation.SATISFIES

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'hypothesis': self.ref.key,
            'relation': str(self.relation),
            'severity': str(self.severity),
            'kind': self.kind,
            'review': str(self.review),
            'freshness': str(self.freshness),
            'site': str(self.site) if self.site else None,
            'anchor': self.anchor,
            'informal': self.informal,
            'note': self.note,
            'evidence': self.evidence,
        }

    def __repr__(self) -> str:
        where = f' at {self.site}' if self.site else ''
        return f'<Edge {self.relation} {self.ref.key} [{self.severity}]{where}>'


class Grounding(_Annotation):
    """
    A claim's assertion that it is the empirical shadow of a statement.

    Naming the theorem is what opens the card to an assumption audit: it names
    the hypothesis list every edge then has to account for.
    """

    def __init__(
        self,
        ref,
        informal: str = '',
        note: str = '',
        review=ReviewStatus.DRAFT,
        site=None,
        anchor: str | None = None,
    ) -> None:
        self.ref: HypothesisRef = parse_ref(ref)
        self.informal = informal
        self.note = note
        self.anchor = anchor
        self.review = ReviewStatus(review)
        self.site = CodeSite.parse(site) if isinstance(site, str) else site
        self.freshness = Freshness.UNKNOWN

    @property
    def declaration(self) -> str:
        return self.ref.declaration

    def to_dict(self) -> dict:
        return {
            'theorem': self.ref.declaration,
            'informal': self.informal,
            'note': self.note,
            'review': str(self.review),
            'freshness': str(self.freshness),
            'site': str(self.site) if self.site else None,
            'anchor': self.anchor,
        }

    def __repr__(self) -> str:
        return f'<Grounding {self.ref.declaration}>'


def _predicate(relation: Relation, doc: str):
    """Build one verb. All of them differ only in the relation they assert."""

    def verb(ref, severity=None, **kwargs) -> Edge:
        return Edge(ref, relation=relation, severity=severity, **kwargs)

    verb.__name__ = verb.__qualname__ = str(relation)
    verb.__doc__ = doc + _COMMON_ARGS
    return verb


_COMMON_ARGS = """
    Args:
        ref: the hypothesis, as ``Declaration::binder`` or a
            :class:`~magnet.theory.model.Hypothesis`.
        severity: how much the departure costs. Defaults per relation.
        informal: what the code does about the hypothesis, at this site.
        note: caveats about the correspondence itself.
        site: ``module:line`` when declared away from the code it describes.
        anchor: a literal expected on that line, so a shift is detected.

    Returns:
        Edge
"""

satisfies = _predicate(
    Relation.SATISFIES,
    'Declare that the experiment establishes the hypothesis.',
)
approximates = _predicate(
    Relation.APPROXIMATES,
    'Declare a finite or numerical version of the same object.',
)
substitutes = _predicate(
    Relation.SUBSTITUTES,
    'Declare that a different object stands in for the one the theorem names.',
)
assumes = _predicate(
    Relation.ASSUMES,
    'Declare a hypothesis relied on and neither established nor checked.',
)
ignores = _predicate(
    Relation.IGNORES,
    'Declare a side condition delimiting the regime, dropped.',
)
violates = _predicate(
    Relation.VIOLATES,
    'Declare a hypothesis known to fail. Pass ``evidence=`` where there is a proof.',
)


def grounds(ref, **kwargs) -> Grounding:
    """
    Declare that a claim is the empirical shadow of a statement.

    Args:
        ref: the declaration the claim shadows.
        informal: how the claim corresponds to the conclusion.
        note: caveats about the correspondence itself.
        review: this correspondence is itself an authored claim.

    Example:
        >>> from magnet.theory import grounds
        >>> @grounds('Paper.TheoryPractice.EmpiricalCrossBudgetMAEClaim',
        ...          informal='the card asserts exactly this proposition at n=32')
        ... def claim(results):
        ...     assert results['a'] <= results['b']
        >>> claim.__magnet_groundings__[0].declaration
        'Paper.TheoryPractice.EmpiricalCrossBudgetMAEClaim'
    """
    return Grounding(ref, **kwargs)
