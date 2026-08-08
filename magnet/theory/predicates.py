"""
The predicates: how code stands with respect to an idealized hypothesis.

Six verbs, one object. Each reads as a true sentence at the site it annotates::

    satisfies     the experiment establishes it
    approximates  the same object, in a finite or numerical version
    substitutes   a *different* object stands in
    assumes       relied on; nothing establishes or checks it
    ignores       a side condition delimiting the regime, dropped
    violates      known to fail -- ideally with proof, via ``evidence=``

plus one that resolves at run time::

    checks        tested during the run; comes out satisfied or violated

Every one of them returns an :class:`Edge`, and an ``Edge`` activates three
ways with identical registration semantics:

.. code:: python

    # (a) bare -- the gap is an *absence* of code, so there is no site
    assumes('Paper.main::hlipschitz', informal='never tested')

    # (b) decorator -- site is the decorated object; each call is an observation
    @approximates('Paper.main::hcover', informal='fixed pool')
    def __init__(self, num_example_runs=64): ...

    # (c) context manager -- site is the with-statement, plus explicit witnesses
    with substitutes('Paper.main::hpsi') as edge:
        coords = embed(texts)
        edge.witness(embedder='nomic-embed-text-v2-moe', dim=coords.shape[1])

The edge is registered when the predicate is *called*, in all three forms, so
the assumption ledger does not depend on a ``with`` block being reached. What
activation adds is the site and the observations.

The decorator form is defined as wrapping the body in the context manager, so
the two cannot drift, and it witnesses the call's arguments automatically --
which covers the case that actually recurs, a hyperparameter standing in for an
idealized quantity.
"""
import functools
import inspect

from magnet.theory.model import (
    RELATION_DEFAULT_SEVERITY,
    Check,
    CodeSite,
    Freshness,
    HypothesisRef,
    Observation,
    Relation,
    ReviewStatus,
    Severity,
    parse_ref,
)
from magnet.theory.registry import REGISTRY

__all__ = [
    'Edge',
    'Grounding',
    'satisfies',
    'approximates',
    'substitutes',
    'assumes',
    'ignores',
    'violates',
    'checks',
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
        'checks',
        'grounds',
    }
)


class _Activatable:
    """
    Shared activation machinery: bare call, decorator, or context manager.

    Subclasses supply the payload; this supplies the three ways to attach it to
    code, and the guarantee that decorating a function means exactly running its
    body inside the context manager.
    """

    site: CodeSite | None
    observations: list

    # ------------------------------------------------------------ decorator

    def __call__(self, obj):
        self.site = CodeSite.from_object(obj)
        _stamp(obj, self)

        if not self.observe or not (inspect.isfunction(obj) or inspect.ismethod(obj)):
            # Classes cannot have a body wrapped in a context manager, and
            # observe=False is the escape hatch for hot paths. Annotate only.
            return obj

        signature = _signature_or_none(obj)

        @functools.wraps(obj)
        def wrapper(*args, **kwargs):
            with self:
                self.witness(**_bind(signature, args, kwargs, self.witness_params))
                return obj(*args, **kwargs)

        _stamp(wrapper, self)
        return wrapper

    # ------------------------------------------------------ context manager

    def __enter__(self):
        if self.site is None:
            self.site = CodeSite.from_caller(depth=2)
        self.observations.append(Observation())
        return self

    def __exit__(self, exc_type, exc, tb):
        return False  # never swallow

    # ----------------------------------------------------------- witnessing

    def witness(self, **values):
        """
        Record values observed during the current activation.

        Outside an activation this is a no-op rather than an error: annotated
        code should behave identically whether or not anyone is auditing it.
        """
        if values and self.observations:
            self.observations[-1].witness.update(values)
        return self

    @property
    def witnessed(self) -> dict:
        """Merged witness across every activation, most recent winning."""
        merged: dict = {}
        for observation in self.observations:
            merged.update(observation.witness)
        return merged


class Edge(_Activatable):
    """
    One hypothesis-to-code correspondence.

    Constructed through the predicate verbs rather than directly, so the
    relation is always set by the verb that reads correctly at the site.
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
        observe: bool = True,
        witness_params=None,
        registry=None,
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
        #: ``estimator-swap`` under ``substitutes``). Never carries the relation
        #: itself -- that lesson is why the verb exists.
        self.kind = kind
        self.site = CodeSite.parse(site) if isinstance(site, str) else site
        self.observe = observe
        self.witness_params = tuple(witness_params) if witness_params else None
        self.observations: list[Observation] = []
        self.freshness = Freshness.UNKNOWN
        (registry if registry is not None else REGISTRY).add_edge(self)

    # ----------------------------------------------------------------- checks

    def record_check(self, result) -> Check:
        """Attach a :class:`Check` to the current activation."""
        check = result if isinstance(result, Check) else Check(ok=bool(result), value=result)
        if self.observations:
            self.observations[-1].check = check
        return check

    @property
    def check_outcome(self) -> str | None:
        """``passed``, ``failed``, or ``not-run`` for a ``checks`` edge."""
        if self.relation is not Relation.CHECKS:
            return None
        results = [o.check for o in self.observations if o.check is not None]
        if not results:
            return 'not-run'
        return 'passed' if all(c.ok for c in results) else 'failed'

    @property
    def resolved_relation(self) -> Relation:
        """
        The relation to account with.

        A ``checks`` edge is not a relation until it runs: passing resolves it
        to ``satisfies``, failing to ``violates``, and never running leaves the
        hypothesis merely assumed -- the state it was in before someone wrote
        the checker.
        """
        if self.relation is not Relation.CHECKS:
            return self.relation
        return {
            'passed': Relation.SATISFIES,
            'failed': Relation.VIOLATES,
            'not-run': Relation.ASSUMES,
        }[self.check_outcome]

    @property
    def resolved_severity(self) -> Severity:
        """A failed check is high severity regardless of what was declared."""
        resolved = self.resolved_relation
        if self.relation is Relation.CHECKS:
            return RELATION_DEFAULT_SEVERITY[resolved]
        return self.severity

    # ------------------------------------------------------------- reporting

    @property
    def is_gap(self) -> bool:
        return self.resolved_relation is not Relation.SATISFIES

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'hypothesis': self.ref.key,
            'relation': str(self.relation),
            'resolved_relation': str(self.resolved_relation),
            'severity': str(self.resolved_severity),
            'declared_severity': str(self.severity),
            'kind': self.kind,
            'review': str(self.review),
            'freshness': str(self.freshness),
            'site': str(self.site) if self.site else None,
            'informal': self.informal,
            'note': self.note,
            'evidence': self.evidence,
            'check_outcome': self.check_outcome,
            'observations': [o.to_dict() for o in self.observations],
            'witness': self.witnessed,
        }

    def __repr__(self) -> str:
        where = f' at {self.site}' if self.site else ''
        return f'<Edge {self.relation} {self.ref.key} [{self.resolved_severity}]{where}>'


class Grounding(_Activatable):
    """
    A claim's assertion that it is the empirical shadow of a statement.

    Separate from :class:`Edge` because it is a different axis: claim to
    theorem, not code to hypothesis. Naming the theorem is what opens the card
    up to an assumption audit, since it names the hypothesis list every edge
    then has to account for.
    """

    def __init__(
        self,
        ref,
        informal: str = '',
        note: str = '',
        formalization=None,
        review=ReviewStatus.DRAFT,
        site=None,
        observe: bool = True,
        witness_params=None,
        registry=None,
    ) -> None:
        self.ref: HypothesisRef = parse_ref(ref)
        self.informal = informal
        self.note = note
        self.formalization = formalization
        self.review = ReviewStatus(review)
        self.site = CodeSite.parse(site) if isinstance(site, str) else site
        self.observe = observe
        self.witness_params = tuple(witness_params) if witness_params else None
        self.observations: list[Observation] = []
        self.freshness = Freshness.UNKNOWN
        (registry if registry is not None else REGISTRY).add_grounding(self)

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
            'witness': self.witnessed,
        }

    def __repr__(self) -> str:
        return f'<Grounding {self.ref.declaration}>'


class _CheckEdge(Edge):
    """
    An :class:`Edge` whose decorator form also captures the checker's result.

    Everything else about it is an ordinary edge -- the bare and
    context-manager forms behave identically, and inside a ``with`` block the
    caller records the outcome with :meth:`Edge.record_check`.
    """

    def __call__(self, func):
        wrapped = super().__call__(func)

        @functools.wraps(func)
        def runner(*args, **kwargs):
            result = wrapped(*args, **kwargs)
            self.record_check(result)
            return result

        _stamp(runner, self)
        return runner


def _stamp(obj, annotation) -> None:
    """Attach the annotation to the object, for collection without a registry."""
    attr = '__magnet_groundings__' if isinstance(annotation, Grounding) else '__magnet_edges__'
    try:
        setattr(obj, attr, getattr(obj, attr, ()) + (annotation,))
    except AttributeError:
        pass  # builtins and slotted objects; the registry still has it


def _signature_or_none(func):
    try:
        return inspect.signature(func)
    except (TypeError, ValueError):
        return None


def _bind(signature, args, kwargs, only) -> dict:
    """
    Auto-witness a call's arguments.

    Only simple scalars are recorded. A witness exists to be read on a dashboard
    and diffed across runs, so a dataframe or a benchmark-suite handle would be
    noise; ``self`` is dropped for the same reason.
    """
    if signature is None:
        return {}
    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError:
        return {}
    bound.apply_defaults()
    out = {}
    for name, value in bound.arguments.items():
        if name in {'self', 'cls'}:
            continue
        if only is not None and name not in only:
            continue
        if isinstance(value, (int, float, str, bool, type(None))):
            out[name] = value
    return out


def _predicate(relation: Relation):
    """Build one verb. All of them differ only in the relation they assert."""

    def verb(ref, severity=None, **kwargs) -> Edge:
        return Edge(ref, relation=relation, severity=severity, **kwargs)

    verb.__name__ = str(relation)
    verb.__qualname__ = str(relation)
    return verb


satisfies = _predicate(Relation.SATISFIES)
satisfies.__doc__ = """
Declare that the experiment establishes the hypothesis.

Recording this is what distinguishes "we checked" from "nobody looked" -- the
difference between a discharged assumption and one that merely never came up.

Example:
    >>> from magnet.theory import satisfies
    >>> edge = satisfies('Paper.main::hbounded',
    ...                  informal='scores are benchmark accuracies in [0, 1]')
    >>> edge.is_gap
    False
"""

approximates = _predicate(Relation.APPROXIMATES)
approximates.__doc__ = """
Declare that the code uses a finite or numerical version of the same object.

The asymptotic hypothesis met with one finite draw; the density hypothesis met
with a fixed pool.

Example:
    >>> from magnet.theory import approximates
    >>> @approximates('Paper.main::hcover', informal='fixed pool, no density')
    ... def build_pool(num_example_runs=64):
    ...     return list(range(num_example_runs))
    >>> _ = build_pool()
    >>> build_pool.__magnet_edges__[0].witnessed
    {'num_example_runs': 64}
"""

substitutes = _predicate(Relation.SUBSTITUTES)
substitutes.__doc__ = """
Declare that a *different* object stands in for the one the theorem is about.

A different estimator, a different embedding, a different metric. Presumptively
high severity, because the proved statement is then not about the artifact.

Example:
    >>> from magnet.theory import substitutes
    >>> edge = substitutes('Paper.main::hmetric', kind='metric-swap',
    ...                    informal='card reports MAE; the theorem controls MSE')
    >>> edge.severity
    <Severity.HIGH: 'high'>
"""

assumes = _predicate(Relation.ASSUMES)
assumes.__doc__ = """
Declare that the hypothesis is relied on and never established.

The gap here is an *absence* of code, which is why the bare form matters: there
is no line to decorate. Each of these is also a work item -- write a
:func:`checks` for it and the relation stops being an assumption.

Example:
    >>> from magnet.theory import assumes
    >>> edge = assumes('Paper.main::hlipschitz',
    ...                informal='close-in-embedding implies close-in-score; never tested')
    >>> edge.site is None
    True
"""

ignores = _predicate(Relation.IGNORES)
ignores.__doc__ = """
Declare that a side condition delimiting the theorem's regime is dropped.

Distinct from :func:`assumes` in that the condition is typically checkable and
simply is not checked -- a numeric smallness bound, a rank condition.

Example:
    >>> from magnet.theory import ignores
    >>> edge = ignores('Paper.main::hsmall', informal='no smallness check at this budget')
    >>> str(edge.relation)
    'ignores'
"""

violates = _predicate(Relation.VIOLATES)
violates.__doc__ = """
Declare that the hypothesis is known to fail.

Pass ``evidence`` naming a proved counterexample where one exists. That is the
difference between "we suspect this does not hold" and "it is proved that it
does not", and only the second can be checked by someone else.

Example:
    >>> from magnet.theory import violates
    >>> edge = violates('Paper.main::haffine',
    ...                 evidence='Paper.OLS.lipschitz_not_sufficient_for_affineRealizability')
    >>> edge.severity
    <Severity.HIGH: 'high'>
"""


def checks(ref, severity=None, **kwargs) -> Edge:
    """
    Test a hypothesis at run time.

    The decorated function returns a :class:`~magnet.theory.model.Check` (or
    anything truthy) and its result is recorded against the hypothesis. The
    relation is then *computed* rather than declared: a passing check resolves
    to ``satisfies``, a failing one to ``violates``, and a checker that never
    ran leaves the hypothesis assumed.

    A failing check is not a missing annotation. It is a measured contradiction
    between the run and the theorem the card claims to shadow.

    Example:
        >>> from magnet.theory import checks
        >>> from magnet.theory.model import Check
        >>> @checks('Paper.main::hrank', informal='truncation rank within budget')
        ... def rank_within_budget(rank, budget):
        ...     return Check(ok=rank <= budget, value=rank,
        ...                  detail=f'rank {rank} vs budget {budget}')
        >>> _ = rank_within_budget(7, 8)
        >>> edge = rank_within_budget.__magnet_edges__[0]
        >>> edge.check_outcome
        'passed'
        >>> edge.resolved_relation
        <Relation.SATISFIES: 'satisfies'>
        >>> _ = rank_within_budget(9, 8)
        >>> edge.check_outcome, edge.resolved_relation
        ('failed', <Relation.VIOLATES: 'violates'>)
    """
    return _CheckEdge(ref, relation=Relation.CHECKS, severity=severity, **kwargs)


def grounds(ref, **kwargs) -> Grounding:
    """
    Declare that a claim is the empirical shadow of a statement.

    Args:
        ref: the declaration the claim shadows.
        informal: how the claim corresponds to the conclusion.
        note: caveats about the correspondence itself.
        formalization: the body it came from, for provenance.
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
