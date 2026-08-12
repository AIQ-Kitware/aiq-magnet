"""
The data model: formalizations, theorems, hypotheses, and the edges to code.

A theorem holds given a list of hypotheses; an experiment rarely satisfies them
literally. An **edge** records one correspondence::

    (a named hypothesis of a theorem) -> (the code that stands in for it)

annotated with the relation between the two. The relation is the verb, and it
is the field that matters: ``satisfies`` and ``violates`` are opposite claims
about the same pair.

Three status axes are tracked separately because they fail independently:

    proof      the kernel's result: proved, sorry, unknown
    review     a human's judgement: draft ... accepted, rejected
    freshness  whether the reference still resolves at the pinned commit

Proved, draft and stale at once is a common state, not a contradiction.
"""
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterable, Iterator, Sequence

__all__ = [
    'Relation',
    'Severity',
    'ProofStatus',
    'ReviewStatus',
    'Freshness',
    'Hypothesis',
    'Theorem',
    'Formalization',
    'CodeSite',
    'HypothesisRef',
    'parse_ref',
    'KERNEL_AXIOMS',
    'RELATION_DEFAULT_SEVERITY',
    'GAP_RELATIONS',
]


#: Axioms Lean's kernel permits in a fully proved development. A theorem whose
#: ``#print axioms`` output is a subset of these is proved outright; anything
#: else (notably ``sorryAx``) means something is still assumed.
KERNEL_AXIOMS = frozenset({'propext', 'Classical.choice', 'Quot.sound'})

#: Separates a declaration from a hypothesis binder in a reference string.
REF_SEPARATOR = '::'


class Relation(StrEnum):
    """
    How a piece of code stands with respect to an idealized hypothesis.

    These are the predicates. Each reads as a true sentence at the code site,
    which is the test a candidate relation has to pass to belong here.
    """

    #: The experiment establishes it.
    SATISFIES = 'satisfies'

    #: The same object, in a finite or numerical version of it.
    APPROXIMATES = 'approximates'

    #: A *different* object stands in for the one the theorem is about.
    SUBSTITUTES = 'substitutes'

    #: Relied upon; nothing establishes or checks it.
    ASSUMES = 'assumes'

    #: A side condition delimiting the theorem's regime, simply dropped.
    IGNORES = 'ignores'

    #: Known to fail -- ideally with a proof, via ``evidence``.
    VIOLATES = 'violates'


#: Relations that represent a departure from the theorem.
GAP_RELATIONS = frozenset(
    {
        Relation.APPROXIMATES,
        Relation.SUBSTITUTES,
        Relation.ASSUMES,
        Relation.IGNORES,
        Relation.VIOLATES,
    }
)


class Severity(StrEnum):
    """
    How load-bearing the gap is.

    ``HIGH`` has a specific meaning worth holding to: *the proved theorem does
    not cover the artifact*. Reserve it for that.
    """

    NONE = 'none'
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'


#: Per-relation default severity. An author can always override; these encode
#: that substituting a different object is presumptively worse than
#: approximating the right one.
RELATION_DEFAULT_SEVERITY = {
    Relation.SATISFIES: Severity.NONE,
    Relation.APPROXIMATES: Severity.MEDIUM,
    Relation.ASSUMES: Severity.MEDIUM,
    Relation.IGNORES: Severity.MEDIUM,
    Relation.SUBSTITUTES: Severity.HIGH,
    Relation.VIOLATES: Severity.HIGH,
}


class ProofStatus(StrEnum):
    """The kernel's verdict. Not a human judgment."""

    #: ``#print axioms`` is within :data:`KERNEL_AXIOMS`.
    PROVED = 'proved'

    #: Something is still assumed -- ``sorryAx`` or another extra axiom.
    SORRY = 'sorry'

    #: Never reported. A statement authored for a card that has no
    #: formalization behind it yet lands here or in ``SORRY``.
    UNKNOWN = 'unknown'


class ReviewStatus(StrEnum):
    """
    A human's verdict, on a formalization, a theorem, or an edge.

    Edges carry this too: an edge is itself an authored claim ("this code is
    what stands in for that hypothesis"), and its severity is disputable.
    """

    DRAFT = 'draft'
    WIP = 'wip'
    SELF_ASSESSED = 'self-assessed'
    EXPERT_REVIEWED = 'expert-reviewed'
    ACCEPTED = 'accepted'
    DISPUTED = 'disputed'
    REJECTED = 'rejected'


class Freshness(StrEnum):
    """Whether a reference still resolves against the pinned formalization."""

    #: Resolves, at the recorded commit.
    CURRENT = 'current'

    #: The formalization moved since the reference was drawn.
    STALE = 'stale'

    #: The declaration or binder no longer exists.
    BROKEN = 'broken'

    #: Not checked.
    UNKNOWN = 'unknown'


@dataclass(frozen=True)
class HypothesisRef:
    """
    A parsed ``Declaration::binder`` reference.

    References are strings so they can be extracted without importing the
    annotated module. :func:`magnet.theory.static.lint` does the validation the
    object form used to do at import time.
    """

    declaration: str
    hypothesis: str | None = None

    @property
    def key(self) -> str:
        if self.hypothesis is None:
            return self.declaration
        return f'{self.declaration}{REF_SEPARATOR}{self.hypothesis}'

    def __str__(self) -> str:
        return self.key


def parse_ref(ref) -> HypothesisRef:
    """
    Coerce a reference to a :class:`HypothesisRef`.

    Accepts the string form, an already-parsed ref, or a :class:`Hypothesis`
    object -- so code that has a real formalization loaded can keep passing
    objects while the extractor reads strings out of the same source.

    Example:
        >>> parse_ref('Paper.main::hcover').hypothesis
        'hcover'
        >>> parse_ref('Paper.main').hypothesis is None
        True
    """
    if isinstance(ref, HypothesisRef):
        return ref
    if isinstance(ref, Hypothesis):
        return HypothesisRef(declaration=ref.declaration, hypothesis=ref.name)
    if isinstance(ref, Theorem):
        return HypothesisRef(declaration=ref.declaration)
    if isinstance(ref, str):
        declaration, sep, hypothesis = ref.partition(REF_SEPARATOR)
        return HypothesisRef(declaration=declaration, hypothesis=hypothesis if sep else None)
    raise TypeError(f'cannot interpret {ref!r} as a hypothesis reference')


@dataclass(frozen=True)
class Hypothesis:
    """
    One named assumption of a theorem.

    ``name`` is the binder as it appears in the statement (``hgap``,
    ``hcompetitive``, ``fit``). Binders are what make an edge stable: they
    survive refactors that invalidate file and line.

    ``structural`` marks a binder supplying an *object* rather than a
    proposition -- the estimator, the embedding, the baseline. They still get
    edges, since the theorem is about that object and the code may use another,
    but they read differently in a report.
    """

    name: str
    informal: str = ''
    lean: str | None = None
    structural: bool = False
    declaration: str | None = None  # owning theorem; set by Theorem

    @property
    def key(self) -> str:
        if self.declaration:
            return f'{self.declaration}{REF_SEPARATOR}{self.name}'
        return self.name

    @property
    def ref(self) -> HypothesisRef:
        return HypothesisRef(declaration=self.declaration, hypothesis=self.name)

    def __str__(self) -> str:
        return self.key


@dataclass(frozen=True)
class Theorem:
    """
    A statement, referenced by fully-qualified declaration name.

    Note "statement", not "proved theorem". A card whose team has no
    formalization still gets one -- conclusion plus named hypotheses, with
    ``proof=SORRY``. The statement is the schema for the assumption ledger, and
    writing it is the forcing function: you cannot state it without naming the
    hypotheses, and naming them is most of the value.
    """

    declaration: str
    informal: str = ''
    conclusion: str = ''
    hypotheses: tuple[Hypothesis, ...] = ()
    axioms: tuple[str, ...] | None = None
    file: str | None = None
    line: int | None = None
    review: ReviewStatus = ReviewStatus.DRAFT
    note: str = ''

    def __init__(
        self,
        declaration: str,
        informal: str = '',
        conclusion: str = '',
        hypotheses: Iterable['Hypothesis | str'] = (),
        axioms: Iterable[str] | None = None,
        file: str | None = None,
        line: int | None = None,
        review: ReviewStatus | str = ReviewStatus.DRAFT,
        note: str = '',
    ) -> None:
        bound = []
        for hyp in hypotheses:
            if isinstance(hyp, str):
                hyp = Hypothesis(name=hyp)
            bound.append(
                Hypothesis(
                    name=hyp.name,
                    informal=hyp.informal,
                    lean=hyp.lean,
                    structural=hyp.structural,
                    declaration=declaration,
                )
            )
        object.__setattr__(self, 'declaration', declaration)
        object.__setattr__(self, 'informal', informal)
        object.__setattr__(self, 'conclusion', conclusion)
        object.__setattr__(self, 'hypotheses', tuple(bound))
        object.__setattr__(self, 'axioms', None if axioms is None else tuple(axioms))
        object.__setattr__(self, 'file', file)
        object.__setattr__(self, 'line', line)
        object.__setattr__(self, 'review', ReviewStatus(review))
        object.__setattr__(self, 'note', note)

    @property
    def name(self) -> str:
        """The final component of the declaration."""
        return self.declaration.rsplit('.', 1)[-1]

    @property
    def hypotheses_enumerated(self) -> bool:
        """
        Whether the hypothesis list is known at all.

        Empty means *not enumerated*, never *has none*. A manifest names
        capstones without their binders, and reporting such a theorem as fully
        covered would let a missing exporter pass for a clean bill of health.
        """
        return bool(self.hypotheses)

    @property
    def proof(self) -> ProofStatus:
        """
        The kernel's verdict.

        Example:
            >>> Theorem('A', axioms=['propext']).proof
            <ProofStatus.PROVED: 'proved'>
            >>> Theorem('A', axioms=['sorryAx']).proof
            <ProofStatus.SORRY: 'sorry'>
            >>> Theorem('A').proof
            <ProofStatus.UNKNOWN: 'unknown'>
        """
        if self.axioms is None:
            return ProofStatus.UNKNOWN
        if set(self.axioms) <= KERNEL_AXIOMS:
            return ProofStatus.PROVED
        return ProofStatus.SORRY

    @property
    def extra_axioms(self) -> tuple[str, ...]:
        """Axioms used beyond the kernel's."""
        if self.axioms is None:
            return ()
        return tuple(a for a in self.axioms if a not in KERNEL_AXIOMS)

    def hypothesis(self, name: str) -> Hypothesis:
        """
        Look up a hypothesis by binder name.

        Raises if the hypotheses are enumerated and ``name`` is not among them.
        If they are not enumerated the hypothesis is created on demand, and
        coverage over this theorem stays "unknown".

        Example:
            >>> Theorem('A', hypotheses=['hgap'])['hgap'].key
            'A::hgap'
            >>> Theorem('A', hypotheses=['hgap'])['hgep']
            Traceback (most recent call last):
              ...
            KeyError: "theorem 'A' has no hypothesis 'hgep'; known: hgap"
        """
        for hyp in self.hypotheses:
            if hyp.name == name:
                return hyp
        if self.hypotheses_enumerated:
            known = ', '.join(h.name for h in self.hypotheses)
            raise KeyError(f'theorem {self.declaration!r} has no hypothesis {name!r}; known: {known}')
        return Hypothesis(name=name, declaration=self.declaration)

    def __getitem__(self, name: str) -> Hypothesis:
        return self.hypothesis(name)

    def to_dict(self) -> dict:
        out = {
            'declaration': self.declaration,
            'informal': self.informal,
            'conclusion': self.conclusion,
            'proof': str(self.proof),
            'review': str(self.review),
            'axioms': list(self.axioms) if self.axioms is not None else None,
            'file': self.file,
            'line': self.line,
            'hypotheses': [
                {
                    'name': h.name,
                    'informal': h.informal,
                    'lean': h.lean,
                    'structural': h.structural,
                }
                for h in self.hypotheses
            ],
        }
        if self.note:
            out['note'] = self.note
        return out

    def __str__(self) -> str:
        return self.declaration


@dataclass
class Formalization:
    """
    A body of statements -- typically one repository at one commit.

    Pinning ``commit`` is what makes :class:`Freshness` computable. An edge
    drawn against a moving formalization is an edge that quietly stops meaning
    what it said, which is not hypothetical: every file and line in the first
    hand-drawn DKPS edge table was invalid five weeks later.
    """

    name: str
    repository: str | None = None
    commit: str | None = None
    source: str | None = None
    review: ReviewStatus = ReviewStatus.DRAFT
    note: str = ''
    theorems: dict[str, Theorem] = field(default_factory=dict)

    def __init__(
        self,
        name: str,
        repository: str | None = None,
        commit: str | None = None,
        source: str | None = None,
        review: ReviewStatus | str = ReviewStatus.DRAFT,
        note: str = '',
        theorems: 'Iterable[Theorem] | dict[str, Theorem]' = (),
    ) -> None:
        self.name = name
        self.repository = repository
        self.commit = commit
        self.source = source
        self.review = ReviewStatus(review)
        self.note = note
        if isinstance(theorems, dict):
            self.theorems = dict(theorems)
        else:
            self.theorems = {t.declaration: t for t in theorems}

    def add(self, theorem: Theorem) -> Theorem:
        self.theorems[theorem.declaration] = theorem
        return theorem

    def theorem(self, declaration: str) -> Theorem:
        """Look up by declaration name, suggesting near misses on a failure."""
        try:
            return self.theorems[declaration]
        except KeyError:
            import difflib

            close = difflib.get_close_matches(declaration, self.theorems, n=3, cutoff=0.4)
            hint = f'; did you mean {", ".join(close)}' if close else ''
            raise KeyError(
                f'{self.name!r} has no theorem {declaration!r} ({len(self.theorems)} known){hint}'
            ) from None

    def resolve(self, ref) -> 'Hypothesis | Theorem':
        """
        Resolve a reference to the object it names.

        Raises :class:`KeyError` if the declaration or binder is absent -- which
        is what the linter reports as :attr:`Freshness.BROKEN`.
        """
        parsed = parse_ref(ref)
        theorem = self.theorem(parsed.declaration)
        if parsed.hypothesis is None:
            return theorem
        return theorem.hypothesis(parsed.hypothesis)

    def __getitem__(self, key: str) -> Theorem:
        return self.theorem(key)

    def __contains__(self, declaration: object) -> bool:
        return declaration in self.theorems

    def __iter__(self) -> Iterator[Theorem]:
        return iter(self.theorems.values())

    def __len__(self) -> int:
        return len(self.theorems)

    def to_dict(self) -> dict:
        return {
            'version': 1,
            'project': {
                'name': self.name,
                'repository': self.repository,
                'commit': self.commit,
                'review': str(self.review),
                'note': self.note,
            },
            'theorems': [t.to_dict() for t in self],
        }


@dataclass(frozen=True)
class CodeSite:
    """Where in the empirical codebase a relation was declared."""

    module: str | None = None
    qualname: str | None = None
    file: str | None = None
    line: int | None = None

    @classmethod
    def from_object(cls, obj) -> 'CodeSite':
        """Best-effort location of a function, method, or class."""
        import inspect

        try:
            file = inspect.getsourcefile(obj)
            _, line = inspect.getsourcelines(obj)
        except (TypeError, OSError):
            file, line = None, None
        return cls(
            module=getattr(obj, '__module__', None),
            qualname=getattr(obj, '__qualname__', None) or getattr(obj, '__name__', None),
            file=file,
            line=line,
        )

    @classmethod
    def from_caller(cls, depth: int = 2) -> 'CodeSite':
        """Location of the frame ``depth`` levels up -- used by ``with`` entry."""
        import inspect

        frame = inspect.currentframe()
        try:
            for _ in range(depth):
                if frame is None:
                    break
                frame = frame.f_back
            if frame is None:
                return cls()
            return cls(
                module=frame.f_globals.get('__name__'),
                qualname=frame.f_code.co_qualname,
                file=frame.f_code.co_filename,
                line=frame.f_lineno,
            )
        finally:
            del frame

    @classmethod
    def parse(cls, text: str) -> 'CodeSite':
        """
        Parse the ``module.qualname:line`` form used in declared edges.

        Example:
            >>> CodeSite.parse('pkg.mod.Class.method:234').line
            234
        """
        base, sep, line = text.rpartition(':')
        if not sep or not line.isdigit():
            base, line = text, None
        module, _, qualname = base.partition('::')
        if not qualname:
            module, qualname = None, base
        return cls(
            module=module,
            qualname=qualname,
            line=int(line) if line else None,
        )

    def __str__(self) -> str:
        if self.qualname and self.module and not self.qualname.startswith(self.module):
            base = f'{self.module}.{self.qualname}'
        else:
            base = self.qualname or self.module or self.file or '<unlocated>'
        return f'{base}:{self.line}' if self.line else base

    def to_dict(self) -> dict:
        return {
            'module': self.module,
            'qualname': self.qualname,
            'file': self.file,
            'line': self.line,
        }


def _jsonable(value):
    """
    Reduce a value to something a card can serialize.

    Witnessed values are arbitrary -- arrays, fitted models, benchmark suite
    handles. Anything not obviously representable is summarized by its type
    rather than dropped, so the record shows that something was witnessed even
    when it cannot show what.
    """
    import json

    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return f'<{type(value).__name__}>'


def severity_rank(severity: Severity) -> int:
    """Order severities so ``max`` works on them."""
    return {
        Severity.NONE: 0,
        Severity.LOW: 1,
        Severity.MEDIUM: 2,
        Severity.HIGH: 3,
    }[Severity(severity)]


def max_severity(severities: Sequence[Severity]) -> Severity:
    if not severities:
        return Severity.NONE
    return max(severities, key=severity_rank)
