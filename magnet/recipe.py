"""
Evaluation recipes: cards defined in Python instead of YAML.

A recipe is the *program* half of an evaluation — symbols, sweeps, and a claim.
Running it produces an :class:`~magnet.card.EvaluationResultCard`.

MAGNET's YAML cards express the same program as a dependency graph of Python
snippets held in strings. That is fine for ``threshold: 0.1`` and poor for
anything substantial: a string gets no linting, no import resolution, no type
checking, no test coverage, and a traceback that points into ``exec``. Writing
the same graph as decorated functions gets all of it back, and dependencies come
from the function signature rather than a hand-maintained ``depends_on`` list
that can silently disagree with the code beneath it.

Writing the claim as a function also gives later work something to decorate --
an annotation recording what the experiment assumed can attach to the code that
assumes it. :meth:`EvaluationRecipe.basis` is the seam where that arrives; it
returns ``None`` here, and this module needs nothing else to be judged.

Nothing here is on the path of an existing evaluation. A recipe compiles back to
the YAML card schema (:meth:`EvaluationRecipe.to_schema_dict`), so the runner is
not forked; :class:`magnet.evaluation.EvaluationCard` is untouched, and teams
who write YAML keep writing YAML.

Example:
    >>> from magnet.recipe import recipe, symbol, claim, Sweep
    >>>
    >>> @recipe(title='Addition commutes', version='1.0')
    ... class Commutativity:
    ...     offset = Sweep([1, 2, 3])
    ...
    ...     @symbol
    ...     def evens():
    ...         return [n for n in range(-10, 11) if n % 2 == 0]
    ...
    ...     @claim
    ...     def commutes(evens, offset):
    ...         for even in evens:
    ...             assert even + offset == offset + even
    >>>
    >>> card = Commutativity.evaluate()
    >>> card.verdict
    <Verdict.VERIFIED: 'VERIFIED'>
    >>> sorted(card.empirical.counts.items())
    [('VERIFIED', 3)]
"""
import inspect
from dataclasses import dataclass, field
from graphlib import TopologicalSorter
from itertools import product
from typing import Any, Callable, Iterable, Sequence

from magnet.card import ClaimOutcome, EmpiricalResult, EvaluationResultCard, Verdict

__all__ = [
    'recipe',
    'symbol',
    'claim',
    'Sweep',
    'Symbol',
    'Claim',
    'EvaluationRecipe',
]

DEFAULT_CLAIM_AGGREGATION_STRATEGY = {'type': 'all'}


@dataclass(frozen=True)
class Sweep:
    """
    A symbol that takes each of several values in turn.

    The claim is evaluated once per point of the cartesian product of every
    sweep in the recipe, exactly as the YAML runner does.
    """

    values: tuple
    type: str | None = None

    def __init__(self, values: Iterable, type: str | None = None) -> None:
        object.__setattr__(self, 'values', tuple(values))
        object.__setattr__(self, 'type', type)


@dataclass
class Symbol:
    """One node of a recipe's dependency graph."""

    name: str
    func: Callable | None = None
    value: Any = None
    sweep: tuple | None = None
    type: str | None = None
    display_name: str | None = None
    display: bool | None = None
    metric: dict | None = None
    dependencies: tuple[str, ...] = ()

    @property
    def is_computed(self) -> bool:
        return self.func is not None

    def compute(self, **kwargs) -> Any:
        if self.func is None:
            return self.value
        return self.func(**kwargs)


@dataclass
class Claim:
    """An assertion over resolved symbols."""

    name: str
    func: Callable
    dependencies: tuple[str, ...] = ()

    def evaluate(self, context: dict) -> ClaimOutcome:
        """
        Run the claim and translate exceptions into a verdict.

        The mapping matches the YAML runner's: an ``AssertionError`` is a
        falsification (the claim was tested and is false), anything else is
        inconclusive (the claim was never actually tested).
        """
        kwargs = {dep: context[dep] for dep in self.dependencies if dep in context}
        missing = [dep for dep in self.dependencies if dep not in context]
        if missing:
            return ClaimOutcome(
                claim=self.name,
                verdict=Verdict.INCONCLUSIVE,
                message=f'SymbolNotResolved: {", ".join(missing)}',
                symbols=dict(context),
            )
        try:
            self.func(**kwargs)
        except AssertionError as ex:
            return ClaimOutcome(
                claim=self.name,
                verdict=Verdict.FALSIFIED,
                message=f'Assertion does not hold: {ex}',
                symbols=dict(context),
            )
        except Exception as ex:
            return ClaimOutcome(
                claim=self.name,
                verdict=Verdict.INCONCLUSIVE,
                message=f'ERROR evaluating claim: {ex!r}',
                symbols=dict(context),
            )
        return ClaimOutcome(claim=self.name, verdict=Verdict.VERIFIED, symbols=dict(context))


def symbol(
    func: Callable | None = None,
    type: str | None = None,
    display_name: str | None = None,
    display: bool | None = None,
    metric: dict | None = None,
):
    """
    Mark a function as a symbol of the recipe.

    Its parameter names are its dependencies — there is no separate
    ``depends_on`` list to fall out of sync with the body. Usable bare
    (``@symbol``) or called (``@symbol(type='float')``).

    Example:
        >>> @symbol(type='float')
        ... def ratio(numerator, denominator):
        ...     return numerator / denominator
        >>> ratio.__magnet_symbol__['type']
        'float'
    """

    def _decorate(fn):
        fn.__magnet_symbol__ = {
            'type': type,
            'display_name': display_name,
            'display': display,
            'metric': metric,
        }
        return fn

    if func is not None:
        return _decorate(func)
    return _decorate


def claim(func: Callable | None = None, name: str | None = None):
    """
    Mark a function as a claim: it asserts, and raises if the claim is false.

    Its parameter names are the symbols it needs.

    Example:
        >>> @claim
        ... def bounded(score, threshold):
        ...     assert score < threshold, f'{score} exceeds {threshold}'
        >>> bounded.__magnet_claim__['name'] is None
        True
    """

    def _decorate(fn):
        fn.__magnet_claim__ = {'name': name}
        return fn

    if func is not None:
        return _decorate(func)
    return _decorate


@dataclass
class EvaluationRecipe:
    """
    A runnable evaluation program.

    Built by the :func:`recipe` decorator rather than constructed directly.
    """

    title: str = ''
    description: str = ''
    version: str = ''
    category: str | None = None
    organizations: tuple[str, ...] = ()
    submitter: dict[str, str] | None = None
    tags: tuple[str, ...] = ()
    links: tuple[dict, ...] = ()
    claim_aggregation_strategy: dict = field(
        default_factory=lambda: dict(DEFAULT_CLAIM_AGGREGATION_STRATEGY)
    )
    symbols: dict[str, Symbol] = field(default_factory=dict)
    claims: dict[str, Claim] = field(default_factory=dict)
    source_class: type | None = None

    # ------------------------------------------------------------------ graph

    def static_order(self) -> list[str]:
        """Symbol names in dependency order."""
        graph = {name: set(sym.dependencies) for name, sym in self.symbols.items()}
        unknown = {
            f'{name} -> {dep}'
            for name, deps in graph.items()
            for dep in deps
            if dep not in self.symbols
        }
        if unknown:
            raise ValueError(f'symbols depend on undefined names: {sorted(unknown)}')
        return list(TopologicalSorter(graph).static_order())

    def sweep_symbols(self) -> list[Symbol]:
        return [sym for sym in self.symbols.values() if sym.sweep is not None]

    def _sweep_dependent(self) -> set[str]:
        """
        Names whose value can change across the sweep.

        Everything else is resolved once and shared. The YAML runner recomputes
        the whole graph at every sweep point; for a recipe whose first symbol
        loads a benchmark suite off disk, that is the difference between one
        load and thirty-two.
        """
        dependent = {sym.name for sym in self.sweep_symbols()}
        # static_order guarantees dependencies precede dependents
        for name in self.static_order():
            sym = self.symbols[name]
            if any(dep in dependent for dep in sym.dependencies):
                dependent.add(name)
        return dependent

    # --------------------------------------------------------------- resolve

    def compute(self, name: str, **kwargs) -> Any:
        """
        Resolve one symbol from its dependencies.

        Public because the YAML export calls it: an exported card's ``python:``
        block imports the recipe and calls back here, so the exported card runs
        the same code the recipe does rather than a transcription of it.
        """
        return self.symbols[name].compute(**kwargs)

    def check(self, name: str, **kwargs) -> None:
        """Run one claim, raising as it would in-process. See :meth:`compute`."""
        self.claims[name].func(**kwargs)

    def _resolve(self, names: Sequence[str], context: dict) -> tuple[dict, str | None]:
        """Resolve symbols into a copy of ``context``; return (context, error)."""
        context = dict(context)
        for name in names:
            if name in context:
                continue
            sym = self.symbols[name]
            try:
                kwargs = {dep: context[dep] for dep in sym.dependencies}
                context[name] = sym.compute(**kwargs)
            except Exception as ex:
                return context, f'ERROR resolving symbol {name!r}: {ex!r}'
        return context, None

    # -------------------------------------------------------------- evaluate

    def evaluate(self, provenance: bool = True) -> EvaluationResultCard:
        """
        Run the recipe and produce a card.

        Resolution failures make a sweep point INCONCLUSIVE rather than raising,
        matching the YAML runner: a claim that could not be tested is not a
        claim that was refuted, and the distinction is the whole reason
        INCONCLUSIVE exists.
        """
        order = self.static_order()
        dependent = self._sweep_dependent()
        shared_names = [n for n in order if n not in dependent]
        dynamic_names = [n for n in order if n in dependent]

        shared, error = self._resolve(shared_names, {})

        sweeps = self.sweep_symbols()
        if sweeps:
            points = [
                dict(zip([s.name for s in sweeps], values))
                for values in product(*[s.sweep for s in sweeps])
            ]
        else:
            points = [{}]

        outcomes: list[ClaimOutcome] = []
        for point in points:
            if error is not None:
                context, point_error = dict(shared, **point), error
            else:
                context, point_error = self._resolve(dynamic_names, dict(shared, **point))
            for name, claim_obj in self.claims.items():
                if point_error is not None:
                    outcomes.append(
                        ClaimOutcome(
                            claim=name,
                            verdict=Verdict.INCONCLUSIVE,
                            message=point_error,
                            symbols=context,
                        )
                    )
                else:
                    outcomes.append(claim_obj.evaluate(context))

        verdict = _reduce([o.verdict for o in outcomes], self.claim_aggregation_strategy)

        empirical = EmpiricalResult(
            verdict=verdict,
            outcomes=tuple(outcomes),
            aggregation=dict(self.claim_aggregation_strategy),
            provenance=EmpiricalResult.default_provenance() if provenance else {},
        )
        return EvaluationResultCard(
            title=self.title,
            description=self.description,
            version=self.version,
            organizations=tuple(self.organizations),
            tags=tuple(self.tags),
            recipe=self._qualified_name(),
            empirical=empirical,
            theoretical=self.basis(),
        )

    # ----------------------------------------------------------------- basis

    def basis(self) -> Any | None:
        """
        What a card produced from this recipe is standing on, or None.

        Placeholder, and the recipe half of the seam described in
        :attr:`magnet.card.EvaluationResultCard.theoretical`. Separate work
        gathers this from annotations on the claim and symbol functions, which
        is available precisely because they are functions. Until then a card
        records no basis rather than an empty one, and every other method here
        behaves the same either way.
        """
        return None

    # ---------------------------------------------------------------- export

    def _qualified_name(self) -> str | None:
        if self.source_class is None:
            return None
        return f'{self.source_class.__module__}.{self.source_class.__qualname__}'

    def to_schema_dict(self) -> dict:
        """
        Compile to the existing YAML evaluation-card schema.

        Computed symbols become ``python:`` blocks that import this recipe and
        call back into it. That keeps the export faithful — the emitted card
        runs the recipe's actual functions rather than a copy of their source —
        at the cost of requiring the recipe module to be importable wherever the
        card runs. For a card that already imports a team's predictor package,
        that costs nothing.
        """
        qualname = self._qualified_name()
        if qualname is None:
            raise ValueError('recipe has no source class; cannot be exported')
        module, _, attr = qualname.rpartition('.')
        preamble = f'from {module} import {attr}'

        symbols: dict[str, Any] = {}
        for name, sym in self.symbols.items():
            spec: dict[str, Any] = {}
            if sym.type:
                spec['type'] = sym.type
            metadata = {}
            if sym.display_name is not None:
                metadata['display_name'] = sym.display_name
            if sym.display is not None:
                metadata['display'] = sym.display
            if sym.metric is not None:
                metadata['define_metric'] = sym.metric
            if metadata:
                spec['metadata'] = metadata

            if sym.sweep is not None:
                spec['sweep'] = list(sym.sweep)
            elif sym.is_computed:
                if sym.dependencies:
                    spec['depends_on'] = list(sym.dependencies)
                args = ', '.join(f'{dep}={dep}' for dep in sym.dependencies)
                call = f'{attr}.compute({name!r}{", " if args else ""}{args})'
                spec['python'] = f'{preamble}\n{name} = {call}\n'
            else:
                spec['value'] = sym.value
            symbols[name] = spec

        claim_lines = [preamble]
        for name, claim_obj in self.claims.items():
            args = ', '.join(f'{dep}={dep}' for dep in claim_obj.dependencies)
            claim_lines.append(f'{attr}.check({name!r}{", " if args else ""}{args})')

        out: dict[str, Any] = {
            'title': self.title,
            'description': self.description,
            'version': self.version,
            'organizations': list(self.organizations),
            'submitter': self.submitter,
            'tags': list(self.tags),
            'links': [dict(link) for link in self.links],
            'claim': {'python': '\n'.join(claim_lines) + '\n'},
            'symbols': symbols,
        }
        if self.category:
            out['category'] = self.category
        if self.claim_aggregation_strategy != DEFAULT_CLAIM_AGGREGATION_STRATEGY:
            out['claim_aggregation_strategy'] = dict(self.claim_aggregation_strategy)
        return out

    def to_schema(self):
        """
        Validate the export against :class:`magnet.schema.EvaluationCardSchema`.

        Requires a MAGNET that carries the card schema module; older ones
        validate cards only implicitly, by running them.
        """
        try:
            from magnet.schema import EvaluationCardSchema
        except ImportError as ex:
            raise ImportError(
                'magnet.schema is unavailable in this MAGNET; '
                'use to_schema_dict() and validate the card by running it'
            ) from ex

        return EvaluationCardSchema.model_validate(self.to_schema_dict())

    def to_yaml(self) -> str:
        """Render the compiled card as YAML."""
        import yaml

        class _Dumper(yaml.SafeDumper):
            pass

        def _str_representer(dumper, data):
            # Emit the generated `python:` bodies as literal blocks; an exported
            # card is meant to be read and reviewed like a written one.
            style = '|' if '\n' in data else None
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style=style)

        _Dumper.add_representer(str, _str_representer)
        return yaml.dump(self.to_schema_dict(), Dumper=_Dumper, sort_keys=False)

    def write_card(self, path) -> None:
        """Write the compiled YAML card."""
        import ubelt as ub

        path = ub.Path(path)
        path.parent.ensuredir()
        path.write_text(self.to_yaml())


def recipe(
    title: str = '',
    description: str = '',
    version: str = '',
    category: str | None = None,
    organizations: Iterable[str] = (),
    submitter: dict[str, str] | None = None,
    tags: Iterable[str] = (),
    links: Iterable[dict] = (),
    claim_aggregation_strategy: dict | None = None,
):
    """
    Turn a class body into an :class:`EvaluationRecipe`.

    The class is a declaration, not a namespace to instantiate: the decorator
    reads it and returns the recipe, so the name it was defined under refers to
    the recipe afterwards.

    Members are read as follows:

    ``@symbol`` function
        a computed symbol; its parameters are its dependencies
    ``@claim`` function
        an assertion over resolved symbols
    :class:`Sweep` attribute
        a symbol taking each value in turn
    any other plain attribute
        a constant symbol

    Undecorated methods are ignored, so a recipe can keep helpers next to the
    symbols that use them.

    Example:
        >>> @recipe(title='Threshold', version='1.0')
        ... class Simple:
        ...     threshold = 0.5
        ...
        ...     @symbol
        ...     def score():
        ...         return 0.25
        ...
        ...     @claim
        ...     def under(score, threshold):
        ...         assert score < threshold
        >>> Simple.symbols['threshold'].value
        0.5
        >>> Simple.claims['under'].dependencies
        ('score', 'threshold')
    """

    def _decorate(cls):
        symbols: dict[str, Symbol] = {}
        claims: dict[str, Claim] = {}

        for name, member in vars(cls).items():
            if name.startswith('_'):
                continue
            if isinstance(member, (staticmethod, classmethod)):
                member = member.__func__

            if callable(member) and hasattr(member, '__magnet_claim__'):
                meta = member.__magnet_claim__
                claims[meta['name'] or name] = Claim(
                    name=meta['name'] or name,
                    func=member,
                    dependencies=_parameters(member),
                )
            elif callable(member) and hasattr(member, '__magnet_symbol__'):
                meta = member.__magnet_symbol__
                symbols[name] = Symbol(
                    name=name,
                    func=member,
                    type=meta['type'],
                    display_name=meta['display_name'],
                    display=meta['display'],
                    metric=meta['metric'],
                    dependencies=_parameters(member),
                )
            elif isinstance(member, Sweep):
                symbols[name] = Symbol(name=name, sweep=member.values, type=member.type)
            elif not callable(member):
                symbols[name] = Symbol(name=name, value=member, type=_type_name(member))

        obj = EvaluationRecipe(
            title=title or cls.__name__,
            description=description or (inspect.getdoc(cls) or ''),
            version=version,
            category=category,
            organizations=tuple(organizations),
            submitter=submitter,
            tags=tuple(tags),
            links=tuple(links),
            claim_aggregation_strategy=(
                dict(claim_aggregation_strategy)
                if claim_aggregation_strategy
                else dict(DEFAULT_CLAIM_AGGREGATION_STRATEGY)
            ),
            symbols=symbols,
            claims=claims,
            source_class=cls,
        )
        obj.static_order()  # fail at definition time on a bad graph, not at run time
        return obj

    return _decorate


def _parameters(func: Callable) -> tuple[str, ...]:
    """Positional parameter names, which are a function's symbol dependencies."""
    signature = inspect.signature(func)
    return tuple(
        name
        for name, param in signature.parameters.items()
        if param.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )


def _type_name(value: Any) -> str | None:
    if isinstance(value, bool):
        return 'bool'
    if isinstance(value, (int, float, str)):
        return type(value).__name__
    return None


def _reduce(verdicts: Sequence[Verdict], strategy: dict) -> Verdict:
    """
    Reduce per-sweep-point verdicts to one card-level verdict.

    Mirrors ``magnet.evaluation._reduce_results``; ``tests/test_recipe.py``
    asserts the two agree, since a recipe and its exported YAML card giving
    different answers would be worse than either being wrong.
    """
    total = len(verdicts)
    if total == 0:
        return Verdict.INCONCLUSIVE

    verified = sum(v is Verdict.VERIFIED for v in verdicts)
    falsified = sum(v is Verdict.FALSIFIED for v in verdicts)
    inconclusive = sum(v is Verdict.INCONCLUSIVE for v in verdicts)

    kind = strategy.get('type', 'all')
    if kind == 'all':
        if falsified:
            return Verdict.FALSIFIED
        if inconclusive:
            return Verdict.INCONCLUSIVE
        return Verdict.VERIFIED
    if kind == 'any':
        if verified:
            return Verdict.VERIFIED
        if inconclusive:
            return Verdict.INCONCLUSIVE
        return Verdict.FALSIFIED
    if kind == 'fraction':
        threshold = (strategy.get('parameters') or {}).get('threshold')
        if threshold is None:
            raise ValueError('reduce type=fraction requires `threshold`')
        return Verdict.VERIFIED if verified / total >= threshold else Verdict.FALSIFIED
    raise ValueError(f'Unknown reduce type: {kind!r}')
