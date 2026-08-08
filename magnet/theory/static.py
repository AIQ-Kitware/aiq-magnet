"""
Reading the assumption ledger out of source, without executing it.

The runtime registry only knows what has been imported, which makes it the
wrong authority on what a codebase declares. Teams annotate repositories whose
dependencies we cannot always install, in environments where importing their
predictor would pull in a serving stack. And an annotation that lives behind an
``if`` nobody took is still a declaration.

So the ledger is extracted by parsing. This module finds every predicate call in
a file or tree, in any of its four syntactic positions, resolves the hypothesis
reference where it can, and *reports* what it cannot resolve rather than
dropping it -- a team gets told "line 234 declares an edge we could not read",
which is the whole point of doing this statically.

The one API consequence: references should be **string literals**. The object
form (``QE['hcover']``) was validated at import; a string is validated by
:func:`lint` against a theorem index, which is the same check done earlier and
without needing the code to run. Simple module-level bindings are resolved too,
so an existing object-style codebase is not left out.

Example:
    >>> source = '''
    ... from magnet.theory import approximates, assumes
    ...
    ... assumes('Paper.main::hlipschitz', informal='never tested')
    ...
    ... @approximates('Paper.main::hcover', severity='high')
    ... def build_pool(num_example_runs=64):
    ...     ...
    ... '''
    >>> ledger = extract_source(source, filename='predictor.py')
    >>> for found in ledger.annotations:
    ...     print(found.predicate, found.ref, found.form)
    assumes Paper.main::hlipschitz bare
    approximates Paper.main::hcover decorator
    >>> ledger.annotations[1].options['severity']
    'high'
"""
import ast
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

from magnet.theory.predicates import PREDICATE_NAMES

__all__ = [
    'ExtractedAnnotation',
    'StaticLedger',
    'Issue',
    'extract_source',
    'extract_file',
    'extract_tree',
    'lint',
    'THEORY_MODULES',
    'SKIP_DIRECTORIES',
]

#: Modules a predicate may be imported from. ``magnet_theory`` is the
#: dependency-free shim teams vendor; both spell the same API.
THEORY_MODULES = frozenset(
    {
        'magnet.theory',
        'magnet.theory.predicates',
        'magnet_theory',
    }
)

SKIP_DIRECTORIES = frozenset(
    {
        '.git',
        '.hg',
        '.tox',
        '.venv',
        'venv',
        '__pycache__',
        'node_modules',
        'build',
        'dist',
        '.mypy_cache',
        '.pytest_cache',
        '.ruff_cache',
        'site-packages',
    }
)


@dataclass
class ExtractedAnnotation:
    """One predicate call found in source."""

    predicate: str
    ref: str | None
    ref_expr: str
    form: str
    file: str | None = None
    line: int = 0
    qualname: str = ''
    target: str | None = None
    options: dict = field(default_factory=dict)
    unreadable_options: tuple[str, ...] = ()

    @property
    def resolved(self) -> bool:
        return self.ref is not None

    @property
    def declaration(self) -> str | None:
        if self.ref is None:
            return None
        return self.ref.split('::', 1)[0]

    @property
    def hypothesis(self) -> str | None:
        if self.ref is None or '::' not in self.ref:
            return None
        return self.ref.split('::', 1)[1]

    @property
    def site(self) -> str:
        where = self.qualname or '<module>'
        return f'{where}:{self.line}'

    def to_dict(self) -> dict:
        return {
            'predicate': self.predicate,
            'ref': self.ref,
            'ref_expr': self.ref_expr,
            'resolved': self.resolved,
            'form': self.form,
            'file': self.file,
            'line': self.line,
            'qualname': self.qualname,
            'target': self.target,
            'options': self.options,
            'unreadable_options': list(self.unreadable_options),
        }


@dataclass
class StaticLedger:
    """Everything a source tree declares about its relationship to theory."""

    annotations: list[ExtractedAnnotation] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def edges(self) -> list[ExtractedAnnotation]:
        return [a for a in self.annotations if a.predicate != 'grounds']

    @property
    def groundings(self) -> list[ExtractedAnnotation]:
        return [a for a in self.annotations if a.predicate == 'grounds']

    @property
    def unresolved(self) -> list[ExtractedAnnotation]:
        return [a for a in self.annotations if not a.resolved]

    def declarations(self) -> set[str]:
        return {a.declaration for a in self.annotations if a.declaration}

    def extend(self, other: 'StaticLedger') -> 'StaticLedger':
        self.annotations.extend(other.annotations)
        self.errors.extend(other.errors)
        return self

    def to_dict(self) -> dict:
        return {
            'annotations': [a.to_dict() for a in self.annotations],
            'errors': [{'file': f, 'error': e} for f, e in self.errors],
        }

    def __iter__(self) -> Iterator[ExtractedAnnotation]:
        return iter(self.annotations)

    def __len__(self) -> int:
        return len(self.annotations)


def extract_source(source: str, filename: str | None = None) -> StaticLedger:
    """Extract from a source string."""
    ledger = StaticLedger()
    try:
        tree = ast.parse(source, filename=filename or '<string>')
    except SyntaxError as ex:
        ledger.errors.append((filename or '<string>', f'SyntaxError: {ex}'))
        return ledger

    aliases = _predicate_aliases(tree)
    if not aliases.names and not aliases.modules:
        return ledger  # nothing here imports the API

    bindings = _statement_bindings(tree)
    parents = _parent_map(tree)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        predicate = aliases.predicate_of(node.func)
        if predicate is None:
            continue
        ledger.annotations.append(
            _build(node, predicate, bindings, parents, filename)
        )
    ledger.annotations.sort(key=lambda a: a.line)
    return ledger


def extract_file(path) -> StaticLedger:
    """Extract from one file."""
    import ubelt as ub

    path = ub.Path(path)
    try:
        source = path.read_text()
    except (OSError, UnicodeDecodeError) as ex:
        ledger = StaticLedger()
        ledger.errors.append((str(path), f'{type(ex).__name__}: {ex}'))
        return ledger
    return extract_source(source, filename=str(path))


def extract_tree(root, skip: Iterable[str] = SKIP_DIRECTORIES) -> StaticLedger:
    """
    Extract from every ``.py`` file under ``root``.

    Skips the usual noise directories; a vendored dependency's annotations are
    not the audited repository's declarations.
    """
    import ubelt as ub

    root = ub.Path(root)
    skip = set(skip)
    ledger = StaticLedger()
    if root.is_file():
        return extract_file(root)
    for path in sorted(root.rglob('*.py')):
        if any(part in skip for part in path.parts):
            continue
        ledger.extend(extract_file(path))
    return ledger


# --------------------------------------------------------------------- issues


@dataclass
class Issue:
    """One problem found by :func:`lint`."""

    kind: str
    message: str
    annotation: ExtractedAnnotation | None = None

    def __str__(self) -> str:
        where = f'{self.annotation.file}:{self.annotation.line}: ' if self.annotation else ''
        return f'{where}{self.kind}: {self.message}'

    def to_dict(self) -> dict:
        return {
            'kind': self.kind,
            'message': self.message,
            'file': self.annotation.file if self.annotation else None,
            'line': self.annotation.line if self.annotation else None,
        }


def lint(ledger: StaticLedger, formalizations: Sequence = ()) -> list[Issue]:
    """
    Check a ledger, optionally against loaded formalizations.

    This is where the validation the object form used to do at import time
    happens instead -- and it does more, because it can see a reference that
    names a declaration which no longer exists, which no amount of importing
    would reveal.

    Without formalizations it still catches the errors that need no index:
    unreadable references, edges that name a theorem but no binder, and
    duplicate ids.
    """
    issues: list[Issue] = []

    for path, error in ledger.errors:
        issues.append(Issue('unparseable', f'{path}: {error}'))

    for annotation in ledger.annotations:
        if not annotation.resolved:
            issues.append(
                Issue(
                    'unresolved-reference',
                    f'cannot read the reference {annotation.ref_expr!r} statically; '
                    f'use a string literal such as "Some.Declaration::binder"',
                    annotation,
                )
            )
            continue
        if annotation.predicate != 'grounds' and annotation.hypothesis is None:
            issues.append(
                Issue(
                    'missing-binder',
                    f'{annotation.predicate}({annotation.ref!r}) names a statement but no '
                    f'hypothesis; edges attach to a binder',
                    annotation,
                )
            )

    seen: dict[str, ExtractedAnnotation] = {}
    for annotation in ledger.annotations:
        identifier = annotation.options.get('id')
        if not identifier:
            continue
        if identifier in seen:
            issues.append(
                Issue(
                    'duplicate-id',
                    f'id {identifier!r} is also used at '
                    f'{seen[identifier].file}:{seen[identifier].line}',
                    annotation,
                )
            )
        else:
            seen[identifier] = annotation

    if formalizations:
        issues.extend(_lint_against(ledger, formalizations))
    return issues


def _lint_against(ledger: StaticLedger, formalizations: Sequence) -> list[Issue]:
    issues: list[Issue] = []
    for annotation in ledger.annotations:
        if not annotation.resolved:
            continue
        declaration = annotation.declaration
        holder = next((f for f in formalizations if declaration in f), None)
        if holder is None:
            issues.append(
                Issue(
                    'unknown-declaration',
                    f'{declaration!r} is not in any loaded formalization',
                    annotation,
                )
            )
            continue
        theorem = holder[declaration]
        binder = annotation.hypothesis
        if binder is None or not theorem.hypotheses_enumerated:
            continue
        if binder not in {h.name for h in theorem.hypotheses}:
            known = ', '.join(h.name for h in theorem.hypotheses)
            issues.append(
                Issue(
                    'unknown-binder',
                    f'{declaration} has no hypothesis {binder!r}; known: {known}',
                    annotation,
                )
            )
    return issues


# ------------------------------------------------------------------ internals


@dataclass
class _Aliases:
    """Local names under which the predicates are reachable in one module."""

    names: dict = field(default_factory=dict)  # local name -> predicate
    modules: set = field(default_factory=set)  # local name for a theory module

    def predicate_of(self, func: ast.expr) -> str | None:
        if isinstance(func, ast.Name):
            return self.names.get(func.id)
        if isinstance(func, ast.Attribute) and func.attr in PREDICATE_NAMES:
            base = _dotted(func.value)
            if base is not None and base in self.modules:
                return func.attr
        return None


def _predicate_aliases(tree: ast.AST) -> _Aliases:
    """
    Map local names back to predicates.

    Handles ``from magnet.theory import assumes``, ``... as skips``, and
    ``import magnet.theory as th`` followed by ``th.assumes(...)``.
    """
    aliases = _Aliases()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module in THEORY_MODULES:
                for alias in node.names:
                    if alias.name in PREDICATE_NAMES:
                        aliases.names[alias.asname or alias.name] = alias.name
                    elif alias.name == 'predicates':
                        aliases.modules.add(alias.asname or alias.name)
            elif node.module and node.module.rsplit('.', 1)[0] in THEORY_MODULES:
                for alias in node.names:
                    if alias.name in PREDICATE_NAMES:
                        aliases.names[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in THEORY_MODULES:
                    # `import magnet.theory` binds `magnet`; `as th` binds `th`.
                    aliases.modules.add(alias.asname or alias.name)
                    if alias.asname is None:
                        aliases.modules.add(alias.name)
    return aliases


def _statement_bindings(tree: ast.AST) -> dict:
    """
    Resolve module-level names that hold a declaration.

    Supports the object-style spellings that appear in code written before the
    string form -- ``QE = DKPS['Some.Decl']`` and ``QE = DKPS.theorem('Some.Decl')``
    -- so ``QE['hcover']`` can still be read out of source.
    """
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if value is None:
            continue
        declaration = None
        if isinstance(value, ast.Subscript) and isinstance(value.slice, ast.Constant):
            if isinstance(value.slice.value, str):
                declaration = value.slice.value
        elif isinstance(value, ast.Call) and value.args:
            func = value.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, 'id', None)
            if name in {'theorem', 'resolve', 'statement'}:
                first = value.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    declaration = first.value
        elif isinstance(value, ast.Constant) and isinstance(value.value, str):
            declaration = value.value
        if declaration is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                bindings[target.id] = declaration
    return bindings


def _build(
    node: ast.Call,
    predicate: str,
    bindings: dict,
    parents: dict,
    filename: str | None,
) -> ExtractedAnnotation:
    ref_expr = ast.unparse(node.args[0]) if node.args else ''
    ref = _resolve_ref(node.args[0], bindings) if node.args else None

    options: dict = {}
    unreadable: list[str] = []
    # The first positional is the reference; the second, where given, is
    # severity, matching the predicate signatures.
    if len(node.args) > 1:
        value, ok = _literal(node.args[1])
        if ok:
            options['severity'] = value
    for keyword in node.keywords:
        if keyword.arg is None:
            unreadable.append('**kwargs')
            continue
        value, ok = _literal(keyword.value)
        if ok:
            options[keyword.arg] = value
        else:
            unreadable.append(keyword.arg)

    form, target = _form_of(node, parents)
    if ref is None and form == 'bare' and predicate == 'grounds':
        pass  # a grounding may legitimately name only a declaration

    return ExtractedAnnotation(
        predicate=predicate,
        ref=ref,
        ref_expr=ref_expr,
        form=form,
        file=filename,
        line=node.lineno,
        qualname=_qualname(node, parents),
        target=target,
        options=options,
        unreadable_options=tuple(unreadable),
    )


def _resolve_ref(node: ast.expr, bindings: dict) -> str | None:
    """
    Read a hypothesis reference out of an expression.

    Handles the string literal; an f-string or ``+`` concatenation built from
    module-level string constants, which is how anyone writing more than two
    edges against a ninety-character declaration name will actually spell it;
    the module-level binding subscripted by a binder name; and an explicit
    ``.hypothesis('name')`` call. Anything else is left unresolved and reported.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, (ast.JoinedStr, ast.BinOp, ast.Name)):
        return _join_string(node, bindings)
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        binder = node.slice.value
        base = _dotted(node.value)
        if isinstance(binder, str) and base is not None:
            declaration = bindings.get(base)
            if declaration is not None:
                return f'{declaration}::{binder}'
            # A direct `FORMALIZATION['Some.Decl']` names a statement, not a
            # binder -- valid for `grounds`.
            if '.' in binder:
                return binder
    if isinstance(node, ast.Call) and node.args:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == 'hypothesis':
            first = node.args[0]
            base = _dotted(func.value)
            if isinstance(first, ast.Constant) and isinstance(first.value, str) and base:
                declaration = bindings.get(base)
                if declaration is not None:
                    return f'{declaration}::{first.value}'
    return None


def _join_string(node: ast.expr, bindings: dict) -> str | None:
    """
    Fold an f-string or ``+`` chain of string constants into one value.

    Only constants and module-level names bound to string constants
    participate; anything else makes the whole expression unresolvable, which is
    the honest answer rather than a partial reference.

    Example:
        >>> tree = ast.parse("f'{DECL}::hgap'", mode='eval')
        >>> _join_string(tree.body, {'DECL': 'Paper.main'})
        'Paper.main::hgap'
    """
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.Name):
        return bindings.get(node.id)
    if isinstance(node, ast.FormattedValue):
        if node.format_spec is not None or node.conversion not in (-1, None):
            return None
        return _join_string(node.value, bindings)
    if isinstance(node, ast.JoinedStr):
        parts = [_join_string(v, bindings) for v in node.values]
        return None if any(p is None for p in parts) else ''.join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _join_string(node.left, bindings)
        right = _join_string(node.right, bindings)
        return None if left is None or right is None else left + right
    return None


def _literal(node: ast.expr):
    """Evaluate a literal argument; ``(value, ok)``."""
    try:
        return ast.literal_eval(node), True
    except (ValueError, SyntaxError, TypeError):
        return None, False


def _parent_map(tree: ast.AST) -> dict:
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _ancestors(node: ast.AST, parents: dict) -> Iterator[ast.AST]:
    current = parents.get(node)
    while current is not None:
        yield current
        current = parents.get(current)


def _form_of(node: ast.Call, parents: dict) -> tuple[str, str | None]:
    """
    Which syntactic position the call occupies.

    ``decorator``, ``with``, ``annotation``, or ``bare``. The distinction is
    reported rather than normalized away because it tells a reader how the edge
    behaves at run time -- whether it observes, and what it witnesses.
    """
    previous: ast.AST = node
    for parent in _ancestors(node, parents):
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if any(previous is dec for dec in parent.decorator_list):
                return 'decorator', parent.name
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if parent.returns is not None and previous is parent.returns:
                    return 'annotation', parent.name
            return 'bare', None
        if isinstance(parent, ast.withitem):
            return 'with', None
        if isinstance(parent, ast.arg):
            return 'annotation', parent.arg
        if isinstance(parent, ast.AnnAssign):
            if previous is parent.annotation:
                name = parent.target.id if isinstance(parent.target, ast.Name) else None
                return 'annotation', name
            return 'bare', None
        previous = parent
    return 'bare', None


def _qualname(node: ast.AST, parents: dict) -> str:
    """Dotted names of the enclosing definitions, outermost first."""
    names = []
    for parent in _ancestors(node, parents):
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(parent.name)
    return '.'.join(reversed(names))


def _dotted(node: ast.expr) -> str | None:
    """Render a dotted name expression, or None if it is not one."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f'{base}.{node.attr}' if base else None
    return None
