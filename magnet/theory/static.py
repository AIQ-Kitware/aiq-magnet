"""
Read theory links out of source, without importing it.

One canonical spelling is accepted::

    import magnet.theory as theory        # or: import magnet_theory as theory

    @theory.tests('Literal.Reference')
    def experiment(): ...

    with theory.approximates('Literal.Reference'):
        ...

The reference has to be a literal string, and the call has to be a decorator or
a ``with`` item. Anything else is ignored, which keeps the walk short and keeps
a reader's model of what counts accurate.
"""
import ast
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import ubelt as ub

from magnet.theory.links import RELATIONS

__all__ = ['Link', 'extract', 'extract_tree']

#: Module names that may be aliased to the relation namespace.
THEORY_MODULES = ('magnet.theory', 'magnet_theory')


@dataclass(frozen=True)
class Link:
    """One ``practice <relation> theory`` annotation, as found in source."""

    relation: str
    ref: str
    file: str
    line: int
    qualname: str

    def to_dict(self) -> dict:
        return {
            'relation': self.relation,
            'ref': self.ref,
            'file': self.file,
            'line': self.line,
            'qualname': self.qualname,
        }


@dataclass
class _Namespaces:
    """Names bound to the theory module in this file."""

    aliases: set = field(default_factory=set)

    def visit_import(self, node) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in THEORY_MODULES:
                    self.aliases.add(alias.asname or alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module == 'magnet' and any(
                    a.name == 'theory' for a in node.names):
                for a in node.names:
                    if a.name == 'theory':
                        self.aliases.add(a.asname or 'theory')

    def is_relation_call(self, call: ast.Call) -> str | None:
        """The relation name, when this call is ``<alias>.<relation>(...)``."""
        func = call.func
        if not isinstance(func, ast.Attribute):
            return None
        if func.attr not in RELATIONS:
            return None
        value = func.value
        if isinstance(value, ast.Name) and value.id in self.aliases:
            return func.attr
        # `magnet.theory.tests(...)` spelled out in full.
        if isinstance(value, ast.Attribute) and value.attr == 'theory':
            if isinstance(value.value, ast.Name) and value.value.id == 'magnet':
                return func.attr
        return None


def _literal_ref(call: ast.Call) -> str | None:
    if not call.args:
        return None
    first = call.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def extract_tree(tree: ast.AST, fpath: str) -> list[Link]:
    """
    Collect the links in one parsed module.

    Args:
        tree (ast.AST): a parsed module.
        fpath (str): path recorded on each link.

    Returns:
        list[Link]

    Example:
        >>> import ast
        >>> from magnet.theory.static import extract_tree
        >>> source = ub.codeblock(
        ...     '''
        ...     import magnet.theory as theory
        ...
        ...     @theory.tests('A.b')
        ...     def experiment():
        ...         with theory.motivates('C.d'):
        ...             pass
        ...     ''')
        >>> links = extract_tree(ast.parse(source), 'demo.py')
        >>> [(l.relation, l.ref, l.qualname) for l in links]
        [('tests', 'A.b', 'experiment'), ('motivates', 'C.d', 'experiment')]
    """
    namespaces = _Namespaces()
    for node in ast.walk(tree):
        namespaces.visit_import(node)
    if not namespaces.aliases:
        return []

    links: list[Link] = []

    def record(call: ast.Call, qualname: str) -> None:
        relation = namespaces.is_relation_call(call)
        if relation is None:
            return
        ref = _literal_ref(call)
        if ref is None:
            return
        links.append(Link(relation=relation, ref=ref, file=fpath,
                          line=call.lineno, qualname=qualname))

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)):
                qualname = f'{prefix}.{child.name}' if prefix else child.name
                for decorator in child.decorator_list:
                    if isinstance(decorator, ast.Call):
                        record(decorator, qualname)
                walk(child, qualname)
            elif isinstance(child, (ast.With, ast.AsyncWith)):
                for item in child.items:
                    if isinstance(item.context_expr, ast.Call):
                        record(item.context_expr, prefix)
                walk(child, prefix)
            else:
                walk(child, prefix)

    walk(tree, '')
    return links


def extract(paths: Sequence[str]) -> list[Link]:
    """
    Collect links from files and directories.

    Args:
        paths (Sequence[str]): files, or directories walked for ``*.py``.

    Returns:
        list[Link]: in file then line order.
    """
    links: list[Link] = []
    for fpath in _python_files(paths):
        try:
            tree = ast.parse(fpath.read_text())
        except SyntaxError:
            continue
        links.extend(extract_tree(tree, str(fpath)))
    links.sort(key=lambda link: (link.file, link.line))
    return links


def _python_files(paths: Sequence[str]) -> Iterator[ub.Path]:
    for raw in paths:
        path = ub.Path(raw)
        if path.is_dir():
            yield from sorted(path.rglob('*.py'))
        elif path.suffix == '.py':
            yield path
