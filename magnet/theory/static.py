"""
Read theory annotations out of Python source without importing it.

A recognized annotation is a decorator or ``with`` item on the namespace bound
by ``magnet.theory`` or the standalone ``magnet_theory`` package. References and
optional notes must be literal strings. Once a recognized relation is used as
an annotation, malformed arguments are errors rather than annotations that
disappear from the report.
"""
import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from magnet_theory import PREMISE_RELATIONS, RELATIONS, STATEMENT_RELATIONS
from magnet.theory.links import Link, split_ref

__all__ = ['Link', 'TheoryAnnotationError', 'extract', 'extract_tree']

THEORY_MODULES = ('magnet.theory', 'magnet_theory')


class TheoryAnnotationError(ValueError):
    """A recognized theory annotation has an invalid static form."""


@dataclass
class _Namespaces:
    """Names bound to the theory annotation namespace in one source file."""

    aliases: set[str] = field(default_factory=set)

    def visit_import(self, node: ast.AST) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in THEORY_MODULES:
                    self.aliases.add(alias.asname or alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module == 'magnet':
                for alias in node.names:
                    if alias.name == 'theory':
                        self.aliases.add(alias.asname or 'theory')
            # A vendored annotation module naturally lives inside the team's
            # package, e.g. ``from .. import magnet_theory as theory``.
            for alias in node.names:
                if alias.name == 'magnet_theory':
                    self.aliases.add(alias.asname or 'magnet_theory')

    def relation_name(self, call: ast.Call) -> str | None:
        func = call.func
        if not isinstance(func, ast.Attribute) or func.attr not in RELATIONS:
            return None

        value = func.value
        if isinstance(value, ast.Name) and value.id in self.aliases:
            return func.attr

        # ``magnet.theory.tests(...)`` spelled out in full.
        if isinstance(value, ast.Attribute) and value.attr == 'theory':
            if isinstance(value.value, ast.Name) and value.value.id == 'magnet':
                return func.attr
        return None


def _annotation_error(call: ast.Call, fpath: str, message: str) -> TheoryAnnotationError:
    return TheoryAnnotationError(f'{fpath}:{call.lineno}: {message}')


def _parse_annotation(call: ast.Call, relation: str, fpath: str) -> Link:
    if len(call.args) != 1:
        raise _annotation_error(
            call,
            fpath,
            f'theory.{relation} requires exactly one literal reference argument',
        )

    first = call.args[0]
    if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
        raise _annotation_error(
            call,
            fpath,
            f'theory.{relation} reference must be a literal string',
        )
    ref = first.value

    note = ''
    for keyword in call.keywords:
        if keyword.arg != 'note':
            name = '**kwargs' if keyword.arg is None else keyword.arg
            raise _annotation_error(
                call,
                fpath,
                f'theory.{relation} has unsupported keyword {name!r}',
            )
        value = keyword.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            raise _annotation_error(
                call,
                fpath,
                f'theory.{relation} note must be a literal string',
            )
        note = value.value

    try:
        _, premise_id = split_ref(ref)
    except ValueError as ex:
        raise _annotation_error(call, fpath, str(ex)) from ex

    if premise_id is None and relation not in STATEMENT_RELATIONS:
        raise _annotation_error(
            call,
            fpath,
            f'theory.{relation} targets a premise; use an EntryId::binder reference',
        )
    if premise_id is not None and relation not in PREMISE_RELATIONS:
        raise _annotation_error(
            call,
            fpath,
            f'theory.{relation} targets a whole statement; remove ::binder',
        )

    return Link(relation=relation, ref=ref, note=note)


def extract_tree(tree: ast.AST, fpath: str) -> list[Link]:
    """Collect valid theory annotations in one parsed Python module."""
    namespaces = _Namespaces()
    for node in ast.walk(tree):
        namespaces.visit_import(node)
    if not namespaces.aliases:
        return []

    links: list[Link] = []

    def record(call: ast.Call, qualname: str) -> None:
        relation = namespaces.relation_name(call)
        if relation is None:
            return
        link = _parse_annotation(call, relation, fpath)
        links.append(
            Link(
                relation=link.relation,
                ref=link.ref,
                note=link.note,
                file=fpath,
                line=call.lineno,
                qualname=qualname,
            )
        )

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
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


def _display_path(
    path: Path, raw: Path, resolved_source: Path, root: Path
) -> str:
    if raw.is_absolute():
        return os.path.relpath(path, root).replace(os.sep, '/')
    if resolved_source.is_dir():
        rel = path.relative_to(resolved_source)
        return str(raw / rel).replace(os.sep, '/')
    return str(raw).replace(os.sep, '/')


def _python_files(
    paths: Sequence[str], root: str | os.PathLike[str]
) -> Iterator[tuple[Path, str]]:
    root_path = Path(root).resolve()
    for raw_text in paths:
        raw = Path(raw_text)
        path = raw if raw.is_absolute() else root_path / raw
        path = path.resolve()
        if path.is_dir():
            for fpath in sorted(path.rglob('*.py')):
                yield fpath, _display_path(fpath, raw, path, root_path)
        elif path.suffix == '.py':
            yield path, _display_path(path, raw, path, root_path)
        else:
            raise ValueError(
                f'empirical source {raw_text!r} is neither a Python file nor directory'
            )


def extract(
    paths: Sequence[str], *, root: str | os.PathLike[str] = '.'
) -> list[Link]:
    """Collect static theory links from explicitly declared empirical sources."""
    links: list[Link] = []
    for fpath, display_path in _python_files(paths, root):
        tree = ast.parse(fpath.read_text(), filename=str(fpath))
        links.extend(extract_tree(tree, display_path))
    links.sort(key=lambda link: (link.file, link.line or 0))
    return links
