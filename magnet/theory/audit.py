"""
Audit a repository's theory annotations without running any of it.

Parses a source tree, resolves what it declares against one or more theorem
indexes, and prints the assumption ledger: which hypotheses are discharged,
which are relaxed and how badly, and which nobody has looked at.

Nothing here withholds or downgrades a verdict. The output is context to hang
next to one -- and a work list, since every ``assumes`` in it is an invitation
to write a :func:`~magnet.theory.predicates.checks` and find out.

CommandLine:
    python -m magnet.theory.audit --source path/to/team/repo \\
        --index theory/indexes/some-formalization.yaml

    python -m magnet.theory.audit --source . --index idx.yaml --format json
"""
import sys

import kwconf
import ubelt as ub

__all__ = ['TheoryAuditCLI', 'audit']


class TheoryAuditCLI(kwconf.Config):
    """
    Report what a codebase assumes, and what proves it.
    """

    __command__ = 'audit'

    source: list[str] = kwconf.Value(
        '.',
        position=1,
        nargs='+',
        help='source file or directory to parse; repeatable',
        tags=['in_path'],
    )
    index: list[str] | None = kwconf.Value(
        None,
        nargs='+',
        help=ub.paragraph(
            """
            Theorem index or formalization.yaml to resolve references against;
            repeatable. Without one, references cannot be checked and coverage
            is reported as unknown.
            """
        ),
        tags=['in_path'],
    )
    site_root: list[str] | None = kwconf.Value(
        None,
        nargs='+',
        help=ub.paragraph(
            """
            ``package=directory`` mapping used to check that externally
            declared ``site=`` references still point at real code; repeatable.
            An edge declared away from its code site carries a string, and
            nothing about a string keeps it true -- this is what catches it
            going stale.
            """
        ),
    )
    format: str = kwconf.Value(
        'text',
        choices=['text', 'json'],
        help='text for a readable report, json for a machine-readable one',
    )
    out: str | None = kwconf.Value(
        None, help='write the report here instead of stdout', tags=['out_path']
    )
    strict: bool = kwconf.Value(
        False,
        isflag=True,
        help=ub.paragraph(
            """
            Exit nonzero if the lint finds problems. Lint problems are broken
            references and unreadable annotations -- defects in the ledger
            itself. Unaccounted hypotheses are never an error; they are the
            report's whole point.
            """
        ),
    )

    @classmethod
    def main(cls, argv=None, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        report = audit(
            sources=_as_list(config['source']),
            indexes=_as_list(config['index']),
            site_roots=_parse_roots(_as_list(config['site_root'])),
        )

        if config['format'] == 'json':
            import json

            text = json.dumps(report.to_dict(), indent=2, ensure_ascii=False, default=str)
        else:
            text = report.render()

        if config['out']:
            path = ub.Path(config['out'])
            path.parent.ensuredir()
            path.write_text(text + '\n')
            print(f'Wrote report to: {path}')
        else:
            print(text)

        if config['strict'] and report.issues:
            sys.exit(1)


class AuditReport:
    """The result of auditing a tree: the ledger, the lint, and the coverage."""

    def __init__(self, ledger, issues, basis, formalizations):
        self.ledger = ledger
        self.issues = issues
        self.basis = basis
        self.formalizations = formalizations

    def to_dict(self) -> dict:
        return {
            'formalizations': [
                {
                    'name': f.name,
                    'repository': f.repository,
                    'commit': f.commit,
                    'review': str(f.review),
                    'statements': len(f),
                }
                for f in self.formalizations
            ],
            'ledger': self.ledger.to_dict(),
            'issues': [i.to_dict() for i in self.issues],
            'basis': self.basis.to_dict(),
        }

    def render(self) -> str:
        lines = []
        for formalization in self.formalizations:
            pin = (formalization.commit or 'unpinned')[:12]
            lines.append(
                f'formalization: {formalization.name}  '
                f'[{len(formalization)} statements, {formalization.review}, pin {pin}]'
            )
        if self.formalizations:
            lines.append('')

        annotations = self.ledger.annotations
        lines.append(
            f'declared: {len(annotations)} annotations '
            f'({len(self.ledger.groundings)} groundings, {len(self.ledger.edges)} edges) '
            f'across {len({a.file for a in annotations})} files'
        )
        by_form: dict[str, int] = {}
        for annotation in annotations:
            by_form[annotation.form] = by_form.get(annotation.form, 0) + 1
        if by_form:
            lines.append('          ' + ', '.join(f'{v} {k}' for k, v in sorted(by_form.items())))
        lines.append('')

        if self.issues:
            lines.append(f'issues: {len(self.issues)}')
            for issue in self.issues:
                lines.append(f'  {issue}')
            lines.append('')

        lines.append(self.basis.coverage().summary())
        lines.append('')
        lines.append(self.basis.render())
        return '\n'.join(lines)


def audit(sources, indexes=(), site_roots=None) -> AuditReport:
    """
    Parse ``sources``, resolve against ``indexes``, and report.

    Args:
        sources: files or directories to parse.
        indexes: theorem indexes or ``formalization.yaml`` files.
        site_roots: package -> directory, for validating declared ``site=``
            references. Omitting it skips that check rather than passing it.
    """
    from magnet.theory.basis import TheoreticalBasis
    from magnet.theory.index import load
    from magnet.theory.static import (
        StaticLedger,
        check_sites,
        extract_tree,
        lint,
    )

    formalizations = [load(path) for path in indexes]

    ledger = StaticLedger()
    for source in sources:
        ledger.extend(extract_tree(source))

    issues = lint(ledger, formalizations)
    if site_roots:
        issues.extend(check_sites(ledger, site_roots))
    basis = TheoreticalBasis.from_ledger(ledger, formalizations)
    return AuditReport(ledger, issues, basis, formalizations)


def _parse_roots(items) -> dict:
    """Parse ``package=directory`` arguments."""
    roots = {}
    for item in items:
        package, sep, directory = str(item).partition('=')
        if not sep:
            raise ValueError(f'--site-root expects package=directory, got {item!r}')
        roots[package.strip()] = directory.strip()
    return roots


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


__cli__ = TheoryAuditCLI

if __name__ == '__main__':
    __cli__.main()
