"""The static annotation vocabulary and versioned theory index."""
import ast
import textwrap
from pathlib import Path

import pytest

import magnet.theory as theory
import magnet_theory
from magnet.theory.index import Entry, Premise, TheoryIndex, load_index
from magnet.theory.static import TheoryAnnotationError, extract, extract_tree


def test_annotations_are_runtime_inert():
    @theory.tests('A.Statement')
    @theory.assumes('A.Statement::hpremise')
    def experiment(value):
        return value * 2

    assert experiment(21) == 42
    with theory.checks('A.Statement::hpremise') as link:
        result = 1 + 1
    assert result == 2
    assert (link.relation, link.ref) == ('checks', 'A.Statement::hpremise')


def test_annotation_package_is_dependency_free_and_compatible():
    source = Path(magnet_theory.__file__).read_text()
    assert 'import magnet' not in source
    assert getattr(magnet_theory.tests, '__test__') is False
    assert magnet_theory.satisfies('A::h').ref == 'A::h'
    assert theory.tests is magnet_theory.tests
    assert theory.assumes is magnet_theory.assumes


SOURCE = textwrap.dedent(
    '''
    import magnet.theory as theory

    @theory.tests('Examples.Stability.Theorem')
    @theory.satisfies('Examples.Stability.Theorem::hbounded')
    def exact(n):
        with theory.assumes(
                'Examples.Stability.Theorem::hiid',
                note='sampling is treated as IID'):
            return n

    class Estimator:
        @theory.approximates(
            'Examples.Stability.Population', note='finite sample')
        def estimate(self):
            with theory.checks('Examples.Stability.Population::hpositive'):
                return 0
    '''
)


def test_extraction_records_statement_and_premise_links():
    links = extract_tree(ast.parse(SOURCE), 'demo.py')
    found = [(link.relation, link.ref, link.qualname) for link in links]
    assert found == [
        ('tests', 'Examples.Stability.Theorem', 'exact'),
        ('satisfies', 'Examples.Stability.Theorem::hbounded', 'exact'),
        ('assumes', 'Examples.Stability.Theorem::hiid', 'exact'),
        ('approximates', 'Examples.Stability.Population', 'Estimator.estimate'),
        ('checks', 'Examples.Stability.Population::hpositive', 'Estimator.estimate'),
    ]
    assert links[0].target_kind == 'statement'
    assert links[1].target_kind == 'premise'
    assert links[2].note == 'sampling is treated as IID'


@pytest.mark.parametrize(
    'import_line',
    [
        'import magnet_theory as theory',
        'from .. import magnet_theory as theory',
        'from magnet import theory',
    ],
)
def test_supported_annotation_namespace_imports(import_line):
    source = SOURCE.replace('import magnet.theory as theory', import_line)
    links = extract_tree(ast.parse(source), 'demo.py')
    assert [link.relation for link in links] == [
        'tests', 'satisfies', 'assumes', 'approximates', 'checks'
    ]


def test_a_file_that_never_imports_theory_is_skipped():
    source = textwrap.dedent(
        '''
        import something_else as theory

        @theory.tests('A.Statement')
        def experiment():
            pass
        '''
    )
    assert extract_tree(ast.parse(source), 'demo.py') == []


@pytest.mark.parametrize(
    'call,match',
    [
        ('theory.tests(REF)', 'literal string'),
        ("theory.tests('A.' + 'b')", 'literal string'),
        ('theory.tests()', 'exactly one'),
        ("theory.tests('A', extra='x')", 'unsupported keyword'),
        ("theory.satisfies('A')", 'EntryId::binder'),
        ("theory.tests('A::h')", 'whole statement'),
    ],
)
def test_malformed_recognized_annotations_are_errors(call, match):
    source = textwrap.dedent(
        f'''
        import magnet.theory as theory
        REF = 'A.Statement'

        @{call}
        def experiment():
            pass
        '''
    )
    with pytest.raises(TheoryAnnotationError, match=match):
        extract_tree(ast.parse(source), 'demo.py')


def test_unknown_relation_and_bare_calls_are_not_annotations():
    source = textwrap.dedent(
        '''
        import magnet.theory as theory

        @theory.believes('A.Statement')
        def experiment():
            theory.tests('A.Statement')
        '''
    )
    assert extract_tree(ast.parse(source), 'demo.py') == []


def test_extract_is_strict_and_records_declared_relative_paths(tmp_path):
    source_dpath = tmp_path / 'src'
    source_dpath.mkdir()
    (source_dpath / 'annotated.py').write_text(SOURCE)
    links = extract(['src'], root=tmp_path)
    assert links[0].file == 'src/annotated.py'

    (source_dpath / 'broken.py').write_text('def (:\n')
    with pytest.raises(SyntaxError):
        extract(['src'], root=tmp_path)


def test_versioned_index_resolves_entries_and_premises(tmp_path):
    fpath = tmp_path / 'theory.yaml'
    fpath.write_text(
        textwrap.dedent(
            '''
            schema_version: 1
            formalization:
              system: lean4
              repository: https://example.invalid/formalization.git
              revision: deadbeef
            entries:
              - id: Examples.Stability.Theorem
                kind: theorem
                declaration: MagnetExamples.Stability.theorem
                source_path: MagnetExamples/Stability.lean
                premises:
                  - id: hbounded
                    type: Bounded xs
                  - id: hiid
                    statement: samples are independent and identically distributed
            '''
        )
    )
    index = load_index(fpath)
    entry = index['Examples.Stability.Theorem']
    assert entry.formalization is not None
    assert entry.formalization.system == 'lean4'
    assert entry.formalization.revision == 'deadbeef'
    premise = index.resolve('Examples.Stability.Theorem::hbounded')
    assert isinstance(premise, Premise)
    assert premise.type == 'Bounded xs'
    assert index.unresolved(
        ['Examples.Stability.Theorem::hbounded', 'Examples.Stability.Theorem::hmissing']
    ) == ['Examples.Stability.Theorem::hmissing']


def test_index_schema_version_is_required(tmp_path):
    fpath = tmp_path / 'theory.yaml'
    fpath.write_text(
        'entries:\n  - id: A.Statement\n    kind: theorem\n    statement: demo\n'
    )
    with pytest.raises(ValueError, match='schema_version'):
        load_index(fpath)


def test_duplicate_entry_and_premise_ids_are_rejected(tmp_path):
    duplicate_entries = tmp_path / 'entries.yaml'
    duplicate_entries.write_text(
        textwrap.dedent(
            '''
            schema_version: 1
            entries:
              - {id: A.Statement, kind: theorem, statement: one}
              - {id: A.Statement, kind: theorem, statement: two}
            '''
        )
    )
    with pytest.raises(ValueError, match='duplicate theory entry'):
        load_index(duplicate_entries)

    duplicate_premises = tmp_path / 'premises.yaml'
    duplicate_premises.write_text(
        textwrap.dedent(
            '''
            schema_version: 1
            entries:
              - id: A.Statement
                kind: theorem
                statement: demo
                premises:
                  - {id: h}
                  - {id: h}
            '''
        )
    )
    with pytest.raises(ValueError, match='duplicate premise'):
        load_index(duplicate_premises)


def test_entry_kind_and_identity_are_explicit(tmp_path):
    missing_kind = tmp_path / 'missing-kind.yaml'
    missing_kind.write_text(
        'schema_version: 1\nentries:\n  - id: A.Statement\n    statement: demo\n'
    )
    with pytest.raises(ValueError, match='kind None'):
        load_index(missing_kind)

    empty = tmp_path / 'empty.yaml'
    empty.write_text(
        'schema_version: 1\nentries:\n  - id: A.Statement\n    kind: theorem\n'
    )
    with pytest.raises(ValueError, match='statement or declaration'):
        load_index(empty)


def test_index_membership():
    index = TheoryIndex([Entry('A.Statement', 'conjecture', statement='demo')])
    assert 'A.Statement' in index
    assert 'A.Missing' not in index
