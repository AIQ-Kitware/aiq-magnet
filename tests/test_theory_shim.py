"""
Tests for the dependency-free shim teams vendor.

Two things have to hold or the shim is worse than useless: annotated code must
behave identically with the shim or the real package installed, and the shim
must spell exactly the same API, since a team's annotations are read by the
real extractor.
"""
import textwrap

from magnet.theory import shim
from magnet.theory.predicates import PREDICATE_NAMES
from magnet.theory.static import extract_source


def test_shim_spells_the_same_predicates():
    # A predicate that exists in one and not the other is an annotation a team
    # can write and we cannot read, or vice versa.
    assert set(shim.PREDICATES) == set(PREDICATE_NAMES)
    for name in PREDICATE_NAMES:
        assert callable(getattr(shim, name))


def test_shim_imports_nothing_beyond_the_standard_library():
    # Checked against the parsed imports rather than the text, since the
    # docstring necessarily shows `from magnet_theory import ...`.
    import ast
    import sys

    tree = ast.parse(open(shim.__file__).read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split('.')[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split('.')[0])

    assert imported <= sys.stdlib_module_names, f'non-stdlib imports: {imported}'


def test_shim_annotations_are_inert():
    @shim.approximates('Paper.main::hcover', severity='high')
    def build(n=64):
        return n * 2

    assert build(8) == 16
    assert build.__name__ == 'build'


def test_shim_activates_both_ways():
    edge = shim.assumes('Paper.main::h')
    assert edge.ref == 'Paper.main::h'

    @shim.satisfies('Paper.main::h')
    def annotated():
        return 'ok'

    assert annotated() == 'ok'


def test_shim_leaves_classes_alone():
    @shim.assumes('Paper.main::h')
    class Predictor:
        pass

    assert isinstance(Predictor, type)
    assert Predictor.__name__ == 'Predictor'


def test_extractor_reads_shim_annotated_source():
    # The whole point: a team annotates with the vendored shim, and MAGNET
    # reads their source without importing either.
    ledger = extract_source(
        textwrap.dedent(
            '''
            from magnet_theory import approximates, grounds

            @grounds('Paper.claim')
            def claim(): ...

            @approximates('Paper.main::hcover', severity='high')
            def build(n=64): ...
            '''
        ),
        filename='team/predictor.py',
    )
    assert [(a.predicate, a.ref) for a in ledger] == [
        ('grounds', 'Paper.claim'),
        ('approximates', 'Paper.main::hcover'),
    ]


def test_shim_installs_a_copy(tmp_path):
    destination = shim._install(tmp_path)
    assert destination.endswith('magnet_theory.py')
    assert 'def approximates' not in open(destination).read()  # built by a factory
    assert 'PREDICATES' in open(destination).read()
