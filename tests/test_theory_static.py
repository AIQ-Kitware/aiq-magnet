"""
Tests for static extraction.

The premise of extracting from source is that a repository can be audited
without installing its dependencies or executing it. So the properties that
matter are: every syntactic position is found, references that cannot be read
are *reported* rather than dropped, and nothing here ever imports the parsed
module.
"""
import textwrap

import pytest

from magnet.theory import Formalization, Hypothesis, Theorem
from magnet.theory.basis import TheoreticalBasis
from magnet.theory.static import extract_source, extract_tree, lint


def parse(source):
    return extract_source(textwrap.dedent(source), filename='team/predictor.py')


def test_finds_every_activation_form():
    ledger = parse(
        '''
        from magnet.theory import approximates, assumes, grounds, substitutes

        assumes('Paper.main::hlip')

        @approximates('Paper.main::hcover')
        def build(n=64): ...

        def predict():
            with substitutes('Paper.main::hpsi'):
                pass

        @grounds('Paper.claim')
        def claim(): ...
        '''
    )
    found = {(a.predicate, a.form) for a in ledger}
    assert found == {
        ('assumes', 'bare'),
        ('approximates', 'decorator'),
        ('substitutes', 'with'),
        ('grounds', 'decorator'),
    }


def test_finds_annotated_parameters():
    # Several real assumptions are about one knob, not about a function.
    ledger = parse(
        '''
        from typing import Annotated
        from magnet.theory import approximates

        def __init__(self, num_example_runs: Annotated[int, approximates('Paper.main::hcover')] = 64): ...
        '''
    )
    (found,) = ledger.annotations
    assert found.form == 'annotation'
    assert found.target == 'num_example_runs'
    assert found.ref == 'Paper.main::hcover'


def test_records_enclosing_qualname_and_line():
    ledger = parse(
        '''
        from magnet.theory import assumes

        class Predictor:
            def predict(self):
                with assumes('Paper.main::hrank'):
                    pass
        '''
    )
    (found,) = ledger.annotations
    assert found.qualname == 'Predictor.predict'
    assert found.line == 6
    assert found.file == 'team/predictor.py'


def test_reads_literal_options():
    ledger = parse(
        '''
        from magnet.theory import substitutes

        substitutes('Paper.main::hpsi', severity='high', kind='different-object',
                    id='team-psi', informal='an off-the-shelf embedder')
        '''
    )
    (found,) = ledger.annotations
    assert found.options['severity'] == 'high'
    assert found.options['kind'] == 'different-object'
    assert found.options['id'] == 'team-psi'


def test_reads_positional_severity():
    ledger = parse(
        """
        from magnet.theory import approximates
        approximates('Paper.main::hcover', 'high')
        """
    )
    assert ledger.annotations[0].options['severity'] == 'high'


def test_unreadable_options_are_named_not_dropped():
    ledger = parse(
        '''
        from magnet.theory import assumes
        assumes('Paper.main::h', informal=compute_description())
        '''
    )
    (found,) = ledger.annotations
    assert found.unreadable_options == ('informal',)


@pytest.mark.parametrize(
    'source',
    [
        "from magnet.theory import assumes\nassumes('Paper.main::h')",
        "from magnet.theory import assumes as skips\nskips('Paper.main::h')",
        "from magnet_theory import assumes\nassumes('Paper.main::h')",
        "import magnet.theory as th\nth.assumes('Paper.main::h')",
    ],
)
def test_import_spellings(source):
    ledger = extract_source(source, filename='x.py')
    assert [a.ref for a in ledger] == ['Paper.main::h']


def test_ignores_unrelated_functions_with_the_same_name():
    # `checks` and `assumes` are ordinary words; matching on the name alone
    # would produce false positives in unrelated code.
    ledger = parse(
        '''
        from mypackage import assumes
        assumes('this is not a hypothesis reference')
        '''
    )
    assert len(ledger) == 0


def test_resolves_fstring_and_concatenated_references():
    # Nobody writes a ninety-character declaration name six times.
    ledger = parse(
        '''
        from magnet.theory import assumes

        DECL = 'Paper.Section.main'

        assumes(f'{DECL}::hgap')
        assumes(DECL + '::hcompetitive')
        assumes(DECL)
        '''
    )
    assert [a.ref for a in ledger] == [
        'Paper.Section.main::hgap',
        'Paper.Section.main::hcompetitive',
        'Paper.Section.main',
    ]


def test_resolves_object_style_references():
    # Code written against the object API is not left out.
    ledger = parse(
        '''
        from magnet.theory import assumes

        QE = FORMALIZATION['Paper.Section.main']

        assumes(QE['hgap'])
        '''
    )
    assert [a.ref for a in ledger] == ['Paper.Section.main::hgap']


def test_unresolvable_reference_is_reported_not_dropped():
    ledger = parse(
        '''
        from magnet.theory import assumes
        assumes(pick_hypothesis_at_random())
        '''
    )
    (found,) = ledger.annotations
    assert not found.resolved
    assert found.ref_expr == 'pick_hypothesis_at_random()'

    (issue,) = lint(ledger)
    assert issue.kind == 'unresolved-reference'


def test_syntax_errors_are_recorded():
    ledger = extract_source('def broken(:\n', filename='bad.py')
    assert ledger.errors
    assert lint(ledger)[0].kind == 'unparseable'


def test_extract_tree_skips_noise_directories(tmp_path):
    (tmp_path / 'pkg').mkdir()
    (tmp_path / 'pkg' / 'a.py').write_text(
        "from magnet.theory import assumes\nassumes('Paper.main::h')\n"
    )
    (tmp_path / '.venv').mkdir()
    (tmp_path / '.venv' / 'b.py').write_text(
        "from magnet.theory import assumes\nassumes('Vendored.thing::h')\n"
    )
    ledger = extract_tree(tmp_path)
    assert [a.ref for a in ledger] == ['Paper.main::h']


# ----------------------------------------------------------------------- lint


@pytest.fixture
def formalization():
    return Formalization(
        name='Example',
        theorems=[
            Theorem(
                declaration='Paper.main',
                hypotheses=[Hypothesis('hcover'), Hypothesis('hgap')],
            )
        ],
    )


def test_lint_catches_a_renamed_binder(formalization):
    # The check the object form used to do at import, now done without one.
    ledger = parse(
        '''
        from magnet.theory import assumes
        assumes('Paper.main::h_renamed_upstream')
        '''
    )
    (issue,) = lint(ledger, [formalization])
    assert issue.kind == 'unknown-binder'
    assert 'hcover, hgap' in issue.message


def test_lint_catches_an_unknown_declaration(formalization):
    ledger = parse(
        '''
        from magnet.theory import assumes
        assumes('Paper.renamed::hcover')
        '''
    )
    (issue,) = lint(ledger, [formalization])
    assert issue.kind == 'unknown-declaration'


def test_lint_catches_an_edge_with_no_binder(formalization):
    ledger = parse(
        '''
        from magnet.theory import assumes
        assumes('Paper.main')
        '''
    )
    kinds = {i.kind for i in lint(ledger, [formalization])}
    assert 'missing-binder' in kinds


def test_lint_allows_a_grounding_without_a_binder(formalization):
    ledger = parse(
        '''
        from magnet.theory import grounds
        grounds('Paper.main')
        '''
    )
    assert lint(ledger, [formalization]) == []


def test_lint_catches_duplicate_ids(formalization):
    ledger = parse(
        '''
        from magnet.theory import assumes
        assumes('Paper.main::hcover', id='dup')
        assumes('Paper.main::hgap', id='dup')
        '''
    )
    (issue,) = lint(ledger, [formalization])
    assert issue.kind == 'duplicate-id'


def test_clean_ledger_lints_clean(formalization):
    ledger = parse(
        '''
        from magnet.theory import assumes, grounds
        grounds('Paper.main')
        assumes('Paper.main::hcover', id='a')
        assumes('Paper.main::hgap', id='b')
        '''
    )
    assert lint(ledger, [formalization]) == []


# -------------------------------------------------------- static -> reporting


def test_static_ledger_produces_the_same_report_shape(formalization):
    ledger = parse(
        '''
        from magnet.theory import approximates, assumes, grounds

        grounds('Paper.main')

        @approximates('Paper.main::hcover', severity='high', informal='fixed pool')
        def build(n=64): ...
        '''
    )
    basis = TheoreticalBasis.from_ledger(ledger, [formalization])
    coverage = basis.coverage()

    assert [h.name for h in coverage.unaccounted] == ['hgap']
    (edge,) = coverage.per_theorem[0].gaps
    assert edge.informal == 'fixed pool'
    assert str(edge.severity) == 'high'
