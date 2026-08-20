"""
The three relations, and reading them back out of source.
"""
import ast

import pytest
import ubelt as ub

import magnet.theory as theory
from magnet.theory.index import Entry, TheoryIndex, load_index
from magnet.theory.static import extract, extract_tree


def test_a_relation_is_inert_as_a_decorator():
    @theory.tests('A.b')
    def experiment(value):
        return value * 2

    assert experiment(21) == 42


def test_a_relation_is_inert_as_a_context_manager():
    with theory.approximates('A.b') as link:
        result = 1 + 1
    assert result == 2
    assert (link.relation, link.ref) == ('approximates', 'A.b')


def test_the_shim_matches_the_real_relations():
    from magnet.theory import shim

    @shim.motivates('A.b')
    def experiment():
        return 'unchanged'

    assert experiment() == 'unchanged'
    with shim.tests('A.b'):
        pass


SOURCE = ub.codeblock(
    '''
    import magnet.theory as theory

    @theory.tests('Examples.CoinFlip.Binomial')
    def exact(n):
        return n

    class Estimator:
        @theory.approximates('Examples.Dice.SumSevenProbability')
        def estimate(self):
            with theory.motivates('Examples.TrainingOrder.Why'):
                return 0
    ''')


def test_extraction_records_relation_ref_and_site():
    links = extract_tree(ast.parse(SOURCE), 'demo.py')
    found = [(link.relation, link.ref, link.qualname) for link in links]
    assert found == [
        ('tests', 'Examples.CoinFlip.Binomial', 'exact'),
        ('approximates', 'Examples.Dice.SumSevenProbability',
         'Estimator.estimate'),
        ('motivates', 'Examples.TrainingOrder.Why', 'Estimator.estimate'),
    ]
    assert all(link.file == 'demo.py' for link in links)
    assert links[0].line == 3


def test_the_vendored_alias_is_read_the_same_way():
    source = SOURCE.replace('import magnet.theory as theory',
                            'import magnet_theory as theory')
    links = extract_tree(ast.parse(source), 'demo.py')
    assert [link.relation for link in links] == [
        'tests', 'approximates', 'motivates']


def test_a_file_that_never_imports_theory_is_skipped():
    source = ub.codeblock(
        '''
        import something_else as theory

        @theory.tests('A.b')
        def experiment():
            pass
        ''')
    assert extract_tree(ast.parse(source), 'demo.py') == []


@pytest.mark.parametrize('call', [
    "theory.tests(REF)",             # a name, not a literal
    "theory.tests('A.' + 'b')",      # concatenated
    "theory.tests()",                # no reference at all
    "theory.believes('A.b')",        # not one of the three relations
])
def test_only_a_literal_reference_in_a_known_relation_counts(call):
    source = ub.codeblock(
        f'''
        import magnet.theory as theory
        REF = 'A.b'

        @{call}
        def experiment():
            pass
        ''')
    assert extract_tree(ast.parse(source), 'demo.py') == []


def test_a_bare_call_is_not_an_annotation():
    # Only a decorator or a with-item counts, so a stray call in a function
    # body cannot quietly add a link.
    source = ub.codeblock(
        '''
        import magnet.theory as theory

        def experiment():
            theory.tests('A.b')
        ''')
    assert extract_tree(ast.parse(source), 'demo.py') == []


def test_extract_walks_a_directory(tmp_path):
    (tmp_path / 'annotated.py').write_text(SOURCE)
    (tmp_path / 'plain.py').write_text('x = 1\n')
    (tmp_path / 'broken.py').write_text('def (:\n')
    links = extract([str(tmp_path)])
    assert [link.relation for link in links] == [
        'tests', 'approximates', 'motivates']


def test_an_index_resolves_references(tmp_path):
    fpath = tmp_path / 'theory.yaml'
    fpath.write_text(ub.codeblock(
        '''
        entries:
          - id: Examples.CoinFlip.Binomial
            kind: theorem
            statement: P(X = k) = C(n, k) p^k (1-p)^(n-k)
          - id: Examples.TrainingOrder.Why
            kind: question
            statement: Why does order change the learned solution?
        '''))
    index = load_index(fpath)
    assert len(index) == 2
    assert index['Examples.TrainingOrder.Why'].kind == 'question'
    assert index.unresolved(
        ['Examples.CoinFlip.Binomial', 'Nope.Missing']) == ['Nope.Missing']


def test_an_unknown_kind_is_rejected(tmp_path):
    fpath = tmp_path / 'theory.yaml'
    fpath.write_text('entries:\n  - id: A.b\n    kind: vibes\n')
    with pytest.raises(ValueError, match='vibes'):
        load_index(fpath)


def test_an_entry_without_an_id_is_rejected(tmp_path):
    fpath = tmp_path / 'theory.yaml'
    fpath.write_text('entries:\n  - kind: theorem\n')
    with pytest.raises(ValueError, match='no id'):
        load_index(fpath)


def test_index_membership():
    index = TheoryIndex([Entry('A.b', 'conjecture')])
    assert 'A.b' in index
    assert 'A.c' not in index
