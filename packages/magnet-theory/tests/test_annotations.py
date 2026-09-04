from __future__ import annotations

import magnet_theory as theory


def test_annotations_are_runtime_inert():
    @theory.tests('A.Statement')
    @theory.assumes('A.Statement::hpremise')
    def experiment(value: int) -> int:
        return value * 2

    assert experiment(21) == 42
    with theory.checks('A.Statement::hpremise') as link:
        result = 1 + 1
    assert result == 2
    assert (link.relation, link.ref) == ('checks', 'A.Statement::hpremise')


def test_public_relation_vocabulary_is_complete():
    assert set(theory.RELATIONS) == {
        'tests',
        'approximates',
        'motivates',
        'satisfies',
        'substitutes',
        'assumes',
        'ignores',
        'violates',
        'checks',
    }
    assert getattr(theory.tests, '__test__') is False
