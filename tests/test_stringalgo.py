"""Regression coverage for pygtrie's typed step API."""

import pytest

pytest.importorskip('pygtrie')

from magnet.utils.util_stringalgo import shortest_unique_prefixes, shortest_unique_suffixes


def test_character_prefixes():
    items = ['zebra', 'dog', 'duck', 'dove']
    assert shortest_unique_prefixes(items) == ['z', 'dog', 'du', 'dov']


def test_separator_aware_prefixes():
    paths = ['params.metrics.foo.mean', 'params.metrics.foo.std', 'params.metrics.foo.count']
    reversed_paths = [path[::-1] for path in paths]
    prefixes = shortest_unique_prefixes(reversed_paths, sep='.', min_length=2)
    assert [prefix[::-1] for prefix in prefixes] == [
        'foo.mean', 'foo.std', 'foo.count'
    ]


def test_prefix_of_another_requires_allow_end():
    with pytest.raises(AssertionError, match='no unique prefix'):
        shortest_unique_prefixes(['ab', 'abc'], allow_end=False)
    assert shortest_unique_prefixes(['ab', 'abc'], allow_end=True) == ['ab', 'abc']


def test_suffixes():
    assert shortest_unique_suffixes(['zebra', 'dog', 'duck', 'dove']) == [
        'a', 'g', 'k', 'e'
    ]
