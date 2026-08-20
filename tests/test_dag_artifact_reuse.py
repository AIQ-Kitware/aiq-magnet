"""
A card change must not recompute the cells it did not change.

kwdagger identifies a node by hashing its own configuration, so an unchanged
node keeps its id when a different part of the card changes. Rooting the DAG
under a hash of the whole card threw that away and recomputed everything.

The card, its pipeline and the program the pipeline runs are all built in
``tmp_path``, so these tests do not depend on any shipped card.
"""

import json
import sys

import pytest
import ubelt as ub
import yaml

from magnet.evaluation import EvaluationCard

# The whole pipeline: one node whose output depends only on its own parameters.
#
# It is invoked as ``<sys.executable> <this file>``: kwdagger nodes shell out,
# and there is no guarantee of a bare ``python`` on PATH or of ``tmp_path``
# being importable by the subprocess.
_NODE_SOURCE = ub.codeblock(
    '''
    import json
    import sys

    args = dict(arg.lstrip('-').split('=', 1) for arg in sys.argv[1:])
    seed = int(args['seed'])
    with open(args['results_fpath'], 'w') as file:
        json.dump({'result': {'seed': seed, 'doubled': seed * 2}}, file)
    ''')


@pytest.fixture
def card_factory(tmp_path, monkeypatch):
    """
    Build cards that differ only in which seeds they sweep.

    Returns:
        Callable[[list], EvaluationCard]
    """
    # A test must not queue jobs into tmux, which is the default backend.
    monkeypatch.setenv('MAGNET_QUEUE_BACKEND', 'serial')

    dpath = ub.Path(tmp_path)
    node_fpath = dpath / 'demo_node.py'
    node_fpath.write_text(_NODE_SOURCE)

    def make(seeds):
        card = {
            'title': 'Doubling',
            'description': 'Doubling a seed doubles it',
            'version': '1.0',
            'organizations': ['Kitware'],
            'submitter': {'name': 'Kitware TA2 Team', 'email': 'x@kitware.com'},
            'tags': ['test'],
            'links': [{
                'title': 'MAGNET',
                'url': 'https://github.com/AIQ-Kitware/aiq-magnet',
                'type': 'software',
            }],
            'claim': {'python': 'assert doubled == seed * 2'},
            'pipeline': {
                'demo_node': {
                    'executable': f'{sys.executable} {node_fpath}',
                    'algo_params': {'seed': list(seeds)},
                    'out_paths': {'results_fpath': 'results.json'},
                },
            },
        }
        stem = '_'.join(str(seed) for seed in seeds)
        card_fpath = dpath / f'card_{stem}.yaml'
        card_fpath.write_text(yaml.safe_dump(card, sort_keys=False))
        return EvaluationCard(card_fpath, dpath / 'runs')

    return make


def _artifacts(output_dpath):
    """
    Every cell's artifact under ``output_dpath``, keyed by node id.

    Keyed by node id rather than by path: a cell recomputed somewhere else
    then appears as a second path under the same id.

    Returns:
        Dict[str, Dict[ub.Path, float]]: node id -> artifact path -> mtime
    """
    found = {}
    for fpath in sorted(ub.Path(output_dpath).glob('**/results.json')):
        found.setdefault(fpath.parent.name, {})[fpath] = fpath.stat().st_mtime
    return found


def test_adding_a_cell_reuses_the_cells_that_did_not_change(
        card_factory, tmp_path):
    output_dpath = ub.Path(tmp_path) / 'runs'

    card = card_factory([1, 2])
    assert card.evaluate() == 'VERIFIED'

    before = _artifacts(output_dpath)
    assert len(before) == 2, 'the first run should have computed two cells'

    # The same two cells, plus a third. Seeds 1 and 2 are configured as
    # before, so their node ids do not move.
    card2 = card_factory([1, 2, 3])
    assert card2.evaluate() == 'VERIFIED'

    after = _artifacts(output_dpath)

    for node_id, artifacts in before.items():
        assert node_id in after, f'{node_id} was orphaned by the card change'
        strays = sorted(set(after[node_id]) - set(artifacts))
        assert not strays, (
            f'{node_id} did not change, but the card change relocated it '
            f'and it was recomputed at {strays}')
        assert after[node_id] == artifacts, (
            f'{node_id} did not change, but was recomputed in place')

    assert len(after) == 3, 'the added seed should have computed a third cell'

    # Each card version keeps its own provenance directory ...
    run_dpaths = [
        p for p in sorted(output_dpath.iterdir())
        if p.is_dir() and not p.name.startswith('_')
    ]
    assert len(run_dpaths) == 2, run_dpaths

    # ... but they share one DAG root, which is what makes the reuse possible.
    assert card2.dag_root_dpath == card.dag_root_dpath
    assert card.dag_root_dpath.name == '_kwdagger'


def test_a_changed_cell_is_not_reused(card_factory, tmp_path):
    """A changed seed is a different node id, so its cell is not reused."""
    output_dpath = ub.Path(tmp_path) / 'runs'

    assert card_factory([1]).evaluate() == 'VERIFIED'
    before = _artifacts(output_dpath)
    assert len(before) == 1

    # The shared root must not hand back the cell computed for seed 1.
    assert card_factory([7]).evaluate() == 'VERIFIED'
    after = _artifacts(output_dpath)

    new_ids = set(after) - set(before)
    assert len(new_ids) == 1, 'seed 7 should have computed its own cell'
    new_fpath, = after[new_ids.pop()]
    assert json.loads(new_fpath.read_text())['result'] == {
        'seed': 7, 'doubled': 14}
