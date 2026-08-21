"""
Tests for the ``kwdagger.result_node`` declaration.

A card that declares a result node is stating which node produces its result.
MAGNET reads each configured instance's own artifact and evaluates the claim
once per instance -- one cell of the card each.
"""

import json

import pytest
import ubelt as ub

from magnet._kwdagger import KWDaggerProcessor


def test_result_node_is_kept_out_of_the_scheduled_spec():
    # kwdagger does not know about result_node; passing it through would be
    # rejected by the schedule config.
    processor = KWDaggerProcessor(
        {
            'pipeline': 'some.module.some_pipeline()',
            'matrix': {'a.b': [1, 2]},
            'result_node': 'summary',
        },
        root_dpath=ub.Path('.'),
    )
    assert processor.result_node == 'summary'
    assert 'result_node' not in processor.spec
    assert set(processor.spec) == {'pipeline', 'matrix'}


def test_absent_result_node_keeps_the_rediscovery_path():
    processor = KWDaggerProcessor(
        {'pipeline': 'some.module.some_pipeline()', 'matrix': {}},
        root_dpath=ub.Path('.'),
    )
    assert processor.result_node is None


def test_collect_result_cells_requires_a_declaration():
    processor = KWDaggerProcessor(
        {'pipeline': 'some.module.some_pipeline()', 'matrix': {}},
        root_dpath=ub.Path('.'),
    )
    with pytest.raises(ValueError, match='result_node'):
        processor.collect_result_cells()


class _FakeNode:
    def __init__(self, name, dpath, config=None, process_id=None):
        self.name = name
        self.final_node_dpath = ub.Path(dpath)
        self.out_paths = {'o': 'out.json'}
        self.primary_out_key = 'o'
        self.config = config or {}
        self.process_id = process_id or f'{name}_id_{ub.Path(dpath).name}'


class _FakeDag:
    def __init__(self, nodes):
        self.nodes = nodes


class _FakeJob:
    def __init__(self, name, stat_fpath):
        self.name = name
        self.stat_fpath = stat_fpath


class _FakeQueue:
    def __init__(self, jobs):
        self.jobs = jobs


def _processor_with_dag(nodes, root_dpath, result_node='summary'):
    processor = KWDaggerProcessor(
        {
            'pipeline': 'some.module.some_pipeline()',
            'matrix': {},
            'result_node': result_node,
        },
        root_dpath=root_dpath,
    )
    processor.dag = _FakeDag(nodes)
    return processor


def _fresh(name):
    dpath = ub.Path.appdir(f'magnet/tests/{name}')
    ub.delete(dpath)
    return dpath.ensuredir()


def _write(dpath, payload):
    artifact = ub.Path(dpath).ensuredir() / 'out.json'
    artifact.write_text(json.dumps(payload))
    return artifact


def test_a_cell_carries_its_results_and_its_artifact():
    dpath = _fresh('result_ok')
    artifact = _write(dpath / 'summary' / 'abc', {'mae': 0.03, '_hidden': 1})

    processor = _processor_with_dag(
        {'summary_id_abc': _FakeNode('summary', dpath / 'summary' / 'abc')},
        root_dpath=dpath,
    )
    cells = processor.collect_result_cells()

    assert len(cells) == 1
    assert cells[0]['results'] == {'metrics.summary.mae': 0.03}
    assert cells[0]['key'] == 'summary_id_abc'
    assert cells[0]['artifact'] == str(artifact)


def test_a_cell_is_identified_by_the_node_that_produced_it():
    # The key is the instance's own identity, not something derived from what
    # else happened to be scheduled. A card run one cell at a time gets the
    # same key it would as part of a sweep.
    dpath = _fresh('result_single')
    _write(dpath / 'summary' / 'only', {'mae': 0.05})

    processor = _processor_with_dag(
        {'summary_id_only': _FakeNode(
            'summary', dpath / 'summary' / 'only', {'dataset': 'one'})},
        root_dpath=dpath,
    )
    cells = processor.collect_result_cells()

    assert [cell['key'] for cell in cells] == ['summary_id_only']
    assert cells[0]['params'] == {'dataset': 'one'}


def test_a_fanned_out_result_node_yields_one_cell_each():
    # Several configured instances is a gather with group_by, or a swept
    # parameter. Each is one cell of the card, consumed independently.
    dpath = _fresh('result_fanout')
    _write(dpath / 'summary' / 'a', {'mae': 0.01})
    _write(dpath / 'summary' / 'b', {'mae': 0.02})

    processor = _processor_with_dag(
        {
            'summary_id_a': _FakeNode(
                'summary', dpath / 'summary' / 'a',
                {'dataset': 'one', 'workers': 4}),
            'summary_id_b': _FakeNode(
                'summary', dpath / 'summary' / 'b',
                {'dataset': 'two', 'workers': 4}),
        },
        root_dpath=dpath,
    )
    cells = processor.collect_result_cells()

    assert len(cells) == 2
    assert sorted(cell['key'] for cell in cells) == [
        'summary_id_a', 'summary_id_b']
    assert sorted(cell['results']['metrics.summary.mae'] for cell in cells) == [
        0.01, 0.02]


def test_each_instance_is_asked_for_its_own_artifact():
    # The DAG root is shared across card versions, so an earlier version's
    # artifact can sit beside this one under its own node id. Globbing the
    # node directory would find both; asking the instance cannot.
    dpath = _fresh('result_shared_root')
    _write(dpath / 'summary' / 'mine', {'mae': 0.01})
    _write(dpath / 'summary' / 'from_an_older_card', {'mae': 0.99})

    processor = _processor_with_dag(
        {'summary_id_mine': _FakeNode('summary', dpath / 'summary' / 'mine')},
        root_dpath=dpath,
    )
    cells = processor.collect_result_cells()

    assert len(cells) == 1
    assert cells[0]['results'] == {'metrics.summary.mae': 0.01}


def test_unknown_result_node_names_the_available_ones():
    dpath = _fresh('result_unknown')
    processor = _processor_with_dag(
        {'other_id_abc': _FakeNode('other', dpath / 'other' / 'abc')},
        root_dpath=dpath,
        result_node='summary',
    )
    with pytest.raises(ValueError, match="available: \\['other'\\]"):
        processor.collect_result_cells()


def test_a_partial_run_reports_the_cells_it_has():
    # One node failing does not discard the cells that succeeded.
    dpath = _fresh('result_partial')
    _write(dpath / 'summary' / 'ran', {'mae': 0.01})

    processor = _processor_with_dag(
        {
            'summary_id_ran': _FakeNode('summary', dpath / 'summary' / 'ran'),
            'summary_id_failed': _FakeNode(
                'summary', dpath / 'summary' / 'failed'),
        },
        root_dpath=dpath,
    )
    cells = processor.collect_result_cells()

    assert [cell['key'] for cell in cells] == ['summary_id_ran']
    assert [e['key'] for e in processor.incomplete] == ['summary_id_failed']


def test_a_run_that_produced_nothing_is_empty_not_an_error():
    dpath = _fresh('result_missing')
    processor = _processor_with_dag(
        {'summary_id_abc': _FakeNode('summary', dpath / 'summary' / 'abc')},
        root_dpath=dpath,
    )
    assert processor.collect_result_cells() == []


def test_an_instance_with_no_job_has_not_run():
    dpath = _fresh('result_pending')
    processor = _processor_with_dag(
        {'summary_id_abc': _FakeNode('summary', dpath / 'summary' / 'abc')},
        root_dpath=dpath,
    )
    processor.collect_result_cells()

    entry, = processor.incomplete
    assert entry['status'] == 'pending'
    assert entry['returncode'] is None
    assert entry['expected'].endswith('out.json')


@pytest.mark.parametrize('returncode,status', [(3, 'failed'), (0, 'empty')])
def test_an_instance_that_ran_reports_its_exit_code(
        tmp_path, returncode, status):
    dpath = _fresh('result_exit')
    stat_fpath = ub.Path(tmp_path) / 'job.stat'
    stat_fpath.write_text(json.dumps({'ret': returncode, 'name': 'x'}))

    processor = _processor_with_dag(
        {'summary_id_abc': _FakeNode('summary', dpath / 'summary' / 'abc')},
        root_dpath=dpath,
    )
    processor.queue = _FakeQueue([_FakeJob('summary_id_abc', stat_fpath)])
    processor.collect_result_cells()

    entry, = processor.incomplete
    assert entry['status'] == status
    assert entry['returncode'] == returncode
