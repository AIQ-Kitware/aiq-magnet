"""
Tests for the ``kwdagger.terminal_node`` declaration.

A card that declares a terminal node is stating that one pipeline node
produces its whole result.  MAGNET then reads that artifact instead of
rediscovering outputs by globbing the run tree, and evaluates the claim
once against what the pipeline actually computed.
"""

import json

import pytest
import ubelt as ub

from magnet.evaluation import KWDaggerProcessor


def test_terminal_node_is_kept_out_of_the_scheduled_spec():
    # kwdagger does not know about terminal_node; passing it through would
    # be rejected by the schedule config.
    processor = KWDaggerProcessor(
        {
            'pipeline': 'some.module.some_pipeline()',
            'matrix': {'a.b': [1, 2]},
            'terminal_node': 'summary',
        },
        root_dpath=ub.Path('.'),
    )
    assert processor.terminal_node == 'summary'
    assert 'terminal_node' not in processor.spec
    assert set(processor.spec) == {'pipeline', 'matrix'}


def test_absent_terminal_node_keeps_legacy_behaviour():
    processor = KWDaggerProcessor(
        {'pipeline': 'some.module.some_pipeline()', 'matrix': {}},
        root_dpath=ub.Path('.'),
    )
    assert processor.terminal_node is None


def test_collect_terminal_result_requires_a_declaration():
    processor = KWDaggerProcessor(
        {'pipeline': 'some.module.some_pipeline()', 'matrix': {}},
        root_dpath=ub.Path('.'),
    )
    with pytest.raises(ValueError, match='terminal_node'):
        processor.collect_terminal_result()


class _FakeNode:
    def __init__(self, name, out_paths, primary_out_key):
        self.name = name
        self.out_paths = out_paths
        self.primary_out_key = primary_out_key


class _FakeDag:
    def __init__(self, nodes):
        self.nodes = nodes


def _processor_with_dag(nodes, root_dpath, terminal_node='summary'):
    processor = KWDaggerProcessor(
        {
            'pipeline': 'some.module.some_pipeline()',
            'matrix': {},
            'terminal_node': terminal_node,
        },
        root_dpath=root_dpath,
    )
    processor.dag = _FakeDag(nodes)
    return processor


def test_collect_terminal_result_reads_the_declared_artifact():
    dpath = ub.Path.appdir('magnet/tests/terminal_ok').ensuredir()
    ub.delete(dpath)
    dpath.ensuredir()

    payload = {'status': 'VERIFIED', 'metrics': {'mae': 0.03}}
    artifact = (dpath / 'summary' / 'summary_id_abc').ensuredir() / 'out.json'
    artifact.write_text(json.dumps(payload))

    processor = _processor_with_dag(
        {'summary_id_abc': _FakeNode('summary', {'o': 'out.json'}, 'o')},
        root_dpath=dpath,
    )
    result = processor.collect_terminal_result()

    assert result['status'] == 'VERIFIED'
    assert result['metrics']['mae'] == 0.03
    # Provenance for the artifact that produced the card's status.
    assert result['_terminal_artifact_fpath'] == str(artifact)


def test_unknown_terminal_node_names_the_available_ones():
    dpath = ub.Path.appdir('magnet/tests/terminal_unknown').ensuredir()
    processor = _processor_with_dag(
        {'other_id_abc': _FakeNode('other', {'o': 'out.json'}, 'o')},
        root_dpath=dpath,
        terminal_node='summary',
    )
    with pytest.raises(ValueError, match="available: \\['other'\\]"):
        processor.collect_terminal_result()


def test_a_fanned_out_terminal_node_is_rejected():
    # Two configured instances means the node summarizes a slice, not the
    # card.  Reporting one of them as the whole finding would be wrong.
    dpath = ub.Path.appdir('magnet/tests/terminal_fanout').ensuredir()
    processor = _processor_with_dag(
        {
            'summary_id_a': _FakeNode('summary', {'o': 'out.json'}, 'o'),
            'summary_id_b': _FakeNode('summary', {'o': 'out.json'}, 'o'),
        },
        root_dpath=dpath,
    )
    with pytest.raises(RuntimeError, match='not\\s+terminal'):
        processor.collect_terminal_result()


def test_missing_artifact_points_at_the_run_directory():
    dpath = ub.Path.appdir('magnet/tests/terminal_missing').ensuredir()
    ub.delete(dpath)
    dpath.ensuredir()

    processor = _processor_with_dag(
        {'summary_id_abc': _FakeNode('summary', {'o': 'out.json'}, 'o')},
        root_dpath=dpath,
    )
    with pytest.raises(RuntimeError, match='produced no out.json'):
        processor.collect_terminal_result()
