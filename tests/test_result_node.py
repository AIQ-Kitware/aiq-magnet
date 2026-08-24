"""Tests for the new evaluator's kwdagger evidence / request boundary."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest
import ubelt as ub

from magnet._kwdagger import KWDaggerProcessor


def _processor(root_dpath='.', result_node='summary'):
    data = {
        'pipeline': 'some.module.some_pipeline()',
        'matrix': {'a.b': [1, 2]},
    }
    if result_node is not None:
        data['result_node'] = result_node
    return KWDaggerProcessor(data, root_dpath=ub.Path(root_dpath))


def test_result_node_is_kept_out_of_the_scheduled_spec():
    processor = _processor()
    assert processor.result_node == 'summary'
    assert 'result_node' not in processor.params
    assert set(processor.params) == {'pipeline', 'matrix'}


def test_available_result_rows_require_a_result_node():
    processor = _processor(result_node=None)
    with pytest.raises(ValueError, match='result_node'):
        processor.load_available_result_rows()


def test_available_evidence_uses_kwdagger_aggregate_rows(
        monkeypatch, tmp_path):
    """MAGNET consumes KWDagger's qualified row instead of parsing artifacts."""
    import kwdagger.aggregate_loader

    artifact = ub.Path(tmp_path) / 'summary' / 'proc123' / 'out.json'
    artifact.parent.ensuredir()
    artifact.write_text('{}')

    fake_dag = SimpleNamespace(node_dict={'summary': object()})
    processor = _processor(root_dpath=tmp_path)
    monkeypatch.setattr(processor, '_coerce_aggregate_pipeline', lambda: fake_dag)

    def fake_build_tables(
        root_dpath, dag, io_workers, eval_nodes, cache_resolved_results
    ):
        assert ub.Path(root_dpath) == ub.Path(tmp_path)
        assert dag is fake_dag
        assert eval_nodes == ['summary']
        assert cache_resolved_results is True
        return {
            'summary': {
                'fpath': pd.DataFrame({'fpath': [artifact]}),
                'index': pd.DataFrame({'node': ['summary']}),
                'metrics': pd.DataFrame({'metrics.summary.mae': [0.03]}),
                'requested_params': pd.DataFrame({
                    'params.predict.model': ['model-a']
                }),
                'resolved_params': pd.DataFrame({
                    'resolved_params.predict.model': ['model-a']
                }),
                'other': pd.DataFrame({
                    'context.predict.uuid': ['uuid-1'],
                    'unused.nan': [float('nan')],
                }),
            }
        }

    monkeypatch.setattr(
        kwdagger.aggregate_loader, 'build_tables', fake_build_tables
    )

    rows = processor.load_available_result_rows()
    assert rows == [{
        'key': 'proc123',
        'artifact': str(artifact),
        'row': {
            'fpath': str(artifact),
            'node': 'summary',
            'metrics.summary.mae': 0.03,
            'params.predict.model': 'model-a',
            'resolved_params.predict.model': 'model-a',
            'context.predict.uuid': 'uuid-1',
        },
    }]


def test_available_evidence_is_independent_of_the_current_request(
        monkeypatch, tmp_path):
    """A current schedule does not bound what historical evidence is visible."""
    import kwdagger.aggregate_loader

    old_artifact = ub.Path(tmp_path) / 'summary' / 'oldproc' / 'out.json'
    old_artifact.parent.ensuredir()
    old_artifact.write_text('{}')

    processor = _processor(root_dpath=tmp_path)
    processor.request_dag = SimpleNamespace(nodes={
        'newproc': SimpleNamespace(name='summary')
    })
    monkeypatch.setattr(
        processor,
        '_coerce_aggregate_pipeline',
        lambda: SimpleNamespace(node_dict={'summary': object()}),
    )
    monkeypatch.setattr(
        kwdagger.aggregate_loader,
        'build_tables',
        lambda *args, **kwargs: {
            'summary': {
                'fpath': pd.DataFrame({'fpath': [old_artifact]}),
                'metrics': pd.DataFrame({'metrics.summary.mae': [0.01]}),
            }
        },
    )

    rows = processor.load_available_result_rows()
    assert [row['key'] for row in rows] == ['oldproc']


class _FakeNode:
    def __init__(self, dpath, *, name='summary', enabled=True):
        self.name = name
        self.enabled = enabled
        self.primary_out_key = 'out'
        self.out_paths = {'out': 'out.json'}
        self.final_node_dpath = ub.Path(dpath)
        self.final_out_paths = {
            'out': self.final_node_dpath / self.out_paths['out']
        }


class _FakeJob:
    def __init__(self, name, stat_fpath=None, log_fpath=None):
        self.name = name
        self.stat_fpath = stat_fpath
        self.log_fpath = log_fpath


def _with_request(tmp_path, node, job=None):
    processor = _processor(root_dpath=tmp_path)
    process_id = 'summary-proc'
    processor.request_dag = SimpleNamespace(nodes={process_id: node})
    processor.queue = SimpleNamespace(
        named_jobs={} if job is None else {process_id: job},
        jobs=[] if job is None else [job],
    )
    return processor


def test_request_state_is_separate_from_result_availability(tmp_path):
    node = _FakeNode(ub.Path(tmp_path) / 'summary' / 'summary-proc')
    node.final_node_dpath.ensuredir()
    node.final_out_paths['out'].write_text('{}')
    processor = _with_request(tmp_path, node, job=None)

    record, = processor.inspect_requested_runs()
    assert record['schedule_status'] == 'skipped'
    assert record['attempt_status'] == 'not_attempted'
    assert record['output_available'] is True


@pytest.mark.parametrize(
    'stat,expected_status',
    [
        (None, 'not_started'),
        ({'ret': None}, 'running'),
        ({'ret': 0}, 'passed'),
        ({'ret': 3}, 'failed'),
        ({'ret': 126}, 'skipped'),
    ],
)
def test_requested_attempt_status(tmp_path, stat, expected_status):
    node = _FakeNode(ub.Path(tmp_path) / 'summary' / 'summary-proc')
    stat_fpath = ub.Path(tmp_path) / 'job.stat'
    if stat is not None:
        stat_fpath.write_text(json.dumps(stat))
    job = _FakeJob('summary-proc', stat_fpath=stat_fpath)
    processor = _with_request(tmp_path, node, job=job)

    record, = processor.inspect_requested_runs()
    assert record['schedule_status'] == 'new_submission'
    assert record['attempt_status'] == expected_status
    assert record['output_available'] is False


def test_disabled_request_is_reported_as_operational_state(tmp_path):
    node = _FakeNode(
        ub.Path(tmp_path) / 'summary' / 'summary-proc', enabled=False
    )
    processor = _with_request(tmp_path, node)

    record, = processor.inspect_requested_runs()
    assert record['schedule_status'] == 'disabled'
    assert record['attempt_status'] == 'not_attempted'


def test_failed_rerun_can_coexist_with_an_older_available_output(tmp_path):
    """Attempt state and reusable evidence are independent observations."""
    node = _FakeNode(ub.Path(tmp_path) / 'summary' / 'summary-proc')
    node.final_node_dpath.ensuredir()
    node.final_out_paths['out'].write_text('{}')

    stat_fpath = ub.Path(tmp_path) / 'job.stat'
    stat_fpath.write_text(json.dumps({'ret': 3}))
    job = _FakeJob('summary-proc', stat_fpath=stat_fpath)
    processor = _with_request(tmp_path, node, job=job)

    record, = processor.inspect_requested_runs()
    assert record['schedule_status'] == 'new_submission'
    assert record['attempt_status'] == 'failed'
    assert record['output_available'] is True
