"""
Declarative kwdagger nodes using MAGNET's single process-node surface.

A plain inline YAML node has no MAGNET execution capabilities. Naming
``magnet.process_node.MagnetProcessNode`` opts that node into the same independent
container and leasing policies used by Python-defined DAGs, while preserving
kwdagger's YAML-only metadata hooks.
"""

import kwdagger
import pytest
from kwdagger.pipeline import coerce_pipeline
from kwdagger.yaml_pipeline import YamlProcessNode

from magnet import containers, leasing
from magnet._kwdagger import _check_container_settings_apply
from magnet.containers import ContainerCapability
from magnet.process_node import MagnetProcessNode

import shlex as _shlex
import sys as _sys

HOST_PY = _shlex.quote(_sys.executable)
IMAGE = 'aiq-eval-node:latest'
MAGNET_CLASS = 'magnet.process_node.MagnetProcessNode'


class Infer(MagnetProcessNode):
    """Python-defined node that fixes which parameter holds its endpoint."""

    endpoint_params = ('model_id',)


class BareContainerNode(ContainerCapability, kwdagger.ProcessNode):
    """Test-only non-YAML capability composition used to pin loader guards."""

    @property
    def command(self):
        return containers.render_container_command(self, super().command)


def _spec(node_class=None, **extra):
    node = {
        'executable': 'python -m pkg.work',
        'algo_params': {'task': None},
        'out_paths': {'results_fpath': 'results.json'},
        'primary_out_key': 'results_fpath',
        **extra,
    }
    if node_class is not None:
        node['class'] = node_class
    return {'nodes': {'work': node}}


def _settings(image=IMAGE, mounts='/repo'):
    return containers.ContainerSettings.coerce(image=image, mounts=mounts)


def _node(spec, config={'task': 't'}, settings=None, lease=None):
    pipeline = coerce_pipeline(spec)
    (settings or containers.ContainerSettings()).apply(pipeline)
    (lease or leasing.LeaseSettings()).apply(pipeline)
    node = pipeline.node_dict['work']
    node.configure(config)
    return node


def test_magnet_process_node_is_the_one_declarative_integration_surface():
    assert issubclass(MagnetProcessNode, kwdagger.ProcessNode)
    assert issubclass(MagnetProcessNode, YamlProcessNode)
    node = _node(_spec(MAGNET_CLASS))
    assert isinstance(node, MagnetProcessNode)
    assert containers.is_container_capable(node)
    assert leasing.is_lease_capable(node)


def test_a_plain_yaml_node_still_cannot_containerize():
    command = _node(_spec(), settings=_settings()).command
    assert 'docker run' not in command


def test_a_declarative_magnet_node_runs_in_the_image():
    command = _node(_spec(MAGNET_CLASS), settings=_settings()).command
    assert command.startswith('docker run --rm ')
    assert f' {IMAGE} ' in command
    assert 'python -m pkg.work' in command
    assert command.rstrip().endswith('--task=t')


def test_it_is_inert_until_an_image_is_named():
    command = _node(_spec(MAGNET_CLASS)).command
    assert 'docker run' not in command
    assert command.startswith(HOST_PY + ' -m pkg.work')


def test_the_declarative_extras_survive():
    node = _node(_spec(MAGNET_CLASS, load_result='pkg.results.load'))
    assert node._load_result_ref == 'pkg.results.load'
    assert hasattr(node, 'load_result')


def test_a_non_yaml_capability_class_is_still_rejected_with_yaml_extras():
    """Capabilities are reusable, but the declarative integration is explicit."""
    with pytest.raises(ValueError, match='YamlProcessNode'):
        coerce_pipeline(
            _spec(f'{__name__}.BareContainerNode', load_result='pkg.results.load')
        )


def test_metrics_metadata_survives():
    node = _node(_spec(MAGNET_CLASS, metrics=[{'name': 'auc'}]))
    assert node.default_metrics() == [{'name': 'auc'}]


def test_the_lease_stays_outside_the_container():
    spec = {'nodes': {'work': {
        'class': f'{__name__}.Infer',
        'executable': 'python -m pkg.infer',
        'algo_params': {'model_id': None},
        'out_paths': {'results_fpath': 'results.json'},
        'load_result': 'pkg.results.load',
    }}}
    command = _node(
        spec,
        {'model_id': 'qwen3-8b'},
        settings=_settings(),
        lease=leasing.LeaseSettings(enabled=True),
    ).command
    assert command.index('infer-stack run') < command.index('docker run')


def test_the_same_integration_node_can_lease_without_a_container():
    spec = {'nodes': {'work': {
        'class': f'{__name__}.Infer',
        'executable': 'python -m pkg.infer',
        'algo_params': {'model_id': None},
        'out_paths': {'results_fpath': 'results.json'},
    }}}
    command = _node(
        spec,
        {'model_id': 'qwen3-8b'},
        lease=leasing.LeaseSettings(enabled=True),
    ).command
    assert command.startswith('infer-stack run')
    assert 'docker run' not in command


def _kwdagger_widens_the_node_spec():
    """Whether installed kwdagger honours `extra_node_spec_keys` (>=0.4.1)."""
    import inspect
    from kwdagger import yaml_pipeline
    return 'extra_node_spec_keys' in inspect.getsource(yaml_pipeline)


@pytest.mark.skipif(
    not _kwdagger_widens_the_node_spec(),
    reason='needs kwdagger >= 0.4.1 for extra_node_spec_keys',
)
def test_a_card_can_declare_endpoint_params():
    spec = {'nodes': {'work': {
        'class': MAGNET_CLASS,
        'endpoint_params': ['model_id'],
        'lease_ttl': '2h',
        'executable': 'python -m pkg.infer',
        'algo_params': {'model_id': None},
        'out_paths': {'results_fpath': 'results.json'},
    }}}
    node = _node(
        spec,
        {'model_id': 'qwen3-8b'},
        settings=_settings(),
        lease=leasing.LeaseSettings(enabled=True),
    )
    assert node.endpoint_params == ['model_id']
    assert '--endpoint qwen3-8b' in node.command
    assert '--ttl 2h' in node.command


@pytest.mark.skipif(
    not _kwdagger_widens_the_node_spec(),
    reason='needs kwdagger >= 0.4.1 for extra_node_spec_keys',
)
def test_an_unknown_node_spec_key_is_still_refused():
    spec = {'nodes': {'work': {
        'class': MAGNET_CLASS,
        'not_a_real_key': 1,
        'executable': 'python -m pkg.infer',
        'out_paths': {'results_fpath': 'results.json'},
    }}}
    with pytest.raises(ValueError, match='not_a_real_key'):
        coerce_pipeline(spec)


# --- the guard: an execution setting that reaches nothing is a failed run ----


def _pipeline(*node_classes):
    nodes = {
        f'n{idx}': {
            'executable': 'python -m pkg.work',
            'out_paths': {'results_fpath': 'results.json'},
            **({'class': cls} if cls else {}),
        }
        for idx, cls in enumerate(node_classes)
    }
    return coerce_pipeline({'nodes': nodes})


def test_an_image_that_reaches_no_node_is_an_error():
    with pytest.raises(ValueError) as excinfo:
        _check_container_settings_apply(_pipeline(None, None), _settings())
    message = str(excinfo.value)
    assert 'MagnetProcessNode' in message
    assert IMAGE in message


def test_no_image_means_no_opinion():
    _check_container_settings_apply(_pipeline(None, None), _settings(image=''))


def test_an_image_every_node_can_use_is_fine():
    _check_container_settings_apply(
        _pipeline(MAGNET_CLASS, MAGNET_CLASS), _settings()
    )


def test_a_mixed_dag_warns_but_runs(caplog):
    _check_container_settings_apply(_pipeline(MAGNET_CLASS, None), _settings())


def test_the_guard_runs_before_anything_is_submitted(monkeypatch):
    from magnet import _kwdagger

    submitted = []
    monkeypatch.setattr(
        _kwdagger,
        'build_schedule',
        lambda config: submitted.append(config) or (None, None),
    )
    processor = _kwdagger.KWDaggerProcessor(
        {'result_node': 'n0',
         'pipeline': {'nodes': {'n0': {
             'executable': 'python -m pkg.work',
             'out_paths': {'results_fpath': 'results.json'}}}}},
        '.',
    )
    with pytest.raises(ValueError):
        processor.schedule(container_settings=_settings())
    assert submitted == []
