"""
Containerized execution for cards that declare their DAG in YAML.

A card that inlines ``kwdagger.pipeline.nodes`` gets
:class:`~kwdagger.yaml_pipeline.YamlProcessNode`, a *sibling* of
:class:`~magnet.containers.ContainerProcessNode`. So ``--container_image`` was
accepted, stored, and never read: the run went green having containerized
nothing. These tests pin the composition that fixes it.
"""

import kwdagger
import pytest
from kwdagger.pipeline import coerce_pipeline
from kwdagger.yaml_pipeline import YamlProcessNode

from magnet import containers, leasing
from magnet._kwdagger import _check_container_settings_apply
from magnet.containers import ContainerYamlProcessNode
from magnet.leasing import LeasedYamlProcessNode

IMAGE = 'aiq-eval-node:latest'

CONTAINER_CLASS = 'magnet.containers.ContainerYamlProcessNode'


class Infer(LeasedYamlProcessNode):
    """Leasing needs `endpoint_params`, which has no node-spec key, so a card
    that leases declares a subclass and names that. Module scope on purpose:
    `class:` is resolved by importing the module and walking attributes, so a
    class defined inside a test function cannot be found."""

    endpoint_params = ('model_id',)


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
    """One node out of a built DAG, carrying this test's execution settings.

    Settings are applied to the pipeline, not to the node, because that is how
    they arrive in a real run: MAGNET never constructs these nodes -- kwdagger's
    loader does, from the class the card names -- so the DAG is the only place
    it can reach them.
    """
    pipeline = coerce_pipeline(spec)
    containers.apply_settings(
        pipeline, settings or containers.ContainerSettings()
    )
    leasing.apply_settings(pipeline, lease or leasing.LeaseSettings())
    node = pipeline.node_dict['work']
    node.configure(config)
    return node


def test_both_families_are_honoured():
    """The point of the class: container behaviour *and* declarative data."""
    assert issubclass(ContainerYamlProcessNode, containers.ContainerProcessNode)
    assert issubclass(ContainerYamlProcessNode, YamlProcessNode)
    assert issubclass(LeasedYamlProcessNode, leasing.LeasedProcessNode)
    assert issubclass(LeasedYamlProcessNode, YamlProcessNode)


def test_a_plain_yaml_node_still_cannot_containerize():
    """The defect this class exists for, stated as a fact about the old path.

    Not an aspiration to fix in place: a card that names no class keeps the
    behaviour it has today, which is what makes this addition safe.
    """
    command = _node(_spec(), settings=_settings()).command
    assert 'docker run' not in command


def test_a_declarative_node_runs_in_the_image():
    command = _node(_spec(CONTAINER_CLASS), settings=_settings()).command
    assert command.startswith('docker run --rm ')
    assert f' {IMAGE} ' in command
    assert 'python -m pkg.work' in command
    assert command.rstrip().endswith('--task=t')


def test_it_is_inert_until_an_image_is_named():
    """The same card must run on the host during development."""
    command = _node(_spec(CONTAINER_CLASS)).command
    assert 'docker run' not in command
    assert command.startswith('python -m pkg.work')


def test_the_declarative_extras_survive():
    """kwdagger rejects `load_result` for a non-YamlProcessNode `class`.

    That rejection is why containerized execution and declarative readout
    were mutually exclusive, and why inheriting from both is the fix rather
    than an alias.
    """
    node = _node(_spec(CONTAINER_CLASS, load_result='pkg.results.load'))
    assert node._load_result_ref == 'pkg.results.load'
    assert hasattr(node, 'load_result')


def test_a_non_yaml_class_is_still_rejected_with_its_extras():
    """The guard we are satisfying, not circumventing."""
    with pytest.raises(ValueError, match='YamlProcessNode'):
        coerce_pipeline(
            _spec('magnet.containers.ContainerProcessNode',
                  load_result='pkg.results.load')
        )


def test_metrics_metadata_survives():
    node = _node(_spec(CONTAINER_CLASS, metrics=[{'name': 'auc'}]))
    assert node.default_metrics() == [{'name': 'auc'}]


def test_the_lease_stays_outside_the_container():
    """Acquiring a lease needs the host daemon and ledger; consuming the
    endpoint happens inside. Documented in magnet.containers."""
    spec = {
        'nodes': {
            'work': {
                'class': f'{__name__}.Infer',
                'executable': 'python -m pkg.infer',
                'algo_params': {'model_id': None},
                'out_paths': {'results_fpath': 'results.json'},
                'load_result': 'pkg.results.load',
            }
        }
    }
    command = _node(
        spec, {'model_id': 'qwen3-8b'},
        settings=_settings(), lease=leasing.LeaseSettings(enabled=True),
    ).command
    assert command.index('infer-stack run') < command.index('docker run')


def test_the_node_is_a_kwdagger_process_node():
    """kwdagger's loader checks this before anything else."""
    assert issubclass(ContainerYamlProcessNode, kwdagger.ProcessNode)


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
    """The defect this guard exists for: a green run that containerized
    nothing produces evidence indistinguishable from the real thing."""
    with pytest.raises(ValueError) as excinfo:
        _check_container_settings_apply(_pipeline(None, None), _settings())
    message = str(excinfo.value)
    # The message has to carry the fix, not just the complaint.
    assert 'ContainerYamlProcessNode' in message
    assert IMAGE in message


def test_no_image_means_no_opinion():
    """Running on the host is the default, not a degraded mode."""
    _check_container_settings_apply(_pipeline(None, None), _settings(image=''))


def test_an_image_every_node_can_use_is_fine():
    _check_container_settings_apply(
        _pipeline(CONTAINER_CLASS, CONTAINER_CLASS), _settings()
    )


def test_a_mixed_dag_warns_but_runs(caplog):
    """Legitimate: an analysis step may belong on the host beside a
    containerized model step. Say which nodes stay behind; do not refuse."""
    _check_container_settings_apply(
        _pipeline(CONTAINER_CLASS, None), _settings()
    )


def test_the_guard_runs_before_anything_is_submitted(monkeypatch):
    """A late error would leave a queue half-built and jobs running."""
    from magnet import _kwdagger

    submitted = []
    monkeypatch.setattr(
        _kwdagger, 'build_schedule',
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
