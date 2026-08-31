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
from magnet.containers import ContainerYamlProcessNode
from magnet.leasing import LeasedYamlProcessNode

IMAGE = 'aiq-eval-node:latest'

CONTAINER_CLASS = 'magnet.containers.ContainerYamlProcessNode'


@pytest.fixture(autouse=True)
def _clean_settings():
    containers.configure()
    leasing.configure(False)
    yield
    containers.configure()
    leasing.configure(False)


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


def _node(spec, config={'task': 't'}):
    pipeline = coerce_pipeline(spec)
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
    containers.configure(image=IMAGE, mounts='/repo')
    command = _node(_spec()).command
    assert 'docker run' not in command


def test_a_declarative_node_runs_in_the_image():
    containers.configure(image=IMAGE, mounts='/repo')
    command = _node(_spec(CONTAINER_CLASS)).command
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

    class Infer(LeasedYamlProcessNode):
        endpoint_params = ('model_id',)

    containers.configure(image=IMAGE, mounts='/repo')
    leasing.configure(True)
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
    command = _node(spec, {'model_id': 'qwen3-8b'}).command
    assert command.index('infer-stack run') < command.index('docker run')


def test_the_node_is_a_kwdagger_process_node():
    """kwdagger's loader checks this before anything else."""
    assert issubclass(ContainerYamlProcessNode, kwdagger.ProcessNode)
