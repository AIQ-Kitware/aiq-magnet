"""Containerized EvaluationNodes: preflight resolves where execution runs (M3/M8).

Runs a real ``docker run``. The engine worker interpreter is mounted only
inside the container (at ``/opt/aiq-inspect-worker``), so a preflight on the
scheduling host could not resolve it; the node must preflight through its
container wrapper, exactly like its command.

Needs ``MAGNET_TEST_DOCKER=1``, a usable ``docker``, ``$AIQ_EVALS_INSPECT_PYTHON``,
and ``$MAGNET_TEST_CONTAINER_VENV``: a MAGNET environment (with
``magnet_evals``) whose interpreter also works inside a plain ``ubuntu:24.04``
image, e.g. a venv on a uv-managed Python. Its and the worker's base
interpreters are bind-mounted at their host paths.
"""
import json
import os
from pathlib import Path

from aiq_evals_support import (
    HAS_DOCKER,
    INSPECT_PYTHON,
    evaluate,
    evaluation_node,
    evaluations,
    needs,
    recipe_rows,
    require_magnet_evals,
    write_recipe,
)

magnet_evals = require_magnet_evals()

IMAGE = os.environ.get('MAGNET_TEST_CONTAINER_IMAGE', 'ubuntu:24.04')
CONTAINER_VENV = os.environ.get('MAGNET_TEST_CONTAINER_VENV')
WORKER_MOUNT = '/opt/aiq-inspect-worker'


def _venv_base(venv: Path) -> Path:
    home = next(
        line.split('=', 1)[1].strip()
        for line in (venv / 'pyvenv.cfg').read_text().splitlines()
        if line.split('=', 1)[0].strip() == 'home'
    )
    return Path(home).parent


@needs(HAS_DOCKER and INSPECT_PYTHON is not None and CONTAINER_VENV is not None,
       'needs MAGNET_TEST_DOCKER=1, $AIQ_EVALS_INSPECT_PYTHON, and $MAGNET_TEST_CONTAINER_VENV')
def test_container_only_worker_resolves_and_executes_in_the_container(tmp_path):
    import magnet
    from magnet.containers import ContainerSettings

    assert INSPECT_PYTHON is not None and CONTAINER_VENV is not None
    inspect_venv = Path(INSPECT_PYTHON).parent.parent
    container_venv = Path(CONTAINER_VENV)
    assert not Path(WORKER_MOUNT).exists(), 'the worker must exist only in the container'
    mounts = sorted({
        str(_venv_base(container_venv)),
        str(_venv_base(inspect_venv)),
        str(container_venv),
        str(Path(magnet.__file__).resolve().parents[1]),        # MAGNET source (editable)
        str(Path(magnet_evals.__file__).resolve().parents[1]),  # aiq-magnet-evals source
        str(tmp_path),
    })
    settings = ContainerSettings.coerce(
        image=IMAGE,
        mounts=mounts,
        env={'PATH': f'{container_venv}/bin:/usr/bin:/bin'},
        docker_args=f'-v {inspect_venv}:{WORKER_MOUNT}:ro',
    )
    algo = {
        'engine': 'inspect_ai',
        'task': 'python:magnet_evals.examples.inspect_tasks:generation',
        'data_revision': 'example-v1',
        'models': [{'role': 'primary', 'model': 'local', 'provider': 'aiq_example', 'revision': 'local-v1'}],
        'engine_options': {'registration_modules': ['magnet_evals.examples.inspect_tasks']},
        'select': {'scorer': 'match', 'metric': 'accuracy'},
    }
    fpath = write_recipe(
        tmp_path, {'evaluate': evaluation_node(algo, worker=f'{WORKER_MOUNT}/bin/python')},
        claim='assert metrics.evaluate.score == 1.0',
    )
    out = tmp_path / 'out'
    recipe, card = evaluate(fpath, out, container_settings=settings)
    assert card.result == 'VERIFIED'
    (record_path,) = evaluations(out)
    record = json.loads(record_path.read_text())
    # Preflight (in the container) and execution (in the container) agree.
    assert record['preflight_identity'] == record['measurement_identity']['digest']
    assert record['measurement_identity']['reusable']
    (row,) = recipe_rows(recipe)
    assert row['row']['metrics.evaluate.score'] == 1.0

    # A second schedule preflights in the container again and reuses the node.
    _, again = evaluate(fpath, out, container_settings=settings)
    assert again.result == 'VERIFIED' and len(evaluations(out)) == 1
