"""Shared helpers for the aiq-magnet-evals integration tests.

The integration tests need ``magnet_evals`` and, for end-to-end checks, real
engine worker interpreters:

* HELM: ``$AIQ_EVALS_HELM_PYTHON``, default this interpreter (MAGNET's
  ``helm`` extra installs ``crfm-helm``);
* Inspect: ``$AIQ_EVALS_INSPECT_PYTHON``;
* OLMo Eval: ``$AIQ_EVALS_OLMO_PYTHON``;
* an aiq-magnet-evals source checkout (native multi-result fixtures and
  test-only plugins): ``$AIQ_EVALS_REPO``, default the checkout ``magnet_evals``
  is imported from, when it is one;
* Docker (container preflight/execution): ``$MAGNET_TEST_DOCKER=1``.

Unavailable prerequisites skip, unless ``$MAGNET_REQUIRE_AIQ_EVALS=1``: then
they fail, so a CI job that is meant to exercise the integration cannot pass
by skipping it.
"""
from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import ubelt as ub
import yaml

REQUIRED = os.environ.get('MAGNET_REQUIRE_AIQ_EVALS', '').strip().lower() in {'1', 'true', 'yes'}


def require_magnet_evals():
    """Import ``magnet_evals`` or skip the module (fail when required)."""
    try:
        import magnet_evals
    except ImportError:
        if REQUIRED:
            raise
        pytest.skip(
            'magnet_evals is not installed (MAGNET_REQUIRE_AIQ_EVALS=1 makes this fail)',
            allow_module_level=True,
        )
    return magnet_evals


def needs(available: bool, reason: str):
    """Skip when a prerequisite is unavailable, or fail when it is required."""
    if available:
        return lambda func: func
    if REQUIRED:
        def fail(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                pytest.fail(f'MAGNET_REQUIRE_AIQ_EVALS=1 but unavailable: {reason}')
            return wrapper
        return fail
    return pytest.mark.skip(reason=reason)


def _python_has(python: str | None, *modules: str) -> bool:
    if not python or not Path(python).exists():
        return False
    code = 'import importlib.util, sys; sys.exit(any(importlib.util.find_spec(m) is None for m in sys.argv[1:]))'
    return subprocess.run([python, '-c', code, *modules], capture_output=True).returncode == 0


def _env_python(var: str) -> str | None:
    python = os.environ.get(var)
    return python if python and Path(python).exists() else None


HELM_PYTHON = os.environ.get('AIQ_EVALS_HELM_PYTHON', sys.executable)
HAS_HELM = _python_has(HELM_PYTHON, 'helm')
INSPECT_PYTHON = _env_python('AIQ_EVALS_INSPECT_PYTHON')
HAS_INSPECT = _python_has(INSPECT_PYTHON, 'inspect_ai')
OLMO_PYTHON = _env_python('AIQ_EVALS_OLMO_PYTHON')
HAS_OLMO = _python_has(OLMO_PYTHON, 'olmo_eval')


def _aiq_evals_repo() -> Path | None:
    configured = os.environ.get('AIQ_EVALS_REPO')
    if configured:
        root = Path(configured)
    else:
        try:
            import magnet_evals
        except ImportError:
            return None
        root = Path(magnet_evals.__file__).resolve().parents[1]
    return root if (root / 'tests' / 'fixtures').is_dir() else None


REPO = _aiq_evals_repo()


def _docker_usable() -> bool:
    if os.environ.get('MAGNET_TEST_DOCKER', '').strip().lower() not in {'1', 'true', 'yes'}:
        return False
    if shutil.which('docker') is None:
        return False
    return subprocess.run(['docker', 'info'], capture_output=True).returncode == 0


HAS_DOCKER = _docker_usable()

needs_helm = needs(HAS_HELM, 'needs crfm-helm in $AIQ_EVALS_HELM_PYTHON (default: this interpreter)')
needs_inspect = needs(HAS_INSPECT, 'needs $AIQ_EVALS_INSPECT_PYTHON with inspect-ai')
needs_olmo = needs(HAS_OLMO, 'needs $AIQ_EVALS_OLMO_PYTHON with olmo-eval')
needs_repo = needs(REPO is not None, 'needs an aiq-magnet-evals source checkout ($AIQ_EVALS_REPO)')


def repo() -> Path:
    """The aiq-magnet-evals checkout (tests using it are marked ``needs_repo``)."""
    assert REPO is not None
    return REPO


def repo_on_worker_path(monkeypatch) -> None:
    """Let workers import the checkout's test-only ``tests.native`` modules."""
    assert REPO is not None
    old = os.environ.get('PYTHONPATH')
    monkeypatch.setenv('PYTHONPATH', str(REPO) if not old else f'{REPO}{os.pathsep}{old}')


HELM_ALGO: dict[str, Any] = {
    'engine': 'helm',
    'task': 'simple_mcqa',
    'task_revision': 'helm-builtin',
    'data_revision': 'helm-builtin',
    'models': [{'role': 'primary', 'model': 'simple/model1', 'revision': 'local-v1'}],
    'task_options': {'max_eval_instances': 1},
}
HELM_SELECT = {'metric': 'exact_match', 'group': 'test', 'score': None}


def evaluation_node(algo: dict[str, Any], *, worker: str | None = HELM_PYTHON, **perf: Any) -> dict[str, Any]:
    return {
        'class': 'magnet.backends.aiq_evals.EvaluationNode',
        'algo_params': dict(algo),
        'perf_params': {'worker_python': worker, **perf},
    }


def write_recipe(
    dpath: Any,
    nodes: dict[str, Any],
    *,
    claim: str = 'assert metrics.evaluate.score >= 0',
    matrix: dict[str, Any] | None = None,
    edges: list[str] | None = None,
    scope: str = 'requested',
    name: str = 'aiq_evals_probe',
) -> Path:
    pipeline: dict[str, Any] = {'nodes': nodes}
    if edges:
        pipeline['edges'] = edges
    kwdagger: dict[str, Any] = {'result_node': 'evaluate', 'pipeline': pipeline}
    if matrix:
        kwdagger['matrix'] = matrix
    fpath = Path(dpath) / f'{name}.yaml'
    fpath.write_text(yaml.safe_dump({
        'name': name,
        'title': name,
        'description': 'aiq-magnet-evals integration test recipe',
        'version': '1.0',
        'organizations': ['Kitware'],
        'submitter': {'name': 't', 'email': 't@example.com'},
        'links': [],
        'tags': ['test'],
        'claim': {'python': claim},
        'evidence': {'scope': scope},
        'kwdagger': kwdagger,
    }, sort_keys=False))
    return fpath


def evaluate(recipe_fpath: Any, out_dpath: Any, params: Any = None, **options: Any):
    """Run ``evaluate_new`` on a recipe; returns ``(recipe, card)``."""
    from magnet.evaluation_new import NewEvaluationRecipe

    recipe = NewEvaluationRecipe(recipe_fpath, ub.Path(out_dpath), validate='off')
    if params is not None:
        recipe.apply_params(params)
    options.setdefault('backend', 'serial')
    card = recipe.evaluate(**options)
    return recipe, card


def recipe_rows(recipe) -> list[dict[str, Any]]:
    from magnet._kwdagger import KWDaggerProcessor

    processor = KWDaggerProcessor(recipe.kwdagger, root_dpath=recipe.kwdagger_dpath)
    return processor.load_available_result_rows()


def evaluations(out_dpath: Any, node: str = 'evaluate') -> list[Path]:
    return sorted((Path(out_dpath) / '_kwdagger' / node).glob('*/evaluation.json'))


def store_attempts(out_dpath: Any) -> list[Path]:
    return sorted((Path(out_dpath) / '_kwdagger' / '_aiq_evals_store' / 'attempts').rglob('ATTEMPT_TERMINAL'))


def verdicts(out_dpath: Any) -> list[dict[str, Any]]:
    """Per-row verdicts of every invocation under ``out_dpath``."""
    return [json.loads(p.read_text()) for p in Path(out_dpath).glob('*/results/*/verdict.json')]


def latest_run_dir(out_dpath: Any) -> Path:
    """The most recent invocation's run directory (names start with a recipe hash)."""
    runs = [p for p in Path(out_dpath).iterdir() if p.is_dir() and (p / 'card.yaml').exists()]
    return max(runs, key=lambda p: (p / 'card.yaml').stat().st_mtime_ns)


def run_verdicts(run_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(run_dir.glob('results/*/verdict.json'))]
