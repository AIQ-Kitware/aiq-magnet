r"""
magnet.backends.helm.cli.materialize_helm_run_from_spec
=======================================================

Faithful-replay sibling of :mod:`materialize_helm_run`.

Why this exists (the central insight)
-------------------------------------
A run-entry string (``mmlu:subject=...,model=X``) is the *pre-expansion* input
to ``helm-run``: it is fed through ``construct_run_specs`` + run-expanders, which
**derive** ``metric_specs``, fill ``adapter_spec`` defaults, and apply ``model=``
expansion using whatever the *currently installed* crfm-helm version's defaults
are. A ``run_spec.json`` is written *after* all of that — it is the **fully
resolved** recipe (exact ``scenario_spec``, ``adapter_spec``, ``metric_specs``,
``annotators``, ``data_augmenter_spec``).

So the sibling :mod:`materialize_helm_run`, which reconstructs a run-entry string
and re-parses it through ``helm-run``, re-derives the recipe under *today's*
library defaults — a silent drift surface. This module instead **deserializes the
``run_spec.json`` directly** (HELM's symmetric ``helm.common.codec.from_json``
codec) and hands the resolved :class:`RunSpec` straight to
``helm.benchmark.run.run_benchmarking``. For a paper whose claim is
"same recipe -> same metrics," replaying the resolved spec is the defensible
choice.

In *discovery* mode the run-entry string's only remaining job is to **locate**
the official run dir (directory-name matching, which is robust); the authoritative
``run_spec.json`` then drives execution. With an explicit ``--run-spec-json``
path we skip even that.

Two input modes (dual-input, plan section 4)
--------------------------------------------
* **Explicit** ``--run-spec-json <path>`` -> read that file directly. Works on
  *any* spec file (a modified/hand-authored spec, a non-public spec, an EEE
  sidecar, or one outside a ``benchmark_output/`` tree). The explicit path wins
  when both are given.
* **Discovery** ``--run-entry <str>`` + ``--precomputed-root <root>`` -> locate
  the matching official run dir via :func:`find_best_precomputed_run` and read
  its ``run_spec.json``. This is what the eval_audit pipeline uses (it is keyed
  on run-entry strings and the manifest carries no per-entry paths).

Substitution is by-name only (plan section 5)
---------------------------------------------
This CLI never rewrites ``adapter_spec.model`` / ``model_deployment``. The
``run_spec.json`` keeps its official deployment name; the locally registered
override (``model_deployments.yaml``, copied in by
:func:`prepare_local_helm_config`) binds that name to a local
``HuggingFaceClient``. The only field this CLI mutates is
``adapter_spec.max_eval_instances`` (when ``--max-eval-instances`` is set), so
the produced run dir name stays ``run_spec.name`` and downstream indexing /
``logical_run_key`` / planner pairing are untouched.

Version drift fails two ways (plan section 1)
---------------------------------------------
* **Class availability is loud.** A renamed / not-yet-existing
  scenario/metric/annotator class surfaces as an ``ImportError`` (or
  ``AttributeError``) naming the exact missing class. We convert this into an
  explicit *preflight* (:func:`preflight_resolve_classes`) so a version mismatch
  is an actionable message, not a mid-run crash.
* **Field/shape drift is silent.** HELM's codec is a plain ``cattrs.Converter``
  that ignores unknown keys and fills missing optionals with defaults, so a spec
  written by an older crfm-helm deserializes *successfully* with the old field
  dropped. ``from_json`` cannot detect that the *source* shape differed; the
  schema-drift unit test (``tests/test_materialize_from_spec.py``) guards this by
  asserting no raw JSON key is silently dropped on round-trip.
"""

from __future__ import annotations

import dataclasses
import os
import time
import traceback
from pathlib import Path
from typing import Any, Iterator

import kwutil
import scriptconfig as scfg

from loguru import logger

# Reuse the run-entry materializer's scaffolding verbatim (plan section 4). These
# are package-internal helpers; importing them keeps the two CLIs symmetric and
# means a fix to discovery/matching/local-config benefits both paths.
from magnet.backends.helm.cli.materialize_helm_run import (
    find_best_precomputed_run,
    find_run_in_out_dpath,
    prepare_local_helm_config,
    write_helm_log_config,
    _capture_process_context,
    _normalize_optional_pathish,
    _CMD_STREAM_TAIL_BYTES,
)


class MaterializeHelmRunFromSpecConfig(scfg.DataConfig):
    """
    Replay a single fully-resolved HELM ``run_spec.json`` faithfully.

    Accepts the same flags as :class:`MaterializeHelmRunConfig` (so the
    containerized docker node can render them from ``final_config`` unchanged),
    plus ``run_spec_json`` for the explicit-path input mode. There are no
    ``--model`` / ``--model-deployment`` flags: substitution is by-name only.
    """

    run_entry = scfg.Value(
        None,
        help=(
            "HELM run-entry description string, e.g. "
            "'mmlu:subject=philosophy,model=openai/gpt2'. In discovery mode this "
            "is used only to LOCATE the official run dir (whose run_spec.json is "
            "replayed); it is never re-parsed through helm-run."
        ),
        tags=['algo_param'],
        type=str,
    )

    run_spec_json = scfg.Value(
        None,
        help=(
            "Explicit path to a run_spec.json to replay. When set, discovery is "
            "skipped entirely and this exact recipe is used (the most faithful "
            "form). Wins over --run-entry/--precomputed-root if both are given."
        ),
        tags=['algo_param'],
        type=str,
    )

    suite = scfg.Value(
        'default-suite',
        help='HELM suite name to use for output layout (and run_benchmarking --suite).',
        tags=['algo_param'],
    )

    out_dpath = scfg.Value(
        None,
        help='Output directory (kwdagger node output directory).',
        tags=['out_path'],
    )

    precomputed_root = scfg.Value(
        [],
        help=(
            'In discovery mode this is the RECIPE SOURCE: the root searched for '
            'the official run dir whose run_spec.json is replayed (not just a '
            'reuse cache). Required when --run-spec-json is not given.'
        ),
        tags=['in_param'],
    )

    max_eval_instances = scfg.Value(
        None,
        type=int,
        help=(
            'When set, the ONLY field substituted into the resolved recipe: '
            'adapter_spec.max_eval_instances is replaced verbatim (truncating to '
            "HELM's deterministic instance prefix). When unset, replay exactly."
        ),
        tags=['algo_param'],
    )

    require_per_instance_stats = scfg.Value(
        True,
        help='Require per_instance_stats.json when locating the produced run dir.',
        tags=['algo_param'],
    )

    mode = scfg.Value(
        'compute_if_missing',
        choices=['reuse_only', 'compute_if_missing', 'force_recompute'],
        help=(
            'Accepted for flag-compatibility with the run-entry materializer. The '
            'from-spec path ALWAYS replays the recipe (precomputed_root is the '
            'recipe source, not a reuse cache); "already computed" is delegated to '
            "kwdagger's skip_existing DONE sentinel. reuse_only is rejected."
        ),
        tags=['perf_param'],
    )

    materialize = scfg.Value(
        'symlink',
        choices=['symlink', 'copy'],
        help='Accepted for flag-compatibility; unused (the from-spec path computes).',
        tags=['perf_param'],
    )

    num_threads = scfg.Value(
        1,
        type=int,
        help='Passed to run_benchmarking (parallelism).',
        tags=['perf_param'],
    )

    local_path = scfg.Value(
        'prod_env',
        type=str,
        help='HELM local config path. Relative paths are resolved inside out_dpath.',
        tags=['perf_param'],
    )

    model_deployments_fpath = scfg.Value(
        None,
        type=str,
        help=(
            'Optional path to a HELM model_deployments.yaml override copied into '
            '<local_path>/model_deployments.yaml before replay. This is the '
            'by-name substitution mechanism (binds the official deployment name '
            'to a local HuggingFaceClient).'
        ),
        tags=['algo_param'],
    )

    enable_huggingface_models = scfg.Value(
        None,
        type=str,
        help='Optional YAML-encoded list, mirrored from helm-run --enable-huggingface-models.',
        tags=['algo_param'],
    )

    enable_local_huggingface_models = scfg.Value(
        None,
        type=str,
        help='Optional YAML-encoded list, mirrored from helm-run --enable-local-huggingface-models.',
        tags=['algo_param'],
    )

    done_fname = scfg.Value(
        'DONE',
        help='Name of the sentinel file written last when the node is complete.',
        tags=['out_path', 'primary'],
    )

    manifest_fname = scfg.Value(
        'adapter_manifest.json',
        help='Name of the JSON manifest describing what happened.',
        tags=['out_path'],
    )

    @classmethod
    def main(cls, argv=None, **kwargs) -> dict:
        """
        Entry point. Returns (and writes) the adapter manifest.

        Example:
            >>> # xdoctest: +REQUIRES(env:HELM_RUN_AVAILABLE)
            >>> import ubelt as ub
            >>> from magnet.backends.helm.cli.materialize_helm_run_from_spec import (
            ...     MaterializeHelmRunFromSpecConfig)
            >>> dpath = ub.Path.appdir('magnet/tests/materialize_from_spec').delete().ensuredir()
            >>> MaterializeHelmRunFromSpecConfig.main([
            ...   '--run-entry', 'mmlu:subject=philosophy,model=openai/gpt2',
            ...   '--precomputed-root', '/data/crfm-helm-public',
            ...   '--suite', 'my-suite',
            ...   '--max-eval-instances', '2',
            ...   '--out-dpath', str(dpath),
            ... ])
        """
        config = cls.cli(argv=argv, data=kwargs, verbose='auto')
        config.run_entry = _normalize_optional_pathish(config.run_entry)
        config.run_spec_json = _normalize_optional_pathish(config.run_spec_json)
        config.precomputed_root = _normalize_optional_pathish(config.precomputed_root)
        config.model_deployments_fpath = _normalize_optional_pathish(
            config.model_deployments_fpath
        )
        config.enable_huggingface_models = kwutil.Yaml.coerce(
            config.enable_huggingface_models
        )
        config.enable_local_huggingface_models = kwutil.Yaml.coerce(
            config.enable_local_huggingface_models
        )

        if config.suite is None:
            raise SystemExit('Missing required --suite')
        if config.out_dpath is None:
            raise SystemExit('Missing required --out-dpath')
        if config.mode == 'reuse_only':
            raise SystemExit(
                'mode=reuse_only is meaningless for the from-spec replay path '
                '(it always recomputes the recipe). Use the run-entry '
                'materializer if you want to reuse precomputed official outputs.'
            )

        out_dpath = Path(config.out_dpath).expanduser().resolve()
        out_dpath.mkdir(parents=True, exist_ok=True)

        done_fpath = out_dpath / config.done_fname
        manifest_fpath = out_dpath / config.manifest_fname

        # 1) Resolve the recipe path (explicit wins; else discovery).
        run_spec_path, recipe_source = _resolve_run_spec_path(config)
        logger.info('Replaying run_spec.json ({}): {}', recipe_source, run_spec_path)

        manifest: dict = {
            'requested': {
                'run_entry': config.run_entry,
                'run_spec_json': config.run_spec_json,
                'suite': config.suite,
                'max_eval_instances': config.max_eval_instances,
                'require_per_instance_stats': config.require_per_instance_stats,
                'local_path': config.local_path,
                'model_deployments_fpath': config.model_deployments_fpath,
                'enable_huggingface_models': list(config.enable_huggingface_models or []),
                'enable_local_huggingface_models': list(
                    config.enable_local_huggingface_models or []
                ),
            },
            'recipe': {
                'run_spec_path': str(run_spec_path),
                'source': recipe_source,
            },
            'substitution': 'by-name only (deployment names registered via model_deployments.yaml)',
            'status': None,
            'replay': None,
            'out_dpath': str(out_dpath),
            'timestamp': time.time(),
        }
        process_context = _capture_process_context(out_dpath, config)
        manifest['process_context_fpath'] = str(out_dpath / 'process_context.json')
        manifest['process_context'] = process_context

        # 2) Deserialize the resolved recipe (symmetric cattrs codec).
        from helm.benchmark.run_spec import RunSpec
        from helm.common.codec import from_json

        raw_text = run_spec_path.read_text()
        run_spec = from_json(raw_text, RunSpec)
        manifest['recipe']['run_spec_name'] = run_spec.name
        logger.info('Deserialized RunSpec: {}', run_spec.name)

        # 3) Prepare the local HELM config (copies the by-name override yaml).
        (out_dpath / 'benchmark_output').mkdir(exist_ok=True)
        prepared_local_path = prepare_local_helm_config(
            out_dpath=out_dpath,
            local_path=config.local_path,
            model_deployments_fpath=config.model_deployments_fpath,
        )

        # 4) Register the full environment BEFORE the preflight, mirroring the
        #    helm_run preamble verbatim so the preflight resolves classes in
        #    exactly the environment the run will use.
        _register_helm_environment(prepared_local_path, config)

        # 5) Preflight: resolve every class_name reachable from the spec (loud
        #    failure on version drift).
        preflight_resolve_classes(run_spec)

        # 6) Substitute (minimal): only max_eval_instances.
        applied_max_eval_instances = None
        if config.max_eval_instances is not None:
            adapter_spec = dataclasses.replace(
                run_spec.adapter_spec,
                max_eval_instances=int(config.max_eval_instances),
            )
            run_spec = dataclasses.replace(run_spec, adapter_spec=adapter_spec)
            applied_max_eval_instances = int(config.max_eval_instances)
            logger.info(
                'Applied max_eval_instances={} to adapter_spec',
                applied_max_eval_instances,
            )
        manifest['replay'] = {'applied_max_eval_instances': applied_max_eval_instances}

        # 7) Run, in-process. Wrap so a Python exception is captured to
        #    cmd_stderr.txt for the failure classifier (a hard crash —
        #    OOM-kill/segfault — bypasses this, but the missing DONE sentinel +
        #    non-zero exit still flag the failure).
        output_path = out_dpath / 'benchmark_output'
        _replay_run_spec(
            run_spec=run_spec,
            suite=config.suite,
            output_path=output_path,
            local_path=prepared_local_path,
            num_threads=int(config.num_threads),
            out_dpath=out_dpath,
            manifest=manifest,
            manifest_fpath=manifest_fpath,
        )

        # 8) Locate the produced run dir + finalize.
        requested_desc = config.run_entry or run_spec.name
        computed_run_dir = find_run_in_out_dpath(
            out_dpath=out_dpath,
            suite=config.suite,
            requested_desc=requested_desc,
            max_eval_instances=config.max_eval_instances,
            require_per_instance_stats=config.require_per_instance_stats,
        )
        if computed_run_dir is None:
            logger.warning(
                'Could not locate run via suite path; falling back to full scan'
            )
            fallback = find_best_precomputed_run(
                precomputed_root=out_dpath,
                requested_desc=requested_desc,
                max_eval_instances=None,
                require_per_instance_stats=False,
            )
            computed_run_dir = fallback.run_dir if fallback else None

        if computed_run_dir is None:
            manifest['status'] = 'error'
            manifest_fpath.write_text(kwutil.Json.dumps(manifest, indent=2))
            raise RuntimeError(
                'run_benchmarking completed, but the produced run directory '
                'could not be located/validated under out_dpath'
            )

        manifest['status'] = 'replayed'
        manifest['replay'].update(
            {
                'computed_run_dir': str(computed_run_dir),
                'computed_run_name': computed_run_dir.name,
            }
        )

        manifest_fpath.write_text(kwutil.Json.dumps(manifest, indent=2))
        logger.info('Wrote manifest: {}', manifest_fpath)

        # Sentinel last: node complete, outputs ready.
        done_fpath.write_text('ok\n')
        logger.success('Wrote DONE sentinel: {}', done_fpath)
        return manifest


def _resolve_run_spec_path(config) -> tuple[Path, str]:
    """Resolve the ``run_spec.json`` to replay (plan section 4, step 1).

    Returns ``(path, source)`` where ``source`` is ``'explicit'`` or
    ``'discovery'``. Raises ``SystemExit`` when no recipe can be resolved — we
    cannot replay without the recipe.
    """
    if config.run_spec_json:
        path = Path(config.run_spec_json).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f'--run-spec-json does not exist: {path}')
        return path, 'explicit'

    if not config.run_entry:
        raise SystemExit(
            'from-spec replay requires either --run-spec-json (explicit) or '
            '--run-entry + --precomputed-root (discovery)'
        )
    if not config.precomputed_root:
        raise SystemExit(
            'discovery mode requires --precomputed-root (the recipe source from '
            'which the official run_spec.json is read)'
        )
    match = find_best_precomputed_run(
        precomputed_root=config.precomputed_root,
        requested_desc=config.run_entry,
        max_eval_instances=None,
        require_per_instance_stats=False,
    )
    if match is None:
        raise SystemExit(
            f'No official run dir matched run_entry={config.run_entry!r} under '
            f'{config.precomputed_root!r}; cannot locate a run_spec.json to replay'
        )
    run_spec_path = Path(match.run_dir) / 'run_spec.json'
    if not run_spec_path.is_file():
        raise SystemExit(
            f'Matched official run dir {match.run_dir} has no run_spec.json'
        )
    return run_spec_path, 'discovery'


def _register_helm_environment(prepared_local_path: Path, config) -> None:
    """Mirror the ``helm_run`` registration preamble verbatim (run.py:284-302).

    Same calls, same order, so the replay environment is identical to a normal
    ``helm-run`` — notably ``load_entry_point_plugins()`` (entry-point plugins
    contribute scenarios / run-spec functions / model metadata; skipping them
    both breaks the run and could make the preflight false-positive on a
    plugin-provided class).
    """
    from helm.benchmark.config_registry import (
        register_builtin_configs_from_helm_package,
        register_configs_from_directory,
    )
    from helm.benchmark.run import load_entry_point_plugins

    register_builtin_configs_from_helm_package()
    register_configs_from_directory(os.fspath(prepared_local_path))
    load_entry_point_plugins()

    if config.enable_huggingface_models:
        from helm.benchmark.huggingface_registration import (
            register_huggingface_hub_model_from_flag_value,
        )

        for name in config.enable_huggingface_models:
            register_huggingface_hub_model_from_flag_value(str(name))

    if config.enable_local_huggingface_models:
        from helm.benchmark.huggingface_registration import (
            register_huggingface_local_model_from_flag_value,
        )

        for path in config.enable_local_huggingface_models:
            register_huggingface_local_model_from_flag_value(str(path))


def _iter_object_specs(value: Any) -> Iterator[Any]:
    """Yield every ObjectSpec / ``{'class_name': ...}`` dict reachable from ``value``.

    After ``from_json``, ``ObjectSpec.args`` is typed ``Dict[str, Any]``, so
    nested specs inside ``args`` (e.g. an annotator's judge model spec) come back
    as plain dicts rather than ObjectSpec instances. We therefore recurse through
    both shapes so the preflight resolves nested ``class_name`` values too.
    """
    from helm.common.object_spec import ObjectSpec

    if isinstance(value, ObjectSpec):
        yield value
        for sub in value.args.values():
            yield from _iter_object_specs(sub)
    elif isinstance(value, dict):
        if isinstance(value.get('class_name'), str):
            yield value
        for sub in value.values():
            yield from _iter_object_specs(sub)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_object_specs(item)


def collect_class_names(run_spec) -> list[str]:
    """Collect every distinct ``class_name`` referenced by a RunSpec.

    Roots are ``scenario_spec``, each ``metric_specs`` entry, and each
    ``annotators`` entry; :func:`_iter_object_specs` recurses into nested
    ``args`` (plan section 4, step 5). ``adapter_spec`` is intentionally
    excluded — it carries deployment *names*, not importable class paths
    (substitution there is by-name).
    """
    from helm.common.object_spec import ObjectSpec

    roots: list[Any] = [run_spec.scenario_spec, *(run_spec.metric_specs or [])]
    if run_spec.annotators:
        roots.extend(run_spec.annotators)

    names: list[str] = []
    seen: set[str] = set()
    for root in roots:
        for spec in _iter_object_specs(root):
            class_name = (
                spec.class_name if isinstance(spec, ObjectSpec) else spec.get('class_name')
            )
            if isinstance(class_name, str) and class_name not in seen:
                seen.add(class_name)
                names.append(class_name)
    return names


def preflight_resolve_classes(run_spec) -> None:
    """Resolve every class referenced by the spec; fail fast and actionably.

    Two distinct failure causes both surface here as an unresolvable class, and
    both are worth reporting up front rather than as a mid-run crash inside the
    Runner:

    * **version drift** — a renamed / not-yet-existing class raises
      ``ImportError`` (module gone) or ``AttributeError`` (module present, class
      renamed);
    * **recipe/environment filter reason** — a scenario whose module imports a
      missing optional extra raises HELM's ``OptionalDependencyNotInstalled``
      (e.g. vision-language scenarios needing ``latex``/``image2struct``). Per
      the research taxonomy this is an environment constraint, not a
      reproducibility failure — such runs are normally filtered upstream, but the
      preflight must degrade gracefully if one reaches it.

    ``get_class_by_name`` imports the class's module, so it can raise *any*
    exception the module's top-level code raises. We therefore catch broadly and
    record ``(class_name, "<ExcType>: <msg>")`` for every class that fails to
    resolve, then raise once listing all of them — so one unresolvable class is
    reported, never crashed on, and the per-class error type lets a human tell
    drift (ImportError) from a missing extra (OptionalDependencyNotInstalled).
    """
    from helm.common.object_spec import get_class_by_name

    unresolved: list[tuple[str, str]] = []
    for class_name in collect_class_names(run_spec):
        if '.' not in class_name:
            # Not a dotted path: get_class_by_name cannot resolve it, and a
            # fully-resolved run_spec.json should never carry bare names. Skip
            # rather than manufacture a false failure.
            continue
        try:
            get_class_by_name(class_name)
        except Exception as ex:  # noqa: BLE001 - preflight reports, never crashes
            unresolved.append((class_name, f'{type(ex).__name__}: {ex}'))

    if unresolved:
        detail = '\n'.join(f'  - {name}: {err}' for name, err in unresolved)
        raise SystemExit(
            'Preflight failed: this crfm-helm build cannot resolve '
            f'{len(unresolved)} class(es) referenced by the run_spec.json:\n'
            f'{detail}\n'
            'Either the installed crfm-helm version differs from the one that '
            'produced this run (pin the helm submodule to the run\'s era), or a '
            'required optional dependency is missing (an environment/recipe '
            'filter reason, not a reproducibility failure).'
        )


def _replay_run_spec(
    run_spec,
    suite: str,
    output_path: Path,
    local_path: Path,
    num_threads: int,
    out_dpath: Path,
    manifest: dict,
    manifest_fpath: Path,
) -> None:
    """Drive ``run_benchmarking`` in-process (plan section 4, step 7).

    Mirrors the tail of ``helm_run``: ensure + set the benchmark output path,
    then call ``run_benchmarking`` on the single resolved spec. A Python
    exception is captured to ``cmd_stderr.txt`` (so the existing failure
    classifier has content) and re-raised.
    """
    from helm.benchmark.run import run_benchmarking
    from helm.benchmark.runner import set_benchmark_output_path
    from helm.common.authentication import Authentication
    from helm.common.general import ensure_directory_exists
    from helm.common.hierarchical_logger import setup_default_logging

    ensure_directory_exists(os.fspath(output_path))
    set_benchmark_output_path(os.fspath(output_path))

    try:
        setup_default_logging(os.fspath(write_helm_log_config(out_dpath)))
    except Exception:
        logger.exception('Failed to configure HELM logging (continuing)')

    try:
        run_benchmarking(
            run_specs=[run_spec],
            auth=Authentication(''),
            url=None,
            local_path=os.fspath(local_path),
            num_threads=num_threads,
            output_path=os.fspath(output_path),
            suite=suite,
            dry_run=False,
            skip_instances=False,
            cache_instances=False,
            cache_instances_only=False,
            skip_completed_runs=False,
            exit_on_error=True,
            runner_class_name=None,
        )
    except BaseException:
        tb = traceback.format_exc()
        _persist_stderr(out_dpath, tb)
        manifest['status'] = 'error'
        manifest['error'] = tb.strip().splitlines()[-1] if tb.strip() else None
        try:
            manifest_fpath.write_text(kwutil.Json.dumps(manifest, indent=2))
        except Exception:
            pass
        raise


def _persist_stderr(out_dpath: Path, text: str) -> None:
    """Write a tail of ``text`` to ``cmd_stderr.txt`` (best-effort)."""
    try:
        tail = (
            text[-_CMD_STREAM_TAIL_BYTES:]
            if len(text) > _CMD_STREAM_TAIL_BYTES
            else text
        )
        (out_dpath / 'cmd_stderr.txt').write_text(tail)
    except Exception:
        pass


__cli__ = MaterializeHelmRunFromSpecConfig

if __name__ == '__main__':
    __cli__.main()
