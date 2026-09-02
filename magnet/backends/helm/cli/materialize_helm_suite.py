r"""
magnet.backends.helm.cli.materialize_helm_suite
================================================

A kwdagger-shaped node that puts a *set* of precomputed HELM runs where a
downstream node expects a suite directory, cache-first.

:mod:`materialize_helm_run` handles one run entry. Several evaluation cards
read a whole suite -- every run of one scenario across the models a version
holds -- and today that suite is staged by hand into a path the card is then
pointed at. This node makes acquisition part of the pipeline: declare the
benchmark, the version and a run pattern, and the node produces
``<out>/benchmark_output/runs/<version>/<run>`` for every matching run,
symlinked from a precomputed root when one holds it and downloaded from the
public bucket otherwise. A ``DONE`` sentinel and then the JSON manifest are
written last; the manifest is the node's primary output, because kwdagger's
generic result loader parses the primary output as JSON and a bare sentinel
would fail that parse for every downstream aggregate.

Identity: benchmark, version and pattern are algorithm parameters (they say
*which* suite). Where the bytes come from -- ``precomputed_roots``, whether a
download is allowed -- is not: two nodes asking for the same suite are the
same computation whichever cache served it.

Usage::

    python -m magnet.backends.helm.cli.materialize_helm_suite \
        --benchmark lite --version v1.0.0 --runs 'regex:med_qa.*' \
        --precomputed_roots /data/crfm-helm-public \
        --suite_dpath ./node_out/benchmark_output/runs/v1.0.0 \
        --done_fpath ./node_out/DONE

In a declarative pipeline::

    nodes:
      materialize_lite:
        executable: 'python -m magnet.backends.helm.cli.materialize_helm_suite'
        algo_params: {benchmark: lite, version: v1.0.0, runs: 'regex:med_qa.*'}
        perf_params: {precomputed_roots: '', download: auto}
        out_paths:
          suite_dpath: benchmark_output/runs/v1.0.0
          manifest_fpath: materialize_manifest.json
          done_fpath: DONE
        primary_out_key: manifest_fpath
      score:
        in_paths: [helm_suite_path]
        ...
    edges:
      - materialize_lite.suite_dpath -> score.helm_suite_path
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import kwconf
import ubelt as ub
from loguru import logger

__all__ = ['MaterializeHelmSuiteConfig', 'materialize_suite', 'main']


class MaterializeHelmSuiteConfig(kwconf.Config):
    """Materialize every run of a HELM benchmark version that matches a pattern."""

    benchmark: str = kwconf.Value('lite', help='HELM benchmark name, e.g. lite, heim, ewok.')
    version: str = kwconf.Value(None, help='Benchmark version, e.g. v1.0.0.')
    runs: str = kwconf.Value(
        None,
        help=(
            'Which runs: a kwutil MultiPattern over run ids, e.g. '
            "'regex:(med_qa|legalbench).*' or a comma-separated list. "
            'None takes every run in the version.'
        ),
    )
    precomputed_roots: str | list[str] | None = kwconf.Value(
        None,
        help=(
            'Colon-separated local roots laid out like the public bucket '
            '(<root>/<benchmark>/benchmark_output/runs/<version>/<run>). A run '
            'found here is symlinked, never fetched.'
        ),
    )
    download: str = kwconf.Value(
        'auto', choices=['auto', 'never', 'always'],
        help=(
            'auto: fetch only what no precomputed root holds; never: refuse '
            'to fetch (an absent run is an error); always: fetch everything.'
        ),
    )
    bucket: str = kwconf.Value(
        'gs://crfm-helm-public',
        help='Remote bucket, or a local directory with the same layout (tests).',
    )
    materialize: str = kwconf.Value(
        'symlink', choices=['symlink', 'copy'],
        help='How a precomputed run is placed into the suite.',
    )
    suite_dpath: str = kwconf.Value(
        None, help='Output suite directory; the downstream node reads this.'
    )
    done_fpath: str | None = kwconf.Value(
        None, help='Sentinel written last. Default: <suite_dpath>/../../../DONE.'
    )
    manifest_fpath: str | None = kwconf.Value(
        None, help='JSON manifest, the primary output. Default: beside the sentinel.'
    )


def _coerce_roots(raw) -> list[Path]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        items = [str(x) for x in raw]
    else:
        items = str(raw).split(':')
    return [Path(x).expanduser() for x in items if x.strip()]


def _local_candidates(roots, benchmark, version) -> dict[str, Path]:
    """run id -> directory, first root wins."""
    found: dict[str, Path] = {}
    for root in roots:
        version_dpath = root / benchmark / 'benchmark_output' / 'runs' / version
        if not version_dpath.is_dir():
            continue
        for child in sorted(version_dpath.iterdir()):
            if child.is_dir() and child.name not in found:
                found[child.name] = child
    return found


def _make_store(bucket: str):
    from magnet.backends.helm.cli.download_helm_results import HelmRemoteStore
    local = Path(bucket).expanduser()
    if local.is_dir():
        from magnet.demo.helm_demodata import LocalHelmStorageBackend
        return HelmRemoteStore(bucket=str(local), backend=LocalHelmStorageBackend(local))
    return HelmRemoteStore(bucket)


def materialize_suite(config) -> dict:
    """
    Do the work; returns the manifest written beside the sentinel.
    """
    import kwutil
    from magnet.backends.helm.cli.download_helm_results import filter_runs
    from magnet.backends.helm.cli.materialize_helm_run import ensure_symlink

    # Validated here rather than in __post_init__ so the class can be
    # constructed with defaults (which the CLI machinery and its tests do).
    if not config.version:
        raise ValueError('--version is required (e.g. v1.0.0)')
    if not config.suite_dpath:
        raise ValueError('--suite_dpath is required')
    benchmark, version = config.benchmark, config.version
    suite_dpath = Path(config.suite_dpath)
    done_fpath = Path(config.done_fpath) if config.done_fpath else suite_dpath.parent.parent.parent / 'DONE'
    roots = _coerce_roots(config.precomputed_roots)
    pattern = kwutil.MultiPattern.coerce(config.runs) if config.runs else None

    local = _local_candidates(roots, benchmark, version)
    remote_ids: list[str] = []
    store = None
    need_remote = config.download == 'always' or (config.download == 'auto' and (not local or pattern is None))
    if config.download != 'never' and (need_remote or pattern is not None):
        # The remote listing is what a pattern is matched against when the
        # local roots may be partial; if it is unreachable, fall back to what
        # the roots hold and say so.
        try:
            store = _make_store(config.bucket)
            remote_ids = list(store.list_runs(benchmark, version))
        except Exception as ex:  # noqa: BLE001
            logger.warning('remote listing unavailable ({}); using precomputed roots only', ex)
            store = None
    universe = sorted(set(local) | set(remote_ids))
    if pattern is not None:
        wanted = filter_runs(universe, pattern)
    else:
        wanted = universe
    if not wanted:
        raise RuntimeError(
            f'no runs of {benchmark}/{version} match {config.runs!r}; '
            f'{len(local)} local, {len(remote_ids)} remote'
        )

    suite_dpath.mkdir(parents=True, exist_ok=True)
    manifest = {'benchmark': benchmark, 'version': version, 'runs': config.runs,
                'precomputed_roots': [os.fspath(r) for r in roots], 'members': {}}
    to_fetch = []
    for run_id in wanted:
        dst = suite_dpath / run_id
        if config.download != 'always' and run_id in local:
            src = local[run_id]
            if dst.is_symlink() or dst.exists():
                if dst.is_symlink():
                    dst.unlink()
                else:
                    shutil.rmtree(dst)
            if config.materialize == 'symlink':
                ensure_symlink(src.resolve(), dst)
            else:
                shutil.copytree(src, dst)
            manifest['members'][run_id] = {'source': os.fspath(src), 'how': config.materialize}
        else:
            to_fetch.append(run_id)
    if to_fetch:
        if config.download == 'never' or store is None:
            raise RuntimeError(
                f'{len(to_fetch)} run(s) are in no precomputed root and download is '
                f'{config.download!r} (remote {"unreachable" if store is None else "allowed"}): '
                f'{to_fetch[:5]}{"..." if len(to_fetch) > 5 else ""}'
            )
        logger.info('fetching {} run(s) of {}/{} into {}', len(to_fetch), benchmark, version, suite_dpath)
        store.download_runs(benchmark, version, ub.Path(suite_dpath), to_fetch)
        for run_id in to_fetch:
            manifest['members'][run_id] = {'source': f'{config.bucket}/{benchmark}/benchmark_output/runs/{version}/{run_id}', 'how': 'download'}

    manifest_fpath = Path(config.manifest_fpath) if config.manifest_fpath else done_fpath.parent / 'materialize_manifest.json'
    manifest_fpath.parent.mkdir(parents=True, exist_ok=True)
    done_fpath.parent.mkdir(parents=True, exist_ok=True)
    done_fpath.write_text('DONE\n')
    # Last, because it is the primary output kwdagger's completion check
    # and generic loader look at.
    manifest_fpath.write_text(json.dumps(manifest, indent=2) + '\n')
    logger.success('{} runs in {} ({} linked, {} fetched); wrote {}',
                   len(manifest['members']), suite_dpath,
                   len(manifest['members']) - len(to_fetch), len(to_fetch), done_fpath)
    return manifest


def main(argv=None, **kwargs) -> int:
    config = MaterializeHelmSuiteConfig.cli(argv=argv, data=kwargs, strict=True)
    materialize_suite(config)
    return 0


__cli__ = MaterializeHelmSuiteConfig

if __name__ == '__main__':
    raise SystemExit(main())
