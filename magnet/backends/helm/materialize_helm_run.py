r"""
magnet.backends.helm.materialize_helm_run
=========================================

This module implements a small command line script that computes (or reuses)
*one* HELM run result for a single run-entry description.

Design goals
------------
1) **Deterministic node outputs (kwdagger-friendly)**

   The node writes a small "DONE" sentinel file *last* to indicate the node
   completed successfully. This guards against confusing / partially written
   outputs when a job is interrupted.

2) **Reuse precomputed HELM outputs when available**

   We may have existing HELM run directories on disk (e.g. downloaded bundles).
   If a matching run exists, we "materialize" it into the node output directory
   via a symlink (default) or a copy.

3) **No incremental caching assumption**

   Per your conclusion: HELM does **not** incrementally extend a prior run when
   max-eval-instances is increased. Therefore we treat ``--max-eval-instances``
   as an algorithm parameter that *changes* the identity of the output.

Practical note on normalization
-------------------------------
HELM run directories are often named after the run-entry description string,
but the exact name may be *normalized* by HELM:

- HELM may inject default parameters into the folder name (e.g. ``method=...``)
- HELM may canonicalize model names (e.g. ``openai/gpt2`` -> ``openai_gpt2``)

To avoid depending on HELM's exact naming logic, this script uses a robust
matching strategy:

- Parse the requested run-entry into required tokens (benchmark + key=value)
- Canonicalize the model token by replacing ``/`` with ``_``
- Consider a candidate directory a match if it contains **all required tokens**
  (it may contain extras due to default parameters)

Then (optionally) verify the requested ``max_eval_instances`` by inspecting
the number of instances in ``scenario.json`` when present.

Usage (CLI)
-----------
Example (compute if missing):

    python -m magnet.backends.helm.materialize_helm_run \
        --run_entry "mmlu:subject=philosophy,model=openai/gpt2" \
        --suite my-suite \
        --max_eval_instances 10 \
        --out_dpath ./node_out \
        --precomputed_roots /data/crfm-helm-public

The output directory will contain:

    node_out/
      benchmark_output/
        runs/
          my-suite/
            <run_dir_name>/   (symlinked or computed run)
              run_spec.json
              scenario_state.json
              stats.json
              ...
      adapter_manifest.json
      DONE

Doctests
--------
This module includes doctests for token parsing and matching helpers.

Run doctests (example):

    xdoctest -m magnet.backends.helm.materialize_helm_run


NOTES
-----
* We probably want to support calling HELM via docker to avoid environment
  issues. Punt on this until we need it.

"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import ubelt as ub
import kwutil
import scriptconfig as scfg
import loguru


# We rely on MAGNET's HELM output exploration helpers.
# These are already present in aiq-magnet and know how to load / validate
# the standard json files produced by helm-run.
from magnet.helm_outputs import HelmOutputs


class MaterializeHelmRunConfig(scfg.DataConfig):
    """
    Materialize HELM results either by computing them directly or pulling them
    from a precomputed cache.
    """

    run_entry = scfg.Value(
        None,
        help="Single HELM run-entry description string, e.g. 'mmlu:subject=philosophy,model=openai/gpt2'",
        tags=['algo_param'],
        type=str,
    )

    suite = scfg.Value(
        None,
        help="HELM suite name to use for output layout (and for helm-run --suite).",
        tags=['algo_param'],
    )

    out_dpath = scfg.Value(
        None,
        help="Output directory (kwdagger node output directory).",
        tags=['out_path'],
    )

    precomputed_roots = scfg.Value(
        [],
        nargs='*',
        help="0+ directories to search for existing HELM outputs (may contain nested benchmark_output dirs).",
        tags=['in_param'],
    )

    max_eval_instances = scfg.Value(
        None,
        type=int,
        help="Treat as part of identity. If set, only reuse runs matching this instance count (when inferable).",
        tags=['algo_param'],
    )

    require_per_instance_stats = scfg.Value(
        True,
        help="Require per_instance_stats.json to exist when reusing / validating outputs.",
        tags=['algo_param'],
    )

    mode = scfg.Value(
        'compute_if_missing',
        choices=['reuse_only', 'compute_if_missing', 'force_recompute'],
        help="reuse_only: never compute; compute_if_missing: reuse else run helm; force_recompute: always run helm.",
        tags=['perf_param'],
    )

    materialize = scfg.Value(
        'symlink',
        choices=['symlink', 'copy'],
        help="How to materialize reused outputs into out_dpath.",
        tags=['perf_param'],
    )

    num_threads = scfg.Value(
        1,
        type=int,
        help="Passed to helm-run --num-threads.",
        tags=['perf_param'],
    )

    # extra_helm_args = scfg.Value(
    #     [],
    #     nargs='*',
    #     help="Extra args appended to helm-run command (advanced use).",
    #     tags=['algo_param'],
    # )

    done_fname = scfg.Value(
        'DONE',
        help="Name of sentinel file written in out_dpath when the node is complete.",
        tags=['out_path', 'primary'],
    )

    manifest_fname = scfg.Value(
        'adapter_manifest.json',
        help="Name of a small JSON manifest written in out_dpath describing what happened.",
        tags=['out_path'],
    )

    @classmethod
    def main(cls, argv=None, **kwargs) -> dict:
        """
        Main entry point.

        Returns:
            dict: manifest information (also written to disk).

        Example:
            >>> # This doctest is illustrative only; it requires helm-run installed.
            >>> # xdoctest: +REQUIRES(env:HELM_RUN_AVAILABLE)
            >>> from magnet.backends.helm.materialize_helm_run import main
            >>> dpath = ub.Path.appdir('magnet/tests/materialize').delete().ensuredir()
            >>> main([
            ...   '--run-entry', 'mmlu:subject=philosophy,model=openai/gpt2',
            ...   '--suite', 'my-suite',
            ...   '--max-eval-instances', '2',
            ...   '--out-dpath', str(dpath),
            ...   '--mode', 'compute_if_missing',
            ... ])
        """
        config = MaterializeHelmRunConfig.cli(argv=argv, data=kwargs, verbose='auto')

        if config.run_entry is None:
            raise SystemExit("Missing required --run-entry")
        if config.suite is None:
            raise SystemExit("Missing required --suite")
        if config.out_dpath is None:
            raise SystemExit("Missing required --out-dpath")

        out_dpath = Path(config.out_dpath).expanduser().resolve()
        out_dpath.mkdir(parents=True, exist_ok=True)

        done_fpath = out_dpath / config.done_fname
        manifest_fpath = out_dpath / config.manifest_fname

        # NOTE: if we enable updating some shared cache directory then
        # we will need to do some file locking.

        # If DONE exists, we consider the node complete, unless forcing recompute.
        if done_fpath.exists() and config.mode != 'force_recompute':
            # Load existing manifest (if present) to return something useful.
            if manifest_fpath.exists():
                try:
                    return kwutil.Json.load(manifest_fpath, backend='orjson')
                except Exception:
                    return {"status": "done", "out_dpath": str(out_dpath)}
            return {"status": "done", "out_dpath": str(out_dpath)}

        # If forcing recompute, clean the previous DONE to avoid confusion.
        if config.mode == 'force_recompute' and done_fpath.exists():
            done_fpath.unlink()

        manifest: dict = {
            "requested": {
                "run_entry": config.run_entry,
                "suite": config.suite,
                "max_eval_instances": config.max_eval_instances,
                "require_per_instance_stats": config.require_per_instance_stats,
                "mode": config.mode,
                "materialize": config.materialize,
            },
            "status": None,
            "reuse": None,
            "computed": None,
            "out_dpath": str(out_dpath),
            "timestamp": time.time(),
        }

        # 1) Try reuse
        match = None
        if config.mode != 'force_recompute' and config.precomputed_roots:
            match = find_best_precomputed_run(
                precomputed_roots=config.precomputed_roots,
                requested_desc=config.run_entry,
                max_eval_instances=config.max_eval_instances,
                require_per_instance_stats=config.require_per_instance_stats,
            )

        if match is not None:
            # Materialize into out_dpath in the suite layout we want.
            target_run_dir = out_dpath / 'benchmark_output' / 'runs' / config.suite / match.run_name
            if config.materialize == 'symlink':
                ensure_symlink(match.run_dir, target_run_dir)
            else:
                ensure_copytree(match.run_dir, target_run_dir)

            manifest["status"] = "reused"
            manifest["reuse"] = {
                "source_run_dir": str(match.run_dir),
                "matched_run_name": match.run_name,
                "materialized_run_dir": str(target_run_dir),
                "source_benchmark_output_dir": str(match.source_root),
            }

        else:
            # 2) Compute (unless reuse-only)
            if config.mode == 'reuse_only':
                manifest["status"] = "missing"
                manifest_fpath.write_text(kwutil.Json.dump(manifest, indent=2))
                raise SystemExit("No reusable HELM run found and mode=reuse_only")

            # Ensure benchmark_output exists (helm-run will create, but pre-creating is fine)
            (out_dpath / 'benchmark_output').mkdir(exist_ok=True)

            run_helm(
                requested_desc=config.run_entry,
                suite=config.suite,
                out_dpath=out_dpath,
                max_eval_instances=config.max_eval_instances,
                num_threads=config.num_threads,
                # extra_args=config.extra_helm_args,
            )

            # Locate what helm-run produced.
            computed_run_dir = find_run_in_out_dpath(
                out_dpath=out_dpath,
                suite=config.suite,
                requested_desc=config.run_entry,
                max_eval_instances=config.max_eval_instances,
                require_per_instance_stats=config.require_per_instance_stats,
            )
            if computed_run_dir is None:
                # Fall back: scan everything under benchmark_output for any match
                match2 = find_best_precomputed_run(
                    precomputed_roots=[out_dpath],
                    requested_desc=config.run_entry,
                    max_eval_instances=config.max_eval_instances,
                    require_per_instance_stats=config.require_per_instance_stats,
                )
                computed_run_dir = match2.run_dir if match2 else None

            if computed_run_dir is None:
                manifest["status"] = "error"
                manifest_fpath.write_text(kwutil.Json.dump(manifest, indent=2))
                raise RuntimeError("helm-run completed, but the run directory could not be located/validated")

            manifest["status"] = "computed"
            manifest["computed"] = {
                "computed_run_dir": str(computed_run_dir),
                "computed_run_name": computed_run_dir.name,
            }

        # Write manifest first (helpful for debugging even if DONE is missing)
        manifest_fpath.write_text(kwutil.Json.dumps(manifest, indent=2))

        # Write sentinel last: indicates the node is complete and outputs are ready.
        done_fpath.write_text("ok\n")

        return manifest


# -----------------------------
# Token parsing / normalization
# -----------------------------

def parse_run_entry_description(desc: str) -> Tuple[str, Dict[str, object]]:
    """
    Parse a run-entry description into (benchmark, tokens).

    The run-entry description format commonly looks like:

        "<benchmark>:k1=v1,k2=v2,..."

    The parser also supports "flag" tokens with no ``=`` (rare, but possible in
    some naming conventions); those are stored as ``True``.

    Example:
        >>> parse_run_entry_description("mmlu:subject=philosophy,model=openai/gpt2")
        ('mmlu', {'subject': 'philosophy', 'model': 'openai/gpt2'})

        >>> parse_run_entry_description("ifeval:model=openai_gpt2")
        ('ifeval', {'model': 'openai_gpt2'})

        >>> # Values may contain ':' (e.g. AWS model ids like ':0')
        >>> parse_run_entry_description("ifeval:model=amazon_nova-premier-v1:0")
        ('ifeval', {'model': 'amazon_nova-premier-v1:0'})

    Notes:
        - This script expects the HELM-style "benchmark:params" form.
          If you have an exotic description with multiple colons, treat it as
          unsupported for now (you can extend parsing later if needed).
    """
    if ':' not in desc:
        raise ValueError("Run entry description must contain ':' separating benchmark and parameters")
    bench, rest = desc.split(':', 1)
    bench = bench.strip()
    if not bench:
        raise ValueError(f"Invalid benchmark in {desc!r}")
    tokens: Dict[str, object] = {}
    rest = rest.strip()
    if rest:
        for part in rest.split(','):
            part = part.strip()
            if not part:
                continue
            if '=' in part:
                k, v = part.split('=', 1)
                tokens[k.strip()] = v.strip()
            else:
                # Bare token / flag
                tokens[part] = True
    return bench, tokens


def canonicalize_requested_tokens(tokens: Dict[str, object]) -> Dict[str, object]:
    """
    Apply small, conservative normalizations that we have observed in practice.

    Currently:
    - If the run entry includes a ``model`` token, replace ``/`` with ``_``.

    This matches common HELM directory naming behavior:
        openai/gpt2 -> openai_gpt2

    Example:
        >>> canonicalize_requested_tokens({'model': 'openai/gpt2', 'subject': 'philosophy'})
        {'model': 'openai_gpt2', 'subject': 'philosophy'}
    """
    tokens = dict(tokens)
    model = tokens.get('model', None)
    if isinstance(model, str):
        tokens['model'] = model.replace('/', '_')
    return tokens


def _split_run_dir_tokens(run_dir_name: str) -> Tuple[str, List[str]]:
    """
    Split a run directory name into (benchmark, [token_str, ...]).

    Example:
        >>> _split_run_dir_tokens("mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2")
        ('mmlu', ['subject=philosophy', 'method=multiple_choice_joint', 'model=openai_gpt2'])
    """
    if ':' not in run_dir_name:
        return '', []
    bench, rest = run_dir_name.split(':', 1)
    rest = rest.strip()
    tokens = [t.strip() for t in rest.split(',') if t.strip()]
    return bench.strip(), tokens


def run_dir_matches_requested(run_dir_name: str, requested_desc: str) -> bool:
    """
    Return True if `run_dir_name` likely corresponds to `requested_desc`.

    Matching policy:
    - benchmark prefix must match (before ':')
    - all required tokens from the requested description must be present in the
      candidate run directory name (token-subset match)
    - candidate may contain extra tokens (HELM defaults / normalization)

    Example:
        >>> requested = "mmlu:subject=philosophy,model=openai/gpt2"
        >>> run_dir_matches_requested("mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2", requested)
        True
        >>> run_dir_matches_requested("mmlu:subject=anatomy,method=multiple_choice_joint,model=openai_gpt2", requested)
        False
        >>> run_dir_matches_requested("ifeval:model=openai_gpt2", requested)
        False
    """
    req_bench, req_tokens = parse_run_entry_description(requested_desc)
    req_tokens = canonicalize_requested_tokens(req_tokens)

    cand_bench, cand_tokens = _split_run_dir_tokens(run_dir_name)
    if cand_bench != req_bench:
        return False

    cand_set = set(cand_tokens)

    # Required tokens are represented as strings in the on-disk naming scheme.
    required = []
    for k, v in req_tokens.items():
        if v is True:
            required.append(str(k))
        else:
            required.append(f"{k}={v}")

    return all(t in cand_set for t in required)


def match_score(run_dir_name: str, requested_desc: str) -> Tuple[int, int, str]:
    """
    Produce a deterministic score used to select the "best" match when multiple
    candidates satisfy token-subset matching.

    Lower score is better.

    Heuristics:
    - Exact string match is best (score 0)
    - Fewer "extra" tokens beyond the requested ones is better
    - Finally tie-break by lexicographic name

    Example:
        >>> requested = "mmlu:subject=philosophy,model=openai/gpt2"
        >>> a = "mmlu:subject=philosophy,model=openai_gpt2"
        >>> b = "mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2"
        >>> match_score(a, requested) < match_score(b, requested)
        True
    """
    if run_dir_name == requested_desc:
        # Some bundles may keep the exact string (rare with model '/')
        return (0, 0, run_dir_name)

    req_bench, req_tokens = parse_run_entry_description(requested_desc)
    req_tokens = canonicalize_requested_tokens(req_tokens)
    _, cand_tokens = _split_run_dir_tokens(run_dir_name)

    required = []
    for k, v in req_tokens.items():
        required.append(str(k) if v is True else f"{k}={v}")
    required_set = set(required)

    extra = [t for t in cand_tokens if t not in required_set]
    # 1st: exact name? (0/1), 2nd: number of extra tokens, 3rd: stable tie-break
    return (1, len(extra), run_dir_name)


# -----------------------------
# Disk layout discovery helpers
# -----------------------------


def infer_num_instances(run_dir: Path) -> int | None:
    """
    Best-effort infer how many scenario instances were evaluated.

    Priority:
    1) per_instance_stats.json (most reliable when present)
    2) scenario_state.json (only if it contains an obvious per-instance list)

    Example:
        >>> # xdoctest: +SKIP
        >>> suite_path = Path('/data/crfm-helm-public/capabilities/benchmark_output/runs/v1.12.0/')
        >>> run_name = 'gpqa:subset=gpqa_main,use_chain_of_thought=true,use_few_shot=false,model=amazon_nova-premier-v1:0'
        >>> run_dir = suite_path / run_name
        >>> infer_num_instances(run_dir)
        446
    """
    # 1) per_instance_stats.json
    per_inst_fpath = run_dir / 'per_instance_stats.json'
    if per_inst_fpath.exists():
        try:
            data = kwutil.Json.load(per_inst_fpath, backend='orjson')
            if isinstance(data, list):
                ids = []
                for item in data:
                    if isinstance(item, dict) and 'instance_id' in item:
                        ids.append(item['instance_id'])
                if ids:
                    return len(set(ids))
                # Fallback: if schema unexpected, fall back to list length
                return len(data)
        except Exception:
            pass

    return None


def is_complete_run_dir(run_dir: Path, require_per_instance_stats: bool = True) -> bool:
    """
    Determine if a run directory is "complete enough" to reuse.

    Since you do not need helm-summarize, we only check helm-run artifacts.

    Minimal required files:
    - run_spec.json
    - scenario_state.json
    - stats.json

    Optionally required:
    - per_instance_stats.json (often needed by downstream analysis)

    Example:
        >>> # doctest: +SKIP
        >>> is_complete_run_dir(Path('.../mmlu:...'))
        True
    """
    required = [
        run_dir / 'run_spec.json',
        run_dir / 'scenario_state.json',
        run_dir / 'stats.json',
    ]
    if require_per_instance_stats:
        required.append(run_dir / 'per_instance_stats.json')
    return all(p.exists() for p in required)


# -----------------------------
# Materialization / computation
# -----------------------------

@dataclass
class MatchResult:
    run_dir: Path
    run_name: str
    source_root: Path


def discover_benchmark_output_dirs(roots: Iterable[os.PathLike]) -> Iterator[Path]:
    """
    Yield directories named ``benchmark_output`` under the given roots.

    This supports both layouts you've seen:

    (A) Local helm-run output:
        <root>/benchmark_output/...

    (B) Downloaded bundle:
        <root>/<suite>/benchmark_output/...

    Example:
        >>> # xdoctest: +SKIP
        >>> list(discover_benchmark_output_dirs(['/data/crfm-helm-public']))  # doctest: +ELLIPSIS
        [...]
    """
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        if root.name == 'benchmark_output' and root.is_dir():
            yield root
            continue
        # Recursive discovery: find any nested benchmark_output directories
        for p in root.rglob('benchmark_output'):
            if p.is_dir():
                yield p


def find_best_precomputed_run(
    precomputed_roots: List[os.PathLike],
    requested_desc: str,
    max_eval_instances: Optional[int] = None,
    require_per_instance_stats: bool = True,
) -> Optional[MatchResult]:
    """
    Search for a reusable run directory under one or more precomputed roots.

    Strategy:
    - Discover nested ``benchmark_output`` dirs
    - Coerce each to `HelmOutputs` (MAGNET helper)
    - Iterate suites and runs
    - Keep candidates that:
        * are complete (per required files)
        * match requested tokens
        * (optional) match max_eval_instances (when inferable)

    Returns:
        MatchResult or None

    Example:
        >>> # xdoctest: +SKIP
        >>> from pathlib import Path
        >>> from magnet.backends.helm.materialize_helm_run import (
        ...     find_best_precomputed_run, infer_num_instances
        ... )
        >>> root = Path('/data/crfm-helm-public')
        >>> assert root.exists(), 'CRFM_HELM_PUBLIC is set but /data/crfm-helm-public is missing'

        >>> # Pick any existing run directory under the public bundle.
        >>> # Layout (as you described):
        >>> #   /data/crfm-helm-public/<suite>/benchmark_output/runs/<version>/<run_name>
        >>> run_dirs = sorted(root.glob('*/benchmark_output/runs/*/*:*'))
        >>> assert len(run_dirs) > 0, 'expected at least one HELM run directory'
        >>> chosen = run_dirs[0]
        >>> requested_desc = chosen.name

        >>> # Sanity: ensure the run looks complete enough for reuse.
        >>> # We don't *require* per_instance_stats here because some suites/versions
        >>> # might omit it.
        >>> result = find_best_precomputed_run(
        ...     precomputed_roots=[root],
        ...     requested_desc=requested_desc,
        ...     require_per_instance_stats=False,
        ... )
        >>> assert result is not None
        >>> assert result.run_name == requested_desc
        >>> assert Path(result.run_dir).name == requested_desc

        >>> # If we can infer the number of evaluated instances, test the filter.
        >>> n = infer_num_instances(Path(result.run_dir))
        >>> if n is not None:
        ...     result2 = find_best_precomputed_run(
        ...         precomputed_roots=[root],
        ...         requested_desc=requested_desc,
        ...         max_eval_instances=n,
        ...         require_per_instance_stats=False,
        ...     )
        ...     assert result2 is not None
        ...     assert result2.run_name == requested_desc
        ...     # Asking for a greater instance count should yield no match
        ...     result3 = find_best_precomputed_run(
        ...         precomputed_roots=[root],
        ...         requested_desc=requested_desc,
        ...         max_eval_instances=n + 1,
        ...         require_per_instance_stats=False,
        ...     )
        ...     assert result3 is None
    """
    candidates: List[MatchResult] = []

    for bo in discover_benchmark_output_dirs(precomputed_roots):
        try:
            outputs = HelmOutputs.coerce(bo)
        except Exception:
            continue

        for suite in outputs.suites(pattern='*'):
            # suite.runs() already filters for ':' in directory name.
            runs = suite.runs(pattern='*')
            for run in runs:
                run_dir = Path(run.path)
                if not is_complete_run_dir(run_dir, require_per_instance_stats=require_per_instance_stats):
                    continue
                if not run_dir_matches_requested(run.name, requested_desc):
                    continue
                if max_eval_instances is not None:
                    n = infer_num_instances(run_dir)
                    if n is not None and n != max_eval_instances:
                        continue
                candidates.append(MatchResult(run_dir=run_dir, run_name=run.name, source_root=bo))

    if not candidates:
        return None

    # Pick best-scoring match deterministically
    candidates.sort(key=lambda c: match_score(c.run_name, requested_desc))
    return candidates[0]


def ensure_symlink(src: Path, dst: Path) -> None:
    """
    Create a symlink `dst` -> `src`, replacing an existing path if needed.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        # If already correct, do nothing
        if dst.is_symlink():
            try:
                if Path(os.readlink(dst)) == src:
                    return
            except OSError:
                pass
        ub.Path(dst).delete()
    os.symlink(src, dst)


def ensure_copytree(src: Path, dst: Path) -> None:
    """
    Copy a directory tree, replacing `dst` if it already exists.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        ub.Path(dst).delete()
    ub.copytree(src, dst)


def run_helm(
    requested_desc: str,
    suite: str,
    out_dpath: Path,
    max_eval_instances: Optional[int],
    num_threads: int,
    extra_args: List[str],
) -> None:
    """
    Execute helm-run in `out_dpath`, writing outputs under out_dpath/benchmark_output.

    We do not run helm-summarize by design.
    """
    cmd = ['helm-run', '--run-entries', requested_desc, '--suite', suite]
    if max_eval_instances is not None:
        cmd += ['--max-eval-instances', str(max_eval_instances)]
    if num_threads is not None:
        cmd += ['--num-threads', str(num_threads)]
    cmd += list(extra_args or [])
    ub.cmd(cmd, cwd=out_dpath, verbose=3, system=True).check_returncode()


def find_run_in_out_dpath(
    out_dpath: Path,
    suite: str,
    requested_desc: str,
    max_eval_instances: Optional[int],
    require_per_instance_stats: bool,
) -> Optional[Path]:
    """
    After helm-run finishes, locate the run directory it produced.

    We search under:
        out_dpath/benchmark_output/runs/<suite>/*

    and choose the best token-subset match.
    """
    bo = out_dpath / 'benchmark_output'
    if not bo.exists():
        return None
    try:
        outputs = HelmOutputs.coerce(bo)
    except Exception:
        return None

    # In the typical local layout, "suites" are directly under runs/
    suites = {s.name: s for s in outputs.suites(pattern='*')}
    suite_obj = suites.get(suite, None)
    if suite_obj is None:
        return None

    candidates = []
    for run in suite_obj.runs(pattern='*'):
        run_dir = Path(run.path)
        if not is_complete_run_dir(run_dir, require_per_instance_stats=require_per_instance_stats):
            continue
        if not run_dir_matches_requested(run.name, requested_desc):
            continue
        if max_eval_instances is not None:
            n = infer_num_instances(run_dir)
            if n is not None and n != max_eval_instances:
                continue
        candidates.append(run_dir)

    if not candidates:
        return None

    candidates.sort(key=lambda p: match_score(p.name, requested_desc))
    return candidates[0]


__cli__ = MaterializeHelmRunConfig

if __name__ == '__main__':
    __cli__.main()
