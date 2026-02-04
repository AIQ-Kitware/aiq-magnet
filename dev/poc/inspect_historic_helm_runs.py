"""
Compile a reproduction list from existing HELM outputs on disk.

Given one or more roots that contain HELM outputs, discover all run directories
and emit a list of run specs you can feed into kwdagger / helm-run.

Outputs are structured so you can:
- reproduce exact run directories (by using run_entry == run directory name)
- optionally include max_eval_instances inferred from per_instance_stats.json

Ignore:
    LINE_PROFILE=1 python ~/code/aiq-magnet/dev/poc/inspect_historic_helm_runs.py /data/crfm-helm-public
    python ~/code/aiq-magnet/dev/poc/inspect_historic_helm_runs.py /data/Public/AIQ/crfm-helm-public/


"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, List, Dict, Any

import ubelt as ub
import kwutil
import scriptconfig as scfg
from loguru import logger

from magnet.helm_outputs import HelmOutputs, HelmRun

# Reuse your existing discovery + inference logic
from magnet.backends.helm.materialize_helm_run import (
    discover_benchmark_output_dirs,
    infer_num_instances,
    is_complete_run_dir,
)

from line_profiler import profile


class CompileHelmReproListConfig(scfg.DataConfig):
    roots = scfg.Value(
        ['/data/crfm-helm-public'],
        nargs="+",
        help=(
            "One or more roots that either ARE a benchmark_output dir, contain "
            "benchmark_output dirs, or contain suite/benchmark_output dirs."
        ),
        position=1,
    )

    suite_pattern = scfg.Value(
        "*",
        help="Glob applied to benchmark_output/runs/<suite> directories.",
    )

    run_pattern = scfg.Value(
        "*:*",
        help="Glob applied within each suite to select runs (default selects HELM run dirs).",
    )

    require_per_instance_stats = scfg.Value(
        False,
        help="If True, only include runs that have per_instance_stats.json.",
    )

    include_max_eval_instances = scfg.Value(
        False,
        help="If True, infer max_eval_instances from per_instance_stats.json when possible. CAN BE VERY SLOW",
    )

    out_fpath = scfg.Value(
        None,
        help="Where to write JSON output. If omitted, prints to stdout.",
    )

    format = scfg.Value(
        "json",
        choices=["json", "txt"],
        help="Output format. json is structured; txt is one run_entry per line.",
    )

    dedupe = scfg.Value(
        True,
        help="If True, dedupe identical (suite, run_entry, max_eval_instances) rows.",
    )

    @classmethod
    def main(cls, argv=None, **kwargs):
        """
        Example:
            >>> # It's a good idea to setup a doctest.
            >>> import sys, ubelt
            >>> sys.path.append(ubelt.expandpath('~/code/aiq-magnet/dev/poc'))
            >>> from inspect_historic_helm_runs import *  # NOQA
            >>> argv = False
            >>> kwargs = dict()
            >>> cls = CompileHelmReproListConfig
            >>> config = cls(**kwargs)
            >>> cls.main(argv=argv, **config)
        """
        config = cls.cli(argv=argv, data=kwargs, verbose="auto")
        roots = [Path(r).expanduser() for r in config.roots]
        if not roots:
            raise SystemExit("Must provide at least one root")

        rows = compile_repro_rows(
            roots=roots,
            suite_pattern=config.suite_pattern,
            run_pattern=config.run_pattern,
            require_per_instance_stats=config.require_per_instance_stats,
            include_max_eval_instances=config.include_max_eval_instances,
        )

        if config.dedupe:
            rows = dedupe_rows(rows)

        if config.format == "txt":
            text = "\n".join([r["run_entry"] for r in rows]) + ("\n" if rows else "")
            if config.out_fpath:
                Path(config.out_fpath).write_text(text)
                logger.success("Wrote {}", config.out_fpath)
            else:
                print(text, end="")
            return {"num_rows": len(rows)}

        payload = {
            "num_rows": len(rows),
            "rows": rows,
        }

        if config.out_fpath:
            Path(config.out_fpath).write_text(kwutil.Json.dumps(payload, indent=2))
            logger.success("Wrote {}", config.out_fpath)
        else:
            print(kwutil.Json.dumps(payload, indent=2))
        return payload


@profile
def compile_repro_rows(
    roots: Iterable[Path],
    suite_pattern: str = "*",
    run_pattern: str = "*:*",
    require_per_instance_stats: bool = False,
    include_max_eval_instances: bool = True,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    # Discover all benchmark_output dirs under provided roots
    logger.info('Discover benchmarks')
    bo_dirs = list(ub.ProgIter(discover_benchmark_output_dirs(roots), desc='discovering benchmarks', verbose=3, homogeneous=False))
    logger.info('Finished Discover benchmarks')
    if not bo_dirs:
        logger.warning("No benchmark_output dirs found under roots={}", roots)

    for bo in ub.ProgIter(bo_dirs, desc='Check dirs'):
        try:
            outputs = HelmOutputs.coerce(bo)
        except Exception:
            continue

        for suite in outputs.suites(pattern=suite_pattern):
            for run in suite.runs(pattern=run_pattern):
                run_dir = Path(run.path)

                # Only include if it looks “complete enough”
                if not is_complete_run_dir(run_dir, require_per_instance_stats=require_per_instance_stats):
                    continue

                run = HelmRun.coerce(run_dir)

                max_eval_instances = None
                if include_max_eval_instances:
                    max_eval_instances = infer_num_instances(run_dir)

                rows.append({
                    "benchmark_output_dir": str(Path(outputs.root_dir)),
                    "suite": suite.name,
                    # Use run directory name as the canonical "run_entry" to reproduce.
                    # This is faithful even if HELM normalized defaults into the name.
                    "run_entry": run.name,
                    "run_dir": str(run_dir),
                    "max_eval_instances": max_eval_instances,
                })

    # Stable order
    rows.sort(key=lambda r: (r["suite"], r["run_entry"], r["max_eval_instances"] or -1, r["run_dir"]))
    logger.info('Found {len(rows)} run directories')
    return rows


def dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in rows:
        key = (r["suite"], r["run_entry"], r.get("max_eval_instances", None))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


__cli__ = CompileHelmReproListConfig

if __name__ == "__main__":
    __cli__.main()
