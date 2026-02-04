r"""
Compile a reproduction list from existing HELM outputs on disk.

Given one or more roots that contain HELM outputs, discover all run directories
and emit a list of run specs you can feed into kwdagger / helm-run.

Outputs are structured so you can:
- reproduce exact run directories (by using run_entry == run directory name)
- optionally include max_eval_instances inferred from per_instance_stats.json

Ignore:

    ls /data/crfm-helm-public/thaiexam/benchmark_output/runs/v1.1.0/thai_exam:exam=tpat1,method=multiple_choice_joint,model=aisingapore_llama3-8b-cpt-sea-lionv2.1-instruct

    python ~/code/aiq-magnet/dev/poc/inspect_historic_helm_runs.py /data/crfm-helm-public --out_fpath run_specs.yaml

    python ~/code/aiq-magnet/dev/poc/inspect_historic_helm_runs.py /data/Public/AIQ/crfm-helm-public/

    # we need fully featured helm installed
    uv pip install crfm-helm[all]

    # Need to login to huggingface can pass token via --token
    hf auth login

    kwdagger schedule \
      --params="
        pipeline: 'magnet.backends.helm.pipeline.helm_single_run_pipeline()'
        matrix:
          helm.run_entry:
            - __include__: run_specs.yaml
          helm.max_eval_instances:
            - 1000
          helm.precomputed_root: null
      " \
      --root_dpath=$PWD/results \
      --backend=tmux \
      --skip_existing=1 \
      --run=1


"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Any

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
        help="Where to write output. If omitted, prints to stdout.",
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

        suite_pattern = config.suite_pattern
        run_pattern = config.run_pattern
        require_per_instance_stats = config.require_per_instance_stats
        include_max_eval_instances = config.include_max_eval_instances

        runs = gather_runs(
            roots=roots,
            suite_pattern=suite_pattern,
            run_pattern=run_pattern,
            require_per_instance_stats=require_per_instance_stats,
            include_max_eval_instances=include_max_eval_instances,
        )
        rows = build_run_table(runs)

        scenario_histo = ub.dict_hist([r['scenario_class'] for r in rows])
        model_histo = ub.dict_hist([r['model'] for r in rows])
        scenario_histo = ub.udict.sorted_values(scenario_histo)
        model_histo = ub.udict.sorted_values(model_histo)
        print(f'scenario_histo = {ub.urepr(scenario_histo, nl=1)}')
        print(f'model_histo = {ub.urepr(model_histo, nl=1)}')

        from helm.benchmark import config_registry
        from helm.benchmark import  model_deployment_registry
        config_registry.register_builtin_configs_from_helm_package()
        model_rows = []
        for model_name, count in model_histo.items():
            try:
                model_meta = model_deployment_registry.get_model_metadata(model_name)
                model_row = model_meta.__dict__ | {'count': count}
                model_rows.append(model_row)
            except Exception:
                logger.warning(f'missing: model_name = {ub.urepr(model_name, nl=1)}')

        require_tags = {
            'FULL_FUNCTIONALITY_TEXT_MODEL_TAG'
        }
        MAX_PARAMS = 10e9
        # MAX_PARAMS = 200e9

        # Filter to text models that will fit in memory
        chosen_model_rows = [
            r for r in model_rows if (
                set(r['tags']).issuperset(require_tags) and
                (r['num_parameters'] is not None and r['num_parameters'] <= MAX_PARAMS) and
                (r['access'] == 'open')
            )
        ]
        chosen_model_names = {r['name'] for r in chosen_model_rows}
        logger.info('Filter to {} / {} models', len(chosen_model_rows), len(model_rows))

        chosen_rows = [r for r in rows if r['model'] in chosen_model_names]
        logger.info('Filter to {} / {} runs', len(chosen_rows), len(rows))

        logger.info(f'chosen_rows = {ub.urepr(chosen_rows, nl=1)}')

        if 1:
            # Show filtered histograms
            scenario_histo = ub.dict_hist([r['scenario_class'] for r in chosen_rows])
            model_histo = ub.dict_hist([r['model'] for r in chosen_rows])
            scenario_histo = ub.udict.sorted_values(scenario_histo)
            model_histo = ub.udict.sorted_values(model_histo)
            logger.info(f'scenario_histo = {ub.urepr(scenario_histo, nl=1)}')
            logger.info(f'model_histo = {ub.urepr(model_histo, nl=1)}')

        run_spec_names = [r["run_spec_name"] for r in chosen_rows]

        text = kwutil.Yaml.dumps(run_spec_names)
        if config.out_fpath:
            Path(config.out_fpath).write_text(text)
            logger.success("Wrote {}", config.out_fpath)
        else:
            print(text, end="")


def gather_runs(
    roots: Iterable[Path],
    suite_pattern: str = "*",
    run_pattern: str = "*:*",
    require_per_instance_stats: bool = False,
    include_max_eval_instances: bool = True,
) -> list[HelmRun]:

    # Discover all benchmark_output dirs under provided roots
    logger.info('Discover benchmarks')
    bo_dirs = list(ub.ProgIter(discover_benchmark_output_dirs(roots), desc='discovering benchmarks', verbose=3, homogeneous=False))
    logger.info('Finished Discover benchmarks')
    if not bo_dirs:
        logger.warning("No benchmark_output dirs found under roots={}", roots)

    runs: list[HelmRun] = []
    for bo in ub.ProgIter(bo_dirs, desc='Check dirs'):
        try:
            outputs = HelmOutputs.coerce(bo)
        except Exception:
            continue

        for suite in outputs.suites(pattern=suite_pattern):
            for run in suite.runs(pattern=run_pattern):
                run_dir = Path(run.path)

                run = HelmRun(run_dir)

                # TODO: if not run.exists():
                #     ...
                # Only include if it looks “complete enough”
                if not is_complete_run_dir(run_dir, require_per_instance_stats=require_per_instance_stats):
                    continue

                runs.append(run)

    # Stable order
    logger.info('Found {} run directories', len(runs))
    return runs


def build_run_table(runs: list[HelmRun]) -> list[dict]:
    rows = []

    include_max_eval_instances = False
    mismatches = []
    for run in ub.ProgIter(runs, desc='Extract run spec info'):
        max_eval_instances = None
        if include_max_eval_instances:
            max_eval_instances = infer_num_instances(run.path)

        # Not sure if there is an advantage to msgspec or json here
        # ZFS is likely messing up my timings.
        if 1:
            run_spec = run.json.run_spec()
            scenario_class = run_spec['scenario_spec']['class_name']
            model = run_spec['adapter_spec']['model']
            run_spec_name = run_spec['name']
        else:
            run_spec = run.msgspec.run_spec()
            scenario_class = run_spec.scenario_spec.class_name
            model = run_spec.adapter_spec.model
            run_spec_name = run_spec.name

        if run.path.name != run_spec_name.replace('/', '_'):
            mismatches.append({
                'run.path.parent': run.path.parent,
                'run.path.name': run.path.name,
                'run_spec_name': run_spec_name,
            })

        # Hack: run spec names sometimes don't correctly encode the model
        FIX_RUN_SPEC_NAME = True
        if FIX_RUN_SPEC_NAME:
            normalized_model = model.replace('/', '_')
            run_spec_name = run_spec_name.replace(normalized_model, model)

        rows.append({
            # "benchmark_output_dir": str(Path(outputs.root_dir)),
            # "suite": suite.name,
            # # Use run directory name as the canonical "run_entry" to reproduce.
            # # This is faithful even if HELM normalized defaults into the name.

            # Use run directory name as the canonical "run_entry" to reproduce.
            # This is faithful even if HELM normalized defaults into the name.
            "run_spec_name": run_spec_name,
            "run_dir": str(run.path),
            "max_eval_instances": max_eval_instances,
            'model': model,
            'scenario_class': scenario_class,
        })
    logger.warning(f'mismatches = {ub.urepr(mismatches, nl=2, align=":")}')
    rows.sort(key=lambda r: (r["run_dir"]))
    return rows


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
