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

        suite_pattern = config.suite_pattern
        run_pattern = config.run_pattern
        require_per_instance_stats = config.require_per_instance_stat
        include_max_eval_instances = config.include_max_eval_instances

        runs = gather_runs(
            roots=roots,
            suite_pattern=suite_pattern,
            run_pattern=run_pattern,
            require_per_instance_stats=require_per_instance_stats,
            include_max_eval_instances=include_max_eval_instances,
        )
        rows = build_run_table(runs)

        import pandas as pd
        df = pd.DataFrame(rows)
        model_histo = ub.dict_hist([r['model'] for r in rows])
        print(f'model_histo = {ub.urepr(model_histo, nl=1)}')
        print(df['scenario_class'].value_counts().to_string())
        print(df['model'].value_counts().to_string())

        from helm.benchmark.config_registry import (
            register_builtin_configs_from_helm_package,
        )
        from helm.benchmark import  model_deployment_registry
        register_builtin_configs_from_helm_package()
        model_rows = []
        for model_name, count in model_histo.items():
            try:
                model_meta = model_deployment_registry.get_model_metadata(model_name)
                model_row = model_meta.__dict__ | {'count': count}
                model_rows.append(model_row)
            except Exception:
                print(f'missing: model_name = {ub.urepr(model_name, nl=1)}')

        flags = ['FULL_FUNCTIONALITY_TEXT_MODEL_TAG' in r['tags'] for r in model_rows]
        model_rows = list(ub.compress(model_rows, flags))

        model_df = pd.DataFrame(model_rows)
        sub = model_df[model_df['access'] == 'open']
        printable = sub.drop(['description'], axis=1)
        printable = printable.drop(['deployment_names'], axis=1)
        # printable = printable.drop(['tags'], axis=1)
        printable = printable.sort_values('count')
        printable = printable[printable['num_parameters'] < 200e9]
        print(printable.to_string())
        print(printable['num_parameters'].describe().round())

        # {'TEXT_MODEL_TAG': 262,
        #  'FULL_FUNCTIONALITY_TEXT_MODEL_TAG': 66,
        #  'ANTHROPIC_CLAUDE_3_MODEL_TAG': 8,
        #  'VISION_LANGUAGE_MODEL_TAG': 85,
        #  'LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG': 168,
        #  'INSTRUCTION_FOLLOWING_MODEL_TAG': 184,
        #  'PARTIAL_FUNCTIONALITY_TEXT_MODEL_TAG': 34,
        #  'GOOGLE_GEMINI_MODEL_TAG': 30,
        #  'OPENAI_CHATGPT_MODEL_TAG': 28,
        #  'DEPRECATED_MODEL_TAG': 37,
        #  'AUDIO_LANGUAGE_MODEL_TAG': 27,
        #  'ABLATION_MODEL_TAG': 8,
        #  'FULL_FUNCTIONALITY_VLM_TAG': 18,
        #  'NOVA_MODEL_TAG': 4,
        #  'CODE_MODEL_TAG': 2,
        #  'GOOGLE_GEMMA_INSTRUCT_MODEL_TAG': 4,
        #  'TEXT_TO_IMAGE_MODEL_TAG': 25,
        #  'IDEFICS_MODEL_TAG': 5,
        #  'IDEFICS_INSTRUCT_MODEL_TAG': 2,
        #  'GOOGLE_GEMINI_PRO_VISION_V1_TAG': 1,
        #  'LLAVA_MODEL_TAG': 2,
        #  'LIMITED_FUNCTIONALITY_VLM_TAG': 4,
        #  'ANTHROPIC_CLAUDE_1_MODEL_TAG': 3,
        #  'ANTHROPIC_CLAUDE_2_MODEL_TAG': 2,
        #  'GOOGLE_PALM_2_MODEL_TAG': 2,
        #  'OPEN_FLAMINGO_MODEL_TAG': 1}

        # BF16 cap: ~150B
        # INT8 cap: ~300B
        # 4-bit cap: ~600B
        ub.dict_hist(ub.flatten(model_df['tags'].values))

        chosen_names = set(printable.name)
        rows = [r for r in rows if r['model'] in chosen_names]
        logger.info(f'Filter to {len(rows)}')

        # if config.dedupe:
        #     rows = dedupe_rows(rows)

        if config.format == "txt":
            text = "\n".join([r["run_spec_name"] for r in rows]) + ("\n" if rows else "")
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
def gather_runs(
    roots: Iterable[Path],
    suite_pattern: str = "*",
    run_pattern: str = "*:*",
    require_per_instance_stats: bool = False,
    include_max_eval_instances: bool = True,
) -> List[HelmRun]:

    # Discover all benchmark_output dirs under provided roots
    logger.info('Discover benchmarks')
    bo_dirs = list(ub.ProgIter(discover_benchmark_output_dirs(roots), desc='discovering benchmarks', verbose=3, homogeneous=False))
    logger.info('Finished Discover benchmarks')
    if not bo_dirs:
        logger.warning("No benchmark_output dirs found under roots={}", roots)

    runs: List[HelmRun] = []
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
    logger.info(f'Found {len(runs)} run directories')
    return runs


@profile
def build_run_table(runs: list[HelmRun]) -> list[dict]:
    rows = []

    include_max_eval_instances = False
    mismatches = []
    for run in ub.ProgIter(runs, desc='Extract run spec info'):
        max_eval_instances = None
        if include_max_eval_instances:
            max_eval_instances = infer_num_instances(run.path)

        run_spec = run.json.run_spec()
        run_spec['scenario_spec']['class_name']

        run_spec = run.msgspec.run_spec()
        scenario_class = run_spec.scenario_spec.class_name
        model = run_spec.adapter_spec.model

        run_spec_name = run_spec.name
        if run.name != run_spec_name.replace('/', '_'):
            mismatches.append({
                'run.path.parent': run.path.parent,
                'run.name': run.name,
                'run_spec_name': run_spec_name,
            })

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
    print(f'mismatches = {ub.urepr(mismatches, nl=2, align=":")}')
    rows.sort(key=lambda r: (r["run_dir"]))
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
