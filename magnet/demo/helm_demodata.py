import kwconf


class HelmDemoConfig(kwconf.Config):
    """
    Configuration for generating helm demo outputs
    """
    run_entries: list[str] = kwconf.Value(
        [
            "mmlu:subject=philosophy,model=openai/gpt2",
            "mmlu:subject=anatomy,model=openai/gpt2",
            "mmlu:subject=anatomy,model=eleutherai/pythia-1b-v0",
            "mmlu:subject=philosophy,model=eleutherai/pythia-1b-v0",
        ],
        parser='yaml',
        help='Benchmark run entries',
    )
    suite: str = kwconf.Value("my-suite", help="Name of the helm suite")
    max_eval_instances: int = kwconf.Value(7, help="Maximum eval instances")
    num_threads: int = kwconf.Value(1, help="Number of threads")


def _parse_demo_run_entry(entry):
    """Parse the small subset of HELM run-entry syntax needed by fixtures."""
    if ':' in entry:
        scenario_name, argstr = entry.split(':', 1)
        args = dict(part.split('=', 1) for part in argstr.split(',') if part)
    else:
        scenario_name = entry
        args = {}
    return scenario_name, args


def _fixture_run_name(entry):
    """Build stable HELM-like directory names for the local demo fixture."""
    scenario_name, args = _parse_demo_run_entry(entry)
    model = args.get('model', 'openai/gpt2').replace('/', '_')
    if scenario_name == 'mmlu':
        subject = args.get('subject', 'philosophy')
        method = args.get('method', 'multiple_choice_joint')
        return f'mmlu:subject={subject},method={method},model={model}'

    rendered_args = [
        f'{key}={value}'
        for key, value in args.items()
        if key != 'model'
    ]
    rendered_args.append(f'model={model}')
    return scenario_name + ':' + ','.join(rendered_args)


def _build_fixture_run(
    run_dpath,
    entry,
    config,
    run_index,
    *,
    num_stats=162,
    num_per_instance_stats=27,
    stat_split='valid',
    primary_score=None,
    include_perturbation_context=False,
):
    """Write one compact, structurally real HELM run without executing HELM."""
    import dataclasses
    import json

    from helm.benchmark.adaptation.adapter_spec import AdapterSpec
    from helm.benchmark.adaptation.request_state import RequestState
    from helm.benchmark.adaptation.scenario_state import ScenarioState
    from helm.benchmark.augmentations.perturbation_description import (
        PerturbationDescription,
    )
    from helm.benchmark.metrics.metric import MetricSpec, PerInstanceStats
    from helm.benchmark.metrics.metric_name import MetricName
    from helm.benchmark.metrics.statistic import Stat
    from helm.benchmark.run_spec import RunSpec
    from helm.benchmark.scenarios.scenario import Input, Instance, ScenarioSpec
    from helm.common.request import Request

    scenario_name, args = _parse_demo_run_entry(entry)
    model = args.get('model', 'openai/gpt2')
    run_name = run_dpath.name

    adapter_spec = AdapterSpec(
        method=args.get('method', 'multiple_choice_joint'),
        model_deployment=model,
        model=model,
        max_eval_instances=config.max_eval_instances,
    )
    scenario_spec = ScenarioSpec(
        class_name=f'magnet.demo.fixture.{scenario_name}',
        args={k: v for k, v in args.items() if k != 'model'},
    )
    metric_specs = [
        MetricSpec(class_name=f'magnet.demo.fixture.Metric{idx}', args={})
        for idx in range(3)
    ]
    run_spec = RunSpec(
        name=run_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
    )

    num_instances = max(1, min(config.max_eval_instances or 7, 7))
    instances = []
    request_states = []
    for instance_index in range(num_instances):
        instance = Instance(
            input=Input(text=f'Fixture question {instance_index}'),
            references=[],
            split=stat_split,
            id=f'id{instance_index}',
        )
        instances.append(instance)
        request_states.append(RequestState(
            instance=instance,
            reference_index=None,
            request_mode=None,
            train_trial_index=0,
            output_mapping=None,
            request=Request(
                model_deployment=model,
                model=model,
                prompt=f'Fixture question {instance_index}',
                temperature=0,
                num_completions=1,
                max_tokens=1,
            ),
            result=None,
            num_train_instances=0,
            prompt_truncated=False,
        ))
    scenario_state = ScenarioState(
        adapter_spec=adapter_spec,
        request_states=request_states,
    )

    stats = []
    for stat_index in range(num_stats):
        if stat_index == 0:
            metric_name = 'exact_match'
            value = (
                primary_score
                if primary_score is not None
                else ((run_index + stat_index) % 10) / 10
            )
            perturbation = None
        else:
            metric_name = f'fixture_metric_{stat_index:03d}'
            value = ((run_index + stat_index) % 10) / 10
            perturbation = None
            if include_perturbation_context and stat_index == 1:
                perturbation = PerturbationDescription(name='fixture')
        stats.append(
            Stat(
                MetricName(
                    metric_name,
                    split=stat_split,
                    perturbation=perturbation,
                )
            ).add(value)
        )

    per_instance_stats = []
    for instance_index, instance in enumerate(instances):
        instance_stats = []
        for stat_index in range(num_per_instance_stats):
            metric_name = (
                'exact_match'
                if stat_index == 0
                else f'fixture_instance_metric_{stat_index:02d}'
            )
            value = ((run_index + instance_index + stat_index) % 10) / 10
            instance_stats.append(
                Stat(MetricName(metric_name, split=stat_split)).add(value)
            )
        per_instance_stats.append(PerInstanceStats(
            instance_id=instance.id,
            perturbation=None,
            train_trial_index=0,
            stats=instance_stats,
        ))

    payloads = {
        'run_spec.json': dataclasses.asdict(run_spec),
        'scenario_state.json': dataclasses.asdict(scenario_state),
        'stats.json': [dataclasses.asdict(item) for item in stats],
        'per_instance_stats.json': [
            dataclasses.asdict(item) for item in per_instance_stats
        ],
        'scenario.json': {
            'name': scenario_name,
            'instances': [dataclasses.asdict(item) for item in instances],
        },
    }
    for fname, data in payloads.items():
        (run_dpath / fname).write_text(json.dumps(data, indent=2, sort_keys=True))


def ensure_helm_fixture_outputs(**kwargs):
    """
    Create compact deterministic HELM-shaped outputs without running HELM.

    The fixture uses HELM's own dataclasses to produce its JSON schema, so it
    exercises MAGNET's real filesystem, JSON, dataclass, msgspec, and dataframe
    loading paths while avoiding model downloads and benchmark execution.

    Args:
        **kwargs: See :class:`HelmDemoConfig`.

    Returns:
        Path:
            path to demo outputs with a ``benchmark_output`` subdirectory.

    Example:
        >>> # The fixture is built with HELM's own dataclasses, so this needs
        >>> # the extra even though the module imports without it.
        >>> # xdoctest: +REQUIRES(module:helm)
        >>> from magnet.demo.helm_demodata import ensure_helm_fixture_outputs
        >>> dpath = ensure_helm_fixture_outputs()
        >>> assert (dpath / 'benchmark_output/runs/my-suite').is_dir()
    """
    import ubelt as ub

    config = HelmDemoConfig(**kwargs)
    config_dict = config.to_dict()
    depends = {
        'fixture_schema_version': 2,
        'config': config_dict,
    }
    hash_id = ub.hash_data(depends)[0:12]
    base_dpath = ub.Path.appdir('magnet/tests/helm_output/fixture').ensuredir()
    dpath = (base_dpath / hash_id).ensuredir()
    stamp = ub.CacheStamp('helm_fixture_outputs', depends=depends, dpath=dpath)

    if stamp.expired():
        benchmark_dpath = dpath / 'benchmark_output'
        if benchmark_dpath.exists():
            benchmark_dpath.delete()
        suite_dpath = (benchmark_dpath / 'runs' / config.suite).ensuredir()
        for run_index, entry in enumerate(config.run_entries):
            run_dpath = (suite_dpath / _fixture_run_name(entry)).ensuredir()
            _build_fixture_run(run_dpath, entry, config, run_index)
        stamp.renew()

    return dpath


def ensure_helm_llama_fixture_outputs():
    """
    Create the small HELM Lite corpus used by the llama evaluation cards.

    The directory layout mirrors downloaded HELM Lite releases, but contains
    only two MMLU subjects for each model and no external data.

    Returns:
        Path:
            root corresponding to ``crfm-helm-public``.

    Example:
        >>> # xdoctest: +REQUIRES(module:helm)
        >>> from magnet.demo.helm_demodata import ensure_helm_llama_fixture_outputs
        >>> root = ensure_helm_llama_fixture_outputs()
        >>> runs = root / 'lite/benchmark_output/runs'
        >>> assert (runs / 'v1.0.0').is_dir()
        >>> assert (runs / 'v1.2.0').is_dir()
    """
    import ubelt as ub

    models = {
        'meta/llama-2-7b': ('v1.0.0', 0.40),
        'meta/llama-2-13b': ('v1.0.0', 0.46),
        'meta/llama-2-70b': ('v1.0.0', 0.55),
        'meta/llama-65b': ('v1.0.0', 0.50),
        'meta/llama-3-8b': ('v1.2.0', 0.64),
        'meta/llama-3-70b': ('v1.2.0', 0.76),
    }
    subjects = ['abstract_algebra', 'anatomy']
    depends = {
        'fixture_schema_version': 1,
        'models': models,
        'subjects': subjects,
    }
    hash_id = ub.hash_data(depends)[0:12]
    base_dpath = ub.Path.appdir('magnet/tests/helm_llama_fixture').ensuredir()
    root = (base_dpath / hash_id).ensuredir()
    stamp = ub.CacheStamp('helm_llama_fixture', depends=depends, dpath=root)

    if stamp.expired():
        lite_dpath = root / 'lite' / 'benchmark_output' / 'runs'
        if lite_dpath.exists():
            lite_dpath.delete()
        run_index = 0
        for model, (version, base_score) in models.items():
            config = HelmDemoConfig(
                run_entries=[],
                suite=version,
                max_eval_instances=1,
                num_threads=1,
            )
            suite_dpath = (lite_dpath / version).ensuredir()
            for subject_index, subject in enumerate(subjects):
                entry = f'mmlu:subject={subject},model={model}'
                run_dpath = (suite_dpath / _fixture_run_name(entry)).ensuredir()
                # Two neighboring subject scores make the card exercise its
                # groupby/mean path while preserving the intended model gap.
                subject_score = base_score + (subject_index * 0.02 - 0.01)
                _build_fixture_run(
                    run_dpath,
                    entry,
                    config,
                    run_index,
                    num_stats=3,
                    num_per_instance_stats=1,
                    stat_split='test',
                    primary_score=subject_score,
                    include_perturbation_context=True,
                )
                run_index += 1
        stamp.renew()

    return root


class LocalHelmStorageBackend:
    """Filesystem-backed stand-in for the public HELM GCS bucket."""

    def __init__(self, bucket):
        import ubelt as ub

        self.bucket = str(ub.Path(bucket))

    def list_dirs(self, prefix):
        import ubelt as ub

        path = ub.Path(prefix)
        if not path.is_dir():
            return []
        return sorted(p.name for p in path.iterdir() if p.is_dir())

    def download_tree(self, src_prefix, dest_dir, checksum=False):
        import shutil
        import ubelt as ub

        src = ub.Path(src_prefix)
        dest = ub.Path(dest_dir)
        if not src.is_dir():
            raise FileNotFoundError(src)
        dest.parent.ensuredir()
        shutil.copytree(src, dest, dirs_exist_ok=True)


def ensure_helm_remote_store_fixture():
    """
    Create a fake HELM public bucket for downloader/listing tests.

    Returns:
        Path:
            root of a local directory with the same benchmark/runs/version
            hierarchy used by ``HelmRemoteStore``.

    Example:
        >>> from magnet.demo.helm_demodata import ensure_helm_remote_store_fixture
        >>> bucket = ensure_helm_remote_store_fixture()
        >>> assert (bucket / 'lite/benchmark_output/runs/v1.13.0').is_dir()
    """
    import json
    import ubelt as ub

    layout = {
        'classic': {
            'v0.4.0': [f'classic_task_{idx:02d}:model=fixture' for idx in range(8)],
        },
        'image2struct': {
            'v1.0.0': ['image_task:model=fixture'],
        },
        'lite': {
            'v1.0.0': ['gsm:model=meta_llama-2-13b'],
            'v1.12.0': ['mmlu:subject=anatomy,model=meta_llama-2-7b'],
            'v1.13.0': [
                'med_qa:model=deepseek-ai_deepseek-v3',
                'med_qa:model=fixture-other',
            ],
        },
    }
    depends = {'fixture_schema_version': 1, 'layout': layout}
    hash_id = ub.hash_data(depends)[0:12]
    base_dpath = ub.Path.appdir('magnet/tests/helm_remote_fixture').ensuredir()
    bucket = (base_dpath / hash_id).ensuredir()
    stamp = ub.CacheStamp('helm_remote_fixture', depends=depends, dpath=bucket)

    if stamp.expired():
        for child in list(bucket.iterdir()):
            if child.is_dir():
                child.delete()
        # These exercise HelmRemoteStore.list_benchmarks() filtering.
        for blocked in ['assets', 'config', 'prod_env']:
            (bucket / blocked).ensuredir()
        for benchmark, versions in layout.items():
            for version, run_names in versions.items():
                version_dpath = (
                    bucket
                    / benchmark
                    / 'benchmark_output'
                    / 'runs'
                    / version
                ).ensuredir()
                for run_name in run_names:
                    run_dpath = (version_dpath / run_name).ensuredir()
                    for fname in ['run_spec.json', 'stats.json', 'scenario_state.json']:
                        payload = {
                            'fixture': True,
                            'benchmark': benchmark,
                            'version': version,
                            'run': run_name,
                            'file': fname,
                        }
                        (run_dpath / fname).write_text(
                            json.dumps(payload, sort_keys=True)
                        )
        stamp.renew()

    return bucket


def make_helm_remote_fixture_store():
    """Construct ``HelmRemoteStore`` over the local fake public bucket."""
    from magnet.backends.helm.cli.download_helm_results import HelmRemoteStore

    bucket = ensure_helm_remote_store_fixture()
    backend = LocalHelmStorageBackend(bucket)
    return HelmRemoteStore(bucket=str(bucket), backend=backend)


def ensure_helm_demo_outputs(**kwargs):
    """
    Create a cached set of helm outputs by executing HELM.

    Args:
        **kwargs: See :class:`HelmDemoConfig`.

    Returns:
        Path:
            path to demo outputs with "benchmark_output" and "prod_env"
            subdirectories.

    Example:
        >>> # xdoctest: +REQUIRES(env:HELM_RUN_AVAILABLE)
        >>> from magnet.demo.helm_demodata import *  # NOQA
        >>> kwargs = {}
        >>> dpath = ensure_helm_demo_outputs(**kwargs)
    """
    import ubelt as ub
    base_dpath = ub.Path.appdir('magnet/tests/helm_output').ensuredir()
    config = HelmDemoConfig(**kwargs)
    config_dict = config.to_dict()
    hash_id = ub.hash_data(config_dict)[0:12]
    dpath = (base_dpath / hash_id).ensuredir()

    stamp = ub.CacheStamp('helm_demo_outputs', depends=config_dict, dpath=dpath)
    if stamp.expired():

        base_cmd = ["helm-run", "--run-entries"] + config.run_entries + [
            "--suite", config.suite,
            "--max-eval-instances", str(config.max_eval_instances),
            "--num-threads", str(config.num_threads),
        ]
        res = ub.cmd(base_cmd, cwd=dpath, verbose=3, system=True)
        res.check_returncode()

        res = ub.cmd(['helm-summarize', '--suite', config.suite], cwd=dpath, verbose=3)
        res.check_returncode()
        stamp.renew()

    return dpath


def grab_helm_demo_outputs():
    """
    Download official pre-computed results instead of computing them.

    This is an explicit integration helper. Hermetic tests should use one of
    the ``ensure_*_fixture`` helpers above.
    """
    import ubelt as ub
    from magnet.backends.helm import download_helm_results
    base_dpath = ub.Path.appdir('magnet/tests/helm_output/downloaded').ensuredir()
    stamp = ub.CacheStamp('helm_demo_downloads', depends=['version1'],
                          dpath=base_dpath)
    if stamp.expired():
        download_helm_results.main(
            argv=False,
            download_dir=base_dpath,
            benchmark='lite',
            version='v1.13.0',
            runs=[
                'narrative_qa:model=amazon_nova-micro-v1:0',
                'narrative_qa:model=amazon_nova-lite-v1:0',
                'natural_qa:mode=closedbook,model=amazon_nova-lite-v1:0',
                'natural_qa:mode=closedbook,model=deepseek-ai_deepseek-v3',
            ],
        )
        stamp.renew()
    dpath = base_dpath / 'lite'
    return dpath
