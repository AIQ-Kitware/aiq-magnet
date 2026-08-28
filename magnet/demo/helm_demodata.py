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


def _build_fixture_run(run_dpath, entry, config, run_index):
    """Write one compact, structurally real HELM run without executing HELM."""
    import dataclasses
    import json

    from helm.benchmark.adaptation.adapter_spec import AdapterSpec
    from helm.benchmark.adaptation.request_state import RequestState
    from helm.benchmark.adaptation.scenario_state import ScenarioState
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
            split='valid',
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

    # Keep the historical demo cardinalities so existing dataframe and summary
    # doctests continue exercising nontrivial collections without storing a
    # large recorded HELM corpus in the repository.
    stats = []
    for stat_index in range(162):
        metric_name = 'exact_match' if stat_index == 0 else f'fixture_metric_{stat_index:03d}'
        value = ((run_index + stat_index) % 10) / 10
        stats.append(Stat(MetricName(metric_name, split='valid')).add(value))

    per_instance_stats = []
    for instance_index, instance in enumerate(instances):
        instance_stats = []
        for stat_index in range(27):
            metric_name = 'exact_match' if stat_index == 0 else f'fixture_instance_metric_{stat_index:02d}'
            value = ((run_index + instance_index + stat_index) % 10) / 10
            instance_stats.append(Stat(MetricName(metric_name, split='valid')).add(value))
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
        'per_instance_stats.json': [dataclasses.asdict(item) for item in per_instance_stats],
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
        >>> from magnet.demo.helm_demodata import ensure_helm_fixture_outputs
        >>> dpath = ensure_helm_fixture_outputs()
        >>> assert (dpath / 'benchmark_output/runs/my-suite').is_dir()
    """
    import ubelt as ub

    config = HelmDemoConfig(**kwargs)
    config_dict = config.to_dict()
    depends = {
        'fixture_schema_version': 1,
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


def ensure_helm_demo_outputs(**kwargs):
    """
    Create a cached set of helm outputs for testing.

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
    Downloads official pre-computed results instead of computing them.
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
