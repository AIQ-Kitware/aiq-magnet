import scriptconfig as scfg


class HelmDemoConfig(scfg.DataConfig):
    """
    Configuration for generating helm demo outputs
    """
    run_entries = scfg.Value(
        [
            "mmlu:subject=philosophy,model=openai/gpt2",
            "mmlu:subject=anatomy,model=openai/gpt2",
            "mmlu:subject=anatomy,model=eleutherai/pythia-1b-v0",
            "mmlu:subject=philosophy,model=eleutherai/pythia-1b-v0",
        ],
        help='Benchmark run entries',
    )
    suite = scfg.Value("my-suite", help="Name of the helm suite")
    max_eval_instances = scfg.Value(7, help="Maximum eval instances")
    num_threads = scfg.Value(1, help="Number of threads")


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
    Downloads offical pre-computed results instead of computing them
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
