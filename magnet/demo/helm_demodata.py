def ensure_helm_demo_outputs():
    """
    Create a cached set of helm outputs for testing.

    Returns:
        Path:
            path to demo outputs with "benchmark_output" and "prod_env"
            subdirectories.

    Example:
        >>> from magnet.demo.helm_demodata import *  # NOQA
        >>> dpath = ensure_helm_demo_outputs()
    """
    import ubelt as ub
    dpath = ub.Path.appdir('magnet/tests/helm_output').ensuredir()

    stamp = ub.CacheStamp('helm_demo_outputs', depends=[], dpath=dpath)
    if stamp.expired():
        res = ub.cmd('helm-run --run-entries mmlu:subject=philosophy,model=openai/gpt2 --suite my-suite --max-eval-instances 1 --num-threads 1', cwd=dpath, verbose=3)
        res.check_returncode()

        res = ub.cmd('helm-summarize --suite my-suite', cwd=dpath, verbose=3)
        res.check_returncode()
        stamp.renew()

    return dpath
