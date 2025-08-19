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

    stamp = ub.CacheStamp('helm_demo_outputs', depends=['v2'], dpath=dpath)
    if stamp.expired():
        res = ub.cmd(ub.codeblock(
            r'''
            helm-run --run-entries \
                mmlu:subject=philosophy,model=openai/gpt2 \
                mmlu:subject=history,model=openai/gpt2 \
                mmlu:subject=history,model=eleutherai/pythia-1b-v0 \
                mmlu:subject=philosophy,model=eleutherai/pythia-1b-v0 \
            --suite my-suite \
            --max-eval-instances 7 \
            --num-threads 1
            '''), cwd=dpath, verbose=3, system=True)
        res.check_returncode()

        res = ub.cmd('helm-summarize --suite my-suite', cwd=dpath, verbose=3)
        res.check_returncode()
        stamp.renew()

    return dpath
