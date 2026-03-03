def _create_small_helm_output():
    # Do we want a temp dir, or cache outputs for faster test rerun?
    # import tempfile
    # tempdir = tempfile.TemporaryDirectory(prefix='helm-output-test')

    import ubelt as ub

    dpath = ub.Path.appdir('magnet/tests/helm_output').ensuredir()

    res = ub.cmd(
        'helm-run --run-entries mmlu:subject=philosophy,model=openai/gpt2 --suite my-suite --max-eval-instances 1 --num-threads 1',
        cwd=dpath,
        verbose=3,
    )
    res.check_returncode()

    res = ub.cmd('helm-summarize --suite my-suite', cwd=dpath, verbose=3)
    res.check_returncode()
