class HelmOutputs:
    """
    Class to represent and explore helm outputs

    Example:
        >>> from magnet.helm_outputs import *  # NOQA
        >>> self = HelmOutputs.demo()
        >>> self.list_run_specs()
        ['mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2']
        >>> sorted(self.list_suites())
        ['latest', 'my-suite']
    """

    def __init__(self, root_dir):
        self.root_dir = root_dir

    @classmethod
    def demo(cls):
        import magnet
        dpath = magnet.demo.ensure_helm_demo_outputs()
        self = cls(dpath / 'benchmark_output')
        return self

    def list_suites(self):
        # not robust to extra directories being written.  is there a way to
        # determine that these directories are actually suites?
        suite_names = [p.name for p in (self.root_dir / 'runs').glob('*') if p.is_dir()]
        return suite_names

    def list_run_specs(self, suite='*'):
        # not robust to extra directories being written.  is there a way to
        # determine that these directories are actually run specs?
        run_spec_names = [p.name for p in (self.root_dir / 'runs').glob(suite + '/*') if p.is_dir() if ':' in p.name]
        run_spec_names = sorted(set(run_spec_names))
        return run_spec_names
