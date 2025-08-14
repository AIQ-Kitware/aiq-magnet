"""
Object oriented classes to represent, load, and explore the outputs of helm
benchmarks.
"""
import ubelt as ub
import kwutil
import dacite

from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.run_spec import RunSpec
from helm.benchmark.metrics.statistic import Stat

from magnet.utils import util_pandas

# monkey patch until kwutil 0.3.7
MONKEYPATCH_KWUTIL = True
if MONKEYPATCH_KWUTIL:
    from kwutil import util_dotdict
    kwutil.DotDict = util_dotdict.DotDict


class HelmOutputs(ub.NiceRepr):
    """
    Class to represent and explore helm outputs

    Example:
        >>> import magnet
        >>> self = magnet.HelmOutputs.demo()
        >>> print(self)
        <HelmOutputs(.../magnet/tests/helm_output/benchmark_output)>
        >>> [s.name for s in self.suites()]
        ['my-suite']
        >>> self.list_run_specs(suite='*')
        ['mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2']
    """

    def __init__(self, root_dir):
        self.root_dir = root_dir

    def __nice__(self):
        return self.root_dir

    def write_directory_report(self):
        """
        Print an exploratory summary of how much data is available.

        Requires optional dependency: xdev
        """
        import xdev
        dirwalker = xdev.DirectoryWalker(self.root_dir).build()
        dirwalker.write_report(max_depth=4)

    @classmethod
    def demo(cls):
        import magnet
        dpath = magnet.demo.ensure_helm_demo_outputs()
        root_dir = dpath / 'benchmark_output'
        self = cls(root_dir)
        return self

    def suites(self):
        # Note sure if a property or method is best here
        # could do an implicit "view" system like CocoImageView
        # to give best of both worlds in terms of generator / lists but lets
        # also not overcomplicate it.
        return [HelmSuite(p) for p in self._suite_dirs()]

    def _suite_dirs(self):
        # not robust to extra directories being written.  is there a way to
        # determine that these directories are actually suites?
        return [p for p in sorted((self.root_dir / 'runs').glob('*')) if p.is_dir() and p.name != 'latest']

    def list_suites(self):
        # maybe remove
        return [p.name for p in self._suite_dirs()]

    def list_run_specs(self, suite='*'):
        # maybe remove
        # not robust to extra directories being written.  is there a way to
        # determine that these directories are actually run specs?
        run_spec_names = [p.name for p in (self.root_dir / 'runs').glob(suite + '/*') if p.is_dir() if ':' in p.name]
        run_spec_names = sorted(set(run_spec_names))
        return run_spec_names


class HelmSuite(ub.NiceRepr):
    """
    Represents a single suite in a set of benchmark outputs.

    Example:
        >>> from magnet.helm_outputs import *
        >>> self = HelmSuite.demo()
        >>> print(self)
        <HelmSuite(my-suite)>
        >>> print(self.runs())
        [<HelmRun(mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2) at ...>]
    """
    def __init__(self, path):
        self.path = ub.Path(path)
        self.name = self.path.name

    def __nice__(self):
        return self.name

    @classmethod
    def demo(cls):
        self = HelmOutputs.demo().suites()[0]
        return self

    def _run_dirs(self, suite='*'):
        # not robust to extra directories being written.  is there a way to
        # determine that these directories are actually run specs?
        return [p for p in (self.path).glob('*') if p.is_dir() if ':' in p.name]

    def runs(self):
        return [HelmRun(p) for p in self._run_dirs()]


class HelmRun(ub.NiceRepr):
    """
    Represents a single run in a suite.

    Example:
        >>> from magnet.helm_outputs import *
        >>> self = HelmRun.demo()
        >>> print(self)
        <HelmRun(mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2)>
        >>> # HELM objects
        >>> stats = self.stats()
        >>> spec = self.run_spec()
        >>> scenario = self.scenario_state()
        >>> print(f'stats = {ub.urepr(stats, nl=1)}')
        >>> print(f'spec = {ub.urepr(spec, nl=1)}')
        >>> print(f'scenario = {ub.urepr(scenario, nl=1)}')
        >>> # Dataframe objects
        >>> stats_df = self.stats_dataframe()
        >>> spec_df = self.run_spec_dataframe()
        >>> scenario_df = self.scenario_state_dataframe()
        >>> print(stats_df)
        >>> print(spec_df)
        >>> print(scenario_df)
    """
    def __init__(self, path):
        self.path = ub.Path(path)
        self.name = self.path.name

    def __nice__(self):
        return self.name

    def exists(self):
        return all(p.exists() for p in [
            self.path / 'stats.json',
            self.path / 'run_spec.json',
            self.path / 'scenario_state.json',
        ])

    @classmethod
    def demo(cls):
        suite = HelmOutputs.demo().suites()[0]
        self = suite.runs()[0]
        return self

    ## These are alternatives to functions in loaders.py
    ## Still thinking about the best way to structure these.

    def stats(self) -> Stat:
        stats_list = kwutil.Json.load(self.path / 'stats.json')
        stats = [dacite.from_dict(Stat, json_stat) for json_stat in stats_list]
        return stats

    def run_spec(self) -> RunSpec:
        nested = kwutil.Json.load(self.path / 'run_spec.json')
        run_spec = dacite.from_dict(RunSpec, nested)
        return run_spec

    def scenario_state(self) -> ScenarioState:
        nested = kwutil.Json.load(self.path / 'scenario_state.json')
        state = dacite.from_dict(ScenarioState, nested)
        return state

    def stats_dataframe(self) -> util_pandas.DotDictDataFrame:
        stats_list = kwutil.Json.load(self.path / 'stats.json')
        stats_flat = [kwutil.DotDict.from_nested(stats) for stats in stats_list]
        flat_table = util_pandas.DotDictDataFrame(stats_flat)
        # Add a prefix to enable joins for join keys
        flat_table = flat_table.insert_prefix('stat')
        # Enrich with contextual metadata
        flat_table['run_spec.name'] = self.name
        flat_table = flat_table.reorder(head=['run_spec.name'], axis=1)
        return flat_table

    def scenario_state_dataframe(self):
        nested = kwutil.Json.load(self.path / 'scenario_state.json')
        flat_state = kwutil.DotDict.from_nested(nested)
        # Add a prefix to enable joins
        flat_state = flat_state.insert_prefix('scenario')
        flat_table = util_pandas.DotDictDataFrame([flat_state])
        # Enrich with contextual metadata for join keys
        flat_table['run_spec.name'] = self.name
        flat_table['run_spec.name'] = self.name
        flat_table = flat_table.reorder(head=['run_spec.name'], axis=1)
        return flat_table

    def run_spec_dataframe(self):
        nested = kwutil.Json.load(self.path / 'run_spec.json')
        flat_state = kwutil.DotDict.from_nested(nested)
        flat_state = flat_state.insert_prefix('run_spec')
        flat_table = util_pandas.DotDictDataFrame([flat_state])
        return flat_table
