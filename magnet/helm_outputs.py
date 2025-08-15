"""
Object oriented classes to represent, load, and explore the outputs of helm
benchmarks.
"""
import ubelt as ub
import pandas as pd
import kwutil
import dacite

from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.run_spec import RunSpec
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.metric import PerInstanceStats


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

    def suites(self, pattern='*'):
        # Note sure if a property or method is best here
        # could do an implicit "view" system like CocoImageView
        # to give best of both worlds in terms of generator / lists but lets
        # also not overcomplicate it.
        return [HelmSuite(p) for p in self._suite_dirs(pattern)]

    def _suite_dirs(self, pattern='*'):
        # not robust to extra directories being written.  is there a way to
        # determine that these directories are actually suites?
        return [p for p in sorted((self.root_dir / 'runs').glob(pattern)) if p.is_dir() and p.name != 'latest']

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
        <HelmSuiteRuns(1)>
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

    def _run_dirs(self, pattern='*'):
        # not robust to extra directories being written.  is there a way to
        # determine that these directories are actually run specs?
        return [p for p in (self.path).glob(pattern) if p.is_dir() if ':' in p.name]

    def runs(self, pattern='*'):
        paths = self._run_dirs(pattern)
        return HelmSuiteRuns(paths)
        # return [HelmRun(p) for p in self._run_dirs(pattern)]


class HelmSuiteRuns(ub.NiceRepr):
    """
    Represents multiple runs from a suite.

    Example:
        >>> from magnet.helm_outputs import *
        >>> self = HelmSuiteRuns.demo()
        >>> print(self)
        <HelmSuiteRuns(1)>
        >>> self.stats_dataframe()
        >>> self.scenario_state_dataframe()
        >>> self.run_spec_dataframe()
    """
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def existing(self):
        """
        Filter to only existing runs
        """
        return self.__class__([
            p for p in self.paths
            if all(p.exists() for p in [
                p / 'stats.json',
                p / 'run_spec.json',
                p / 'scenario_state.json',
            ])
        ])

    @classmethod
    def demo(cls):
        self = HelmOutputs.demo().suites()[0].runs()
        return self

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [HelmRun(p) for p in self.paths[index]]
        else:
            return HelmRun(self.paths[index])

    def stats_dataframe(self):
        # Could likely be quite a bit more efficient here
        table = pd.concat([r.stats_dataframe() for r in self[:]], axis=0)
        return table

    def scenario_state_dataframe(self):
        # Could likely be quite a bit more efficient here
        table = pd.concat([r.scenario_state_dataframe() for r in self[:]], axis=0)
        return table

    def run_spec_dataframe(self):
        # Could likely be quite a bit more efficient here
        table = pd.concat([r.run_spec_dataframe() for r in self[:]], axis=0)
        return table


class HelmRun(ub.NiceRepr):
    """
    Represents a single run in a suite.

    Note:
        The following is a list of json files that are in a helm run directory.

        Output files from helm-run:
            * run_spec.json,
            * per_instance_stats.json,
            * scenario.json,
            * scenario_state.json,
            * stats.json.

        Output files from helm-summarize:
            * instances.json,
            * display_requests.json,
            * display_predictions.json

    Example:
        >>> from magnet.helm_outputs import *
        >>> self = HelmRun.demo()
        >>> print(self)
        <HelmRun(mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2)>
        >>> # HELM objects
        >>> stats = self.stats()
        >>> spec = self.run_spec()
        >>> scenario_state = self.scenario_state()
        >>> print(f'stats = {ub.urepr(stats, nl=1)}')
        >>> print(f'spec = {ub.urepr(spec, nl=1)}')
        >>> print(f'scenario_state = {ub.urepr(scenario_state, nl=1)}')
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
            # TODO: do we need to add scenario.json and per_instance_stats.json
            # What about the files from helm-summarize?
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

    # -- Pure HELM Loaders

    def stats(self) -> Stat:
        stats_list = kwutil.Json.load(self.path / 'stats.json')
        stats = [dacite.from_dict(Stat, json_stat) for json_stat in stats_list]
        return stats

    def per_instance_stats(self) -> RunSpec:
        nested_items = kwutil.Json.load(self.path / 'per_instance_stats.json')
        items = [dacite.from_dict(PerInstanceStats, item) for item in nested_items]
        return items

    def run_spec(self) -> RunSpec:
        nested = kwutil.Json.load(self.path / 'run_spec.json')
        run_spec = dacite.from_dict(RunSpec, nested)
        return run_spec

    def scenario_state(self) -> ScenarioState:
        nested = kwutil.Json.load(self.path / 'scenario_state.json')
        state = dacite.from_dict(ScenarioState, nested)
        return state

    # Note: not sure how to load scenario.json with dacite, or if it matters

    # -- Data Frame Loaders

    def stats_dataframe(self) -> util_pandas.DotDictDataFrame:
        stats_list = kwutil.Json.load(self.path / 'stats.json')
        stats_flat = [kwutil.DotDict.from_nested(stats) for stats in stats_list]
        flat_table = util_pandas.DotDictDataFrame(stats_flat)
        # Add a prefix to enable joins for join keys
        flat_table = flat_table.insert_prefix('stats')
        # Enrich with contextual metadata
        flat_table['run_spec.name'] = self.name
        flat_table = flat_table.reorder(head=['run_spec.name'], axis=1)
        return flat_table

    def perinstance_stats_dataframe(self) -> util_pandas.DotDictDataFrame:
        stats_list = kwutil.Json.load(self.path / 'per_instance_stats.json')
        stats_flat = [kwutil.DotDict.from_nested(stats) for stats in stats_list]
        flat_table = util_pandas.DotDictDataFrame(stats_flat)
        # Add a prefix to enable joins for join keys
        flat_table = flat_table.insert_prefix('per_instance_stats')
        # Enrich with contextual metadata
        flat_table['run_spec.name'] = self.name
        flat_table = flat_table.reorder(head=['run_spec.name'], axis=1)
        return flat_table

    def scenario_state_dataframe(self):
        nested = kwutil.Json.load(self.path / 'scenario_state.json')
        flat_state = kwutil.DotDict.from_nested(nested)
        # Add a prefix to enable joins
        flat_state = flat_state.insert_prefix('scenario_state')
        flat_table = util_pandas.DotDictDataFrame([flat_state])
        # Enrich with contextual metadata for join keys
        flat_table['run_spec.name'] = self.name
        flat_table = flat_table.reorder(head=['run_spec.name'], axis=1)
        return flat_table

    def run_spec_dataframe(self):
        nested = kwutil.Json.load(self.path / 'run_spec.json')
        flat_state = kwutil.DotDict.from_nested(nested)
        flat_state = flat_state.insert_prefix('run_spec')
        flat_table = util_pandas.DotDictDataFrame([flat_state])
        return flat_table
