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

from typing import Generator

from magnet.utils import util_pandas
from functools import cached_property

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
        >>> [s.name for s in self.suites()]
        ['my-suite']
        >>> self.list_run_specs(suite='*')
        ['mmlu:subject=history,method=multiple_choice_joint,model=eleutherai_pythia-1b-v0', 'mmlu:subject=history,method=multiple_choice_joint,model=openai_gpt2', 'mmlu:subject=philosophy,method=multiple_choice_joint,model=eleutherai_pythia-1b-v0', 'mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2']
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

    def summarize(self):
        # TODO: what is the most useful summary information we can quickly get?
        summary = {}
        suites = self.suites()
        runs_per_suite = []
        for suite in suites:
            runs = suite.runs().run_spec()
            runs_per_suite.append(len(runs))
        summary['num_suites'] = len(self._suite_dirs())
        summary['num_run_specs'] = len(self.list_run_specs())
        return summary

    @classmethod
    def demo(cls, **kwargs):
        import magnet
        dpath = magnet.demo.ensure_helm_demo_outputs(**kwargs)
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
        <HelmSuiteRuns(4)>
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
        <HelmSuiteRuns(4)>
        >>> self.per_instance_stats()
        >>> self.run_spec()
        >>> self.scenario_state()
        >>> self.stats()
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
                p / 'run_spec.json',
                p / 'scenario.json',
                p / 'scenario_state.json',
                p / 'per_instance_stats.json',
                p / 'stats.json',
            ])
        ])

    @classmethod
    def demo(cls):
        self = HelmOutputs.demo().suites()[0].runs()
        return self

    def __getitem__(self, index):
        if isinstance(index, slice):
            return HelmSuiteRuns(self.paths[index])
        else:
            return HelmRun(self.paths[index])

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def per_instance_stats(self):
        # Could likely be quite a bit more efficient here
        table = pd.concat([r.dataframe.per_instance_stats() for r in self], axis=0)
        return table

    def run_spec(self):
        # Could likely be quite a bit more efficient here
        table = pd.concat([r.dataframe.run_spec() for r in self], axis=0)
        return table

    def scenario_state(self):
        # Could likely be quite a bit more efficient here
        table = pd.concat([r.dataframe.scenario_state() for r in self], axis=0)
        return table

    def stats(self):
        # Could likely be quite a bit more efficient here
        table = pd.concat([r.dataframe.stats() for r in self], axis=0)
        return table


class _HelmRunJsonView:
    """
    View that provides simple json readers
    """
    def __init__(self, parent):
        self.parent = parent
        self.backend = 'orjson'  # can be ujson or stdlib, but orjson is fastest

    def per_instance_stats(self) -> list[dict]:
        """
        Example:
            >>> from magnet.helm_outputs import *
            >>> self = HelmRun.demo().json
            >>> print(self.per_instance_stats())
        """
        return kwutil.Json.load(self.parent.path / 'per_instance_stats.json', backend=self.backend)

    def run_spec(self) -> dict:
        """
        Example:
            >>> from magnet.helm_outputs import *
            >>> self = HelmRun.demo().json
            >>> print(self.run_spec())
        """
        return kwutil.Json.load(self.parent.path / 'run_spec.json', backend=self.backend)

    def scenario(self) -> dict:
        """
        Example:
            >>> from magnet.helm_outputs import *
            >>> self = HelmRun.demo().json
            >>> print(self.scenario())
        """
        return kwutil.Json.load(self.parent.path / 'scenario.json', backend=self.backend)

    def scenario_state(self) -> dict:
        """
        Example:
            >>> from magnet.helm_outputs import *
            >>> self = HelmRun.demo().json
            >>> print(self.scenario_state())
        """
        return kwutil.Json.load(self.parent.path / 'scenario_state.json', backend=self.backend)

    def stats(self) -> dict:
        """
        Example:
            >>> from magnet.helm_outputs import *
            >>> self = HelmRun.demo().json
            >>> print(self.stats())
        """
        return kwutil.Json.load(self.parent.path / 'stats.json', backend=self.backend)


class _HelmRunDataclassView:
    """
    Helper to provide access to raw HELM data structures.

    Example:
        >>> from magnet.helm_outputs import *
        >>> self = HelmRun.demo().dataclass
        >>> # Raw HELM objects
        >>> per_instance_stats = self.per_instance_stats()
        >>> stats = self.stats()
        >>> spec = self.run_spec()
        >>> scenario_state = self.scenario_state()
        >>> print(f'per_instance_stats = {ub.urepr(per_instance_stats, nl=1)}')
        >>> print(f'stats = {ub.urepr(stats, nl=1)}')
        >>> print(f'spec = {ub.urepr(spec, nl=1)}')
        >>> print(f'scenario_state = {ub.urepr(scenario_state, nl=1)}')
    """
    def __init__(self, parent):
        self.parent = parent

    def per_instance_stats(self) -> Generator[PerInstanceStats, None, None]:
        """
        per_instance_stats.json contains a serialized list of PerInstanceStats,
        which contains the statistics produced for the metrics for each
        instance (i.e. input).
        """
        from helm.benchmark.augmentations.perturbation_description import PerturbationDescription
        from helm.benchmark.metrics.metric_name import MetricName
        nested_items = self.parent.json.per_instance_stats()
        USE_DACITE = 0
        # nested_items = kwutil.Json.load(self.path / 'per_instance_stats.json', backend='ujson')
        if USE_DACITE:
            DACITE_CONFIG = dacite.Config(
                check_types=False,
                type_hooks={
                    MetricName: lambda d: MetricName(**d),
                    PerturbationDescription: lambda d: PerturbationDescription(**ub.udict.intersection(d, PerturbationDescription.__dataclass_fields__.keys()))
                }
            )
            # dacite seems to have a lot of overhead
            for item in nested_items:
                instance = dacite.from_dict(PerInstanceStats, item, config=DACITE_CONFIG)
                yield instance
        else:
            for item in nested_items:
                # Alternative faster loading, using knowledge about the what
                # the dataclass structure is this is not robust to changes in
                # HELM.
                stats_objs = []
                for stat in item['stats']:
                    name = stat['name']
                    if 'perturbation' in name:
                        name['perturbation'] = PerturbationDescription(name['perturbation'])
                    name = MetricName(**name)
                    stat['name'] = name
                    stat_obj = Stat(**stat)
                    stats_objs.append(stat_obj)
                item['stats'] = stats_objs
                perturbation = item.get('perturbation', None)
                if perturbation is not None:
                    perturbation = ub.udict.intersection(perturbation, PerturbationDescription.__dataclass_fields__.keys())
                    perturbation = PerturbationDescription(**perturbation)
                item['perturbation'] = perturbation
                instance = PerInstanceStats(**item)
                yield instance

    def run_spec(self) -> RunSpec:
        """
        run_spec.json contains the RunSpec, which specifies the scenario,
        adapter and metrics for the run.
        """
        nested = self.parent.json.run_spec()
        run_spec = dacite.from_dict(RunSpec, nested)
        return run_spec

    def scenario(self):
        """
        scenario.json contains a serialized Scenario, which contains the
        scenario for the run and specifies the instances (i.e. inputs) used.
        """
        # Note: not sure how to load scenario.json with dacite, or if it
        # matters
        raise NotImplementedError(ub.paragraph(
            '''
            There does not seem to be a way to create an instance of a raw
            helm.benchmark.scenarios.scenario.Scenario from the json file.
            '''))

    def scenario_state(self) -> ScenarioState:
        """
        scenario_state.json contains a serialized ScenarioState, which contains
        every request to and response from the model.
        """
        nested = self.parent.json.scenario_state()
        state = dacite.from_dict(ScenarioState, nested)
        return state

    def stats(self) -> Generator[Stat, None, None]:
        """
        stats.json contains a serialized list of PerInstanceStats, which
        contains the statistics produced for the metrics, aggregated across all
        instances (i.e. inputs).
        """
        stats_list = self.parent.json.stats()
        stats = (dacite.from_dict(Stat, json_stat) for json_stat in stats_list)
        return stats


def _prepare_registry():
    from magnet.utils import util_msgspec
    from helm.benchmark.adaptation.scenario_state import ScenarioState
    from helm.benchmark.run_spec import RunSpec
    from helm.benchmark.metrics.statistic import Stat
    from helm.benchmark.metrics.metric import PerInstanceStats
    util_msgspec.MSGSPEC_REGISTRY.register(ScenarioState)
    util_msgspec.MSGSPEC_REGISTRY.register(RunSpec)
    util_msgspec.MSGSPEC_REGISTRY.register(Stat)
    util_msgspec.MSGSPEC_REGISTRY.register(PerInstanceStats)

_prepare_registry()


class _HelmRunMsgspecView:
    """
    Helper to provide access to raw HELM data structures.

    Example:
        >>> from magnet.helm_outputs import *
        >>> self = HelmRun.demo().msgspec
        >>> # Raw HELM objects
        >>> per_instance_stats = self.per_instance_stats()
        >>> stats = self.stats()
        >>> spec = self.run_spec()
        >>> scenario_state = self.scenario_state()
        >>> print(f'per_instance_stats = {ub.urepr(per_instance_stats, nl=1)}')
        >>> print(f'stats = {ub.urepr(stats, nl=1)}')
        >>> print(f'spec = {ub.urepr(spec, nl=1)}')
        >>> print(f'scenario_state = {ub.urepr(scenario_state, nl=1)}')
    """
    def __init__(self, parent):
        self.parent = parent

    def per_instance_stats(self) -> list:
        """
        per_instance_stats.json contains a serialized list of PerInstanceStats,
        which contains the statistics produced for the metrics for each
        instance (i.e. input).
        """
        from magnet.utils import util_msgspec
        cls = util_msgspec.MSGSPEC_REGISTRY.cache[PerInstanceStats]
        data = (self.parent.path / 'per_instance_stats.json').read_bytes()
        obj = util_msgspec.MSGSPEC_REGISTRY.decode(data, list[cls])
        return obj

    def run_spec(self) -> object:
        """
        run_spec.json contains the RunSpec, which specifies the scenario,
        adapter and metrics for the run.
        """
        from magnet.utils import util_msgspec
        cls = util_msgspec.MSGSPEC_REGISTRY.cache[RunSpec]
        data = (self.parent.path / 'run_spec.json').read_bytes()
        obj = util_msgspec.MSGSPEC_REGISTRY.decode(data, cls)
        return obj

    def scenario(self):
        """
        scenario.json contains a serialized Scenario, which contains the
        scenario for the run and specifies the instances (i.e. inputs) used.
        """
        # Note: not sure how to load scenario.json with dacite, or if it
        # matters
        raise NotImplementedError(ub.paragraph(
            '''
            There does not seem to be a way to create an instance of a raw
            helm.benchmark.scenarios.scenario.Scenario from the json file.
            '''))

    def scenario_state(self) -> ScenarioState:
        """
        scenario_state.json contains a serialized ScenarioState, which contains
        every request to and response from the model.
        """
        from magnet.utils import util_msgspec
        cls = util_msgspec.MSGSPEC_REGISTRY.cache[ScenarioState]
        data = (self.parent.path / 'scenario_state.json').read_bytes()
        obj = util_msgspec.MSGSPEC_REGISTRY.decode(data, cls)
        return obj

    def stats(self) -> list:
        """
        stats.json contains a serialized list of PerInstanceStats, which
        contains the statistics produced for the metrics, aggregated across all
        instances (i.e. inputs).
        """
        from magnet.utils import util_msgspec
        cls = util_msgspec.MSGSPEC_REGISTRY.cache[Stat]
        data = (self.parent.path / 'stats.json').read_bytes()
        obj = util_msgspec.MSGSPEC_REGISTRY.decode(data, list[cls])
        return obj


class _HelmRunDataFrameView:
    def __init__(self, parent):
        self.parent = parent

    def per_instance_stats(self) -> util_pandas.DotDictDataFrame:
        """
        Dataframe representation of :class:`PerInstanceStats`

        Example:
            >>> from magnet.helm_outputs import *
            >>> self = HelmRun.demo()
            >>> print(self.per_instance_stats())
        """
        instance_stats_list = self.parent.json.per_instance_stats()
        rows = []
        for item in instance_stats_list:
            # Each item should correspond to :class:`PerInstanceStats`
            stats_list = item.pop('stats')

            # TODO: cook up a perturbed instance id by hashing
            # the optional perturbation field with instance-id.

            # TODO: build demodata that contains perturbations

            for stats in stats_list:
                row = kwutil.DotDict.from_nested(stats, prefix='stats')
                row.update(item)
                rows.append(row)
        flat_table = util_pandas.DotDictDataFrame(rows)
        # Add a prefix to enable joins for join keys
        flat_table = flat_table.insert_prefix('per_instance_stats')
        # Enrich with contextual metadata (primary key for run_spec joins)
        flat_table['run_spec.name'] = self.parent.name
        flat_table = flat_table.reorder(head=['run_spec.name'], axis=1)
        return flat_table

    def run_spec(self) -> util_pandas.DotDictDataFrame:
        """
        Dataframe representation of :class:`RunSpec`
        """
        nested = self.parent.json.run_spec()
        flat_state = kwutil.DotDict.from_nested(nested)
        flat_state = flat_state.insert_prefix('run_spec')
        flat_table = util_pandas.DotDictDataFrame([flat_state])
        return flat_table

    def scenario(self):
        raise NotImplementedError('not sure if relevant')

    def scenario_state(self) -> util_pandas.DotDictDataFrame:
        """
        Dataframe representation of :class:`ScenarioState`
        """
        top_level = self.parent.json.scenario_state()
        request_states = top_level.pop('request_states')
        flat_top_level = kwutil.DotDict.from_nested(top_level)
        rows = []
        for item in request_states:
            row = kwutil.DotDict.from_nested(item, prefix='request_states')
            row.update(flat_top_level)
            rows.append(row)
        flat_table = util_pandas.DotDictDataFrame(rows)
        # Add a prefix to enable joins for join keys
        flat_table = flat_table.insert_prefix('scenario_state')
        # Enrich with contextual metadata (primary key for run_spec joins)
        flat_table['run_spec.name'] = self.parent.name
        flat_table = flat_table.reorder(head=['run_spec.name'], axis=1)
        return flat_table

    def stats(self) -> util_pandas.DotDictDataFrame:
        """
        Dataframe representation of :class:`Stat`
        """
        stats_list = self.parent.json.stats()
        # TODO: it might be a good idea to hash the name fields to generate
        # unique ids for "types" of stats.
        stats_flat = [kwutil.DotDict.from_nested(stats) for stats in stats_list]
        flat_table = util_pandas.DotDictDataFrame(stats_flat)
        # Add a prefix to enable joins for join keys
        flat_table = flat_table.insert_prefix('stats')
        # Enrich with contextual metadata (primary key for run_spec joins)
        flat_table['run_spec.name'] = self.parent.name
        flat_table = flat_table.reorder(head=['run_spec.name'], axis=1)
        return flat_table


class HelmRun(ub.NiceRepr):
    """
    Represents a single run in a suite.

    This provides output to postprocessed dataframe representations of HELM
    objects. For access to raw HELM objects, use the ``raw`` attribute.

    Note:
        The following is a list of json files that are in a helm run directory.

        Output files from helm-run:
            * per_instance_stats.json,
            * run_spec.json,
            * scenario.json,
            * scenario_state.json,
            * stats.json,

        See [HelmTutorial]_ for a description of each.

        Output files from helm-summarize:
            * display_predictions.json,
            * display_requests.json,
            * instances.json,

    References:
        .. [HelmTutorial] https://crfm-helm.readthedocs.io/en/v0.3.0/tutorial/

    Example:
        >>> from magnet.helm_outputs import *
        >>> self = HelmRun.demo()
        >>> print(self)
        <HelmRun(mmlu:subject=history,method=multiple_choice_joint,model=eleutherai_pythia-1b-v0)>
        >>> # Dataframe objects
        >>> per_instance_stats_df = self.per_instance_stats()
        >>> stats_df = self.stats()
        >>> spec_df = self.run_spec()
        >>> scenario_df = self.scenario_state()
        >>> print(per_instance_stats_df)
        >>> print(stats_df)
        >>> print(spec_df)
        >>> print(scenario_df)
    """
    def __init__(self, path):
        self.path = ub.Path(path)
        self.name = self.path.name

    @cached_property
    def json(self):
        """
        Access to direct JSON view
        """
        return _HelmRunJsonView(self)

    @cached_property
    def dataclass(self):
        """
        Access HELM dataclass view
        """
        return _HelmRunDataclassView(self)


    @cached_property
    def msgspec(self):
        """
        Much faster access to HELM dataclass-like (msgspec) view
        """
        return _HelmRunMsgspecView(self)

    @cached_property
    def dataframe(self):
        """
        Access flattened dataframe view
        """
        return _HelmRunDataFrameView(self)

    def __nice__(self):
        return self.name

    def exists(self):
        return all(p.exists() for p in [
            # TODO: do we need to add scenario.json and per_instance_stats.json
            # What about the files from helm-summarize?
            # self.path / 'per_instance_stats.json', does this always exist ???
            self.path / 'run_spec.json',
            self.path / 'scenario_state.json',
            # self.path / 'scenario.json', does this always exist ???
            self.path / 'stats.json',
        ])

    @classmethod
    def demo(cls):
        suite = HelmOutputs.demo().suites()[0]
        self = suite.runs()[-1]
        return self

    # These are alternatives to functions in loaders.py

    def per_instance_stats(self) -> util_pandas.DotDictDataFrame:
        """
        Dataframe representation of :class:`PerInstanceStats`
        """
        return self.dataframe.per_instance_stats()

    def run_spec(self) -> util_pandas.DotDictDataFrame:
        """
        Dataframe representation of :class:`RunSpec`
        """
        return self.dataframe.run_spec()

    def scenario_state(self) -> util_pandas.DotDictDataFrame:
        """
        Dataframe representation of :class:`ScenarioState`
        """
        return self.dataframe.scenario_state()

    def stats(self) -> util_pandas.DotDictDataFrame:
        """
        Dataframe representation of :class:`Stat`
        """
        return self.dataframe.stats()
