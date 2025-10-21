"""
Object oriented classes to represent, load, and explore the outputs of helm
benchmarks.

TODO:
    - [ ] move this file into the helm backend directory.
"""
from __future__ import annotations
import ubelt as ub
import pandas as pd
import kwutil
import dacite

from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.run_spec import RunSpec
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.metrics.metric import PerInstanceStats

from typing import Generator, Union, Self

from magnet.utils import util_pandas
from magnet.utils import util_msgspec
from magnet.utils.util_iterable import add_length_hint
from functools import cached_property

# Pre-register msgspec structure variants of the HELM dataclass types
ScenarioStateStruct = util_msgspec.MSGSPEC_REGISTRY.register(ScenarioState, dict=True)
RunSpecStruct = util_msgspec.MSGSPEC_REGISTRY.register(RunSpec)
StatStruct = util_msgspec.MSGSPEC_REGISTRY.register(Stat)
PerInstanceStatsStruct = util_msgspec.MSGSPEC_REGISTRY.register(PerInstanceStats)

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
        >>> suite_names = [s.name for s in self.suites()]
        >>> run_names = self.list_run_specs(suite='*')
        >>> summary = self.summarize()
        >>> print(f'suite_names = {ub.urepr(suite_names, nl=1)}')
        >>> print(f'run_names = {ub.urepr(run_names, nl=1)}')
        >>> print(f'summary = {ub.urepr(summary, nl=1)}')
        suite_names = [
            'my-suite',
        ]
        run_names = [
            'mmlu:subject=anatomy,method=multiple_choice_joint,model=eleutherai_pythia-1b-v0',
            'mmlu:subject=anatomy,method=multiple_choice_joint,model=openai_gpt2',
            'mmlu:subject=philosophy,method=multiple_choice_joint,model=eleutherai_pythia-1b-v0',
            'mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2',
        ]
        summary = {
            'num_suites': 1,
            'num_run_specs': 4,
            'stats':        n_stats  n_metrics  n_perinstance  num_outputs  num_trials  num_train_trials
                     count      4.0        4.0            4.0          4.0         4.0               4.0
                     mean     162.0        3.0            7.0          5.0         1.0               1.0
                     std        0.0        0.0            0.0          0.0         0.0               0.0,
        }

    Example:
        >>> # xdoctest: +REQUIRES(module:xdev)
        >>> import magnet
        >>> self = magnet.HelmOutputs.demo()
        >>> self.write_directory_report()  # xdoctest: +IGNORE_WANT
        ╙── .../magnet/tests/helm_output/5be22292db3f/benchmark_output: txt.size=10.08 KB,txt.files=2,csv.size=158.33 MB,csv.files=179,.size=4.00 KB,.files=1,json.size=2.64 MB,json.files=88,tex.size=13.11 KB,tex.files=42
            ├─╼ scenarios: txt.size=10.08 KB,txt.files=2,csv.size=158.33 MB,csv.files=179
            │   └─╼ mmlu: txt.size=10.08 KB,txt.files=2,csv.size=158.33 MB,csv.files=179
            │       └─╼ data: txt.size=10.08 KB,txt.files=2,csv.size=158.33 MB,csv.files=179
            │           └─╼  ...
            ├─╼ scenario_instances
            └─╼ runs: .size=4.00 KB,.files=1,json.size=2.64 MB,json.files=88,tex.size=13.11 KB,tex.files=42
                ├─╼ latest -> my-suite: size=4.00 KB
                └─╼ my-suite: json.size=2.64 MB,json.files=88,tex.size=13.11 KB,tex.files=42
                    ├─╼ runs_to_run_suites.json: size=0.36 KB
        ...
           files       size                name
        0    181  158.34 MB           scenarios
        1      0    0.00 KB  scenario_instances
        2    131    2.65 MB                runs
        ...
        kind    files       size
        ext
        *null*      1    4.00 KB
        txt         2   10.08 KB
        tex        42   13.11 KB
        json       88    2.64 MB
        csv       179  158.33 MB
        kind     files       size
        ext
        ∑ total    312  160.99 MB
        ...
    """

    def __init__(self, root_dir):
        """
        Args:
            root_dir (str | PathLike):
                The benchmark output directory containing a runs folder with
                multiple suites.
        """
        # TODO: Currently this should be a folder named "benchmark_output",
        # I'm not sure that I like that as it makes the directory structure
        # very inflexible. We might want to make this attribute protected so we
        # can change it.
        self.root_dir = root_dir

    def __nice__(self):
        return self.root_dir

    @classmethod
    def coerce(cls, input) -> Self:
        """
        Convert some reasonable representation into a HelmOutputs object.

        Args:
            input (str | PathLike | HelmOutputs):
                An existing HelmOutputs object or a path to the benchmark.
                If the input is a path, it could be the path containing
                benchmark_output/runs or one of those paths, and we will coerce
                it to the expected input.

        Returns:
            HelmOutputs

        Example:
            >>> from magnet.helm_outputs import *
            >>> self = HelmOutputs.demo()
            >>> assert HelmOutputs.coerce(self) is self, 'check inplace return'
            >>> assert HelmOutputs.coerce(self.root_dir).root_dir == self.root_dir, 'check path coerce'
            >>> assert HelmOutputs.coerce(self.root_dir / 'runs').root_dir == self.root_dir, 'check path coerce'
            >>> assert HelmOutputs.coerce(self.root_dir.parent).root_dir == self.root_dir, 'check path coerce'
        """
        import os
        if isinstance(input, cls):
            # Input is already a object of this type, return it inplace.
            self = input
        elif isinstance(input, (str, os.PathLike)):
            # The input is a type of path, validate it pass it in as we expect.
            root_dir = cls._coerce_input_path(input)
            self = cls(root_dir)
        else:
            raise TypeError(f'Unable to coerce {type(input)}')
        return self

    @classmethod
    def _coerce_input_path(cls, path):
        """
        HELM conventions expect that the input path to a set of suits looks
        like ``<prefix>/benchmark_output/runs``, but specifying prefix with or
        without either of the later two subdirectories is typically
        unambiguous, thus we allow some flexibility in the inputs and resolve
        them to something we expect.

        Returns:
            Path: the path ending with benchmark_output

        Example:
            >>> from magnet.helm_outputs import *
            >>> self = HelmOutputs.demo()
            >>> root = self.root_dir.parent
            >>> result1 = HelmOutputs._coerce_input_path(root)
            >>> result2 = HelmOutputs._coerce_input_path(root / 'benchmark_output')
            >>> result3 = HelmOutputs._coerce_input_path(root / 'benchmark_output/runs')
            >>> assert result1 == result2 == result3
        """
        path = ub.Path(path)
        if path.name == 'benchmark_output':
            return path
        elif path.parts[-2:] == ('benchmark_output', 'runs'):
            return path.parent
        else:
            candidate = path / 'benchmark_output'
            if candidate.exists():
                return candidate
            else:
                raise FileNotFoundError("Unable to find a directory that looks like HELM outputs")

    def write_directory_report(self):
        """
        Print an exploratory summary of how much data is available.

        Requires optional dependency: xdev
        """
        import xdev
        dirwalker = xdev.DirectoryWalker(self.root_dir,
                                         exclude_fnames=['lock', '*.lock']).build()
        dirwalker.write_report(max_depth=4)

    def summarize(self):
        # TODO: what is the most useful summary information we can quickly get?
        summary = {}
        suites = self.suites()
        rows = []
        for suite in suites:
            runs = suite.runs()
            for run in runs:
                n_stats = len(run.msgspec.stats())
                n_perinstance = len(run.msgspec.per_instance_stats())
                run_spec = run.msgspec.run_spec()
                adapter_spec = run.msgspec.scenario_state().adapter_spec
                rows.append({
                    'name': run.name,
                    'n_stats': n_stats,
                    'n_metrics': len(run_spec.metric_specs),
                    'n_perinstance': n_perinstance,
                    'num_outputs': adapter_spec.num_outputs,
                    'num_trials': adapter_spec.num_trials,
                    'num_train_trials': adapter_spec.num_train_trials,
                })

        df = pd.DataFrame(rows)
        stats = df.describe().loc[['count', 'mean', 'std']]
        summary['num_suites'] = len(self._suite_dirs())
        summary['num_run_specs'] = len(self.list_run_specs())
        summary['stats'] = stats
        return summary

    @classmethod
    def demo(cls, method='compute', **kwargs) -> Self:
        import magnet
        if method == 'compute':
            dpath = magnet.demo.helm_demodata.ensure_helm_demo_outputs(**kwargs)
        elif method == 'download':
            dpath = magnet.demo.helm_demodata.grab_helm_demo_outputs(**kwargs)
        else:
            raise KeyError(method)
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
    def demo(cls) -> Self:
        self = HelmOutputs.demo().suites()[0]
        return self

    @classmethod
    def coerce(cls, input) -> Self:
        """
        Convert some reasonable representation of a HelmSuite into an object.

        Args:
            input (str | PathLike | HelmSuite):
                An existing HelmSuite object or a path to the suite.
                We may expand this definition in the future.

        Returns:
            HelmSuite

        Example:
            >>> from magnet.helm_outputs import *
            >>> self = HelmSuite.demo()
            >>> assert HelmSuite.coerce(self) is self, 'check inplace return'
            >>> assert HelmSuite.coerce(self.path).path == self.path, 'check path coerce'
        """
        import os
        if isinstance(input, cls):
            # Input is already a suite object, return it inplace.
            self = input
        elif isinstance(input, (str, os.PathLike)):
            # input is likely a path to a suite, todo: could add validation
            self = cls(input)
        else:
            raise TypeError(f'Unable to coerce {type(input)}')
        return self

    def _run_dirs(self, pattern='*'):
        # not robust to extra directories being written.  is there a way to
        # determine that these directories are actually run specs?
        return sorted([p for p in (self.path).glob(pattern) if p.is_dir() if ':' in p.name])

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
    def coerce(cls, input) -> Self:
        """
        Convert some reasonable representation of a HelmSuiteRuns into an object.

        Args:
            input (str | PathLike | HelmSuite | HelmSuiteRuns | List[str | PathLike]):
                An existing HelmSuiteRuns object, path to runs within a suite,
                a single run path, or a HelmSuite.

        Returns:
            HelmSuiteRuns

        Example:
            >>> from magnet.helm_outputs import *
            >>> self = HelmSuiteRuns.demo()
            >>> assert HelmSuiteRuns.coerce(self) is self, 'check inplace return'
            >>> assert HelmSuiteRuns.coerce(self.paths).paths == self.paths, 'check coerce from List[path]'
            >>> assert HelmSuiteRuns.coerce(self.paths[0]).paths == self.paths[0:1], 'check coerce from path'
            >>> assert len(HelmSuiteRuns.coerce(HelmSuite.demo()).paths) > 2, 'check coerce from HelmSuite'
        """
        import os
        if isinstance(input, cls):
            # return in put inplace
            self = input
        elif isinstance(input, HelmSuite):
            # return in put inplace
            self = input.runs()
        elif isinstance(input, (str, os.PathLike)):
            # input is likely a path to a run, todo: could add validation
            # todo: determine if the path is more likely a run or a suite
            self = cls([ub.Path(input)])
        elif isinstance(input, list):
            self = cls([ub.Path(p) for p in input])
        else:
            raise TypeError(f'Unable to coerce {type(input)}')
        return self

    @classmethod
    def demo(cls):
        self = HelmOutputs.demo().suites()[0].runs()
        return self

    def __getitem__(self, index) -> Union[HelmSuiteRuns, HelmRun]:
        """
        Return an slice of HelmSuiteRuns or a single HelmRun
        """
        if isinstance(index, slice):
            return HelmSuiteRuns(self.paths[index])
        else:
            return HelmRun(self.paths[index])

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def per_instance_stats(self) -> util_pandas.DotDictDataFrame:
        # Could likely be quite a bit more efficient here
        table = pd.concat([r.dataframe.per_instance_stats() for r in self], axis=0)
        return table

    def run_spec(self) -> util_pandas.DotDictDataFrame:
        # Could likely be quite a bit more efficient here
        table = pd.concat([r.dataframe.run_spec() for r in self], axis=0)
        return table

    def scenario_state(self) -> util_pandas.DotDictDataFrame:
        # Could likely be quite a bit more efficient here
        table = pd.concat([r.dataframe.scenario_state() for r in self], axis=0)
        return table

    def stats(self) -> util_pandas.DotDictDataFrame:
        # Could likely be quite a bit more efficient here
        table = pd.concat([r.dataframe.stats() for r in self], axis=0)
        return table

### --- Helm Run View Backends


class _HelmRunJsonView:
    """
    A view of a single HelmRun that provides simple json loading methods.

    Note:
        This can use different json backends, but orjson is fastest

    Example:
        >>> from magnet.helm_outputs import *
        >>> self = HelmRun.demo().json
        >>> per_instance_stats = self.per_instance_stats()
        >>> stats = self.stats()
        >>> spec = self.run_spec()
        >>> scenario_state = self.scenario_state()
        >>> print(f'per_instance_stats = {ub.urepr(per_instance_stats, nl=1)}')
        >>> print(f'stats = {ub.urepr(stats, nl=1)}')
        >>> print(f'spec = {ub.urepr(spec, nl=1)}')
        >>> print(f'scenario_state = {ub.urepr(scenario_state, nl=1)}')
    """
    def __init__(self, parent: HelmRun, backend='orjson'):
        self.parent = parent
        self.backend = backend  # can be ujson or stdlib, but orjson is fastest

    def per_instance_stats(self) -> list[dict]:
        """
        A json view for a list of :class:`PerInstanceStats` objects

        Example:
            >>> from magnet.helm_outputs import *
            >>> self = HelmRun.demo().json
            >>> print(self.per_instance_stats())
        """
        return kwutil.Json.load(self.parent.path / 'per_instance_stats.json', backend=self.backend)

    def run_spec(self) -> dict:
        """
        A json view of :class:`RunSpec` objects

        Example:
            >>> from magnet.helm_outputs import *
            >>> self = HelmRun.demo().json
            >>> print(self.run_spec())
        """
        return kwutil.Json.load(self.parent.path / 'run_spec.json', backend=self.backend)

    def scenario(self) -> dict:
        """
        A json view of a :class:`Scenario` object

        Example:
            >>> from magnet.helm_outputs import *
            >>> self = HelmRun.demo().json
            >>> print(self.scenario())
        """
        return kwutil.Json.load(self.parent.path / 'scenario.json', backend=self.backend)

    def scenario_state(self) -> dict:
        """
        A json view of a :class:`ScenarioState` object

        Example:
            >>> from magnet.helm_outputs import *
            >>> self = HelmRun.demo().json
            >>> print(self.scenario_state())
        """
        return kwutil.Json.load(self.parent.path / 'scenario_state.json', backend=self.backend)

    def stats(self) -> list[dict]:
        """
        A json view of a :class:`Stat` object

        Example:
            >>> from magnet.helm_outputs import *
            >>> self = HelmRun.demo().json
            >>> print(self.stats())
        """
        return kwutil.Json.load(self.parent.path / 'stats.json', backend=self.backend)


class _HelmRunDataclassView:
    """
    A view of a single HelmRun that will return raw HELM dataclasses from its
    loader methods.

    Example:
        >>> from magnet.helm_outputs import *
        >>> self = HelmRun.demo().dataclass
        >>> per_instance_stats = list(self.per_instance_stats())
        >>> stats = list(self.stats())
        >>> spec = self.run_spec()
        >>> scenario_state = self.scenario_state()
        >>> print(f'per_instance_stats = {ub.urepr(per_instance_stats, nl=1)}')
        >>> print(f'stats = {ub.urepr(stats, nl=1)}')
        >>> print(f'spec = {ub.urepr(spec, nl=1)}')
        >>> print(f'scenario_state = {ub.urepr(scenario_state, nl=1)}')
    """
    def __init__(self, parent: HelmRun):
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

        def _gen_per_instance_stats():
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

        return add_length_hint(_gen_per_instance_stats(), len(nested_items), known_length=True)

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
        stats = add_length_hint(stats, len(stats_list), known_length=True)
        return stats


class _HelmRunMsgspecView:
    """
    A view of a single HelmRun that will return MsgSpec structures from its
    loader methods. These are similar to the native HELM dataclasses, but they
    often load much faster.

    Example:
        >>> from magnet.helm_outputs import *
        >>> self = HelmRun.demo().msgspec
        >>> per_instance_stats = self.per_instance_stats()
        >>> stats = self.stats()
        >>> spec = self.run_spec()
        >>> scenario_state = self.scenario_state()
        >>> print(f'per_instance_stats = {ub.urepr(per_instance_stats, nl=1)}')
        >>> print(f'stats = {ub.urepr(stats, nl=1)}')
        >>> print(f'spec = {ub.urepr(spec, nl=1)}')
        >>> print(f'scenario_state = {ub.urepr(scenario_state, nl=1)}')
    """
    def __init__(self, parent: HelmRun):
        self.parent = parent

    def per_instance_stats(self) -> list[PerInstanceStatsStruct]:
        """
        per_instance_stats.json contains a serialized list of PerInstanceStats,
        which contains the statistics produced for the metrics for each
        instance (i.e. input).
        """
        data = (self.parent.path / 'per_instance_stats.json').read_bytes()
        obj = util_msgspec.MSGSPEC_REGISTRY.decode(data, list[PerInstanceStatsStruct])
        return obj

    def run_spec(self) -> RunSpecStruct:
        """
        run_spec.json contains the RunSpec, which specifies the scenario,
        adapter and metrics for the run.
        """
        data = (self.parent.path / 'run_spec.json').read_bytes()
        obj = util_msgspec.MSGSPEC_REGISTRY.decode(data, RunSpecStruct)
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

    def scenario_state(self) -> ScenarioStateStruct:
        """
        scenario_state.json contains a serialized ScenarioState, which contains
        every request to and response from the model.

        FIXME:
            ScenarioState has a __post_init__

        CommandLine:
            xdoctest -m magnet.helm_outputs _HelmRunMsgspecView.scenario_state

        Example:
            >>> from magnet.helm_outputs import *
            >>> run = HelmRun.demo()
            >>> self = run.msgspec
            >>> state1 = self.scenario_state()
            >>> state2 = run.dataclass.scenario_state()
            >>> assert state1.__annotations__.keys() == state2.__annotations__.keys()
        """
        from magnet.utils import util_msgspec
        data = (self.parent.path / 'scenario_state.json').read_bytes()
        obj = util_msgspec.MSGSPEC_REGISTRY.decode(data, ScenarioStateStruct)
        ScenarioState.__post_init__(obj)  # Hack
        return obj

    def stats(self) -> list[StatStruct]:
        """
        stats.json contains a serialized list of PerInstanceStats, which
        contains the statistics produced for the metrics, aggregated across all
        instances (i.e. inputs).
        """
        data = (self.parent.path / 'stats.json').read_bytes()
        obj = util_msgspec.MSGSPEC_REGISTRY.decode(data, list[StatStruct])
        return obj


class _HelmRunDataFrameView:
    """
    A view of a single HelmRun that will return DataFrame objects
    from its loader methods.

    Example:
        >>> from magnet.helm_outputs import *
        >>> self = HelmRun.demo().dataframe
        >>> per_instance_stats = self.per_instance_stats()
        >>> stats = self.stats()
        >>> spec = self.run_spec()
        >>> scenario_state = self.scenario_state()
        >>> print(f'per_instance_stats = {ub.urepr(per_instance_stats, nl=1)}')
        >>> print(f'stats = {ub.urepr(stats, nl=1)}')
        >>> print(f'spec = {ub.urepr(spec, nl=1)}')
        >>> print(f'scenario_state = {ub.urepr(scenario_state, nl=1)}')
    """

    def __init__(self, parent: HelmRun):
        self.parent = parent

    def per_instance_stats(self) -> util_pandas.DotDictDataFrame:
        """
        Dataframe representation of :class:`PerInstanceStats`

        Example:
            >>> from magnet.helm_outputs import *
            >>> self = HelmRun.demo()
            >>> table = (self.per_instance_stats())
            >>> print(table)
            >>> assert len(table) > 180
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

        Example:
            >>> from magnet.helm_outputs import *
            >>> self = HelmRun.demo()
            >>> table = (self.run_spec())
            >>> print(table)
            >>> assert len(table) == 1
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

        Example:
            >>> from magnet.helm_outputs import *
            >>> self = HelmRun.demo()
            >>> table = (self.scenario_state())
            >>> print(table)
            >>> assert len(table) >= 7
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

        Example:
            >>> from magnet.helm_outputs import *
            >>> self = HelmRun.demo()
            >>> table = (self.stats())
            >>> print(table)
            >>> assert len(table) >= 160
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
    objects. For access to raw HELM objects, use the ``dataclass`` attribute.

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
        <HelmRun(mmlu:subject=philosophy,method=multiple_choice_joint,model=openai_gpt2)>
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
    def _json_stdlib(self):
        return _HelmRunJsonView(self, backend='stdlib')

    @cached_property
    def _json_orjson(self):
        return _HelmRunJsonView(self, backend='orjson')

    @cached_property
    def _json_ujson(self):
        return _HelmRunJsonView(self, backend='ujson')

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
        """
        Determine if the expected json files for this run directory exist.
        """
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
    def demo(cls) -> Self:
        suite = HelmOutputs.demo().suites()[0]
        self = suite.runs()[-1]
        return self

    # Default accessors

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
