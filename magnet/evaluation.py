from weasel.util.config import _parse_overrides
import builtins
import json
import sys
from graphlib import TopologicalSorter
from itertools import product
from typing import Any, Dict, List, Self, Tuple, get_args, get_origin

import kwutil
import scriptconfig as scfg
import ubelt as ub
import yaml
from kwdagger import Pipeline, ProcessNode
from kwdagger.schedule import ScheduleEvaluationConfig, build_schedule
from loguru import logger
from rich import print


class EvaluationConfig(scfg.DataConfig):
    """
    Resolve an Evaluation Card
    """

    __epilog__ = """
    Usage:
      ./evaluation.py <evaluation_card_path>

    Examples:
      # Show docs
      python -m magnet.evaluation --help

      # Run example card
      python -m magnet.evaluation magnet/cards/simple.yaml
    """

    path = scfg.Value(
        None, required=True, position=1, help='Path to evaluation card YAML'
    )

    results_path = scfg.Value(
        './results', help='Root data path for saved results'
    )

    override = scfg.Value(
        None,
        #nargs='*', 
        type=str,
        help='Override symbol values (e.g. --override dataset=legalbench, num_replicates=5)',
    )


class EvaluationCard:
    """
    Specification of an empirical claim with resolvable symbols and metadata

    Example:
        >>> from importlib.resources import files
        >>> from magnet.evaluation import EvaluationCard
        >>> card_name = 'simple.yaml'
        >>> card_path = files('magnet') / 'cards' / card_name
        >>> results_path = './results'
        >>> card = EvaluationCard(card_path, results_path)
        >>> card.evaluate()
        'VERIFIED'
        >>> 
        >>> # Replacement example
        >>> import kwutil
        >>> example_symbols = kwutil.Yaml.coerce(
            '''
            symbols:
              data_path:
                type: str
                value: './data/runs'
              confidence:
                type: float
                value: 0.1
              model:
                sweep:
                  - llama-2-13b
                  - gpt-5.4-pro
            ''')
        >>> card.symbols = example_symbols.get('symbols')
        >>> def show_symbol_values(symbols):
        >>>   # Print out symbol resolution
        >>>   for symbol in symbols:
        >>>     if 'sweep' in symbols[symbol]:
        >>>       print(f"{symbol}: {symbols[symbol]['sweep']}")
        >>>     else:
        >>>       print(f"{symbol}: {symbols[symbol]['value']}")
        >>>
        >>> show_symbol_values(card.symbols)
        data_path: ./data/runs
        confidence: 0.1
        model: ['llama-2-13b', 'gpt-5.4-pro']
        >>> override = '''
            confidence: 0.01 
            model: [claude-3.5-sonnet, gemini-1.5-pro-001]
        '''
        >>> card.replace(override)
        >>> show_symbol_values(card.symbols)
        data_path: ./data/runs
        confidence: 0.01
        model: ['llama-2-13b', 'gpt-5.4-pro', 'claude-3.5-sonnet', 'gemini-1.5-pro-001']
    """

    def __init__(self, path, results_path):
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)

        self.results_path = ub.Path(results_path)

        self.title = cfg.get('title', '')
        self.description = cfg.get('description', '')

        self.claim = Claim(cfg.get('claim'))
        self.symbols = cfg.get('symbols', {})

        # explicit kwdagger spec
        self.has_kwdagger = 'kwdagger' in cfg
        self.kwdagger = cfg.get('kwdagger')

        # populate ProcessNode(s) programmatically
        self.has_pipeline = 'pipeline' in cfg
        self.pipeline = cfg.get('pipeline')

        self.evaluations = []

    def status(self) -> str:
        """
        Declaration of card state, whether not started, in progress, or complete
        """
        if self.claim.status == 'UNVERIFIED' and len(self.evaluations) > 0:
            not_evaluated_count = sum(
                [
                    evaluation.claim.status == 'UNVERIFIED'
                    for evaluation in self.evaluations
                ]
            )
            percent_not_evaluated = not_evaluated_count / len(self.evaluations)

            if percent_not_evaluated == 0:
                return 'EVALUATED'
            else:
                return f'{percent_not_evaluated:.2f} REMAINING'
        else:
            return 'EVALUATED'

    def replace(self, override_str):
        """
        Handle overrides in symbol field by replacing 'value' entries and appending to sweeps
        """
        override = kwutil.Yaml.coerce(override_str)

        for key, value in override.items():
            if key not in self.symbols:
                raise ValueError(f"Unknown symbol '{key}' -- available: {list(self.symbols.keys())}")
            if 'value' in self.symbols[key]:
                # replacement
                self.symbols[key]['value'] = value
            elif 'sweep' in self.symbols[key]:
                # accept n entries
                if isinstance(value, list):
                    self.symbols[key]['sweep'].extend(value)
                else:
                    self.symbols[key]['sweep'].append(value)

    def evaluate(self):
        """
        Run the evaluation specification

        1. Resolve symbol definitions
        2. Evaluate claim under symbol values
        3. Summarize general finding
        """
        results = []

        if self.has_kwdagger:
            # Explicit kwdagger pipeline defined
            # Claim node handles symbols outside of EvaluationCard
            results = KWDaggerProcessor(
                self.kwdagger, root_dpath=self.results_path
            ).collect_results()

        elif self.has_pipeline:
            # Implicit pipeline definition needs parsing
            pipeline_runs = GenericPipelineProcessor(
                self.pipeline, root_dpath=self.results_path
            ).collect_symbols()

            # Claim resolution
            results = []

            for run in pipeline_runs:
                run_symbols = pipeline_runs[run]
                self.symbols.update(run_symbols)
                self.evaluations.extend(
                    self.dispatch(Symbols.decompose_symbol_defs(self.symbols))
                )

            for evaluation in self.evaluations:
                status, _ = evaluation.execute()
                results.append(status)

        else:
            # Serial Evaluation Card
            self.evaluations = self.dispatch(
                Symbols.decompose_symbol_defs(self.symbols)
            )

            results = []
            for evaluation in self.evaluations:
                status, _ = evaluation.execute()
                results.append(status)

        total = len(results)

        def percentage(count):
            return count / total

        verified_count = results.count('VERIFIED')
        falsified_count = results.count('FALSIFIED')
        inconclusive_count = results.count('INCONCLUSIVE')

        print('================================')
        print(f'Settings Evaluated: {total}')
        print(f'  Verified:     {percentage(verified_count):.2f}')
        print(f'  Falsified:    {percentage(falsified_count):.2f}')
        print(f'  Inconclusive: {percentage(inconclusive_count):.2f}')
        print('================================')
        print('\n')

        card_result = ''
        if falsified_count:
            card_result = 'FALSIFIED'
        elif inconclusive_count:
            card_result = 'INCONCLUSIVE'
        else:
            card_result = 'VERIFIED'

        self.claim.status = card_result
        return card_result

    def dispatch(self, flattened_sweep):
        return [
            EvaluationTask(Claim({'python': self.claim.claim}), symbols)
            for symbols in flattened_sweep
        ]

    def summarize(self):
        """
        Human-readable summary of card in its current state
        """
        print(f'[bold]Title:[/bold]       {self.title}')
        print(f'[bold]Description:[/bold] {self.description}')
        print('================================')
        # print(f"SYMBOLS:     {self.symbols()}")
        print(f'[bold]CLAIM:[/bold]       \n{self.claim}')

        status = self.status()
        if self.claim.status == 'VERIFIED':
            claim_status_color = 'green'
        elif self.claim.status == 'FALSIFIED':
            claim_status_color = 'red'
        else:
            claim_status_color = 'yellow'

        if status == 'EVALUATED':
            print('================================')
            print(
                f'[bold]RESULT:[/bold]      [bold][{claim_status_color}]{self.claim.status}[/{claim_status_color}][/bold]'
                ''
            )

        print('================================')
        print(f'[bold]CARD STATUS:[/bold] {status}')


class GenericPipelineProcessor:
    """
    Handler for yaml-based pipeline specification

    NOTE:
        *possibly merge with KWDaggerProcessor*

    Example:
        >>> from magnet.evaluation import GenericPipelineProcessor
        >>> import kwutil
        >>> # Example snippet of an Evaluation Card
        >>> example_cfg = kwutil.Yaml.coerce(
            '''
            pipeline:
              predict_node:
                executable: python -m magnet.examples.llama_consistency.llama_predict
                algo_params:
                  base_model: ["meta/llama-2-13b", "meta/llama-2-70b"]
                  comp_model: ["meta/llama-2-7b", "meta/llama-3-70b"]
                out_paths:
                  results_fpath: ./llama_results.json
            ''')
        >>> root_dpath = "."
        >>> pipeline_def = example_cfg['pipeline']
        >>> pipeline = GenericPipelineProcessor(pipeline_def, root_dpath)
        >>> #
        >>> # Construct One Node Pipeline
        >>> pipeline.define_kwdagger()
        ...
        >>> pipeline.dag.print_graphs()

        Process Graph
        ╙── predict_node

        IO Graph
        ╙── predict_node
            ╽
            results_fpath

        >>> for attr in ['name', 'executable', 'algo_params', 'out_paths']:
        >>>    print(getattr(pipeline.dag.nodes['predict_node'], attr))
        predict_node
        python -m magnet.examples.llama_consistency.llama_predict
        ['base_model', 'comp_model']
        {'results_fpath': './llama_results.json'}
        >>> #
        >>> # Parameters matrix
        >>> pipeline.matrix
        {'predict_node.base_model': ['meta/llama-2-13b', 'meta/llama-2-70b'],
        'predict_node.comp_model': ['meta/llama-2-7b', 'meta/llama-3-70b']}
    """

    def __init__(self, pipeline_def, root_dpath):
        self.pipeline = pipeline_def
        self.root_dpath = root_dpath
        self.dag = None
        self.matrix = None
        self.symbols = {}

    def define_kwdagger(self):
        """
        Construct kwdagger pipeline programmatically

        *only verified for one-stage pipeline, needs 'connector' handling*
        """
        nodes = {}

        for node_name in self.pipeline:
            # collect nodes
            node_params = self.pipeline[node_name]

            # FIXME: should update matrix for full pipeline
            node_params, self.matrix = self._parse_params(
                node_name, node_params
            )

            node = ProcessNode(name=node_name, **node_params)
            nodes[node_name] = node

        self.dag = Pipeline(nodes)
        self.dag.build_nx_graphs()

    def dispatch(self, backend='serial', skip_existing=True, **kwargs):
        self.define_kwdagger()

        kwdagger_params = {'pipeline': self.dag, 'matrix': self.matrix}

        kwd_config = ScheduleEvaluationConfig(
            params=kwdagger_params,  # includes pipeline and additional params
            root_dpath=self.root_dpath,
            backend=backend,
            skip_existing=skip_existing,
            run=True,
        )

        dag, queue = build_schedule(kwd_config)

    def collect_symbols(self):
        """
        Collect results (Evaluation Card 'symbols') in place of 'load_result' in the ProcessNode definition
        """
        if not self.symbols:
            self.dispatch()

        # Glob all results json (only one node in pipeline)
        paths = self.root_dpath.glob(
            f'**/{self.dag.nodes[next(iter(self.dag.nodes))].out_paths["results_fpath"]}'
        )

        for symbol_resolution in paths:
            symbols = json.load(open(symbol_resolution, 'r'))
            parent_dir = symbol_resolution.parent.stem
            if 'result' in symbols:
                # assume all fields exist
                for symbol in symbols['result']:
                    # record all sweeps
                    if parent_dir not in self.symbols:
                        self.symbols[parent_dir] = {}

                    self.symbols[parent_dir][symbol] = {
                        'value': symbols['result'][symbol]
                    }

        return self.symbols

    def _parse_params(self, node_name, node_cfg):
        """
        Parse sweepable parameters from definition
        """
        matrix = {}
        for k in node_cfg:
            if isinstance(node_cfg[k], dict) and '_params' in k:
                # TODO: Construct a more robust validator
                for param, v in node_cfg[k].items():
                    matrix[f'{node_name}.{param}'] = v
                # decompose yaml
                node_cfg[k] = list(node_cfg[k].keys())
        return node_cfg, matrix


class KWDaggerProcessor:
    """
    Handler for full kwdagger pipeline specification

    Example
        >>> from magnet.evaluation import KWDaggerProcessor
        >>> from kwdagger.schedule import ScheduleEvaluationConfig, build_schedule
        >>> import kwutil
        >>> # Example snippet of an Evaluation Card (related to GenericPipelineProcessor example)
        >>> example_cfg = kwutil.Yaml.coerce(
            '''
            kwdagger:
              pipeline: magnet.examples.llama_consistency.pipelines.llama_pipeline()
              matrix:
                llama_predict.base_model: ["meta/llama-2-13b", "meta/llama-2-70b"]
                llama_predict.comp_model:  ["meta/llama-2-7b", "meta/llama-3-70b"]
            ''')
        >>> root_dpath = "."
        >>> kwdagger_def = example_cfg['kwdagger']
        >>> pipeline = KWDaggerProcessor(kwdagger_def, root_dpath)
        >>> #
        >>> # Construct Two Node Pipeline (llama_predict -> claim)
        >>> kwdagger_spec = ScheduleEvaluationConfig(params=pipeline.spec, run=False)
        >>> dag, queue = build_schedule(kwdagger_spec)
        ...
        >>> dag.print_graphs()

        Process Graph
        ╙── llama_predict
            ╽
            claim_eval

        IO Graph
        ╙── llama_predict
            ╽
            results_fpath
            ╽
            symbols_fpath
            ╽
            claim_eval
            ╽
            verdict_fpath

        >>> #
        >>> # Parameters matrix
        >>> pipeline.spec['matrix']
        {'llama_predict.base_model': ['meta/llama-2-13b', 'meta/llama-2-70b'],
        'llama_predict.comp_model': ['meta/llama-2-7b', 'meta/llama-3-70b']}
    """

    def __init__(self, pipeline_def, root_dpath):
        self.spec = pipeline_def
        self.root_dpath = root_dpath
        self.results = []

    def dispatch(self, backend='serial', skip_existing=True, **kwargs):
        kwd_config = ScheduleEvaluationConfig(
            params=self.spec,  # includes pipeline and additional params
            root_dpath=self.root_dpath,
            backend=backend,
            skip_existing=skip_existing,
            run=True,
            **kwargs,
        )

        self.dag, queue = build_schedule(kwd_config)

    def collect_results(self):
        if not self.results:
            self.dispatch()

        # Glob all Claim node json files recursively
        paths = self.root_dpath.glob('**/verdict.json')

        # Assumes {result: {status: value}} output format
        for claim_json in paths:
            claim_result = json.load(open(claim_json, 'r'))
            if 'result' in claim_result and 'status' in claim_result['result']:
                self.results.append(claim_result['result']['status'])

        return self.results


class EvaluationTask:
    """
    Singular submission from an Evaluation Card
    """

    def __init__(self, claim, symbols):
        self.claim = claim
        self.symbols = symbols

    def execute(self) -> Tuple[str, str]:
        self.symbols.resolve()
        # x -> y -> z1 -> a1 -> res1
        #           ...
        #           zn -> an -> resn
        # make sure x,y are done once / before sweep
        return self.claim.evaluate(self.symbols())

    def record_run(self):
        # Could log requests from here (i.e. timestamps), I think this was done in other code segments
        # timestamp, symbol value, claim result
        raise NotImplementedError


class Claim:
    """
    Represents a verifiable assertion for a set of resolved symbols

    ***
    Currently assumes
    1. claim is valid and safe python code
    2. all symbols can be resolved from card
    3. No additional dependencies are needed
    4. Any conclusions drawn are as reliable as claim itself (i.e. verification is strictly: 'does code execute without error')
    ***

    Example:
        >>> from magnet.evaluation import Claim
        >>> self = Claim({'python': "assert x + 2 == 4"})
        >>> print(self)
        assert x + 2 == 4
        >>> self.evaluate({'x': 2})
        >>> print(self.status)
        VERIFIED
    """

    def __init__(self, raw):
        self.claim = raw.get('python')
        self.status = 'UNVERIFIED'

    def evaluate(self, symbols: Dict[str, Any] = {}):
        """
        Execute the claim subject to symbols definitions

        if True:
            VERIFIED
        elif AssertionError:
            FALSIFIED
        else:
            INCONCLUSIVE
        """
        try:
            out_msg = ''
            exec(self.claim, symbols)
            self.status = 'VERIFIED'
        except AssertionError as e:
            self.status = 'FALSIFIED'
            out_msg = f'Assertion does not hold: {e}'
        except NameError as e:
            self.status = 'INCONCLUSIVE'
            # This doesn't guarantee the missing variable is a symbol
            out_msg = f'SymbolNotResolved: {e}'
        except Exception as e:
            self.status = 'INCONCLUSIVE'
            out_msg = f'ERROR evaluating claim: {e}'
        finally:
            if out_msg:
                print(out_msg)
            return self.status, out_msg

    def __repr__(self) -> str:
        return self.claim


class Symbol:
    """
    Single resolvable unit of a claim

    Example:
        >>> from magnet.evaluation import Symbol
        >>> x = Symbol('x', {'type': "List[int]", 'python': "x = [10]"})
        >>> x.eval()
        [10]
    """

    def __init__(self, name, spec):
        self.name = name
        self.value = spec.get('value')
        self.sweep = spec.get('sweep')
        self.type = spec.get('type', 'List[int]')
        self.definition = spec.get('python', '')
        self.dependencies = spec.get('depends_on', [])

    def eval(self, context: Dict[str, Any] = {}) -> Any:
        """
        Resolve symbol definition

        FIXME: type verification is currently limited and hacky
        """
        if self.value is None:
            print(f'Resolving: {self.name}')
            exec(self.definition, context)
            if self._check_type(self.type, context[self.name]):
                self.value = context[self.name]
            else:
                raise TypeError(
                    f'{self.name}: {context[self.name]} is not {self.type}'
                )

        return self.value

    def _check_type(self, type_str, value) -> bool:
        """
        Validate value is of type str_type
        """
        # TODO: static 'vocabulary' of allowable types / support more than List[Any], Dict[str, Any]
        str_to_type = {'List': List, 'Dict': Dict, 'Tuple': Tuple, 'Any': Any}
        type = eval(type_str, str_to_type)
        return self._check_collections(type, value)

    def _check_collections(self, target_type, value):
        """
        Recursively evaluate if value is target_type
        """
        collection_type = get_origin(target_type)
        members = get_args(target_type)

        match collection_type:
            case builtins.list:
                if isinstance(value, list):
                    return all(
                        self._check_collections(members[0], entry)
                        for entry in value
                    )
            case builtins.dict:
                if isinstance(value, dict):
                    return all(
                        self._check_collections(members[0], key_entry)
                        and self._check_collections(members[1], value_entry)
                        for key_entry, value_entry in value.items()
                    )
            case builtins.tuple:
                if isinstance(value, tuple) and len(value) == len(members):
                    return all(
                        self._check_collections(type, val)
                        for type, val in zip(members, value)
                    )
            case None:
                # Any or primative
                return target_type is Any or isinstance(value, target_type)
            case _:
                return False


class Symbols:
    """
    Collection of Symbol configurations used as context for claim

    Example:
        >>> from magnet.evaluation import Symbols
        >>> symbols = Symbols({'x': {'type': "List[int]", 'python': "x = [10]"}})
        >>> symbols()
        {'x': None}
        >>> symbols.resolve()
        >>> symbols()
        {'x': [10]}
    """

    def __init__(self, symbol_specs) -> None:
        self.symbols = {
            symbol: Symbol(symbol, definition)
            for symbol, definition in symbol_specs.items()
        }

    @classmethod
    def decompose_symbol_defs(cls, symbol_definitions) -> List[Self]:
        """
        Flatten sweep values into a list of resolvable Symbols
        """
        configurations = []
        aggregate_configuration = cls(symbol_definitions)

        sweep_symbols = aggregate_configuration._find_sweep_symbols()
        if sweep_symbols:
            sweep_values = [sweep.sweep for sweep in sweep_symbols]
            combinations = product(*sweep_values)

            for combo in combinations:
                sweep_fill = dict(
                    zip([symbol.name for symbol in sweep_symbols], combo)
                )
                flattened_symbols = cls(symbol_definitions)
                for k, v in sweep_fill.items():
                    flattened_symbols.symbols[k].value = v
                configurations.append(flattened_symbols)
        else:
            configurations.append(aggregate_configuration)

        return configurations

    def resolve(self):
        """
        Trace dependency graph to resolve each symbol definition

        Values stored in Symbol instances
        """
        symbol_definitions = {}

        for symbol in self._construct_dependency_order():
            symbol_value = self.symbols[symbol]
            symbol_definitions_ = symbol_definitions.copy()
            try:
                symbol_definitions[symbol] = symbol_value.eval(
                    symbol_definitions_
                )
            except Exception as ex:
                error_message = ub.codeblock(
                    f"""
                    Error in resolve. ex={ex}

                    {symbol=!r}
                    {symbol_value=!r}
                    {symbol_definitions_=!r}
                    """
                )
                logger.error(error_message)
                raise

    def _find_sweep_symbols(self) -> List[Symbol]:
        return [symbol for symbol in self.symbols.values() if symbol.sweep]

    def _construct_dependency_order(self) -> List[Symbol]:
        """
        Construct dependency order
        """
        dependency_graph = {
            name: symbol.dependencies for name, symbol in self.symbols.items()
        }
        sorter = TopologicalSorter(dependency_graph)
        return list(sorter.static_order())

    def __call__(self):
        return {symbol: self.symbols[symbol].value for symbol in self.symbols}


def main(argv=None, **kwargs):
    args = EvaluationConfig.cli(
        argv=argv,
        data=kwargs,
        strict=True,
        verbose='auto',
        special_options=False,
    )

    card = EvaluationCard(args.path, args.results_path)
    if args.override is not None:
        card.replace(args.override)

    card.evaluate()
    card.summarize()


__cli__ = EvaluationConfig
__cli__.main = main

if __name__ == '__main__':
    main(sys.argv[1:])

'''
kwutil.Yaml.coerce("threshold: 2\nhelms_run_path: [1,2,3,4]")
Out[103]: {'threshold': 2, 'helms_run_path': [1, 2, 3, 4]}
In [92]: kwutil.Yaml.coerce("helm_runs_path: './data'\nthreshold: 2")
Out[92]: {'helm_runs_path': './data', 'threshold': 2}
In [86]: kwutil.Yaml.coerce("{helm_runs_path: './data', threshold: 2}")
Out[86]: {'helm_runs_path': './data', 'threshold': 2}

In [87]: kwutil.Yaml.coerce("{'helm_runs_path': './data', 'threshold': 2}")
Out[87]: {'helm_runs_path': './data', 'threshold': 2}

In [88]: kwutil.Yaml.coerce("'helm_runs_path': './data'\n'threshold': 2")
Out[88]: {'helm_runs_path': './data', 'threshold': 2}

In [89]: kwutil.Yaml.coerce("helm_runs_path: './data'\nthreshold: 2")
Out[89]: {'helm_runs_path': './data', 'threshold': 2}

In [90]: kwutil.Yaml.coerce("helm_runs_path:  './data'\nthreshold:  2")
Out[90]: {'helm_runs_path': './data', 'threshold': 2}

"helm_runs_path:  './data'\nthreshold:  2" '''