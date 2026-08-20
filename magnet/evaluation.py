import builtins
import json
import os
import sys
from collections.abc import Callable
from datetime import datetime
from graphlib import TopologicalSorter
from itertools import product
from statistics import fmean
from typing import Any, Dict, List, Optional, Self, Tuple, get_args, get_origin

import kwutil
import safer
import kwconf
import ubelt as ub
import yaml
from kwdagger import Pipeline, ProcessNode
from kwdagger.schedule import ScheduleEvaluationConfig, build_schedule
from loguru import logger
from pydantic import ValidationError
from rich import print

from magnet.utils.util_logger import setup_logging
from magnet.schema import EvaluationCardSchema, MetricObjective

SAFER_USE_TEMPFILE = not ub.WIN32

DEFAULT_CLAIM_AGGREGATION_STRATEGY = {'type': 'all'}
DEFAULT_METRIC_AGGREGATION_STRATEGY = {'type': 'mean'}



def resolve_queue_backend(requested: str | None = None) -> str:
    """
    Choose the cmd_queue backend a card's DAG is scheduled onto.

    Args:
        requested (str | None): an explicit choice. ``None`` reads
            ``MAGNET_QUEUE_BACKEND``, then falls back to ``'tmux'``.

    Returns:
        str: a backend cmd_queue reports as available.

    Defaults to ``tmux`` even at size 1: the same jobs run in the same order as
    ``serial``, but with a live monitor and a separate log per job rather than
    one interleaved stream. ``serial`` is right for CI and pytest, so an
    unavailable backend degrades to it with a notice rather than raising.

    Example:
        >>> from magnet.evaluation import resolve_queue_backend
        >>> resolve_queue_backend('serial')
        'serial'
    """
    import cmd_queue

    if requested is None:
        requested = os.environ.get('MAGNET_QUEUE_BACKEND') or 'tmux'
    requested = requested.strip()

    try:
        available = set(cmd_queue.Queue.available_backends())
    except Exception:
        return requested

    if requested in available:
        return requested
    if requested != 'serial':
        logger.warning(
            f'queue backend {requested!r} is not available '
            f'(have: {sorted(available)}); falling back to serial. '
            'Install tmux for a live monitor and per-job logs.'
        )
    return 'serial'


def _queue_name_for(root_dpath) -> str:
    """A tmux queue name that says which run these sessions belong to.

    cmd_queue's tmux backend matches sessions on the queue name to decide which
    ones are "this queue". kwdagger's default cannot help: MAGNET passes the
    pipeline as a DAG object inside ``params``, not as the ``pipeline`` spec
    string kwdagger names itself from, and a Pipeline has no name of its own.
    Every card fell back to the literal 'schedule-eval' and shared one
    namespace, so starting an Incubilate run reported a Princeton run's
    sessions as conflicts.

    ``root_dpath`` is ``<run>/evaluation_runs/<hash>/kwdagger``, so the run
    directory names the queue. Two runs of the same card still share a name,
    which is a real conflict; different cards do not.
    """
    import re

    try:
        parts = ub.Path(root_dpath).absolute().parts
        idx = len(parts) - 1 - parts[::-1].index('evaluation_runs')
        name = parts[idx - 1]
    except (ValueError, IndexError, TypeError):
        return 'schedule-eval'
    name = re.sub(r'[^A-Za-z0-9_.-]', '_', str(name))
    return f'schedule-{name}' if name else 'schedule-eval'


class EvaluationConfig(kwconf.Config):
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

    path: str = kwconf.Value(
        None, required=True, position=1, help='Path to evaluation card YAML'
    )

    output_path: str = kwconf.Value(
        './evaluation_runs', help='Root data path for saved results'
    )

    override: str | None = kwconf.Value(
        None,
        parser=str,
        help='Override symbol values (e.g. --override dataset: legalbench\nnum_replicates: 5)',
    )

    jobs: int = kwconf.Value(
        1,
        parser=int,
        help=(
            'Number of evaluation jobs. Use 1 for serial execution, '
            '-1 for all available CPUs when using joblib.'
        ),
    )

    parallel_backend: str = kwconf.Value(
        'loky',
        parser=str,
        choices=['loky', 'threading', 'multiprocessing'],
        help='Joblib backend used when --jobs is not 1.',
    )

    queue_backend: str | None = kwconf.Value(
        None,
        parser=str,
        help=(
            'cmd_queue backend the card DAG is scheduled onto: tmux, serial, '
            'slurm, airflow. Defaults to $MAGNET_QUEUE_BACKEND, then tmux, '
            'falling back to serial when unavailable. Prefer tmux even at '
            'size 1: same jobs in the same order, but with a live monitor and '
            'a separate log per job instead of one interleaved stream.'
        ),
    )

    verbose: bool = kwconf.Value(
        False, isflag=True, help='Verbose log output', group='logging'
    )

    validate: str = kwconf.Value(
        'error',
        parser=str,
        choices=['only', 'error', 'warning', 'off'],
        help=(
            "'only': validate schema and exit. "
            "'error': validate and raise on failure (default). "
            "'warning': validate, warn on failure, and proceed. "
            "'off': skip validation entirely."
        ),
    )


# Claim Resolution (pulled out as standalone function for
# multiprocessing support)
def _run_one(
    evaluation: 'EvaluationTask', claim_results_path: ub.Path
) -> Tuple[str, ub.Path]:
    status, _ = evaluation.execute()
    results_fpath = (
        claim_results_path / evaluation._execution_hash / 'verdict.json'
    )
    results_fpath.parent.ensuredir()

    with safer.open(results_fpath, 'w', temp_file=SAFER_USE_TEMPFILE) as f:
        json.dump(evaluation.log, f, indent=2, ensure_ascii=False)
        f.write('\n')

    return status, results_fpath


class EvaluationCard:
    """
    Specification of an empirical claim with resolvable symbols and metadata

    Example:
        >>> from importlib.resources import files
        >>> from magnet.evaluation import EvaluationCard
        >>> card_name = 'simple.yaml'
        >>> card_path = files('magnet') / 'cards' / card_name
        >>> output_path = './results'
        >>> card = EvaluationCard(card_path, output_path)
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
        model: ['claude-3.5-sonnet', 'gemini-1.5-pro-001']
    """

    def __init__(
        self, path, output_path: str | os.PathLike[str], validate='error'
    ):
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        if validate in ('error', 'warning'):
            try:
                EvaluationCardSchema.model_validate(cfg)
            except ValidationError as e:
                if validate == 'error':
                    raise e
                logger.warning(
                    f'WARNING! Card validation failed with error:\n{e}'
                )

        self.original_card = cfg
        self.output_path = ub.Path(output_path)

        self.title = cfg.get('title', '')
        self.description = cfg.get('description', '')

        self.claim = Claim(cfg.get('claim'))
        self.claim_aggregation_strategy = cfg.get(
            'claim_aggregation_strategy', DEFAULT_CLAIM_AGGREGATION_STRATEGY
        )
        self.symbols = cfg.get('symbols', {})

        # explicit kwdagger spec
        self.has_kwdagger = 'kwdagger' in cfg
        self.kwdagger = cfg.get('kwdagger')
        if self.has_kwdagger:
            self.kwdagger = _resolve_pipeline_path(
                self.kwdagger, self.card_dpath)

        # populate ProcessNode(s) programmatically
        self.has_pipeline = 'pipeline' in cfg
        self.pipeline = cfg.get('pipeline')

        self.evaluations = []
        self._run_hash_cached: str | None = None

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

    def replace(self, override_str: str) -> None:
        """
        Handle overrides in symbol field by replacing 'value' entries and appending to sweeps
        """
        override = _plain_data(kwutil.Yaml.coerce(override_str))

        for key, value in override.items():
            if key not in self.symbols:
                raise ValueError(
                    f"Unknown symbol '{key}' -- available: {list(self.symbols.keys())}"
                )
            if 'value' in self.symbols[key]:
                # replacement
                self.symbols[key]['value'] = value
            elif 'sweep' in self.symbols[key]:
                if isinstance(value, list):
                    self.symbols[key]['sweep'] = value
                else:
                    self.symbols[key]['sweep'] = [value]

    def evaluate(
        self,
        jobs: int = 1,
        parallel_backend: str = 'loky',
        verbose: bool = False,
    ) -> str:
        """
        Run the evaluation specification

        1. Resolve symbol definitions
        2. Evaluate claim under symbol values
        3. Write out results
        4. Summarize general finding

        Assumes user provides input path up to (*)
        e.g.
        ├── Milestone
        │   ├── Organization
        │   │   └── Consistency_Algorithm (*)
        │   │       └── ab0012cf_2026-04-21__12-23-34
        │   │           ├── card.yaml
        │   │           └── kwdagger
        │   │           └── results
        """
        results = []

        card_output_path = self.output_path / self._run_hash
        card_output_path.ensuredir()

        setup_logging(verbose, card_output_path)

        with safer.open(
            card_output_path / 'card.yaml', 'w', temp_file=SAFER_USE_TEMPFILE
        ) as f:
            yaml.safe_dump(self.original_card, f, sort_keys=False)

        claim_results_path = card_output_path / 'results'

        raw_symbol_metadata = _parse_symbol_metadata(self.symbols)

        if raw_symbol_metadata:
            with safer.open(
                card_output_path / 'symbol_metadata.json',
                'w',
                temp_file=SAFER_USE_TEMPFILE,
            ) as f:
                json.dump(raw_symbol_metadata, f, indent=2, ensure_ascii=False)

        if self.has_kwdagger:
            processor = KWDaggerProcessor(
                self.kwdagger, root_dpath=card_output_path / 'kwdagger'
            )

            if processor.terminal_node:
                # The pipeline declares a single terminal artifact, so the
                # DAG is authoritative: bind its payload as symbols and
                # evaluate the claim exactly once against what the pipeline
                # computed.  No globbing, no re-running the claim per
                # rediscovered result.
                terminal = processor.collect_terminal_result()
                terminal_symbols = {
                    name: {'value': value}
                    for name, value in terminal.items()
                    if not name.startswith('_')
                }
                self.symbols.update(terminal_symbols)
                self.evaluations = self.dispatch(
                    Symbols.decompose_symbol_defs(self.symbols)
                )

                with safer.open(
                    card_output_path / 'terminal_result.json',
                    'w',
                    temp_file=SAFER_USE_TEMPFILE,
                ) as f:
                    json.dump(terminal, f, indent=2, ensure_ascii=False)
                    f.write('\n')
            else:
                # Legacy path: no declared terminal node, so results are
                # rediscovered from the run tree and the claim is replayed
                # for each one.
                kwdagger_results, symbols = processor.collect_results()

                for sweep in symbols:
                    symbol_with_value = {
                        s: {'value': v} for s, v in sweep.items()
                    }
                    self.symbols.update(symbol_with_value)
                    self.evaluations.extend(
                        self.dispatch(
                            Symbols.decompose_symbol_defs(self.symbols)
                        )
                    )

        elif self.has_pipeline:
            # Implicit pipeline definition needs parsing
            pipeline_runs = GenericPipelineProcessor(
                self.pipeline, root_dpath=card_output_path / 'kwdagger'
            ).collect_symbols()

            for run in pipeline_runs:
                run_symbols = pipeline_runs[run]
                self.symbols.update(run_symbols)
                self.evaluations.extend(
                    self.dispatch(Symbols.decompose_symbol_defs(self.symbols))
                )

        else:
            # Serial Evaluation Card
            self.evaluations = self.dispatch(
                Symbols.decompose_symbol_defs(self.symbols)
            )

        if jobs == 1:
            out = [_run_one(e, claim_results_path) for e in self.evaluations]
        else:
            from joblib import Parallel, delayed

            out = Parallel(n_jobs=jobs, backend=parallel_backend, verbose=5)(
                delayed(_run_one)(e, claim_results_path)
                for e in self.evaluations
            )

        results = []
        for status, results_fpath in out:
            results.append(status)
            logger.info(f'Wrote claim output to {results_fpath}')

        calculated_metrics = {}

        if raw_symbol_metadata:
            metric_definitions = Metric.build_metrics_from_symbol_metadata(
                raw_symbol_metadata
            )
            calculated_metrics = _calculate_metrics(
                metric_definitions,
                self.evaluations,
                raw_symbol_metadata,
            )

            if calculated_metrics:
                metric_statement = (
                    '================================\n Evaluation Metrics:\n'
                )
                for metric, value in calculated_metrics.items():
                    metric_statement += f'  {metric}: {value: .3f}\n'
                logger.info(metric_statement[:-1])

        total = len(results)

        def percentage(count):
            return count / total

        verified_count = results.count('VERIFIED')
        falsified_count = results.count('FALSIFIED')
        inconclusive_count = results.count('INCONCLUSIVE')

        logger.info('================================')
        logger.info(f'Settings Evaluated: {total}')
        logger.info(f'  Verified:     {percentage(verified_count):.2f}')
        logger.info(f'  Falsified:    {percentage(falsified_count):.2f}')
        logger.info(f'  Inconclusive: {percentage(inconclusive_count):.2f}')
        logger.info('================================')
        logger.info('\n')

        card_result = _reduce_results(results, self.claim_aggregation_strategy)
        aggregate_verdict = {
            'result': card_result,
            'claim_aggregation_strategy': self.claim_aggregation_strategy,
            'claims': [e._execution_hash for e in self.evaluations],
        }

        if raw_symbol_metadata and calculated_metrics:
            aggregate_verdict['metrics'] = calculated_metrics

        with safer.open(
            card_output_path / 'verdict.json', 'w', temp_file=SAFER_USE_TEMPFILE
        ) as f:
            json.dump(aggregate_verdict, f, indent=2, ensure_ascii=False)
            f.write('\n')

        self.claim.status = card_result
        return card_result

    def dispatch(
        self, flattened_sweep: List['Symbols']
    ) -> List['EvaluationTask']:
        return [
            EvaluationTask(Claim({'python': self.claim.claim}), symbols)
            for symbols in flattened_sweep
        ]

    def summarize(self) -> None:
        """
        Human-readable summary of card in its current state
        """
        logger.info(f'[bold]Title:[/bold]       {self.title}')
        logger.info(f'[bold]Description:[/bold] {self.description}')
        logger.info('================================')
        # logger.info(f"SYMBOLS:     {self.symbols()}")
        logger.info(f'[bold]CLAIM:[/bold]       \n{self.claim}')

        status = self.status()
        if self.claim.status == 'VERIFIED':
            claim_status_color = 'green'
        elif self.claim.status == 'FALSIFIED':
            claim_status_color = 'red'
        else:
            claim_status_color = 'yellow'

        if status == 'EVALUATED':
            logger.info('================================')
            logger.info(
                f'[bold]RESULT:[/bold]      [bold][{claim_status_color}]{self.claim.status}[/{claim_status_color}][/bold]'
                ''
            )

        logger.info('================================')
        logger.info(f'[bold]CARD STATUS:[/bold] {status}')

    @property
    def _card_hash(self) -> str:
        """The card's content; the same card is the same id on any day."""
        return ub.hash_data(self.original_card)[:8]

    @property
    def _run_hash(self) -> str:
        """
        This card's directory: ``<card id>_<when it first ran>``.

        Re-running an unchanged card returns the directory it already has
        rather than a new one stamped with the current second. The DAG's root
        is inside this directory, so a fresh name every run meant
        ``skip_existing`` arrived at an empty tree and refit everything.
        Editing the card changes the id, which starts a new directory and
        invalidates the cells.

        Computed once per instance; recomputing it on every read handed each
        reader a different timestamp.
        """
        if self._run_hash_cached is None:
            existing = [
                p for p in sorted(self.output_path.glob(f'{self._card_hash}_*'))
                if p.is_dir()
            ]
            if existing:
                newest = max(existing, key=lambda p: p.stat().st_mtime)
                self._run_hash_cached = newest.name
            else:
                timestamp = datetime.now().strftime('%Y-%m-%d__%H-%M-%S')
                self._run_hash_cached = f'{self._card_hash}_{timestamp}'
        return self._run_hash_cached


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
        >>>    print(getattr(pipeline.dag.node_dict['predict_node'], attr))
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

    def __init__(
        self, pipeline_def: Dict[str, Any], root_dpath: ub.Path
    ) -> None:
        self.pipeline = pipeline_def
        self.root_dpath = root_dpath
        self.dag = None
        self.matrix = None
        self.symbols = {}

    def define_kwdagger(self) -> None:
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

        self.dag = Pipeline(list(nodes.values()))
        self.dag.build_nx_graphs()

    def dispatch(
        self, backend: str | None = None, skip_existing: bool = True,
        **kwargs: Any
    ) -> None:
        self.define_kwdagger()
        backend = resolve_queue_backend(backend)

        kwdagger_params = {'pipeline': self.dag, 'matrix': self.matrix}

        kwd_config = ScheduleEvaluationConfig(
            params=kwdagger_params,  # includes pipeline and additional params
            root_dpath=self.root_dpath,
            queue_name=_queue_name_for(self.root_dpath),
            backend=backend,
            skip_existing=skip_existing,
            run=True,
        )

        dag, queue = build_schedule(kwd_config)

    def collect_symbols(self) -> Dict[str, Any]:
        """
        Collect results (Evaluation Card 'symbols') in place of 'load_result' in the ProcessNode definition
        """
        if not self.symbols:
            self.dispatch()

        # Glob all results json (only one node in pipeline)
        paths = self.root_dpath.glob(
            f'**/{self.dag.node_dict[next(iter(self.dag.node_dict))].out_paths["results_fpath"]}'
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

    def _parse_params(
        self, node_name: str, node_cfg: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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

    def __init__(
        self, pipeline_def: Dict[str, Any], root_dpath: ub.Path
    ) -> None:
        # ``terminal_node`` is a MAGNET-level declaration, not something
        # kwdagger understands, so keep it out of the scheduled spec.
        self.spec = {
            k: v for k, v in pipeline_def.items() if k != 'terminal_node'
        }
        self.terminal_node = pipeline_def.get('terminal_node')
        self.root_dpath = root_dpath
        self.results = []
        self.symbols = []

    def dispatch(
        self, backend: str | None = None, skip_existing: bool = True,
        **kwargs: Any
    ) -> None:
        backend = resolve_queue_backend(backend)
        kwd_config = ScheduleEvaluationConfig(
            params=self.spec,  # includes pipeline and additional params
            root_dpath=self.root_dpath,
            queue_name=_queue_name_for(self.root_dpath),
            backend=backend,
            skip_existing=skip_existing,
            run=True,
            **kwargs,
        )

        self.dag, queue = build_schedule(kwd_config)

    def collect_terminal_result(self) -> Dict[str, Any]:
        """
        Load the declared terminal node's primary output.

        A pipeline that declares ``terminal_node`` is stating that one node
        produces the card's whole result.  Reading that artifact directly
        keeps the DAG authoritative -- MAGNET does not have to rediscover
        outputs by globbing the run tree, and the claim is evaluated once,
        against what the pipeline actually computed.

        Returns:
            Dict[str, Any]: the parsed terminal artifact.

        Raises:
            ValueError: if no ``terminal_node`` was declared, or it does not
                name a node in the pipeline.
            RuntimeError: if the pipeline produced no terminal artifact, or
                produced more than one (which means the terminal node fanned
                out and is therefore not terminal).
        """
        if not self.terminal_node:
            raise ValueError(
                'collect_terminal_result requires the card to declare '
                "kwdagger.terminal_node"
            )

        if not getattr(self, 'dag', None):
            self.dispatch()

        # build_schedule returns *configured* instances keyed by process id
        # (``card_summary_id_wyaepu4x46tf``); ``node.name`` is the template
        # name the card refers to.
        instances = [
            node
            for node in self.dag.nodes.values()
            if node.name == self.terminal_node
        ]

        if not instances:
            available = sorted({node.name for node in self.dag.nodes.values()})
            raise ValueError(
                f'terminal_node {self.terminal_node!r} is not a node in the '
                f'pipeline; available: {available}'
            )
        if len(instances) > 1:
            # More than one configured instance means the node is still
            # parameterized by something swept, so it summarizes a slice
            # rather than the card.  Catching it here -- before reading any
            # artifact -- is better than silently reporting a partial
            # result as the whole finding.
            raise RuntimeError(
                f'terminal_node {self.terminal_node!r} has '
                f'{len(instances)} configured instances, so it is not '
                f'terminal.  Gather its inputs (or drop its swept '
                f'parameters) so exactly one instance remains.'
            )

        node = instances[0]
        out_fname = node.out_paths[node.primary_out_key]

        node_dpath = self.root_dpath / self.terminal_node
        found = sorted(node_dpath.glob(f'*/{out_fname}'))

        if not found:
            raise RuntimeError(
                f'terminal node {self.terminal_node!r} produced no '
                f'{out_fname} under {node_dpath}.  The pipeline likely '
                f'failed upstream; inspect the kwdagger run directory.'
            )
        if len(found) > 1:
            raise RuntimeError(
                f'terminal node {self.terminal_node!r} produced '
                f'{len(found)} artifacts under {node_dpath}, but the DAG '
                f'configured only one instance.  Stale output from an '
                f'earlier run with different parameters is the usual '
                f'cause. Found: {[str(p) for p in found]}'
            )

        with open(found[0], 'r') as file:
            payload = json.load(file)

        payload.setdefault('_terminal_artifact_fpath', str(found[0]))
        return payload

    def collect_results(self) -> Tuple[List[str], List[Any]]:
        if not self.results:
            self.dispatch()

        # Glob all Claim node json files recursively
        paths = self.root_dpath.glob('**/verdict.json')

        # Assumes {result: {status: value}} output format
        for claim_json in paths:
            claim_result = json.load(open(claim_json, 'r'))
            if 'result' in claim_result and 'status' in claim_result['result']:
                self.results.append(claim_result['result']['status'])
            if 'result' in claim_result and 'symbols' in claim_result['result']:
                self.symbols.append(claim_result['result']['symbols'])

        return self.results, self.symbols


class EvaluationTask:
    """
    Singular submission from an Evaluation Card
    """

    def __init__(self, claim: 'Claim', symbols: 'Symbols') -> None:
        self.claim = claim
        self.symbols = symbols
        self.output_msg = ''
        self.log = ''

    def execute(self) -> Tuple[str, str]:
        self.symbols.resolve()
        # x -> y -> z1 -> a1 -> res1
        #           ...
        #           zn -> an -> resn
        # make sure x,y are done once / before sweep
        self.result, self.output_msg = self.claim.evaluate(self.symbols())
        self.record_run()
        return self.result, self.output_msg

    def record_run(self) -> None:
        completion_time = datetime.now().isoformat()
        self.log = {
            'status': self.result,
            'output': self.output_msg,
            'symbols': self.symbols.simple_view(),
            'timestamp': completion_time,
        }

    @property
    def _execution_hash(self) -> str:
        return ub.hash_data(self.symbols.simple_view())[:12]


def _resolve_pipeline_path(
    kwdagger_spec: Dict[str, Any], card_dpath: ub.Path
) -> Dict[str, Any]:
    """
    Make a relative pipeline file path mean the same thing from any directory.

    A card may name a pipeline file rather than inline the DAG or name a Python
    callable. Such a path resolves against the card's directory, matching how
    the theory block's formalization paths already work.

    Args:
        kwdagger_spec (Dict[str, Any]): the card's ``kwdagger`` block.
        card_dpath (ub.Path): the directory holding the card.

    Returns:
        Dict[str, Any]: the spec, with any relative pipeline path made absolute.

    Example:
        >>> import ubelt as ub
        >>> from magnet.evaluation import _resolve_pipeline_path
        >>> spec = {'pipeline': 'module.func()'}
        >>> _resolve_pipeline_path(spec, ub.Path('/cards'))['pipeline']
        'module.func()'
        >>> spec = {'pipeline': {'nodes': {}}}
        >>> _resolve_pipeline_path(spec, ub.Path('/cards'))['pipeline']
        {'nodes': {}}
        >>> spec = {'pipeline': 'dag.yaml'}
        >>> _resolve_pipeline_path(spec, ub.Path('/cards'))['pipeline']
        '/cards/dag.yaml'
        >>> spec = {'pipeline': '/abs/dag.yaml'}
        >>> _resolve_pipeline_path(spec, ub.Path('/cards'))['pipeline']
        '/abs/dag.yaml'
    """
    pipeline = kwdagger_spec.get('pipeline')
    if not isinstance(pipeline, str):
        return kwdagger_spec
    if '::' in pipeline:
        return kwdagger_spec
    if pipeline.rsplit('.', 1)[-1].lower() not in {'yaml', 'yml', 'json'}:
        return kwdagger_spec

    path = ub.Path(pipeline)
    if not path.is_absolute():
        path = card_dpath / path

    resolved = dict(kwdagger_spec)
    resolved['pipeline'] = os.fspath(path)
    return resolved


def _plain_data(data: Any) -> Any:
    """
    Rebuild YAML data out of plain dicts, lists and scalars.

    The loader returns round-trip types that carry formatting. Those reach
    ``original_card`` through an override and then fail in ``yaml.safe_dump``
    when the run directory's copy of the card is written, with
    ``RepresenterError: cannot represent an object``. Any list-valued override
    hit this, so ``--override 'seed: [1, 2]'`` could not run at all.

    Example:
        >>> import kwutil
        >>> from magnet.evaluation import _plain_data
        >>> data = _plain_data(kwutil.Yaml.coerce('seed: [1, 2]'))
        >>> type(data).__name__, type(data['seed']).__name__
        ('dict', 'list')
        >>> import yaml
        >>> yaml.safe_dump(data)
        'seed:\\n- 1\\n- 2\\n'
        >>> quoted = kwutil.Yaml.coerce("cfg: ['a:b=c']")
        >>> yaml.safe_dump(_plain_data(quoted))
        'cfg:\\n- a:b=c\\n'
    """
    if isinstance(data, dict):
        return {_plain_data(key): _plain_data(value)
                for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [_plain_data(value) for value in data]
    # Scalars need this too: a quoted string loads as a str subclass that
    # remembers its quoting style, which safe_dump also refuses.
    if isinstance(data, str):
        return str(data)
    if isinstance(data, bool):
        return bool(data)
    if isinstance(data, int):
        return int(data)
    if isinstance(data, float):
        return float(data)
    return data


def _parse_symbol_metadata(symbols_spec: Dict[str, Any]) -> Dict[str, Any]:
    metadata = {}
    for name, details in symbols_spec.items():
        symbol_metadata = details.get('metadata')
        if symbol_metadata is not None:
            metadata[name] = symbol_metadata
    return metadata


def _reduce_results(results: List[str], reduce_spec: Dict[str, Any]) -> str:
    """
    Reduce per-sweep-point claim outcomes to a single card-level status.

    reduce_spec: dict with key `type`:
      - {'type': 'all'}               any FALSIFIED -> FALSIFIED; any INCONCLUSIVE -> INCONCLUSIVE; else VERIFIED
      - {'type': 'any'}               any VERIFIED -> VERIFIED; any INCONCLUSIVE (and no VERIFIED) -> INCONCLUSIVE; else FALSIFIED
      - {'type': 'fraction', 'parameters': {'threshold': 0.8}}
                                      VERIFIED_count / total >= threshold -> VERIFIED; else FALSIFIED.
                                      INCONCLUSIVE points count in the denominator but not the numerator.
    """
    total = len(results)
    if total == 0:
        return 'INCONCLUSIVE'

    verified_count = results.count('VERIFIED')
    falsified_count = results.count('FALSIFIED')
    inconclusive_count = results.count('INCONCLUSIVE')

    rtype = reduce_spec.get('type', 'all')
    if rtype == 'all':
        if falsified_count:
            return 'FALSIFIED'
        if inconclusive_count:
            return 'INCONCLUSIVE'
        return 'VERIFIED'
    if rtype == 'any':
        if verified_count:
            return 'VERIFIED'
        if inconclusive_count:
            return 'INCONCLUSIVE'
        return 'FALSIFIED'
    if rtype == 'fraction':
        parameters = reduce_spec.get('parameters', {})
        threshold = parameters.get('threshold')
        if threshold is None:
            raise ValueError('reduce type=fraction requires `threshold`')
        frac = verified_count / total
        final_result = 'VERIFIED' if frac >= threshold else 'FALSIFIED'
        logger.info(
            f'[reduce=fraction] {final_result} {verified_count}/{total} ({frac:.3f}) vs threshold {threshold}'
        )
        return final_result

    raise ValueError(f'Unknown reduce type: {rtype!r}')


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

    def __init__(self, raw: Dict[str, str]) -> None:
        self.claim = raw.get('python', '')
        self.status = 'UNVERIFIED'

    def evaluate(self, symbols: Dict[str, Any] = {}) -> Tuple[str, str]:
        """
        Execute the claim subject to symbols definitions

        if True:
            VERIFIED
        elif AssertionError:
            FALSIFIED
        else:
            INCONCLUSIVE
        """
        out_msg = ''
        try:
            exec(self.claim, symbols)
            self.status = 'VERIFIED'
            out_msg = 'Assertion holds'
            logger.info(out_msg)
        except AssertionError as e:
            self.status = 'FALSIFIED'
            out_msg = f'Assertion does not hold: {e}'
            logger.warning(out_msg)
        except NameError as e:
            self.status = 'INCONCLUSIVE'
            # This doesn't guarantee the missing variable is a symbol
            out_msg = f'SymbolNotResolved: {e}'
            logger.error(out_msg)
        except Exception as e:
            self.status = 'INCONCLUSIVE'
            out_msg = f'ERROR evaluating claim: {e}'
            logger.exception('Unexpected exception while evaluating claim')

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

    def __init__(self, name: str, spec: Dict[str, Any]) -> None:
        self.name = name
        self.value = spec.get('value')
        self.sweep = spec.get('sweep')
        self.type = spec.get('type', 'List[int]')
        self.definition = spec.get('python', '')
        self.dependencies = spec.get('depends_on', [])
        self.metadata = spec.get('metadata')

    def eval(self, context: Dict[str, Any] = {}) -> Any:
        """
        Resolve symbol definition

        FIXME: type verification is currently limited and hacky
        """
        if self.value is None:
            logger.debug(f'Resolving: {self.name}')
            exec(self.definition, context)
            if self._check_type(self.type, context[self.name]):
                self.value = context[self.name]
            else:
                raise TypeError(
                    f'{self.name}: {context[self.name]} is not {self.type}'
                )

        return self.value

    def _check_type(self, type_str: str, value: Any) -> bool:
        """
        Validate value is of type str_type
        """
        # TODO: static 'vocabulary' of allowable types / support more than List[Any], Dict[str, Any]
        str_to_type = {'List': List, 'Dict': Dict, 'Tuple': Tuple, 'Any': Any}
        type = eval(type_str, str_to_type)
        return self._check_collections(type, value)

    def _check_collections(self, target_type: Any, value: Any) -> bool:
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
                return False
            case builtins.dict:
                if isinstance(value, dict):
                    return all(
                        self._check_collections(members[0], key_entry)
                        and self._check_collections(members[1], value_entry)
                        for key_entry, value_entry in value.items()
                    )
                return False
            case builtins.tuple:
                if isinstance(value, tuple) and len(value) == len(members):
                    return all(
                        self._check_collections(type, val)
                        for type, val in zip(members, value)
                    )
                return False
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

    def __init__(self, symbol_specs: Dict[str, Any]) -> None:
        self.symbols = {
            symbol: Symbol(symbol, definition)
            for symbol, definition in symbol_specs.items()
        }

    @classmethod
    def decompose_symbol_defs(
        cls, symbol_definitions: Dict[str, Any]
    ) -> List[Self]:
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

    def resolve(self) -> None:
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

    def _find_sweep_symbols(self) -> List['Symbol']:
        return [symbol for symbol in self.symbols.values() if symbol.sweep]

    def _construct_dependency_order(self) -> List[str]:
        """
        Construct dependency order
        """
        dependency_graph = {
            name: symbol.dependencies for name, symbol in self.symbols.items()
        }
        sorter = TopologicalSorter(dependency_graph)
        return list(sorter.static_order())

    def simple_view(self) -> Dict[str, Any]:
        # TODO: replace with free variables and data attestation
        ALLOWABLE_TYPES = [int, float, str, dict]
        return {
            k: v
            for k, v in self().items()
            if type(v) in ALLOWABLE_TYPES
            or (type(v) == list and type(v[0]) == int)
        }

    def __call__(self) -> Dict[str, Any]:
        return {symbol: self.symbols[symbol].value for symbol in self.symbols}


MetricValue = float
MetricReducer = Callable[[List[float]], MetricValue]


class Metric:
    def __init__(
        self,
        name: str,
        objective: MetricObjective,
        reducer: MetricReducer,
    ) -> None:
        self.name = name
        self.objective = objective
        self.reducer = reducer

    def aggregate_calculate(self, runs: List[float]) -> MetricValue:
        logger.info(f'Computing {self.name} Metric across all runs\n')
        return self.reducer(runs)

    @classmethod
    def build_metrics_from_symbol_metadata(
        cls, symbol_metadata: Dict[str, Any]
    ) -> List[Self]:
        metrics = []
        for name, metadata in symbol_metadata.items():
            metric_metadata = metadata.get('define_metric')
            if metric_metadata is not None:
                agg_strategy = metric_metadata.get(
                    'aggregation_strategy', DEFAULT_METRIC_AGGREGATION_STRATEGY
                )
                strategy_name = agg_strategy.get('type')
                parameters = agg_strategy.get('parameters') or {}
                objective = MetricObjective(
                    metric_metadata.get('objective', MetricObjective.MINIMIZE)
                )

                match strategy_name:
                    case 'max':
                        reducer = max
                    case 'min':
                        reducer = min
                    case 'mean':
                        reducer = fmean
                    case 'custom':
                        # Python function
                        raise NotImplementedError
                    case _:
                        logger.warning(
                            'Unrecognized Metric Aggregation Strategy; Please select one of {max, min, mean, custom}'
                        )
                        reducer = None

                if reducer is not None:
                    metrics.append(cls(name, objective, reducer))
        return metrics


def _calculate_metrics(
    metric_definitions: List[Metric],
    evaluations: List['EvaluationTask'],
    symbol_metadata: Dict[str, Any],
) -> Dict[str, MetricValue]:
    calculated_metrics = {}
    for metric in metric_definitions:
        runs = []
        for evaluation in evaluations:
            symbol_value = evaluation.symbols().get(metric.name)
            if symbol_value is None:
                logger.error(
                    f'Metric {metric.name} cannot be mapped to a Symbol value'
                )
                break
            runs.append(symbol_value)
        else:
            display_name = symbol_metadata[metric.name].get(
                'display_name', metric.name
            )
            calculated_metrics[display_name] = metric.aggregate_calculate(runs)
    return calculated_metrics


def main(argv: Optional[List[str]] = None, **kwargs: Any) -> None:
    args = EvaluationConfig.cli(
        argv=argv,
        data=kwargs,
        strict=True,
        verbose='auto',
        special_options=False,
    )

    # Item access, not `args.validate`: the option shares its name with
    # `kwconf.Config.validate`, and the method wins attribute lookup. Reading it
    # as an attribute yields a bound method, which silently compares unequal to
    # every mode and turns validation off. See tests/test_kwconf_configs.py.
    validate = args['validate']

    if validate == 'only':
        try:
            with open(args.path, 'r') as f:
                cfg = yaml.safe_load(f)
            EvaluationCardSchema.model_validate(cfg)
            print('Card validation succeeded.')
        except ValidationError as e:
            print('Card validation failed.')
            print(e)
            sys.exit(1)
        return

    card = EvaluationCard(args.path, args.output_path, validate=validate)
    if args.override is not None:
        card.replace(args.override)

    if args.queue_backend:
        # One source of truth: the resolver reads this, and so does any nested
        # dispatch. Threading a parameter through evaluate() would miss the
        # dispatch calls that run from collect_terminal_result().
        os.environ['MAGNET_QUEUE_BACKEND'] = args.queue_backend

    card.evaluate(
        jobs=args.jobs,
        parallel_backend=args.parallel_backend,
        verbose=bool(args.verbose),
    )
    card.summarize()


__cli__ = EvaluationConfig
__cli__.main = main

if __name__ == '__main__':
    main(sys.argv[1:])
