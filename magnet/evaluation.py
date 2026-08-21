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
from loguru import logger
from pydantic import ValidationError
from rich import print

from magnet.utils.util_logger import setup_logging
from magnet.schema import EvaluationCardSchema, MetricObjective
from magnet._kwdagger import (
    GenericPipelineProcessor,
    KWDaggerProcessor,
    _resolve_pipeline_path,
)

SAFER_USE_TEMPFILE = not ub.WIN32

DEFAULT_CLAIM_AGGREGATION_STRATEGY = {'type': 'all'}
DEFAULT_METRIC_AGGREGATION_STRATEGY = {'type': 'mean'}


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
) -> Tuple[str, ub.Path, str, Dict[str, Any]]:
    status, _ = evaluation.execute()
    execution_hash = evaluation.cell_id
    resolved_symbols = evaluation.log['symbols']
    results_fpath = claim_results_path / execution_hash / 'verdict.json'
    results_fpath.parent.ensuredir()

    with safer.open(results_fpath, 'w', temp_file=SAFER_USE_TEMPFILE) as f:
        json.dump(evaluation.log, f, indent=2, ensure_ascii=False)
        f.write('\n')

    return status, results_fpath, execution_hash, resolved_symbols


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
        # Paths written inside a card resolve against the card, not the
        # shell's cwd.
        self.card_dpath = ub.Path(path).parent

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
            if not self.kwdagger.get('result_node'):
                raise ValueError(
                    f'{path}: a kwdagger card must declare '
                    f'`kwdagger.result_node`, naming the node whose output is '
                    f'the card result'
                )
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

    @property
    def kwdagger_dpath(self) -> Any:
        """
        Where the DAG's node artifacts live: ``<output>/_kwdagger``.

        Shared across card versions, so a node keeps its artifacts when an
        unrelated part of the card changes. Per-run provenance stays under the
        card-hash directory.
        """
        # Underscore-prefixed so it is not mistaken for a run directory
        # (those are ``<card hash>_<timestamp>``); the collectors already skip
        # ``_``-prefixed names.
        return self.output_path / '_kwdagger'

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
                self.kwdagger, root_dpath=self.kwdagger_dpath
            )

            # The DAG is authoritative: each instance of the result node is
            # one cell of the card, evaluated against what it computed.
            cells = processor.collect_result_cells()

            for cell in cells:
                cell_symbols, measured = _fill_declared_symbols(
                    self.symbols, cell['results'])
                self.evaluations.extend(self.dispatch(
                    Symbols.decompose_symbol_defs(cell_symbols),
                    results=cell['results'],
                    cell_key=cell['key'],
                    measured=measured,
                ))

            with safer.open(
                card_output_path / 'result_cells.json',
                'w',
                temp_file=SAFER_USE_TEMPFILE,
            ) as f:
                json.dump(cells, f, indent=2, ensure_ascii=False)
                f.write('\n')

            _link_dag_root(card_output_path, self.kwdagger_dpath)

        elif self.has_pipeline:
            pipeline_runs = GenericPipelineProcessor(
                self.pipeline, root_dpath=self.kwdagger_dpath
            ).collect_symbols()

            _link_dag_root(card_output_path, self.kwdagger_dpath)

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
        resolved_symbols = []
        claim_hashes = []
        for status, results_fpath, execution_hash, symbols in out:
            results.append(status)
            resolved_symbols.append(symbols)
            claim_hashes.append(execution_hash)
            logger.info(f'Wrote claim output to {results_fpath}')

        calculated_metrics = {}

        if raw_symbol_metadata:
            metric_definitions = Metric.build_metrics_from_symbol_metadata(
                raw_symbol_metadata
            )
            calculated_metrics = _calculate_metrics(
                metric_definitions,
                resolved_symbols,
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
            'claims': claim_hashes,
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
        self,
        flattened_sweep: List['Symbols'],
        results: Dict[str, Any] | None = None,
        cell_key: str | None = None,
        measured: set | None = None,
    ) -> List['EvaluationTask']:
        return [
            EvaluationTask(
                Claim({'python': self.claim.claim}), symbols,
                results=results, cell_key=cell_key, measured=measured,
            )
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

        Holds the per-run record; node artifacts live under
        :attr:`kwdagger_dpath`. An unchanged card reuses the directory it
        already has. Computed once per instance, since the timestamp would
        otherwise differ per read.
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


class EvaluationTask:
    """
    Singular submission from an Evaluation Card
    """

    def __init__(
        self,
        claim: 'Claim',
        symbols: 'Symbols',
        results: Dict[str, Any] | None = None,
        cell_key: str | None = None,
        measured: set | None = None,
    ) -> None:
        self.claim = claim
        self.symbols = symbols
        self.results = Results(results or {})
        self.cell_key = cell_key
        self.measured = measured or set()
        self.output_msg = ''
        self.log: Dict[str, Any] = {}

    def execute(self) -> Tuple[str, str]:
        self.symbols.resolve()
        # x -> y -> z1 -> a1 -> res1
        #           ...
        #           zn -> an -> resn
        # make sure x,y are done once / before sweep
        context = self.symbols()
        for name, value in self.results.bind().items():
            if name in context:
                raise ValueError(
                    f'symbol {name!r} collides with a pipeline result of the '
                    f'same name; rename the symbol'
                )
            context[name] = value

        self.result, self.output_msg = self.claim.evaluate(context)
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
        if self.cell_key:
            self.log['cell'] = self.cell_key
        if self.results.accessed:
            self.log['consumed'] = sorted(self.results.accessed)

    @property
    def cell_id(self) -> str:
        """
        Directory name for this cell's verdict.

        Only has to be unique and stable; nothing reads it back by name.
        """
        if self.cell_key is None:
            return self._execution_hash
        return f'{self.cell_key}_{self._execution_hash}'

    @property
    def _execution_hash(self) -> str:
        view = self.symbols.simple_view()
        # What the card fixed, not what the run measured: a cell that reports a
        # different number is the same cell, and replaces its own verdict.
        view = {k: v for k, v in view.items() if k not in self.measured}
        return ub.hash_data(view)[:12]


class Results:
    """
    Pipeline results addressed by their qualified names.

    kwdagger's convention is a flat dict of ``metrics.<node>.<name>`` keys.
    Those are not Python identifiers, so a claim cannot name them directly;
    this exposes each dotted level as an attribute instead. Reads are
    recorded, so a verdict can say which values its claim consumed.

    Example:
        >>> from magnet.evaluation import Results
        >>> results = Results({'metrics.compare.gap': 0.04,
        ...                    'metrics.compare.threshold': 0.1})
        >>> bound = results.bind()
        >>> bound['metrics'].compare.gap
        0.04
        >>> sorted(results.accessed)
        ['metrics.compare.gap']
        >>> bound['metrics'].compare.missing
        Traceback (most recent call last):
            ...
        AttributeError: no 'missing' under 'metrics.compare'; available: ...
    """

    def __init__(
        self,
        flat: Dict[str, Any],
        prefix: str = '',
        accessed: set | None = None,
    ) -> None:
        self._flat = dict(flat)
        self._prefix = prefix
        self._accessed = set() if accessed is None else accessed

    @property
    def accessed(self) -> set:
        return self._accessed

    def bind(self) -> Dict[str, Any]:
        """Top-level names to inject into a claim's namespace."""
        bound = {}
        for key in self._flat:
            root = key.split('.', 1)[0]
            if root in self._flat:
                bound[root] = self._flat[root]
            else:
                bound[root] = Results(self._flat, f'{root}.', self._accessed)
        return bound

    def as_dict(self) -> Dict[str, Any]:
        """
        The leaf values at this level, unqualified.

        For a claim that would rather work in bare names::

            globals().update(metrics.compare.as_dict())
        """
        depth = self._prefix.count('.') + 1
        return {
            key.split('.')[depth]: value
            for key, value in self._flat.items()
            if key.startswith(self._prefix) and key.count('.') == depth
        }

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            # Pickle probes __getstate__ and friends before __init__ has run,
            # so answering them from self._flat would recurse.
            raise AttributeError(name)
        key = f'{self._prefix}{name}'
        if key in self._flat:
            self._accessed.add(key)
            return self._flat[key]
        deeper = f'{key}.'
        if any(k.startswith(deeper) for k in self._flat):
            return Results(self._flat, deeper, self._accessed)
        available = sorted(
            k[len(self._prefix):].split('.')[0]
            for k in self._flat
            if k.startswith(self._prefix)
        )
        raise AttributeError(
            f'no {name!r} under {self._prefix.rstrip(".")!r}; '
            f'available: {available}'
        )

    def __repr__(self) -> str:
        return f'<Results {self._prefix or "/"}>'


def _fill_declared_symbols(
    symbols: Dict[str, Any], results: Dict[str, Any]
) -> Tuple[Dict[str, Any], set]:
    """
    Give a declared-but-unresolved symbol the result of the same name.

    A card that defines a metric declares the symbol and leaves it to the
    pipeline to supply. Naming it is how the card opts in; everything else the
    node produced stays behind ``metrics.``.

    Returns:
        Tuple[Dict[str, Any], set]: the symbols, and the names filled in.
    """
    filled = set()
    out = {}
    for name, spec in symbols.items():
        spec = dict(spec)
        if not {'value', 'sweep', 'python'} & set(spec):
            for key, value in results.items():
                if key.rsplit('.', 1)[-1] == name:
                    spec['value'] = value
                    filled.add(name)
                    break
        out[name] = spec
    return out, filled


def _link_dag_root(card_output_path: ub.Path, kwdagger_dpath: ub.Path) -> None:
    """
    Point ``<run>/kwdagger`` at the shared DAG root.

    The root moved out of the run directory so nodes survive a card edit, but
    downstream tools glob ``<run>/kwdagger`` for a run's artifacts and figures
    and would silently find none. This is a compatibility shim, not the fix:
    those tools should read the shared root and select the run's own cells,
    which ``result_cells.json`` records. Remove it once they do.
    """
    link = card_output_path / 'kwdagger'
    try:
        ub.symlink(kwdagger_dpath, link, overwrite=True)
    except OSError as ex:
        logger.warning(f'could not link {link} to the DAG root: {ex}')


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

    KNOWN_SPEC_KEYS = frozenset({
        'type', 'value', 'sweep', 'python', 'depends_on', 'depends',
        'metadata',
    })

    def __init__(self, name: str, spec: Dict[str, Any]) -> None:
        self.name = name
        self.value = spec.get('value')
        self.sweep = spec.get('sweep')
        self.type = spec.get('type', 'List[int]')
        self.definition = spec.get('python', '')
        self.dependencies = self._resolve_dependencies(name, spec)
        self.metadata = spec.get('metadata')

        unknown = set(spec) - self.KNOWN_SPEC_KEYS
        if unknown:
            logger.warning(
                f'symbol {name!r}: unrecognized key(s): {sorted(unknown)}'
            )

    @staticmethod
    def _resolve_dependencies(
        name: str, spec: Dict[str, Any]
    ) -> List[str]:
        """
        Read dependencies from `depends_on` or its `depends` alias.

        Args:
            name: Symbol name used in error messages.
            spec: Symbol specification.

        Returns:
            Declared dependency names.

        Raises:
            ValueError: If both spellings are present and disagree.

        Example:
            >>> Symbol._resolve_dependencies('y', {'depends': ['x']})
            ['x']
        """
        canonical = spec.get('depends_on')
        alias = spec.get('depends')

        if (
            canonical is not None
            and alias is not None
            and list(canonical) != list(alias)
        ):
            raise ValueError(
                f'symbol {name!r}: `depends_on` and `depends` disagree'
            )

        dependencies = canonical if canonical is not None else alias
        return [] if dependencies is None else list(dependencies)

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
    evaluations: List['EvaluationTask'] | List[Dict[str, Any]],
    symbol_metadata: Dict[str, Any],
) -> Dict[str, MetricValue]:
    calculated_metrics = {}
    for metric in metric_definitions:
        runs = []
        for evaluation in evaluations:
            if isinstance(evaluation, dict):
                symbols = evaluation
            else:
                symbols = evaluation.symbols()
            symbol_value = symbols.get(metric.name)
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
        # dispatch calls that run from collect_result_cells().
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
