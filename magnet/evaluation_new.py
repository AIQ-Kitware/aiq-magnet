"""
The kwdagger-native evaluation path.

``magnet evaluate`` remains an alias of the legacy evaluator. This module is
where the replacement semantics are allowed to evolve: kwdagger owns
computation and cardinality, MAGNET collects a configured result node, and the
existing Python claim/verdict machinery is used only as a temporary
result-consumption tail.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

import kwconf
import kwutil
import safer
import ubelt as ub
import yaml
from loguru import logger
from pydantic import ValidationError
from rich import print

from magnet._kwdagger import KWDaggerProcessor, _resolve_pipeline_path
from magnet.evaluation import (
    SAFER_USE_TEMPFILE,
    Claim,
    EvaluationCard,
    EvaluationTask,
    Metric,
    Symbols,
    _calculate_metrics,
    _parse_symbol_metadata,
    _plain_data,
    _reduce_results,
)
from magnet.schema import EvaluationCardSchema
from magnet.utils.util_logger import setup_logging


class NewEvaluationConfig(kwconf.Config):
    """Run a kwdagger-native Evaluation Card."""

    __epilog__ = """
    This command intentionally has a smaller surface than `magnet evaluate`.
    It accepts only cards with a `kwdagger:` block. Legacy `pipeline:`
    execution and symbol sweeps belong to `magnet evaluate` /
    `magnet evaluate_legacy` during the migration period.
    """

    path: str = kwconf.Value(
        None, required=True, position=1, help='Path to evaluation card YAML'
    )

    output_path: str = kwconf.Value(
        './evaluation_runs',
        help='Root directory for MAGNET run records and kwdagger artifacts',
    )

    params: str | None = kwconf.Value(
        None,
        parser=str,
        help=(
            "YAML/JSON merged into the card's `kwdagger:` block, or a path "
            'to a file containing it. This uses the same matrix/config '
            'language as `kwdagger schedule --params`.'
        ),
    )

    # Keep these names and defaults aligned with ``kwdagger schedule``.
    # evaluate_new forwards them without adding MAGNET scheduling semantics.
    backend: str = kwconf.Value(
        'tmux',
        parser=str,
        help=(
            'cmd_queue backend used by kwdagger (for example tmux, serial, '
            'slurm, or airflow). Passed directly to kwdagger.'
        ),
    )

    tmux_workers: int = kwconf.Value(
        8,
        parser=int,
        help='Number of tmux workers. Passed directly to kwdagger.',
    )

    skip_existing = kwconf.Value(
        False,
        help=(
            'KWDagger schedule option: do not submit nodes whose expected '
            'products already exist.'
        ),
    )

    cache = kwconf.Flag(
        True,
        help=(
            'KWDagger schedule option: guard each submitted node so it skips '
            'its command when its outputs already exist.'
        ),
    )

    max_configs: int | None = kwconf.Value(
        None,
        parser=int,
        help=(
            'KWDagger schedule option: expand at most this many matrix '
            'configurations.'
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


class Results:
    """Pipeline results addressed through their qualified dotted names."""

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
        """Return the top-level names that a Python claim can consume."""
        bound = {}
        for key in self._flat:
            root = key.split('.', 1)[0]
            if root in self._flat:
                bound[root] = self._flat[root]
            else:
                bound[root] = Results(self._flat, f'{root}.', self._accessed)
        return bound

    def as_dict(self) -> Dict[str, Any]:
        """Return the leaf values at this level, with prefixes removed."""
        depth = self._prefix.count('.') + 1
        return {
            key.split('.')[depth]: value
            for key, value in self._flat.items()
            if key.startswith(self._prefix) and key.count('.') == depth
        }

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            raise AttributeError(name)
        key = f'{self._prefix}{name}'
        if key in self._flat:
            self._accessed.add(key)
            return self._flat[key]
        deeper = f'{key}.'
        if any(k.startswith(deeper) for k in self._flat):
            return Results(self._flat, deeper, self._accessed)
        available = sorted({
            k[len(self._prefix):].split('.')[0]
            for k in self._flat
            if k.startswith(self._prefix)
        })
        raise AttributeError(
            f'no {name!r} under {self._prefix.rstrip(".")!r}; '
            f'available: {available}'
        )

    def __repr__(self) -> str:
        return f'<Results {self._prefix or "/"}>'


class NewEvaluationTask(EvaluationTask):
    """A legacy claim evaluated against one kwdagger result-node cell."""

    def __init__(
        self,
        claim: Claim,
        symbols: Symbols,
        results: Dict[str, Any] | None = None,
        cell_key: str | None = None,
        measured: set | None = None,
    ) -> None:
        super().__init__(claim, symbols)
        self.results = Results(results or {})
        self.cell_key = cell_key
        self.measured = measured or set()

    def execute(self) -> Tuple[str, str]:
        self.symbols.resolve()
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
        super().record_run()
        if self.cell_key:
            self.log['cell'] = self.cell_key
        if self.results.accessed:
            self.log['consumed'] = sorted(self.results.accessed)

    @property
    def cell_id(self) -> str:
        """Stable verdict-directory identity for one result-node cell."""
        if self.cell_key is None:
            return self._execution_hash
        return f'{self.cell_key}_{self._execution_hash}'

    @property
    def _execution_hash(self) -> str:
        view = self.symbols.simple_view()
        view = {k: v for k, v in view.items() if k not in self.measured}
        return ub.hash_data(view)[:12]


class NewEvaluationCard(EvaluationCard):
    """EvaluationCard state needed only by the kwdagger-native evaluator."""

    def __init__(
        self, path, output_path: str | ub.Path, validate: str = 'error'
    ) -> None:
        super().__init__(path, output_path, validate=validate)
        self.card_dpath = ub.Path(path).parent
        if self.has_kwdagger:
            if not self.kwdagger.get('result_node'):
                raise ValueError(
                    f'{path}: a kwdagger card must declare '
                    '`kwdagger.result_node`, naming the node whose output is '
                    'the card result'
                )
            self.kwdagger = _resolve_pipeline_path(
                self.kwdagger, self.card_dpath
            )
        self._run_hash_cached: str | None = None

    def apply_params(self, params: Any) -> None:
        """Merge kwdagger-style params into this card's execution block."""
        params = _plain_data(kwutil.Yaml.coerce(params))
        if not params:
            return
        if not self.has_kwdagger:
            raise ValueError('--params requires a card with `kwdagger:`')
        merged = _deep_merge(self.original_card['kwdagger'], params)
        self.original_card['kwdagger'] = merged
        self.kwdagger = _resolve_pipeline_path(merged, self.card_dpath)

    @property
    def kwdagger_dpath(self) -> ub.Path:
        """Shared DAG artifact root, independent of the card run directory."""
        return self.output_path / '_kwdagger'

    @property
    def _card_hash(self) -> str:
        return ub.hash_data(self.original_card)[:8]

    @property
    def _run_hash(self) -> str:
        if self._run_hash_cached is None:
            existing = [
                p
                for p in sorted(self.output_path.glob(f'{self._card_hash}_*'))
                if p.is_dir()
            ]
            if existing:
                newest = max(existing, key=lambda p: p.stat().st_mtime)
                self._run_hash_cached = newest.name
            else:
                timestamp = datetime.now().strftime('%Y-%m-%d__%H-%M-%S')
                self._run_hash_cached = f'{self._card_hash}_{timestamp}'
        return self._run_hash_cached

    def dispatch(
        self,
        flattened_sweep: List[Symbols],
        results: Dict[str, Any] | None = None,
        cell_key: str | None = None,
        measured: set | None = None,
    ) -> List[NewEvaluationTask]:
        return [
            NewEvaluationTask(
                Claim({'python': self.claim.claim}),
                symbols,
                results=results,
                cell_key=cell_key,
                measured=measured,
            )
            for symbols in flattened_sweep
        ]

    def evaluate(
        self,
        verbose: bool = False,
        **schedule_options: Any,
    ) -> str:
        return evaluate_card_new(
            self, verbose=verbose, **schedule_options
        )


def _run_one_new(
    evaluation: NewEvaluationTask, claim_results_path: ub.Path
) -> Tuple[str, ub.Path, str, Dict[str, Any]]:
    status, _ = evaluation.execute()
    execution_hash = evaluation.cell_id
    resolved_symbols = evaluation.log['symbols']
    results_fpath = claim_results_path / execution_hash / 'verdict.json'
    results_fpath.parent.ensuredir()
    with safer.open(results_fpath, 'w', temp_file=SAFER_USE_TEMPFILE) as file:
        json.dump(evaluation.log, file, indent=2, ensure_ascii=False)
        file.write('\n')
    return status, results_fpath, execution_hash, resolved_symbols


def _deep_merge(base: Any, update: Any) -> Any:
    """Merge mappings recursively; non-mappings and lists replace leaves."""
    if not isinstance(base, dict) or not isinstance(update, dict):
        return update
    merged = dict(base)
    for key, value in update.items():
        merged[key] = _deep_merge(base[key], value) if key in base else value
    return merged


def _fill_declared_symbols(
    symbols: Dict[str, Any], results: Dict[str, Any]
) -> Tuple[Dict[str, Any], set]:
    """Fill unresolved declared symbols from same-named result leaves."""
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
    """Keep the historical ``<run>/kwdagger`` artifact location visible."""
    link = card_output_path / 'kwdagger'
    try:
        ub.symlink(kwdagger_dpath, link, overwrite=True)
    except OSError as ex:
        logger.warning(f'could not link {link} to the DAG root: {ex}')


def _check_new_evaluation_card(card: NewEvaluationCard) -> None:
    """Enforce the computation boundary of the new evaluator."""
    if not card.has_kwdagger:
        if card.has_pipeline:
            detail = 'it uses the legacy `pipeline:` execution block'
        else:
            detail = 'it has no `kwdagger:` execution block'
        raise ValueError(
            f'evaluate_new requires a kwdagger card; {detail}. '
            'Use `magnet evaluate` / `magnet evaluate_legacy` for legacy '
            'cards, or migrate computation into `kwdagger:`.'
        )
    if card.has_pipeline:
        raise ValueError(
            'evaluate_new does not combine `kwdagger:` with the legacy '
            '`pipeline:` executor. Remove the legacy block or use '
            '`magnet evaluate_legacy`.'
        )
    sweep_symbols = sorted(
        name
        for name, spec in card.symbols.items()
        if spec.get('sweep') is not None
    )
    if sweep_symbols:
        raise ValueError(
            'evaluate_new does not execute legacy symbol sweeps. Move '
            'experimental variation into `kwdagger.matrix`, or use '
            '`magnet evaluate_legacy`. Sweep symbols: '
            f'{sweep_symbols}'
        )


def evaluate_card_new(
    card: NewEvaluationCard,
    *,
    verbose: bool = False,
    **schedule_options: Any,
) -> str:
    """Evaluate a card with kwdagger as the sole computation engine."""
    _check_new_evaluation_card(card)

    card_output_path = card.output_path / card._run_hash
    card_output_path.ensuredir()
    setup_logging(verbose, card_output_path)

    with safer.open(
        card_output_path / 'card.yaml', 'w', temp_file=SAFER_USE_TEMPFILE
    ) as file:
        yaml.safe_dump(card.original_card, file, sort_keys=False)

    claim_results_path = card_output_path / 'results'
    raw_symbol_metadata = _parse_symbol_metadata(card.symbols)
    if raw_symbol_metadata:
        with safer.open(
            card_output_path / 'symbol_metadata.json',
            'w',
            temp_file=SAFER_USE_TEMPFILE,
        ) as file:
            json.dump(raw_symbol_metadata, file, indent=2, ensure_ascii=False)

    card.evaluations = []
    processor = KWDaggerProcessor(
        card.kwdagger, root_dpath=card.kwdagger_dpath
    )
    processor.dispatch(**schedule_options)
    cells = processor.collect_result_cells()

    for cell in cells:
        cell_symbols, measured = _fill_declared_symbols(
            card.symbols, cell['results']
        )
        card.evaluations.extend(
            card.dispatch(
                [Symbols(cell_symbols)],
                results=cell['results'],
                cell_key=cell['key'],
                measured=measured,
            )
        )

    with safer.open(
        card_output_path / 'result_cells.json',
        'w',
        temp_file=SAFER_USE_TEMPFILE,
    ) as file:
        json.dump(cells, file, indent=2, ensure_ascii=False)
        file.write('\n')

    if processor.incomplete:
        with safer.open(
            card_output_path / 'incomplete_cells.json',
            'w',
            temp_file=SAFER_USE_TEMPFILE,
        ) as file:
            json.dump(processor.incomplete, file, indent=2, ensure_ascii=False)
            file.write('\n')

    _link_dag_root(card_output_path, card.kwdagger_dpath)

    # Transitional compatibility tail. No legacy joblib execution controls are
    # exposed by evaluate_new.
    out = [_run_one_new(e, claim_results_path) for e in card.evaluations]

    results = []
    resolved_symbols = []
    claim_hashes = []
    for status, results_fpath, execution_hash, symbols in out:
        results.append(status)
        resolved_symbols.append(symbols)
        claim_hashes.append(execution_hash)
        logger.info(f'Wrote claim output to {results_fpath}')

    calculated_metrics: Dict[str, Any] = {}
    if raw_symbol_metadata and resolved_symbols:
        metric_definitions = Metric.build_metrics_from_symbol_metadata(
            raw_symbol_metadata
        )
        calculated_metrics = _calculate_metrics(
            metric_definitions,
            resolved_symbols,
            raw_symbol_metadata,
        )
        if calculated_metrics:
            metric_statement = '================================\n Evaluation Metrics:\n'
            for metric, value in calculated_metrics.items():
                metric_statement += f'  {metric}: {value: .3f}\n'
            logger.info(metric_statement[:-1])

    total = len(results)

    def percentage(count: int) -> float:
        return count / total if total else 0.0

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

    card_result = _reduce_results(results, card.claim_aggregation_strategy)
    aggregate_verdict: Dict[str, Any] = {
        'result': card_result,
        'claim_aggregation_strategy': card.claim_aggregation_strategy,
        'claims': claim_hashes,
    }
    if raw_symbol_metadata and calculated_metrics:
        aggregate_verdict['metrics'] = calculated_metrics

    with safer.open(
        card_output_path / 'verdict.json',
        'w',
        temp_file=SAFER_USE_TEMPFILE,
    ) as file:
        json.dump(aggregate_verdict, file, indent=2, ensure_ascii=False)
        file.write('\n')

    card.claim.status = card_result
    return card_result


def main(argv: list[str] | None = None, **kwargs: Any) -> None:
    args = NewEvaluationConfig.cli(
        argv=argv,
        data=kwargs,
        strict=True,
        verbose='auto',
        special_options=False,
    )

    validate = args['validate']
    if validate == 'only':
        try:
            with open(args.path, 'r') as file:
                cfg = yaml.safe_load(file)
            EvaluationCardSchema.model_validate(cfg)
            print('Card validation succeeded.')
        except ValidationError as ex:
            print('Card validation failed.')
            print(ex)
            sys.exit(1)
        return

    card = NewEvaluationCard(args.path, args.output_path, validate=validate)
    _check_new_evaluation_card(card)
    if args.params is not None:
        card.apply_params(args.params)

    schedule_options = {
        key: args[key]
        for key in [
            'backend',
            'tmux_workers',
            'skip_existing',
            'cache',
            'max_configs',
        ]
    }
    card.evaluate(
        verbose=bool(args.verbose),
        **schedule_options,
    )
    card.summarize()


__cli__ = NewEvaluationConfig
__cli__.main = main

if __name__ == '__main__':
    main(sys.argv[1:])
