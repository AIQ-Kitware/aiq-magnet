"""
The kwdagger-native evaluation path.

``magnet evaluate`` remains the compatibility evaluator. This module is the
migration target: kwdagger owns computation and cardinality, MAGNET collects a
configured result node, and the existing Python claim/verdict machinery is
used only as a temporary result-consumption tail.
"""
from __future__ import annotations

import json
import sys
from typing import Any

import kwconf
import safer
import yaml
from loguru import logger
from pydantic import ValidationError
from rich import print

from magnet._kwdagger import KWDaggerProcessor
from magnet.evaluation import (
    SAFER_USE_TEMPFILE,
    EvaluationCard,
    Metric,
    Symbols,
    _calculate_metrics,
    _fill_declared_symbols,
    _link_dag_root,
    _parse_symbol_metadata,
    _reduce_results,
    _run_one,
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

    backend: str = kwconf.Value(
        'tmux',
        parser=str,
        help=(
            'cmd_queue backend used by kwdagger (for example tmux, serial, '
            'slurm, or airflow). Passed directly to kwdagger; this command '
            'does not use MAGNET_QUEUE_BACKEND.'
        ),
    )

    workers: int | None = kwconf.Value(
        None,
        parser=int,
        help=(
            'Maximum tmux workers. Passed directly to kwdagger as '
            '`tmux_workers`; omitted means kwdagger uses its own default. '
            'This command does not use MAGNET_TMUX_WORKERS.'
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


def _check_new_evaluation_card(card: EvaluationCard) -> None:
    """Enforce the execution boundary of the new evaluator."""
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
    card: EvaluationCard,
    *,
    backend: str = 'tmux',
    workers: int | None = None,
    verbose: bool = False,
) -> str:
    """
    Evaluate a kwdagger card with kwdagger as the sole computation engine.

    Result-node values are still fed into the existing Python claim/verdict
    machinery as a migration adapter. Symbol values and derived symbols may
    participate in that claim context, but symbol sweeps and the legacy
    `pipeline:` executor cannot create computation in this path.
    """
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
    cells = processor.collect_result_cells(
        backend=backend,
        workers=workers,
        use_environment_defaults=False,
    )

    # One kwdagger result-node instance is one card cell. No second MAGNET
    # sweep/decomposition pass is allowed in the new evaluator.
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
            json.dump(
                processor.incomplete, file, indent=2, ensure_ascii=False
            )
            file.write('\n')

    _link_dag_root(card_output_path, card.kwdagger_dpath)

    # Transitional compatibility tail. The new path intentionally does not
    # expose the legacy joblib --jobs / --parallel_backend execution layer.
    out = [_run_one(evaluation, claim_results_path)
           for evaluation in card.evaluations]

    results = []
    resolved_symbols = []
    claim_hashes = []
    for status, results_fpath, execution_hash, symbols in out:
        results.append(status)
        resolved_symbols.append(symbols)
        claim_hashes.append(execution_hash)
        logger.info(f'Wrote claim output to {results_fpath}')

    calculated_metrics: dict[str, Any] = {}
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
            metric_statement = (
                '================================\n Evaluation Metrics:\n'
            )
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
    aggregate_verdict: dict[str, Any] = {
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

    card = EvaluationCard(args.path, args.output_path, validate=validate)
    _check_new_evaluation_card(card)
    if args.params is not None:
        card.apply_params(args.params)

    evaluate_card_new(
        card,
        backend=args.backend,
        workers=args.workers,
        verbose=bool(args.verbose),
    )
    card.summarize()


__cli__ = NewEvaluationConfig
__cli__.main = main

if __name__ == '__main__':
    main(sys.argv[1:])
