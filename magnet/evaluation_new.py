"""
The replacement evaluation API built around kwdagger execution.

``magnet evaluate`` remains an alias of the legacy evaluator. This module owns
new evaluation terminology and execution semantics: a ``NewEvaluationRecipe``
describes what to run, kwdagger owns computation and cardinality, and MAGNET
turns each configured result-node cell into a ``NewEvaluationCellResult``.
Those cell results are then reduced into a ``NewEvaluationResultCard``.

The existing Python claim and verdict vocabulary is retained only as a
transitional result-consumption layer while the replacement API takes shape.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
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
    Metric,
    Symbols,
    _calculate_metrics,
    _parse_symbol_metadata,
    _reduce_results,
)
from magnet.schema import NewEvaluationRecipeSchema
from magnet.utils.util_logger import setup_logging

__all__ = [
    'ClaimResultNamespace',
    'NewEvaluationCLI',
    'NewEvaluationCellResult',
    'NewEvaluationRecipe',
    'NewEvaluationResultCard',
    'evaluate_new_recipe',
]


class NewEvaluationCLI(kwconf.Config):
    """Run a ``NewEvaluationRecipe`` with kwdagger."""

    __epilog__ = """
    This command intentionally has a smaller surface than `magnet evaluate`.
    It accepts only recipes with a `kwdagger:` block. Legacy `pipeline:`
    execution and symbol sweeps belong to `magnet evaluate` /
    `magnet evaluate_legacy` during the migration period.
    """

    path: str = kwconf.Value(
        None, required=True, position=1, help='Path to evaluation recipe YAML'
    )

    output_path: str = kwconf.Value(
        './evaluation_runs',
        help='Root directory for MAGNET run records and kwdagger artifacts',
    )

    params: str | None = kwconf.Value(
        None,
        parser=str,
        help=(
            "YAML/JSON merged into the recipe's `kwdagger:` block, or a path "
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

    @classmethod
    def main(
        cls, argv: list[str] | None = None, **kwargs: Any
    ) -> NewEvaluationResultCard | None:
        args = cls.cli(
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
                NewEvaluationRecipeSchema.model_validate(cfg)
                print('Recipe validation succeeded.')
            except ValidationError as ex:
                print('Recipe validation failed.')
                print(ex)
                raise SystemExit(1)
            return None

        recipe = NewEvaluationRecipe(
            args.path, args.output_path, validate=validate
        )
        if args.params is not None:
            recipe.apply_params(args.params)

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
        result_card = recipe.evaluate(
            verbose=bool(args.verbose),
            **schedule_options,
        )
        recipe.summarize()
        return result_card


class ClaimResultNamespace:
    """
    Attribute-access view of kwdagger result values used by Python claims.

    KWDagger result leaves are collected under qualified names such as
    ``metrics.llama_compare.gap``. A Python claim consumes those values through
    the matching attribute expression, for example
    ``metrics.llama_compare.gap < threshold``.

    This proxy presents the flat result mapping as nested attributes and
    records which leaf values were accessed. The access log is stored with the
    per-cell claim result so the run record shows which kwdagger outputs the
    claim actually consumed.

    This object is claim-evaluation plumbing. It is neither an evaluation
    result nor the aggregate ``NewEvaluationResultCard``.
    """

    def __init__(
        self,
        flat: Dict[str, Any],
        prefix: str = '',
        accessed: set[str] | None = None,
    ) -> None:
        self._flat = dict(flat)
        self._prefix = prefix
        self._accessed = set() if accessed is None else accessed

    @property
    def accessed(self) -> set[str]:
        return self._accessed

    def bind(self) -> Dict[str, Any]:
        """Return the top-level names that a Python claim can consume."""
        bound = {}
        for key in self._flat:
            root = key.split('.', 1)[0]
            if root in self._flat:
                bound[root] = self._flat[root]
            else:
                bound[root] = ClaimResultNamespace(
                    self._flat, f'{root}.', self._accessed
                )
        return bound

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            raise AttributeError(name)
        key = f'{self._prefix}{name}'
        if key in self._flat:
            self._accessed.add(key)
            return self._flat[key]
        deeper = f'{key}.'
        if any(k.startswith(deeper) for k in self._flat):
            return ClaimResultNamespace(self._flat, deeper, self._accessed)
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
        return f'<ClaimResultNamespace {self._prefix or "/"}>'


@dataclass
class NewEvaluationCellResult:
    """
    Result of applying one recipe claim to one kwdagger result-node cell.

    ``result_values`` contains the qualified kwdagger outputs made available to
    the claim. ``consumed`` records the subset the claim actually accessed.
    ``result_id`` is stable for the kwdagger cell and the non-measured recipe
    symbols, so changing a measured output does not create a second identity
    for the same configured cell.
    """

    result_id: str
    status: str
    output: str
    symbols: Dict[str, Any]
    timestamp: str
    cell_key: str
    consumed: List[str] = field(default_factory=list)
    result_values: Dict[str, Any] = field(default_factory=dict, repr=False)

    def as_record(self) -> Dict[str, Any]:
        """Return the persisted per-cell verdict record."""
        record = {
            'status': self.status,
            'output': self.output,
            'symbols': self.symbols,
            'timestamp': self.timestamp,
            'cell': self.cell_key,
        }
        if self.consumed:
            record['consumed'] = self.consumed
        return record


@dataclass
class NewEvaluationResultCard:
    """
    Aggregate result produced by evaluating one ``NewEvaluationRecipe``.

    The card contains the final reduced result, all per-cell claim results, and
    any derived metrics. ``as_record`` preserves the current ``verdict.json``
    representation while the replacement Python API gets distinct recipe and
    result types.
    """

    result: str
    claim_aggregation_strategy: Dict[str, Any]
    cell_results: List[NewEvaluationCellResult]
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def cell_result_ids(self) -> List[str]:
        return [cell.result_id for cell in self.cell_results]

    def as_record(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            'result': self.result,
            'claim_aggregation_strategy': self.claim_aggregation_strategy,
            'claims': self.cell_result_ids,
        }
        if self.metrics:
            record['metrics'] = self.metrics
        return record


class NewEvaluationRecipe(EvaluationCard):
    """
    Input recipe for the replacement kwdagger-native evaluation API.

    The recipe owns MAGNET metadata, the Python claim, declared symbols, and a
    required ``kwdagger:`` execution block. KWDagger owns computation and
    matrix expansion. MAGNET consumes the configured ``result_node`` cells and
    returns a ``NewEvaluationResultCard``.

    The legacy ``EvaluationCard`` base is reused temporarily for common card
    parsing, claim, symbol, metric, and summary behavior. Legacy pipeline
    execution and legacy symbol sweeps are rejected by this class.
    """

    def __init__(
        self, path, output_path: str | ub.Path, validate: str = 'error'
    ) -> None:
        super().__init__(path, output_path, validate='off')
        _check_new_evaluation_recipe(self)

        if validate in ('error', 'warning'):
            try:
                NewEvaluationRecipeSchema.model_validate(self.original_card)
            except ValidationError as ex:
                if validate == 'error':
                    raise
                logger.warning(
                    f'WARNING! Recipe validation failed with error:\n{ex}'
                )

        self.recipe_dpath = ub.Path(path).parent
        self.kwdagger = _resolve_pipeline_path(
            self.kwdagger, self.recipe_dpath
        )
        self.original_card['kwdagger'] = self.kwdagger
        self.result_card: NewEvaluationResultCard | None = None
        self._run_hash_cached: str | None = None

    def apply_params(self, params: Any) -> None:
        """Merge kwdagger-style params into this recipe's execution block."""
        params = kwutil.Yaml.coerce(params, backend='pyyaml')
        if not params:
            return
        merged = _deep_merge(self.original_card['kwdagger'], params)
        self.original_card['kwdagger'] = merged
        self.kwdagger = _resolve_pipeline_path(merged, self.recipe_dpath)
        _check_new_evaluation_recipe(self)

    @property
    def kwdagger_dpath(self) -> ub.Path:
        """Shared DAG artifact root, independent of the recipe run directory."""
        return self.output_path / '_kwdagger'

    @property
    def _recipe_hash(self) -> str:
        return ub.hash_data(self.original_card)[:8]

    @property
    def _run_hash(self) -> str:
        if self._run_hash_cached is None:
            existing = [
                p
                for p in sorted(self.output_path.glob(f'{self._recipe_hash}_*'))
                if p.is_dir()
            ]
            if existing:
                newest = max(existing, key=lambda p: p.stat().st_mtime)
                self._run_hash_cached = newest.name
            else:
                timestamp = datetime.now().strftime('%Y-%m-%d__%H-%M-%S')
                self._run_hash_cached = f'{self._recipe_hash}_{timestamp}'
        return self._run_hash_cached

    def evaluate(
        self,
        verbose: bool = False,
        **schedule_options: Any,
    ) -> NewEvaluationResultCard:
        return evaluate_new_recipe(
            self, verbose=verbose, **schedule_options
        )


def _claim_execution_hash(symbols: Symbols, measured: set[str]) -> str:
    view = symbols.simple_view()
    view = {key: value for key, value in view.items() if key not in measured}
    return ub.hash_data(view)[:12]


def _evaluate_claim_cell(
    claim_text: str,
    symbols: Symbols,
    result_values: Dict[str, Any],
    cell_key: str,
    measured: set[str],
) -> NewEvaluationCellResult:
    """Evaluate the recipe claim for one kwdagger result-node cell."""
    symbols.resolve()
    namespace = ClaimResultNamespace(result_values)
    context = symbols()
    for name, value in namespace.bind().items():
        if name in context:
            raise ValueError(
                f'symbol {name!r} collides with a pipeline result of the '
                f'same name; rename the symbol'
            )
        context[name] = value

    claim = Claim({'python': claim_text})
    status, output = claim.evaluate(context)
    execution_hash = _claim_execution_hash(symbols, measured)
    result_id = f'{cell_key}_{execution_hash}'
    return NewEvaluationCellResult(
        result_id=result_id,
        status=status,
        output=output,
        symbols=symbols.simple_view(),
        timestamp=datetime.now().isoformat(),
        cell_key=cell_key,
        consumed=sorted(namespace.accessed),
        result_values=dict(result_values),
    )


def _write_cell_result(
    cell_result: NewEvaluationCellResult, cell_results_path: ub.Path
) -> ub.Path:
    results_fpath = cell_results_path / cell_result.result_id / 'verdict.json'
    results_fpath.parent.ensuredir()
    with safer.open(results_fpath, 'w', temp_file=SAFER_USE_TEMPFILE) as file:
        json.dump(cell_result.as_record(), file, indent=2, ensure_ascii=False)
        file.write('\n')
    return results_fpath


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
) -> Tuple[Dict[str, Any], set[str]]:
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


def _link_dag_root(recipe_output_path: ub.Path, kwdagger_dpath: ub.Path) -> None:
    """Keep the historical ``<run>/kwdagger`` artifact location visible."""
    link = recipe_output_path / 'kwdagger'
    try:
        ub.symlink(kwdagger_dpath, link, overwrite=True)
    except OSError as ex:
        logger.warning(f'could not link {link} to the DAG root: {ex}')


def _check_new_evaluation_recipe(recipe: NewEvaluationRecipe) -> None:
    """Enforce the execution boundary of the replacement evaluator."""
    if not recipe.has_kwdagger:
        if recipe.has_pipeline:
            detail = 'it uses the legacy `pipeline:` execution block'
        else:
            detail = 'it has no `kwdagger:` execution block'
        raise ValueError(
            f'evaluate_new requires a kwdagger recipe; {detail}. '
            'Use `magnet evaluate` / `magnet evaluate_legacy` for legacy '
            'cards, or migrate computation into `kwdagger:`.'
        )
    if recipe.has_pipeline:
        raise ValueError(
            'evaluate_new does not combine `kwdagger:` with the legacy '
            '`pipeline:` executor. Remove the legacy block or use '
            '`magnet evaluate_legacy`.'
        )
    if not recipe.kwdagger.get('result_node'):
        raise ValueError(
            'evaluate_new requires `kwdagger.result_node`, naming the node '
            'whose configured instances become evaluation cells.'
        )
    sweep_symbols = sorted(
        name
        for name, spec in recipe.symbols.items()
        if spec.get('sweep') is not None
    )
    if sweep_symbols:
        raise ValueError(
            'evaluate_new does not execute legacy symbol sweeps. Move '
            'experimental variation into `kwdagger.matrix`, or use '
            '`magnet evaluate_legacy`. Sweep symbols: '
            f'{sweep_symbols}'
        )


def evaluate_new_recipe(
    recipe: NewEvaluationRecipe,
    *,
    verbose: bool = False,
    **schedule_options: Any,
) -> NewEvaluationResultCard:
    """Evaluate a recipe with kwdagger as the sole computation engine."""
    _check_new_evaluation_recipe(recipe)

    recipe_output_path = recipe.output_path / recipe._run_hash
    recipe_output_path.ensuredir()
    setup_logging(verbose, recipe_output_path)

    with safer.open(
        recipe_output_path / 'card.yaml', 'w', temp_file=SAFER_USE_TEMPFILE
    ) as file:
        yaml.safe_dump(recipe.original_card, file, sort_keys=False)

    cell_results_path = recipe_output_path / 'results'
    raw_symbol_metadata = _parse_symbol_metadata(recipe.symbols)
    if raw_symbol_metadata:
        with safer.open(
            recipe_output_path / 'symbol_metadata.json',
            'w',
            temp_file=SAFER_USE_TEMPFILE,
        ) as file:
            json.dump(raw_symbol_metadata, file, indent=2, ensure_ascii=False)

    processor = KWDaggerProcessor(
        recipe.kwdagger, root_dpath=recipe.kwdagger_dpath
    )
    processor.dispatch(**schedule_options)
    cells = processor.collect_result_cells()

    cell_results = []
    for cell in cells:
        cell_symbols, measured = _fill_declared_symbols(
            recipe.symbols, cell['results']
        )
        cell_result = _evaluate_claim_cell(
            recipe.claim.claim,
            Symbols(cell_symbols),
            cell['results'],
            cell['key'],
            measured,
        )
        cell_results.append(cell_result)
        results_fpath = _write_cell_result(cell_result, cell_results_path)
        logger.info(f'Wrote cell result to {results_fpath}')

    with safer.open(
        recipe_output_path / 'result_cells.json',
        'w',
        temp_file=SAFER_USE_TEMPFILE,
    ) as file:
        json.dump(cells, file, indent=2, ensure_ascii=False)
        file.write('\n')

    if processor.incomplete:
        with safer.open(
            recipe_output_path / 'incomplete_cells.json',
            'w',
            temp_file=SAFER_USE_TEMPFILE,
        ) as file:
            json.dump(processor.incomplete, file, indent=2, ensure_ascii=False)
            file.write('\n')

    _link_dag_root(recipe_output_path, recipe.kwdagger_dpath)

    statuses = [cell.status for cell in cell_results]
    resolved_symbols = [cell.symbols for cell in cell_results]

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

    total = len(statuses)

    def percentage(count: int) -> float:
        return count / total if total else 0.0

    verified_count = statuses.count('VERIFIED')
    falsified_count = statuses.count('FALSIFIED')
    inconclusive_count = statuses.count('INCONCLUSIVE')

    logger.info('================================')
    logger.info(f'Settings Evaluated: {total}')
    logger.info(f'  Verified:     {percentage(verified_count):.2f}')
    logger.info(f'  Falsified:    {percentage(falsified_count):.2f}')
    logger.info(f'  Inconclusive: {percentage(inconclusive_count):.2f}')
    logger.info('================================')
    logger.info('\n')

    aggregate_result = _reduce_results(
        statuses, recipe.claim_aggregation_strategy
    )
    result_card = NewEvaluationResultCard(
        result=aggregate_result,
        claim_aggregation_strategy=recipe.claim_aggregation_strategy,
        cell_results=cell_results,
        metrics=calculated_metrics,
    )

    with safer.open(
        recipe_output_path / 'verdict.json',
        'w',
        temp_file=SAFER_USE_TEMPFILE,
    ) as file:
        json.dump(result_card.as_record(), file, indent=2, ensure_ascii=False)
        file.write('\n')

    recipe.result_card = result_card
    recipe.claim.status = aggregate_result
    return result_card


__cli__ = NewEvaluationCLI

if __name__ == '__main__':
    __cli__.main()
