"""
The replacement evaluation API built around kwdagger experiment execution.

``magnet evaluate`` remains an alias of the legacy evaluator. A
``NewEvaluationRecipe`` can request a finite kwdagger campaign, but the
campaign is not the boundary of the evidence set. After scheduling, MAGNET
uses kwdagger's aggregate loader to discover available result rows for the
configured ``result_node`` in the shared result store. ``evidence.scope`` then
selects either all accumulated rows or only rows corresponding to result-node
computations requested by this invocation. Each selected row is checked against
the claim and becomes a ``NewEvaluationCellResult``; those claim results are
reduced into a ``NewEvaluationResultCard``.

The current request's queued/running/failed state is recorded separately from
claim evaluation. A failed computation is execution provenance, not evidence
that a claim is false.
"""
from __future__ import annotations

import json
import re
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
from magnet.schema import RECIPE_NAME_PATTERN, NewEvaluationRecipeSchema
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
    Attribute-access view of one kwdagger aggregate row for a Python claim.

    KWDagger aggregate rows use qualified columns such as
    ``metrics.llama_compare.gap``, ``params.llama_predict.base_model``, and
    ``resolved_params.llama_predict.base_model``. This proxy exposes the flat
    row through the corresponding nested attribute expressions while recording
    which leaves the claim actually accessed.

    The object is only an access adapter over evidence already loaded by
    kwdagger. It does not discover results, represent an execution attempt, or
    define the aggregate ``NewEvaluationResultCard``.
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
    Result of applying one recipe claim to one available kwdagger result row.

    ``evidence_row`` is the complete qualified aggregate row made available to
    the claim. ``consumed`` records the subset actually accessed. ``symbols``
    is the legacy-compatible dashboard view of the concrete claim inputs: it
    combines resolved recipe symbols with the qualified evidence leaves the
    claim consumed. ``artifact`` points back to the primary kwdagger result
    from which the row was loaded. ``cell_key`` is the artifact/computation
    identity used to keep repeated evaluation of the same available evidence
    stable across MAGNET runs.
    """

    result_id: str
    status: str
    output: str
    symbols: Dict[str, Any]
    timestamp: str
    cell_key: str
    artifact: str | None = None
    consumed: List[str] = field(default_factory=list)
    evidence_row: Dict[str, Any] = field(default_factory=dict, repr=False)

    def as_record(self) -> Dict[str, Any]:
        """Return the persisted per-evidence-row claim record."""
        record = {
            'status': self.status,
            'output': self.output,
            'symbols': self.symbols,
            'timestamp': self.timestamp,
            'cell': self.cell_key,
        }
        if self.artifact is not None:
            record['artifact'] = self.artifact
        if self.consumed:
            record['consumed'] = self.consumed
        if self.evidence_row:
            record['evidence'] = self.evidence_row
        return record


@dataclass
class NewEvaluationResultCard:
    """
    Snapshot of a recipe evaluated against the evidence currently available.

    ``cell_results`` contains claim evaluations for available kwdagger result
    rows. ``requested_work`` summarizes only the finite campaign requested by
    this invocation; failed or unfinished attempts are reported there and do
    not directly alter the claim result.
    """

    result: str
    claim_aggregation_strategy: Dict[str, Any]
    cell_results: List[NewEvaluationCellResult]
    metrics: Dict[str, Any] = field(default_factory=dict)
    requested_work: Dict[str, Any] = field(default_factory=dict)
    evidence_scope: str = 'all'
    evidence_discovered: int = 0

    @property
    def cell_result_ids(self) -> List[str]:
        return [cell.result_id for cell in self.cell_results]

    def as_record(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            'result': self.result,
            'claim_aggregation_strategy': self.claim_aggregation_strategy,
            'claims': self.cell_result_ids,
            'evidence': {
                'scope': self.evidence_scope,
                'available': len(self.cell_results),
                'discovered': self.evidence_discovered,
            },
        }
        if self.requested_work:
            record['requested_work'] = self.requested_work
        if self.metrics:
            record['metrics'] = self.metrics
        return record


class NewEvaluationRecipe(EvaluationCard):
    """
    Input recipe for the replacement kwdagger-native evaluation API.

    The recipe owns MAGNET metadata, the Python claim, declared symbols, and a
    required ``kwdagger:`` execution block. KWDagger owns experiment execution
    and the aggregate result store. ``result_node`` selects which accumulated
    aggregate rows are candidates for the claim. The optional top-level
    ``evidence.scope`` chooses whether the claim sees all accumulated rows or
    only rows corresponding to result-node computations requested by this
    invocation.

    The legacy ``EvaluationCard`` base is reused temporarily for common card
    parsing, claim, symbol, metric, and summary behavior. Legacy pipeline
    execution and legacy symbol sweeps are rejected by this class.
    """

    def __init__(
        self, path, output_path: str | ub.Path, validate: str = 'error'
    ) -> None:
        super().__init__(path, output_path, validate='off')
        #: Short machine identifier for this card, distinct from ``title``.
        self.name = self.original_card.get('name', '')
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
        """Shared KWDagger result/artifact root, independent of the MAGNET run."""
        return self.output_path / '_kwdagger'

    @property
    def evidence_scope(self) -> str:
        """Which available aggregate rows this evaluation snapshot consumes."""
        evidence = self.original_card.get('evidence') or {}
        return evidence.get('scope', 'all')

    @property
    def _recipe_hash(self) -> str:
        return ub.hash_data(self.original_card)[:8]

    @property
    def _run_hash(self) -> str:
        if self._run_hash_cached is None:
            # A MAGNET result card is a snapshot of one invocation. Keep these
            # records distinct even when kwdagger reuses the same computations.
            timestamp = datetime.now().strftime('%Y-%m-%d__%H-%M-%S-%f')
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
    evidence_row: Dict[str, Any],
    cell_key: str,
    measured: set[str],
    artifact: str | None = None,
) -> NewEvaluationCellResult:
    """Evaluate the recipe claim for one available kwdagger aggregate row."""
    symbols.resolve()
    namespace = ClaimResultNamespace(evidence_row)
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

    # Preserve the old dashboard contract without forcing KWDagger experiment
    # parameters back into the recipe's legacy ``symbols:`` section. The
    # dashboard calls this mapping ``symbols``; for the new evaluator it is a
    # flat view of the concrete values that the claim actually consumed.
    dashboard_symbols = symbols.simple_view()
    dashboard_symbols.update({
        key: evidence_row[key]
        for key in sorted(namespace.accessed)
    })

    return NewEvaluationCellResult(
        result_id=result_id,
        status=status,
        output=output,
        symbols=dashboard_symbols,
        timestamp=datetime.now().isoformat(),
        cell_key=cell_key,
        artifact=artifact,
        consumed=sorted(namespace.accessed),
        evidence_row=dict(evidence_row),
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


def _select_evidence_rows(
    evidence_rows: List[Dict[str, Any]],
    requested_runs: List[Dict[str, Any]],
    *,
    result_node: str,
    scope: str,
) -> List[Dict[str, Any]]:
    """Apply the recipe's evidence scope after aggregate discovery."""
    if scope == 'all':
        return list(evidence_rows)
    if scope != 'requested':
        raise ValueError(f'unknown evidence scope: {scope!r}')

    requested_result_ids = {
        row['process_id']
        for row in requested_runs
        if row.get('node') == result_node and row.get('enabled', True)
    }
    return [
        row for row in evidence_rows
        if row['key'] in requested_result_ids
    ]


def _summarize_requested_runs(
    requested_runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Summarize this invocation's operational request without affecting claims."""
    if not requested_runs:
        return {'processes': 0}
    return {
        'processes': len(requested_runs),
        'schedule_status': dict(ub.dict_hist(
            row['schedule_status'] for row in requested_runs
        )),
        'attempt_status': dict(ub.dict_hist(
            row['attempt_status'] for row in requested_runs
        )),
        'outputs_available': sum(
            bool(row['output_available']) for row in requested_runs
        ),
    }


def _link_kwdagger_root(recipe_output_path: ub.Path, kwdagger_dpath: ub.Path) -> None:
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
    if not recipe.name:
        raise ValueError(
            'evaluate_new requires a top-level `name`: a short machine '
            'identifier for this card, distinct from its prose `title`. It '
            'names the card wherever machinery has to refer to it, such as '
            'the execution queue.'
        )
    if not re.match(RECIPE_NAME_PATTERN, recipe.name):
        raise ValueError(
            f'recipe `name` {recipe.name!r} must match '
            f'{RECIPE_NAME_PATTERN} -- it is used as a tmux session name and '
            'a path component, so it is restricted to letters, digits, '
            'underscore and hyphen. Put the readable version in `title`.'
        )
    if not recipe.kwdagger.get('result_node'):
        raise ValueError(
            'evaluate_new requires `kwdagger.result_node`, naming the node '
            'whose accumulated aggregate rows provide claim evidence.'
        )
    evidence_scope = recipe.evidence_scope
    if evidence_scope not in {'all', 'requested'}:
        raise ValueError(
            '`evidence.scope` must be either `all` or `requested`; '
            f'got {evidence_scope!r}'
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
    """Schedule requested work, then evaluate the claim over available evidence."""
    _check_new_evaluation_recipe(recipe)

    recipe_output_path = recipe.output_path / recipe._run_hash
    recipe_output_path.ensuredir()
    setup_logging(verbose, recipe_output_path)

    with safer.open(
        recipe_output_path / 'card.yaml', 'w', temp_file=SAFER_USE_TEMPFILE
    ) as file:
        yaml.safe_dump(recipe.original_card, file, sort_keys=False)

    # The visualization dashboard treats this directory as part of MAGNET's
    # run-bundle contract, including when no evidence rows are currently
    # available.
    cell_results_path = (recipe_output_path / 'results').ensuredir()
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

    # Scheduling is one finite operational request. It may add new results to
    # the shared kwdagger store, reuse results that already exist, or leave
    # some requested work failed / pending.
    processor.schedule(**schedule_options)
    requested_runs = processor.inspect_requested_runs()
    requested_work = _summarize_requested_runs(requested_runs)
    with safer.open(
        recipe_output_path / 'requested_runs.json',
        'w',
        temp_file=SAFER_USE_TEMPFILE,
    ) as file:
        json.dump(requested_runs, file, indent=2, ensure_ascii=False)
        file.write('\n')

    # KWDagger aggregate discovers the available result store independently of
    # this request. The recipe may then use the request as an optional filter;
    # the compiled schedule is never the result-discovery mechanism.
    discovered_evidence_rows = processor.load_available_result_rows()
    evidence_rows = _select_evidence_rows(
        discovered_evidence_rows,
        requested_runs,
        result_node=processor.result_node,
        scope=recipe.evidence_scope,
    )

    cell_results = []
    for evidence in evidence_rows:
        evidence_row = evidence['row']
        cell_symbols, measured = _fill_declared_symbols(
            recipe.symbols, evidence_row
        )
        cell_result = _evaluate_claim_cell(
            recipe.claim.claim,
            Symbols(cell_symbols),
            evidence_row,
            evidence['key'],
            measured,
            artifact=evidence['artifact'],
        )
        cell_results.append(cell_result)
        results_fpath = _write_cell_result(cell_result, cell_results_path)
        logger.info(f'Wrote cell result to {results_fpath}')

    _link_kwdagger_root(recipe_output_path, recipe.kwdagger_dpath)

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
            metric_statement = (
                '================================\n Evaluation Metrics:\n'
            )
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
    logger.info(f'Evidence Scope: {recipe.evidence_scope}')
    logger.info(
        f'Available Evidence Rows: {total} '
        f'(discovered: {len(discovered_evidence_rows)})'
    )
    logger.info(f'  Verified:     {percentage(verified_count):.2f}')
    logger.info(f'  Falsified:    {percentage(falsified_count):.2f}')
    logger.info(f'  Inconclusive: {percentage(inconclusive_count):.2f}')
    logger.info(f'Requested Work: {requested_work}')
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
        requested_work=requested_work,
        evidence_scope=recipe.evidence_scope,
        evidence_discovered=len(discovered_evidence_rows),
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
