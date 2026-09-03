"""
Running a card's pipeline on kwdagger.

Everything that knows about DAGs, schedules and queues lives here, so
:mod:`magnet.evaluation` deals in cards, symbols and claims.
"""
import json
import os
from typing import Any, Dict, List, Tuple

import ubelt as ub
from kwdagger import Pipeline, ProcessNode
from kwdagger.pipeline import coerce_pipeline
from kwdagger.schedule import ScheduleEvaluationConfig, build_schedule
from loguru import logger

__all__ = [
    'GenericPipelineProcessor',
    'KWDaggerProcessor',
]


class GenericPipelineProcessor:
    """
    Handler for yaml-based pipeline specification

    Soft-deprecated: prefer a ``kwdagger:`` block with a ``result_node``.
    Its semantics are kept -- one symbol set per instance, bound as bare names
    -- since most cards still use it.

    Example:
        >>> from magnet._kwdagger import GenericPipelineProcessor
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
        self.compiled_dag = None
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
        self, backend: str = 'serial', skip_existing: bool = True,
        **kwargs: Any
    ) -> None:
        self.define_kwdagger()

        kwdagger_params = {'pipeline': self.dag, 'matrix': self.matrix}

        kwd_config = ScheduleEvaluationConfig(
            params=kwdagger_params,  # includes pipeline and additional params
            root_dpath=self.root_dpath,
            backend=backend,
            skip_existing=skip_existing,
            run=True,
            **kwargs,
        )

        self.compiled_dag, self.queue = build_schedule(kwd_config)

    def collect_symbols(self) -> Dict[str, Any]:
        """
        Collect results (Evaluation Card 'symbols') in place of 'load_result' in the ProcessNode definition

        Each configured instance is asked for its own artifact. Globbing the
        root instead would also return the instances of whatever other card
        versions share it.
        """
        if not self.symbols:
            self.dispatch()

        node_name = next(iter(self.dag.node_dict))
        out_path = self.dag.node_dict[node_name].out_paths['results_fpath']

        for node in self.compiled_dag.nodes.values():
            if node.name != node_name:
                continue
            fpath = node.final_node_dpath / out_path
            if not fpath.exists():
                continue
            payload = json.loads(fpath.read_text())
            # A node writes its values at the top level; `result` is the older
            # nesting, still read so existing nodes keep working.
            values = payload.get('result', payload)
            for symbol, value in values.items():
                if symbol.startswith('_'):
                    continue
                self.symbols.setdefault(node.process_id, {})[symbol] = {
                    'value': value
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
    Adapter between a MAGNET recipe and kwdagger's experiment-runner APIs.

    Scheduling and evidence discovery are deliberately separate. ``schedule``
    submits the finite experiment campaign requested by this invocation.
    ``load_available_result_rows`` scans the shared kwdagger result store using
    kwdagger's aggregate loader, so evidence is not limited to the processes
    that happened to be scheduled by the current invocation.

    The pipeline may be an importable Python pipeline, a YAML file path, or a
    declarative ``nodes`` / ``edges`` mapping embedded directly in the recipe.

    Example:
        >>> from magnet._kwdagger import KWDaggerProcessor
        >>> import kwutil
        >>> example_cfg = kwutil.Yaml.coerce(
        ...     '''
        ...     kwdagger:
        ...       result_node: compare
        ...       pipeline:
        ...         nodes:
        ...           predict:
        ...             executable: "python predict.py"
        ...             algo_params: {model: null}
        ...             out_paths: {result_fpath: result.json}
        ...           compare:
        ...             executable: "python compare.py"
        ...             in_paths: [result_fpath]
        ...             out_paths: {out_fpath: comparison.json}
        ...         edges:
        ...           - predict.result_fpath -> compare.result_fpath
        ...       matrix:
        ...         predict.model: [model-a, model-b]
        ...     '''
        ... )
        >>> processor = KWDaggerProcessor(example_cfg['kwdagger'], '.')
        >>> processor.result_node
        'compare'
        >>> sorted(processor.params['pipeline']['nodes'])
        ['compare', 'predict']
        >>> processor.params['pipeline']['edges']
        ['predict.result_fpath -> compare.result_fpath']
    """

    def __init__(
        self, kwdagger_config: Dict[str, Any], root_dpath: ub.Path
    ) -> None:
        # ``result_node`` is a MAGNET-level declaration, not part of the
        # ``kwdagger schedule --params`` payload.
        self.params = {
            k: v for k, v in kwdagger_config.items() if k != 'result_node'
        }
        self.result_node = kwdagger_config.get('result_node')
        self.root_dpath = ub.Path(root_dpath)
        self.results = []
        self.symbols = []
        self.request_dag = None
        self.queue = None

    def schedule(
        self,
        *,
        dry_run: bool = False,
        container_settings: Any = None,
        lease_settings: Any = None,
        **schedule_options: Any,
    ) -> None:
        """Submit this invocation's requested experiment campaign.

        ``schedule_options`` uses kwdagger's own option names and semantics.
        MAGNET supplies only the pipeline spec, artifact root, and ``run``.
        The returned compiled graph is retained only to report the operational
        state of this request; it is not used to decide what evidence exists.

        ``dry_run`` schedules with ``run=0``. KWDagger still compiles the whole
        matrix and hands back the graph, so the request is reported in full; it
        writes a driver script rather than submitting anything.

        ``container_settings`` and ``lease_settings`` say where this
        invocation's work runs. They are written onto the nodes here rather
        than read from process state when a command renders, so two pipelines
        in one process can differ and a rendered command is explainable from
        its node alone.
        """
        from magnet import containers, leasing

        # Build the DAG once and hand the same object to build_schedule.
        # coerce_pipeline returns a Pipeline unchanged, so the nodes configured
        # here are the nodes that render commands -- which they were not when
        # this built a throwaway pipeline to inspect and let build_schedule
        # construct its own.
        pipeline = coerce_pipeline(self.params['pipeline'])

        container_settings = (
            container_settings or containers.ContainerSettings()
        )
        lease_settings = lease_settings or leasing.LeaseSettings()
        container_settings.apply(pipeline)
        lease_settings.apply(pipeline)

        # Before anything is submitted: an execution setting that cannot reach
        # a single node is a failed invocation, not a default.
        _check_container_settings_apply(pipeline, container_settings)

        kwd_config = ScheduleEvaluationConfig(
            # includes pipeline and matrix/grid controls. The configured
            # Pipeline object goes in place of the spec; self.params keeps the
            # spec, because the aggregate path builds its own pipeline for
            # reading results and has no business inheriting execution settings.
            params=dict(self.params, pipeline=pipeline),
            root_dpath=self.root_dpath,
            run=not dry_run,
            **schedule_options,
        )
        self.request_dag, self.queue = build_schedule(kwd_config)

    def _coerce_aggregate_pipeline(self) -> Any:
        """Configure the logical pipeline used by kwdagger aggregate loading."""
        from kwdagger.pipeline import coerce_pipeline

        if not self.result_node:
            raise ValueError('recipe must declare kwdagger.result_node')

        pipeline = coerce_pipeline(self.params['pipeline'])
        pipeline.configure(config=None, root_dpath=self.root_dpath)
        if self.result_node not in pipeline.node_dict:
            available = sorted(pipeline.node_dict)
            raise ValueError(
                f'result_node {self.result_node!r} is not a node in the '
                f'pipeline; available: {available}'
            )
        return pipeline

    def load_available_result_rows(
        self,
        *,
        io_workers: Any = 'avail',
        cache_resolved_results: bool = True,
    ) -> List[Dict[str, Any]]:
        """Load all currently available evidence for ``result_node``.

        This uses kwdagger's aggregate loader, which scans the shared result
        root from the logical pipeline's output templates and reconstructs the
        qualified result namespace (``metrics.*``, ``params.*``,
        ``resolved_params.*``, ``context.*``, ``resources.*``, etc.). The
        current request graph is intentionally irrelevant here: results from
        prior campaigns remain evidence, while a currently pending or failed
        request contributes no result row until it produces an artifact.

        Returns:
            A list of mappings with ``key`` (the computation/artifact identity),
            ``artifact`` (the primary result path), and ``row`` (the complete
            kwdagger aggregate row made available to a MAGNET claim).
        """
        import pandas as pd
        from kwdagger.aggregate_loader import build_tables

        pipeline = self._coerce_aggregate_pipeline()
        tables_by_node = build_tables(
            self.root_dpath,
            pipeline,
            io_workers,
            [self.result_node],
            cache_resolved_results=cache_resolved_results,
        )
        parts = tables_by_node.get(self.result_node)
        if not parts:
            return []

        table = pd.concat(list(parts.values()), axis=1)
        evidence = []
        for raw_row in table.to_dict(orient='records'):
            row = {}
            for key, value in raw_row.items():
                if _is_missing_aggregate_value(value):
                    continue
                if isinstance(value, os.PathLike):
                    value = os.fspath(value)
                elif type(value).__module__.startswith('numpy') and hasattr(
                    value, 'item'
                ):
                    value = value.item()
                row[key] = value

            artifact = row.get('fpath')
            if artifact is None:
                continue
            artifact = os.fspath(artifact)
            row['fpath'] = artifact
            evidence.append({
                'key': ub.Path(artifact).parent.name,
                'artifact': artifact,
                'row': row,
            })
        evidence.sort(key=lambda item: item['artifact'])
        return evidence

    def inspect_requested_runs(self) -> List[Dict[str, Any]]:
        """Describe execution state for this invocation's finite request.

        These records are operational provenance only. They are kept separate
        from aggregate evidence because a failed, disabled, skipped, queued, or
        not-yet-started request says nothing by itself about the truth of a
        MAGNET claim, and an older successful result may already exist for the
        same computation.
        """
        if self.request_dag is None:
            return []

        named_jobs = getattr(self.queue, 'named_jobs', None) or {}
        if not named_jobs:
            named_jobs = {
                getattr(job, 'name', None): job
                for job in (getattr(self.queue, 'jobs', None) or [])
                if getattr(job, 'name', None) is not None
            }

        records = []
        for process_id, node in self.request_dag.nodes.items():
            job = named_jobs.get(process_id)
            expected = _primary_result_path(node)
            output_available = bool(expected and expected.exists())

            # Mirror kwdagger.pipeline.submit_jobs node-status vocabulary.
            # build_schedule creates a fresh queue for this request, so a
            # concrete process present in that queue is a new submission;
            # enabled processes absent from it were skipped by KWDagger.
            if not getattr(node, 'enabled', True):
                schedule_status = 'disabled'
            elif job is not None:
                schedule_status = 'new_submission'
            else:
                schedule_status = 'skipped'

            returncode = None
            stat_fpath = getattr(job, 'stat_fpath', None) if job else None
            if job is None:
                attempt_status = 'not_attempted'
            elif stat_fpath is None or not ub.Path(stat_fpath).exists():
                attempt_status = 'not_started'
            else:
                stat = json.loads(ub.Path(stat_fpath).read_text())
                returncode = stat.get('ret')
                if returncode is None:
                    attempt_status = 'running'
                elif returncode == 0:
                    attempt_status = 'passed'
                elif returncode == 126:
                    attempt_status = 'skipped'
                else:
                    attempt_status = 'failed'

            record = {
                'process_id': process_id,
                'node': node.name,
                'schedule_status': schedule_status,
                'attempt_status': attempt_status,
                'returncode': returncode,
                'output_available': output_available,
                'enabled': getattr(node, 'enabled', True),
            }
            if expected is not None:
                record['expected_output'] = os.fspath(expected)
            if stat_fpath is not None:
                record['stat_fpath'] = os.fspath(stat_fpath)
            log_fpath = getattr(job, 'log_fpath', None) if job else None
            if log_fpath is not None:
                record['log_fpath'] = os.fspath(log_fpath)
            records.append(record)
        return records

    def collect_results(self) -> Tuple[List[str], List[Any]]:
        """Legacy result collector used by ``magnet evaluate``.

        The legacy evaluator historically expected kwdagger pipelines to emit
        ``verdict.json`` files containing ``result.status`` and
        ``result.symbols``. Keep that behavior isolated here.
        """
        if not self.results:
            self.schedule(backend='serial', skip_existing=True)

        paths = self.root_dpath.glob('**/verdict.json')
        for claim_json in paths:
            claim_result = json.load(open(claim_json, 'r'))
            if 'result' in claim_result and 'status' in claim_result['result']:
                self.results.append(claim_result['result']['status'])
            if 'result' in claim_result and 'symbols' in claim_result['result']:
                self.symbols.append(claim_result['result']['symbols'])

        return self.results, self.symbols


def _is_missing_aggregate_value(value: Any) -> bool:
    """Return True for dataframe missing-value sentinels, but not real None."""
    if value is None:
        return False
    try:
        import pandas as pd

        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, bool):
        return missing
    if type(missing).__module__.startswith('numpy') and hasattr(missing, 'item'):
        # `pd.isna` answers elementwise for a container, so a gathered
        # collection-valued column comes back as an array. A container is not a
        # missing scalar whatever its elements are.
        if getattr(missing, 'size', 1) != 1:
            return False
        return bool(missing.item())
    return False


def _check_container_settings_apply(
    pipeline: Any, settings: Any = None
) -> None:
    """
    Refuse to run when ``--container_image`` would do nothing.

    Containerization is opt-in per node capability: only a node carrying
    :class:`~magnet.containers.ContainerCapability` renders the ``docker run``
    prefix. A card that declares its DAG as data gets kwdagger's plain
    ``YamlProcessNode``, which carries no container capability, so the image
    was accepted, stored, and never read -- a green run that containerized
    nothing, with no warning. Evidence from that run is indistinguishable from evidence produced
    the way the invocation asked for, which is the whole reason it has to be an
    error rather than a note in a log nobody reads.

    Raises:
        ValueError: if an image is configured and no node can use it.
    """
    from magnet import containers

    if settings is None:
        settings = containers.ContainerSettings()
    if not settings.image:
        return

    node_dict = getattr(pipeline, 'node_dict', None) or {}
    inert = sorted(
        name for name, node in node_dict.items()
        if not isinstance(node, containers.ContainerCapability)
    )
    if not node_dict or len(inert) < len(node_dict):
        # A mixed DAG is legitimate -- an analysis step may belong on the host
        # next to a containerized model step -- so name what will not move
        # rather than refusing the run.
        if inert:
            logger.warning(
                f'--container_image is set, but these nodes cannot use it and '
                f'will run on the host: {inert}. Give each a container-capable '
                '`class:` (normally magnet.process_node.MagnetProcessNode) if '
                'that is not intended.'
            )
        return

    raise ValueError(
        f'--container_image={settings.image!r} was given, '
        f'but no node in this pipeline can be containerized, so nothing would '
        f'run in the image and the results would look exactly like a '
        f'containerized run. Nodes: {inert}.\n'
        'Containerization is opt-in per node capability. For a card that '
        'inlines its DAG and may also lease endpoints, use:\n'
        '    class: magnet.process_node.MagnetProcessNode\n'
        'Drop --container_image to run on the host.'
    )


def _primary_result_path(node: Any) -> ub.Path | None:
    """Return a concrete process's primary result path, when it has one."""
    primary_out_key = getattr(node, 'primary_out_key', None)
    if primary_out_key is None:
        return None
    final_out_paths = getattr(node, 'final_out_paths', None)
    if final_out_paths is not None and primary_out_key in final_out_paths:
        return ub.Path(final_out_paths[primary_out_key])
    out_paths = getattr(node, 'out_paths', None) or {}
    if primary_out_key not in out_paths:
        return None
    return ub.Path(node.final_node_dpath) / out_paths[primary_out_key]

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
        >>> from magnet._kwdagger import _resolve_pipeline_path
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

