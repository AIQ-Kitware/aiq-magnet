"""
Running a card's pipeline on kwdagger.

Everything that knows about DAGs, schedules and queues lives here, so
:mod:`magnet.evaluation` deals in cards, symbols and claims.
"""
import json
import os
import warnings
from typing import Any, Dict, List, Tuple

import ubelt as ub
from kwdagger import Pipeline, ProcessNode
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
        warnings.warn(
            "a card's `pipeline:` block is soft-deprecated; declare a "
            '`kwdagger:` block with a `result_node` instead.',
            DeprecationWarning,
            stacklevel=2,
        )
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
        self, backend: str = 'serial', skip_existing: bool = True, **kwargs: Any
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
    Handler for a full kwdagger pipeline specification.

    The pipeline may be an importable Python pipeline, a YAML file path, or a
    declarative ``nodes`` / ``edges`` mapping embedded directly in the card.

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
        >>> sorted(processor.spec['pipeline']['nodes'])
        ['compare', 'predict']
        >>> processor.spec['pipeline']['edges']
        ['predict.result_fpath -> compare.result_fpath']
    """

    def __init__(
        self, pipeline_def: Dict[str, Any], root_dpath: ub.Path
    ) -> None:
        # ``result_node`` is a MAGNET-level declaration, not something
        # kwdagger understands, so keep it out of the scheduled spec.
        self.spec = {
            k: v for k, v in pipeline_def.items() if k != 'result_node'
        }
        self.result_node = pipeline_def.get('result_node')
        self.root_dpath = root_dpath
        self.results = []
        self.symbols = []
        self.queue = None
        self.incomplete = []

    def dispatch(self, **schedule_options: Any) -> None:
        """Run this card through :class:`ScheduleEvaluationConfig`.

        ``schedule_options`` uses kwdagger's own option names and semantics.
        MAGNET supplies only the pipeline spec, artifact root, and ``run=True``.
        """
        kwd_config = ScheduleEvaluationConfig(
            params=self.spec,  # includes pipeline and matrix
            root_dpath=self.root_dpath,
            run=True,
            **schedule_options,
        )
        self.dag, self.queue = build_schedule(kwd_config)

    def collect_result_cells(self) -> List[Dict[str, Any]]:
        """
        Read the result node's output for each of its configured instances.

        One instance is one cell, identified by its kwdagger ``process_id``:
        a property of the computation, so it is stable across runs and does
        not depend on what else was scheduled alongside it. Each instance is
        asked where its own artifact is, since the shared DAG root means
        globbing the node directory can return an older card version's.

        Results are qualified as ``metrics.<node>.<name>`` -- kwdagger's
        convention -- so a pipeline value cannot collide with a card symbol.

        Returns:
            List[Dict[str, Any]]: per instance, its ``key``, the node's own
                resolved ``params``, its ``results``, and the ``artifact``
                they were read from.

        An instance that produced nothing is skipped, not fatal: a card
        reports what its run managed to compute.

        Raises:
            ValueError: if no ``result_node`` was declared, or it does not name
                a node in the pipeline.
        """
        if not self.result_node:
            raise ValueError('card must declare kwdagger.result_node')

        if not getattr(self, 'dag', None):
            self.dispatch()

        # build_schedule returns configured instances keyed by process id;
        # node.name is the template name the card refers to.
        instances = [
            node
            for node in self.dag.nodes.values()
            if node.name == self.result_node
        ]
        if not instances:
            available = sorted({node.name for node in self.dag.nodes.values()})
            raise ValueError(
                f'result_node {self.result_node!r} is not a node in the '
                f'pipeline; available: {available}'
            )

        cells = []
        missing = []
        for node in instances:
            fpath = (
                node.final_node_dpath / node.out_paths[node.primary_out_key]
            )
            if not fpath.exists():
                missing.append(self._instance_status(node, fpath))
                continue
            payload = json.loads(fpath.read_text())
            # A node writes its values at the top level; `result` is the older
            # nesting, still read so existing nodes keep working.
            values = payload.get('result', payload)
            cells.append({
                'key': node.process_id,
                'params': dict(node.config),
                'results': {
                    f'metrics.{self.result_node}.{name}': value
                    for name, value in values.items()
                    if not name.startswith('_')
                },
                'artifact': str(fpath),
            })

        self.incomplete = missing
        if missing:
            counts = ub.dict_hist(entry['status'] for entry in missing)
            logger.warning(
                f'{len(cells)} of {len(instances)} {self.result_node!r} '
                f'instances have a result; the rest: {counts}. '
                f'First: {missing[0]["key"]} ({missing[0]["status"]})'
            )
        return cells

    def collect_results(self) -> Tuple[List[str], List[Any]]:
        """Legacy result collector used by ``magnet evaluate``.

        The legacy evaluator historically expected kwdagger pipelines to emit
        ``verdict.json`` files containing ``result.status`` and
        ``result.symbols``. Keep that behavior isolated here while
        ``evaluate_new`` uses :meth:`collect_result_cells`.
        """
        if not self.results:
            self.dispatch(backend='serial', skip_existing=True)

        paths = self.root_dpath.glob('**/verdict.json')
        for claim_json in paths:
            claim_result = json.load(open(claim_json, 'r'))
            if 'result' in claim_result and 'status' in claim_result['result']:
                self.results.append(claim_result['result']['status'])
            if 'result' in claim_result and 'symbols' in claim_result['result']:
                self.symbols.append(claim_result['result']['symbols'])

        return self.results, self.symbols

    def _instance_status(self, node: Any, expected: Any) -> Dict[str, Any]:
        """
        Why an instance has no result: it failed, or it has not run.

        Returns:
            Dict[str, Any]: its ``key``, a ``status`` of ``failed``,
                ``pending`` or ``empty``, the exit code if it has one, and the
                ``expected`` path.
        """
        entry = {
            'key': node.process_id,
            'status': 'pending',
            'returncode': None,
            'expected': str(expected),
        }
        for job in getattr(self.queue, 'jobs', None) or []:
            if getattr(job, 'name', None) != node.process_id:
                continue
            stat_fpath = getattr(job, 'stat_fpath', None)
            if stat_fpath is None or not ub.Path(stat_fpath).exists():
                break
            returncode = json.loads(ub.Path(stat_fpath).read_text()).get('ret')
            entry['returncode'] = returncode
            # Ran and exited clean, but wrote nothing where the card looks.
            entry['status'] = 'failed' if returncode else 'empty'
            break
        return entry


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

