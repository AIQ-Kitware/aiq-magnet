"""
Sankey DSL + utilities.

- Plan/Root/Group/Bucket/Split implement a progressive groupby / branching tree.
- build_sankey produces a nx.DiGraph with weighted edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import typing
import networkx as nx

if typing.TYPE_CHECKING:
    from typing import Protocol

    Row = Dict[str, Any]
    Grouper = Union[str, Callable[[Row], Any]]

    class PlotlyFigureLike(Protocol):
        def write_image(self, *args, **kwargs): ...
        def update_layout(
            self, dict1=None, overwrite=False, **kwargs
        ) -> 'PlotlyFigureLike': ...


@dataclass(frozen=True)
class Root:
    label: str
    def builder(self) -> PlanBuilder:
        return PlanBuilder(Plan(self))
    def group(self, *, by: Grouper, name: Optional[str] = None) -> PlanBuilder:
        return self.builder().group(by=by, name=name)
    def split(self, *, by: Grouper, name: Optional[str] = None) -> SplitBuilder:
        return self.builder().split(by=by, name=name)


# A terminal label convenience (primarily for branch specs)
@dataclass(frozen=True)
class Node:
    label: str


@dataclass(frozen=True)
class Branch:
    """
    A Split branch:
      - label: optional override for the *split outcome node label*
      - plan: plan to continue after the split outcome node
      - terminal: if True, stop at the outcome node (ignore plan)
    """
    plan: "Plan" = field(default_factory=lambda: Plan())
    label: Optional[str] = None
    terminal: bool = False


BranchLike = Union["Plan", Branch, Node, str]  # str => terminal label


@dataclass(frozen=True)
class Group:
    name: str
    by: Grouper


@dataclass(frozen=True)
class Bucket:
    name: str
    by: Grouper


@dataclass(frozen=True)
class Split:
    name: str
    by: Grouper
    branches: Dict[Any, BranchLike]
    default: Optional[BranchLike] = None

    def __post_init__(self):
        def coerce(x: BranchLike) -> Branch:
            if isinstance(x, Branch):
                return x
            if isinstance(x, Node):
                return Branch(label=x.label, terminal=True)
            if isinstance(x, str):
                return Branch(label=x, terminal=True)
            if isinstance(x, Plan):
                return Branch(plan=x)
            raise TypeError(f"Invalid branch spec: {type(x)}")

        # frozen dataclass: use object.__setattr__ to normalize
        object.__setattr__(
            self, "branches", {k: coerce(v) for k, v in self.branches.items()}
        )
        if self.default is not None:
            object.__setattr__(self, "default", coerce(self.default))


class PlanBuilder:
    def __init__(self, plan: Optional[Plan] = None):
        self.plan = plan or Plan()

    def build(self) -> Plan:
        return self.plan

    def group(self, *, by: Grouper, name: Optional[str] = None) -> "PlanBuilder":
        if name is None:
            name = by if isinstance(by, str) else "group"
        self.plan.steps.append(Group(name, by))
        return self

    def bucket(self, *, by: Grouper, name: Optional[str] = None) -> "PlanBuilder":
        if name is None:
            name = by if isinstance(by, str) else "bucket"
        self.plan.steps.append(Bucket(name, by))
        return self

    def split(self, *, by: Grouper, name: Optional[str] = None) -> "SplitBuilder":
        if name is None:
            name = by if isinstance(by, str) else "split"
        sp = Split(name, by, branches={})
        self.plan.steps.append(sp)
        return SplitBuilder(sp)


class CaseBuilder(PlanBuilder):
    def __init__(self, branch_ref: Dict[Any, Branch], key: Any):
        # branch_ref is the Split.branches dict (already normalized to Branch)
        self._branch_ref = branch_ref
        self._key = key
        super().__init__(plan=self._branch_ref[key].plan)

    def set_label(self, label: str) -> "CaseBuilder":
        br = self._branch_ref[self._key]
        self._branch_ref[self._key] = Branch(plan=br.plan, label=label, terminal=br.terminal)
        return self

    def terminal(self, label: Optional[str] = None) -> "CaseBuilder":
        br = self._branch_ref[self._key]
        self._branch_ref[self._key] = Branch(
            plan=br.plan,
            label=(label if label is not None else br.label),
            terminal=True,
        )
        return self


class SplitBuilder:
    def __init__(self, split: Split):
        self.split = split
        # Split.__post_init__ normalized branches to Branch already
        self.cases: Dict[Any, CaseBuilder] = {}

    def add_case(self, *, value: Any) -> CaseBuilder:
        if value not in self.split.branches:
            # create an empty branch by default
            self.split.branches[value] = Branch(plan=Plan())
        cb = CaseBuilder(self.split.branches, value)
        self.cases[value] = cb
        return cb

    def __getitem__(self, value: Any) -> CaseBuilder:
        # defaultdict-ish
        return self.add_case(value=value)

    def set_default(self) -> CaseBuilder:
        if self.split.default is None:
            # store default as a Branch (not in branches dict)
            object.__setattr__(self.split, "default", Branch(plan=Plan()))
        # expose via a pseudo key
        # (or provide a dedicated DefaultCaseBuilder)
        raise NotImplementedError("Implement a DefaultCaseBuilder if you want this.")


def _by_repr(by: Grouper) -> str:
    if isinstance(by, str):
        return repr(by)
    name = getattr(by, '__name__', None)
    if name:
        return f'<fn {name}>'
    return f'<callable {by!r}>'


@dataclass
class Plan:
    """
    A Plan is a sequence of "steps" (Root/Group/Bucket/Split).

    Semantics:
      - Root: declares the starting node label
      - Group/Bucket: always adds one node (stage:value)
      - Split: adds a node for the split decision, then continues using a subplan

    Example:
        >>> # Basic plan text rendering with a split + branches
        >>> plan = Plan(
        ...     Root("ROOT_NODE"),
        ...     Group("dataset", "dataset"),
        ...     Split(
        ...         "status", "status",
        ...         branches={
        ...             "ok": Plan(Bucket("quality", "quality")),
        ...             "fail": Plan(Bucket("reason", "reason")),
        ...         },
        ...     ),
        ... )
        >>> print(plan.to_text())
        ROOT 'ROOT_NODE'
        GROUP 'dataset' by='dataset'
        SPLIT 'status' by='status'
          BRANCH 'ok':
            BUCKET 'quality' by='quality'
          BRANCH 'fail':
            BUCKET 'reason' by='reason'
    """

    steps: List[Any] = field(default_factory=list)

    def __init__(self, *steps):
        self.steps = list(steps)

    def _find_root_label(self, default='ROOT_NODE') -> str:
        for st in self.steps:
            if isinstance(st, Root):
                return st.label
        return default

    def to_text(self) -> str:
        lines: List[str] = []

        def rec(plan: 'Plan', indent: str = ''):
            for st in plan.steps:
                if isinstance(st, Root):
                    lines.append(f'{indent}ROOT {st.label!r}')
                elif isinstance(st, Group):
                    lines.append(
                        f'{indent}GROUP {st.name!r} by={_by_repr(st.by)}'
                    )
                elif isinstance(st, Bucket):
                    lines.append(
                        f'{indent}BUCKET {st.name!r} by={_by_repr(st.by)}'
                    )
                elif isinstance(st, Split):
                    lines.append(f"{indent}SPLIT {st.name!r} by={_by_repr(st.by)}")
                    for k, br in st.branches.items():
                        extra = []
                        if br.label is not None:
                            extra.append(f"label={br.label!r}")
                        if br.terminal:
                            extra.append("terminal=True")
                        extra_s = ("  [" + ", ".join(extra) + "]") if extra else ""
                        lines.append(f"{indent}  BRANCH {k!r}:{extra_s}")
                        rec(br.plan, indent + "    ")
                    if st.default is not None:
                        # same idea for default
                        ...
                else:
                    lines.append(f'{indent}{type(st).__name__} (?)')

        rec(self, '')
        return '\n'.join(lines)

    def trace(self, row: Row, *, label_fmt='{name}: {value}') -> List[str]:
        """
        Return the node labels (path) this row takes.

        Example:
            >>> # Trace the path a row takes through the plan
            >>> from magnet.utils import sankey
            >>> plan = Plan(
            ...     Root("ROOT_NODE"),
            ...     Group("dataset", "dataset"),
            ...     Split(
            ...         "status", "status",
            ...         branches={
            ...             "ok": Plan(Bucket("quality", "quality")),
            ...             "fail": Plan(Bucket("reason", "reason")),
            ...         },
            ...     ),
            ... )
            >>> row = {"dataset": "A", "status": "ok", "quality": "good"}
            >>> plan.trace(row)
            ['ROOT_NODE', 'dataset: A', 'status: ok', 'quality: good']
            >>> # Custom label formatting
            >>> plan.trace(row, label_fmt="{name}={value}")
            ['ROOT_NODE', 'dataset=A', 'status=ok', 'quality=good']
        """
        root = self._find_root_label(default='ROOT_NODE')
        path = [root]

        def _eval_by(by: Grouper, row: Row):
            return by(row) if callable(by) else row.get(by)

        def run(plan: 'Plan', cur: str):
            node = cur
            for st in plan.steps:
                if isinstance(st, Root):
                    continue
                elif isinstance(st, (Group, Bucket)):
                    val = _eval_by(st.by, row)
                    nxt = label_fmt.format(name=st.name, value=val)
                    path.append(nxt)
                    node = nxt
                elif isinstance(st, Split):
                    key = _eval_by(st.by, row)

                    br = st.branches.get(key) or st.default
                    if br is None:
                        return node

                    # Decide the split outcome node label
                    split_label = (
                        br.label
                        if (br.label is not None)
                        else label_fmt.format(name=st.name, value=key)
                    )
                    path.append(split_label)
                    node = split_label

                    # Terminal branch stops here
                    if br.terminal:
                        return node

                    node = run(br.plan, node)
                else:
                    raise TypeError(f'Unknown step type: {type(st)}')
            return node

        run(self, root)
        return path

    def build_sankey(
        self,
        rows: Iterable[Row],
        *,
        weight: Union[float, Callable[[Row], float]] = 1.0,
        label_fmt: str = '{name}: {value}',
        edge_attr: str = 'value',
    ) -> SankeyDiGraph:
        """
        Build a DiGraph where edges carry aggregated flow in edge_attr.

        Example:
            >>> from magnet.utils.sankey import *  # NOQA
            >>> # Build a sankey graph and check aggregated edge weights
            >>> plan = Plan(
            ...     Root("ROOT_NODE"),
            ...     Group("dataset", "dataset"),
            ...     Split(
            ...         "status", "status",
            ...         branches={
            ...             "ok": Plan(Bucket("quality", "quality")),
            ...             "fail": Plan(Bucket("reason", "reason")),
            ...         },
            ...     ),
            ... )
            >>> rows = [
            ...     {"dataset": "A", "status": "ok", "quality": "good"},
            ...     {"dataset": "A", "status": "ok", "quality": "good"},
            ...     {"dataset": "A", "status": "ok", "quality": "bad"},
            ...     {"dataset": "B", "status": "fail", "reason": "timeout"},
            ... ]
            >>> G = plan.build_sankey(rows)
            >>> nx.write_network_text(G)
            ╙── ROOT_NODE
                ├─╼ dataset: A
                │   └─╼ status: ok
                │       ├─╼ quality: good
                │       └─╼ quality: bad
                └─╼ dataset: B
                    └─╼ status: fail
                        └─╼ reason: timeout
        """
        G = SankeyDiGraph(edge_attr=edge_attr)
        weight_fn = weight if callable(weight) else (lambda r: float(weight))

        def add_edge(u, v, w):
            if G.has_edge(u, v):
                G[u][v][edge_attr] += w
            else:
                G.add_edge(u, v, **{edge_attr: w})

        for row in rows:
            w = weight_fn(row)
            path = self.trace(row, label_fmt=label_fmt)
            for a, b in zip(path, path[1:]):
                add_edge(a, b, w)

        return G


class SankeyDiGraph(nx.DiGraph):
    """
    A DiGraph with convenience methods for Sankey rendering / exporting.

    Notes:
        - Flow is stored on edges in `edge_attr` (default: "value")
        - Plotly node ordering is topological when possible, otherwise insertion order.
    """

    def __init__(self, *args, edge_attr: str = 'value', **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_attr = edge_attr

    @classmethod
    def demo(cls, n=200, seed=0) -> SankeyDiGraph:
        """
        Demodata for tests
        """
        import random

        r = random.Random(seed)

        rows = [
            dict(
                dataset=r.choice(['coco', 'openimages', 'cityscapes']),
                backend=r.choice(['cuda', 'cpu']),
                status=('fail' if r.random() < 0.15 else 'ok'),
            )
            for _ in range(n)
        ]
        for row in rows:
            row['reason'] = (
                r.choice(['oom', 'timeout'])
                if row['status'] == 'fail'
                else None
            )

        plan = Plan(
            Root('All Runs'),
            Group('dataset', 'dataset'),
            Split(
                'status',
                'status',
                branches={
                    'ok': Plan(Group('backend', 'backend')),
                    'fail': Plan(
                        Bucket('reason', 'reason'), Group('backend', 'backend')
                    ),
                },
            ),
        )
        self = plan.build_sankey(rows)
        return self

    # ---- light reporting helpers (optional, but nice) ----

    def summarize(
        self,
        *,
        edge_attr: Optional[str] = None,
        max_edges: Optional[int] = 200,
        sort: str = 'value_desc',
    ) -> str:
        """
        Like Plan.graph_to_text, but bound to the graph.

        Example:
            >>> # xdoctest: +REQUIRES(module:plotly)
            >>> import plotly
            >>> from magnet.utils.sankey import *  # NOQA
            >>> self = SankeyDiGraph.demo()
            >>> print(self.summarize())
            Nodes: 10  Edges: 17
            ...
            Top nodes by outflow/inflow:
              All Runs  out=200 in=0
              status: ok  out=171 in=171
              dataset: cityscapes  out=71 in=71
              ...
            Edges:
              status: ok  ->  backend: cuda   value=94
              status: ok  ->  backend: cpu   value=77
              All Runs  ->  dataset: cityscapes   value=71
              ...
        """
        edge_attr = edge_attr or self.edge_attr
        lines: List[str] = []
        lines.append(
            f'Nodes: {self.number_of_nodes()}  Edges: {self.number_of_edges()}'
        )
        lines.append('')

        def outflow(n):
            return sum(self[n][v].get(edge_attr, 0) for v in self.successors(n))

        def inflow(n):
            return sum(
                self[u][n].get(edge_attr, 0) for u in self.predecessors(n)
            )

        nodes_sorted = sorted(
            self.nodes, key=lambda n: (outflow(n), inflow(n)), reverse=True
        )
        lines.append('Top nodes by outflow/inflow:')
        for n in nodes_sorted[:20]:
            lines.append(f'  {n}  out={outflow(n):g} in={inflow(n):g}')
        lines.append('')

        edges = [(u, v, self[u][v].get(edge_attr, 0)) for u, v in self.edges]
        if sort == 'value_desc':
            edges.sort(key=lambda t: t[2], reverse=True)
        elif sort == 'lex':
            edges.sort(key=lambda t: (str(t[0]), str(t[1])))

        lines.append('Edges:')
        shown = edges if max_edges is None else edges[:max_edges]
        for u, v, val in shown:
            lines.append(f'  {u}  ->  {v}   {edge_attr}={val:g}')
        if max_edges is not None and len(edges) > max_edges:
            lines.append(f'... ({len(edges) - max_edges} more edges)')
        return '\n'.join(lines)

    # ---- core conversions ----

    def _to_sankey_data(
        self,
    ) -> tuple[List[Any], List[int], List[int], List[float]]:
        """
        Convert into (nodes, source, target, value) for Plotly Sankey.

        Example:
            >>> # Convert nx graph to plotly sankey arrays
            >>> from magnet.utils.sankey import *  # NOQA
            >>> import networkx as nx
            >>> G = SankeyDiGraph()
            >>> G.add_edge("A", "B", value=2)
            >>> G.add_edge("A", "C", value=3)
            >>> nodes, source, target, value = G._to_sankey_data()
            >>> set(nodes) == {"A", "B", "C"}
            True
            >>> len(source) == len(target) == len(value) == 2
            True
            >>> sorted(value)
            [2.0, 3.0]
        """
        try:
            nodes = list(nx.topological_sort(self))
        except nx.NetworkXUnfeasible:
            nodes = list(self.nodes)

        idx = {n: i for i, n in enumerate(nodes)}
        source: List[int] = []
        target: List[int] = []
        value: List[float] = []

        for u, v, data in self.edges(data=True):
            source.append(idx[u])
            target.append(idx[v])
            value.append(float(data.get(self.edge_attr, 0)))

        return nodes, source, target, value

    def to_plotly(self, *, title: str = 'Sankey') -> PlotlyFigureLike:
        """
        Build a publishable Plotly Sankey figure.

        Example:
            >>> # xdoctest: +REQUIRES(module:plotly)
            >>> import plotly
            >>> from magnet.utils.sankey import *  # NOQA
            >>> G = SankeyDiGraph.demo(n=20)
            >>> fig = G.to_plotly(title='Demo')
            >>> assert fig.layout.title.text == 'Demo'
            >>> # xdoctest: +REQUIRES(module:kaleido)
            >>> # xdoctest: +REQUIRES(module:kwplot)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> import tempfile
            >>> import os
            >>> with tempfile.TemporaryDirectory() as d:
            ...     fpath = os.path.join(d, "sankey_demo.png")
            ...     fig.write_image(fpath, scale=1)
            ...     assert os.path.exists(fpath)
            ...     kwplot.imshow(fpath)
        """
        import plotly.graph_objects as go

        nodes, source, target, value = self._to_sankey_data()
        fig = go.Figure(
            go.Sankey(
                node=dict(label=nodes, pad=15, thickness=18),
                link=dict(source=source, target=target, value=value),
            )
        )
        fig.update_layout(title_text=title, font_size=14)
        return fig


def demo():
    """
    """
    import kwutil
    import ubelt as ub
    rows = kwutil.Yaml.loads(ub.codeblock(
        '''
        - {run: run1, suite: suite1, retcode: 1, retmsg: 'unsupported', spec_diagnosis: null, metric_iou: null}
        - {run: run2, suite: suite1, retcode: 1, retmsg: 'unsupported', spec_diagnosis: null, metric_iou: null}
        - {run: run3, suite: suite1, retcode: 1, retmsg: 'unsupported', spec_diagnosis: null, metric_iou: null}
        - {run: run4, suite: suite1, retcode: 1, retmsg: 'unsupported', spec_diagnosis: null, metric_iou: null}
        - {run: run5, suite: suite1, retcode: 1, retmsg: 'unsupported', spec_diagnosis: null, metric_iou: null}

        - {run: run1, suite: suite2, retcode: 0, retmsg: '', spec_diagnosis: 'agree', metric_iou: 1.0}
        - {run: run2, suite: suite2, retcode: 0, retmsg: '', spec_diagnosis: 'agree', metric_iou: 1.0}
        - {run: run3, suite: suite2, retcode: 0, retmsg: '', spec_diagnosis: 'agree', metric_iou: 1.0}
        - {run: run4, suite: suite2, retcode: 0, retmsg: '', spec_diagnosis: 'agree', metric_iou: 1.0}
        - {run: run5, suite: suite2, retcode: 0, retmsg: '', spec_diagnosis: 'agree', metric_iou: 1.0}

        - {run: run1, suite: suite3, retcode: 0, retmsg: '', spec_diagnosis: 'agree', metric_iou: 0.9}
        - {run: run2, suite: suite3, retcode: 0, retmsg: '', spec_diagnosis: 'agree', metric_iou: 0.7}
        - {run: run3, suite: suite3, retcode: 0, retmsg: '', spec_diagnosis: 'agree', metric_iou: 0.9}
        - {run: run4, suite: suite3, retcode: 0, retmsg: '', spec_diagnosis: 'agree', metric_iou: 1.0}
        - {run: run5, suite: suite3, retcode: 1, retmsg: 'oom', spec_diagnosis: null, metric_iou: null}
        - {run: run6, suite: suite3, retcode: 1, retmsg: 'oom', spec_diagnosis: null, metric_iou: null}

        - {run: run1, suite: suite4, retcode: 0, retmsg: '', spec_diagnosis: 'disagree-deploy', metric_iou: 0.0}
        - {run: run2, suite: suite4, retcode: 0, retmsg: '', spec_diagnosis: 'disagree-deploy', metric_iou: 0.0}
        - {run: run3, suite: suite4, retcode: 0, retmsg: '', spec_diagnosis: 'disagree-input', metric_iou: 0.3}
        - {run: run4, suite: suite4, retcode: 0, retmsg: '', spec_diagnosis: 'disagree-input', metric_iou: 0.1}
        '''))

    from magnet.utils import sankey_builder
    root = sankey_builder.Root(label="All Attempts")
    bench = root.group(by="suite", name="benchmark")

    splits = bench.group(by="retcode", name="Ran")
    splits[1].label = 'Failed'
    splits[0].label = 'Ran'

    # Do something unique on the fail branch
    splits[1].group(by='retmsg')

    def iou_grouper(row):
        value = row['metric_iou']
        if value == 1:
            return '1'
        elif value > 0.5:
            return '0.5 - 1'
        elif value > 0:
            return '0 - 0.5'
        else:
            return '0'

    diagnosis_buckets = splits[0].group(by="agreement")
    iou_groups = diagnosis_buckets.group(by=iou_grouper)

    # Connect only some of the underlying buckets
    iou_groups['0.5 - 1'].connect('Priority Analysis')
    iou_groups['0 - 0.5'].connect('Priority Analysis')

    graph = root.build_sankey(rows)
    print(root.to_text())
    import networkx as nx
    nx.write_network_text(graph)
