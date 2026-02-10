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
    branches: Dict[Any, 'Plan']
    default: Optional['Plan'] = None


def _eval_by(by: Grouper, row: Row):
    return by(row) if callable(by) else row.get(by)


def _by_repr(by: Grouper) -> str:
    if isinstance(by, str):
        return repr(by)
    name = getattr(by, '__name__', None)
    if name:
        return f'<fn {name}>'
    return f'<callable {by!r}>'


def _label(stage: str, value: Any, fmt: str):
    return fmt.format(name=stage, value=value)


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
                    lines.append(f'{indent}GROUP {st.name!r} by={_by_repr(st.by)}')
                elif isinstance(st, Bucket):
                    lines.append(f'{indent}BUCKET {st.name!r} by={_by_repr(st.by)}')
                elif isinstance(st, Split):
                    lines.append(f'{indent}SPLIT {st.name!r} by={_by_repr(st.by)}')
                    for k, sub in st.branches.items():
                        lines.append(f'{indent}  BRANCH {k!r}:')
                        rec(sub, indent + '    ')
                    if st.default is not None:
                        lines.append(f'{indent}  DEFAULT:')
                        rec(st.default, indent + '    ')
                else:
                    lines.append(f'{indent}{type(st).__name__} (?)')

        rec(self, '')
        return '\n'.join(lines)

    def trace(self, row: Row, *, label_fmt='{name}: {value}') -> List[str]:
        """
        Return the node labels (path) this row takes.

        Example:
            >>> # Trace the path a row takes through the plan
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

        def run(plan: 'Plan', cur: str):
            node = cur
            for st in plan.steps:
                if isinstance(st, Root):
                    continue
                elif isinstance(st, (Group, Bucket)):
                    val = _eval_by(st.by, row)
                    nxt = _label(st.name, val, label_fmt)
                    path.append(nxt)
                    node = nxt
                elif isinstance(st, Split):
                    key = _eval_by(st.by, row)
                    split_node = _label(st.name, key, label_fmt)
                    path.append(split_node)
                    node = split_node
                    branch = st.branches.get(key) or st.default
                    if branch is None:
                        return node
                    node = run(branch, node)
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
            >>> from magnet.backends.helm.rundiff.sankey import *  # NOQA
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
                dataset=r.choice(["coco", "openimages", "cityscapes"]),
                backend=r.choice(["cuda", "cpu"]),
                status=("fail" if r.random() < 0.15 else "ok"),
            )
            for _ in range(n)
        ]
        for row in rows:
            row["reason"] = (r.choice(["oom", "timeout"]) if row["status"] == "fail" else None)

        plan = Plan(
            Root("All Runs"),
            Group("dataset", "dataset"),
            Split(
                "status", "status",
                branches={
                    "ok": Plan(Group("backend", "backend")),
                    "fail": Plan(Bucket("reason", "reason"), Group("backend", "backend")),
                },
            ),
        )
        self = plan.build_sankey(rows)
        return self

    # ---- core conversions ----

    def _to_sankey_data(self) -> Tuple[List[Any], List[int], List[int], List[float]]:
        """
        Convert into (nodes, source, target, value) for Plotly Sankey.

        Example:
            >>> # Convert nx graph to plotly sankey arrays
            >>> from magnet.backends.helm.rundiff.sankey import *  # NOQA
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
            >>> from magnet.backends.helm.rundiff.sankey import *  # NOQA
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

    def write_plotly_figure(
        self,
        out_prefix: str,
        *,
        title: str = 'Sankey',
        scale: int = 2,
        formats: Tuple[str, ...] = ('png', 'pdf', 'svg', 'jpg'),
    ) -> go.Figure:
        """
        Render to plotly and export images. Requires kaleido for export.
        Returns the figure as well.
        """
        fig = self.to_plotly(title=title)
        for ext in formats:
            fig.write_image(f'{out_prefix}.{ext}', scale=scale)
        return fig

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
            >>> import plotly
            >>> from magnet.backends.helm.rundiff.sankey import *  # NOQA
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
            return sum(self[u][n].get(edge_attr, 0) for u in self.predecessors(n))

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
