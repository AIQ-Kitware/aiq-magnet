"""
sankey_builder.py

Fluent Sankey spec builder (mutable DAG), focused on:
- Nodes: Root / Constant / Group
- Group becomes a "split" when cases are configured
- connect() coerces strings to Constant nodes
- trace(row) for debugging
- trace_batch(rows) for fast aggregated execution
- build_sankey(rows) to produce a networkx graph with node+edge flow totals
- to_text() for readable structural representation

Semantics:
- Each row follows ONE path through the spec (row -> labels).
- The sankey graph is the union of these paths (fan-out and fan-in supported).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import itertools
import networkx as nx

Row = Dict[str, Any]
By = Union[str, Callable[[Row], Any]]


def _eval(by: By, row: Row) -> Any:
    return by(row) if callable(by) else row.get(by)


def _by_repr(by: By) -> str:
    if isinstance(by, str):
        return repr(by)
    name = getattr(by, "__name__", None)
    if name:
        return f"<fn {name}>"
    return f"<callable {by!r}>"


def _coerce_target(target: Union["Node", str, None]) -> Optional["Node"]:
    """
    Coerce a connect() target:
    - Node -> itself
    - str  -> new Constant(label=str)
    - None -> None (terminal)
    """
    if target is None:
        return None
    if isinstance(target, Node):
        return target
    if isinstance(target, str):
        return Constant(label=target)
    raise TypeError(f"Cannot connect to target of type {type(target)}")


BatchItem = Tuple[Row, float]
NodeFlow = Dict[str, float]
EdgeFlow = Dict[Tuple[str, str], float]


@dataclass
class BatchTraceResult:
    node_flow: NodeFlow
    edge_flow: EdgeFlow


@dataclass
class Case:
    """
    Case config for a parent Group.

    - label: override emitted label for that key
    - then: override continuation for that key
    """
    label: Optional[str] = None
    then: Optional["Node"] = None

    def group(self, *, by: By, name: Optional[str] = None) -> "Group":
        g = Group(name=name or (by if isinstance(by, str) else "group"), by=by)
        if self.then is not None:
            raise RuntimeError("This case already has a `.then` continuation")
        self.then = g
        return g

    def connect(self, target: Union["Node", str, None]) -> Optional["Node"]:
        self.then = _coerce_target(target)
        return self.then


@dataclass
class CaseSet:
    """
    Handle for bulk operations on multiple cases.
    """
    cases: List[Case]

    def connect(self, target: Union["Node", str, None]) -> Optional["Node"]:
        # If target is a string, create ONE Constant and share it.
        tgt = _coerce_target(target)
        for c in self.cases:
            c.then = tgt
        return tgt

    def set_label(self, label: str) -> "CaseSet":
        for c in self.cases:
            c.label = label
        return self


@dataclass
class Node:
    """
    Base node in the spec.

    Every Node:
    - emits exactly one label per row
    - chooses a continuation Node (or terminates)

    `next` is the fallback continuation.
    """
    next: Optional["Node"] = None

    # ---- fluent building ----

    def group(self, *, by: By, name: Optional[str] = None) -> "Group":
        g = Group(name=name or (by if isinstance(by, str) else "group"), by=by)
        if self.next is not None:
            raise RuntimeError("This node already has a `.next` continuation")
        self.next = g
        return g

    def connect(self, target: Union["Node", str, None]) -> Optional["Node"]:
        self.next = _coerce_target(target)
        return self.next

    # ---- debug execution ----

    def trace(self, row: Row, *, label_fmt: str = "{name}: {value}", max_steps: int = 10_000) -> List[str]:
        """
        Debug trace: realize the label path for a single row.
        """
        labels: List[str] = []
        cur: Optional[Node] = self
        visited: set[int] = set()
        steps = 0
        while cur is not None:
            steps += 1
            if steps > max_steps:
                raise RuntimeError(f"Exceeded max_steps={max_steps}; possible cycle")
            oid = id(cur)
            if oid in visited:
                raise RuntimeError("Cycle detected while tracing (node revisited)")
            visited.add(oid)

            label, cur = cur._step_row(row, label_fmt=label_fmt)
            labels.append(label)
        return labels

    # ---- fast batch execution ----

    def trace_batch(
        self,
        rows: Iterable[Row],
        *,
        weight: Union[float, Callable[[Row], float]] = 1.0,
        label_fmt: str = "{name}: {value}",
        max_depth: int = 10_000,
    ) -> BatchTraceResult:
        """
        Fast batch tracing: aggregate node/edge flows without tracing each row into a full path.

        Returns:
            BatchTraceResult(node_flow, edge_flow)
        """
        weight_fn = weight if callable(weight) else (lambda r: float(weight))
        batch: List[BatchItem] = [(r, float(weight_fn(r))) for r in rows]

        node_flow: NodeFlow = {}
        edge_flow: EdgeFlow = {}

        def bump_node(label: str, wsum: float) -> None:
            node_flow[label] = node_flow.get(label, 0.0) + wsum

        def bump_edge(u: str, v: str, wsum: float) -> None:
            edge_flow[(u, v)] = edge_flow.get((u, v), 0.0) + wsum

        def walk(node: Node, in_batch: List[BatchItem], parent_label: Optional[str], stack: List[int]) -> None:
            if not in_batch:
                return
            if len(stack) > max_depth:
                raise RuntimeError(f"Exceeded max_depth={max_depth}; possible cycle")

            oid = id(node)
            if oid in stack:
                raise RuntimeError("Cycle detected in spec while batch-tracing")
            stack.append(oid)
            try:
                for label, out_batch, wsum, nxt in node._step_batch(in_batch, label_fmt=label_fmt):
                    bump_node(label, wsum)
                    if parent_label is not None:
                        bump_edge(parent_label, label, wsum)
                    if nxt is not None:
                        walk(nxt, out_batch, label, stack)
            finally:
                stack.pop()

        walk(self, batch, parent_label=None, stack=[])
        return BatchTraceResult(node_flow=node_flow, edge_flow=edge_flow)

    def build_sankey(
        self,
        rows: Iterable[Row],
        *,
        weight: Union[float, Callable[[Row], float]] = 1.0,
        edge_attr: str = "value",
        node_attr: str = "count",
        label_fmt: str = "{name}: {value}",
    ) -> SankeyDiGraph:
        """
        Build a SankeyDiGraph using batch tracing.
        """
        res = self.trace_batch(rows, weight=weight, label_fmt=label_fmt)
        G = SankeyDiGraph(edge_attr=edge_attr, node_attr=node_attr)

        for n, c in res.node_flow.items():
            G.add_node(n, **{node_attr: c})
        for (u, v), val in res.edge_flow.items():
            # ensure nodes exist even if someone mutates later
            if not G.has_node(u):
                G.add_node(u, **{node_attr: 0.0})
            if not G.has_node(v):
                G.add_node(v, **{node_attr: 0.0})
            G.add_edge(u, v, **{edge_attr: val})
        return G

    # ---- structure printing ----

    def to_text(self) -> str:
        dumper = _TextDumper()
        dumper.rec(self, "")
        return "\n".join(dumper.lines)

    # ---- subclass hooks ----

    def _step_row(self, row: Row, *, label_fmt: str) -> Tuple[str, Optional["Node"]]:
        raise NotImplementedError

    def _step_batch(
        self,
        batch: List[BatchItem],
        *,
        label_fmt: str,
    ) -> Iterator[Tuple[str, List[BatchItem], float, Optional["Node"]]]:
        """
        Yield (label, out_batch, weight_sum, next_node) for each outgoing route.
        """
        raise NotImplementedError

    def _text_head(self, nid: int) -> str:
        return f"{type(self).__name__}#{nid}"

    def _text_body(self, dumper: "_TextDumper", indent: str, nid: int) -> None:
        if self.next is not None:
            # dumper.lines.append(f"{indent}  NEXT:")
            dumper.rec(self.next, indent + "    ")


@dataclass
class Constant(Node):
    label: str = "CONST"

    def _step_row(self, row: Row, *, label_fmt: str) -> Tuple[str, Optional[Node]]:
        return self.label, self.next

    def _step_batch(
        self,
        batch: List[BatchItem],
        *,
        label_fmt: str,
    ) -> Iterator[Tuple[str, List[BatchItem], float, Optional[Node]]]:
        wsum = sum(w for _, w in batch)
        yield (self.label, batch, wsum, self.next)

    def _text_head(self, nid: int) -> str:
        return f"CONST#{nid} {self.label!r}"


@dataclass
class Root(Constant):
    def _text_head(self, nid: int) -> str:
        return f"ROOT#{nid} {self.label!r}"


@dataclass
class Group(Node):
    """
    A group computes value = by(row), emits a label, and optionally routes based on that value.

    - cases[value] can override:
        - label (Case.label)
        - next node (Case.then)
    - fallback continuation is Group.next
    """
    name: str = "group"
    by: By = ""
    cases: Dict[Any, Case] = field(default_factory=dict)
    default: Optional[Case] = None

    def __getitem__(self, key: Union[Any, Sequence[Any]]) -> Union[Case, CaseSet]:
        if isinstance(key, (list, tuple, set)):
            vals = list(key)
            return CaseSet([self.cases.setdefault(v, Case()) for v in vals])
        return self.cases.setdefault(key, Case())

    def set_default(self) -> Case:
        if self.default is None:
            self.default = Case()
        return self.default

    def _label_for(self, value: Any, case: Optional[Case], *, label_fmt: str) -> str:
        if case is not None and case.label is not None:
            return case.label
        return label_fmt.format(name=self.name, value=value)

    def _next_for(self, case: Optional[Case]) -> Optional[Node]:
        if case is not None and case.then is not None:
            return case.then
        return self.next

    def _step_row(self, row: Row, *, label_fmt: str) -> Tuple[str, Optional[Node]]:
        v = _eval(self.by, row)
        case = self.cases.get(v) or self.default
        label = self._label_for(v, case, label_fmt=label_fmt)
        nxt = self._next_for(case)
        return label, nxt

    def _step_batch(
        self,
        batch: List[BatchItem],
        *,
        label_fmt: str,
    ) -> Iterator[Tuple[str, List[BatchItem], float, Optional[Node]]]:
        # Partition incoming batch by computed value
        groups: Dict[Any, List[BatchItem]] = {}
        sums: Dict[Any, float] = {}
        for r, w in batch:
            v = _eval(self.by, r)
            groups.setdefault(v, []).append((r, w))
            sums[v] = sums.get(v, 0.0) + w

        for v, out_batch in groups.items():
            case = self.cases.get(v) or self.default
            label = self._label_for(v, case, label_fmt=label_fmt)
            nxt = self._next_for(case)
            yield (label, out_batch, sums[v], nxt)

    def _text_head(self, nid: int) -> str:
        return f"GROUP#{nid} {self.name!r} by={_by_repr(self.by)}"

    def _text_body(self, dumper: "_TextDumper", indent: str, nid: int) -> None:
        has_branches = bool(self.cases) or (self.default is not None)
        # Cases first
        if self.cases:
            for k in sorted(self.cases.keys(), key=lambda x: (str(type(x)), repr(x))):
                c = self.cases[k]
                extra = []
                if c.label is not None:
                    extra.append(f"label={c.label!r}")
                hdr = f"{indent}  CASE {k!r}"
                if extra:
                    hdr += " [" + ", ".join(extra) + "]"
                dumper.lines.append(hdr + ":")
                if c.then is not None:
                    dumper.rec(c.then, indent + "    ")
                else:
                    dumper.lines.append(f"{indent}    (fallthrough to NEXT)")

        if self.default is not None:
            c = self.default
            extra = []
            if c.label is not None:
                extra.append(f"label={c.label!r}")
            hdr = f"{indent}  DEFAULT"
            if extra:
                hdr += " [" + ", ".join(extra) + "]"
            dumper.lines.append(hdr + ":")
            if c.then is not None:
                dumper.rec(c.then, indent + "    ")
            else:
                dumper.lines.append(f"{indent}    (fallthrough to NEXT)")

        # Fallback continuation
        if self.next is not None:
            if has_branches:
                dumper.lines.append(f"{indent}  ELSE:")
                dumper.rec(self.next, indent + "    ")
            else:
                dumper.rec(self.next, indent + "    ")


class _TextDumper:
    """
    DAG-aware dumper used by Node.to_text().
    """
    def __init__(self) -> None:
        self.lines: List[str] = []
        self._ids: Dict[int, int] = {}
        self._expanded: set[int] = set()
        self._counter = itertools.count(1)

    def _get_id(self, obj: object) -> int:
        oid = id(obj)
        if oid not in self._ids:
            self._ids[oid] = next(self._counter)
        return self._ids[oid]

    def rec(self, node: Node, indent: str) -> None:
        nid = self._get_id(node)
        head = node._text_head(nid)
        self.lines.append(f"{indent}{head}")

        oid = id(node)
        if oid in self._expanded:
            self.lines[-1] += "  (ref)"
            return
        self._expanded.add(oid)

        node._text_body(self, indent, nid)


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
