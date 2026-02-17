"""magnet.backends.helm.helm_run_diff

Run-to-run comparison built on :class:`~magnet.backends.helm.helm_run_analysis.HelmRunAnalysis`.

Design goals
------------
* Keep the public API tight.
* Cache expensive computations.
* Provide both machine-friendly summaries (dict) and human-friendly reports
  (writer-style output using rich by default).

The diff intentionally leans on :class:`HelmRunAnalysis` for canonicalization
and indexing so both single-run and diff views agree on what a "stat" or
"instance" identity means.

CommandLine:
    xdoctest -m magnet.backends.helm.helm_run_diff __doc__

Example:
    >>> import ubelt as ub
    >>> import kwutil
    >>> from magnet.backends.helm.helm_outputs import HelmRun
    >>> from magnet.backends.helm.helm_run_diff import HelmRunDiff

    >>> run_a = HelmRun.demo()
    >>> dpath = ub.Path.appdir('magnet/tests/helm/helm_run_diff').delete().ensuredir()

    >>> # --- Case 1: identical copy -> perfect agreement -----------------
    >>> same_path = dpath / (run_a.path.name + '_same')
    >>> run_a.path.copy(same_path)
    >>> run_b = HelmRun(same_path)

    >>> rd = HelmRunDiff(run_a, run_b, a_name='orig', b_name='same')
    >>> info = rd.summary_dict(level=10)
    >>> assert info['run_spec_dict_ok'] is True
    >>> assert info['scenario_ok'] in {True, None}
    >>> assert info['value_agreement']['overall']['mismatched'] == 0
    >>> assert info['value_agreement']['overall']['agree_ratio'] == 1.0

    >>> line = rd.summary_text(level=0)
    >>> assert 'orig' in line and 'same' in line
    >>> rd.summary(level=1000)

    >>> # --- Case 2: perturb a single RUN-level stat mean ----------------
    >>> stats_path = dpath / (run_a.path.name + '_statsmod')
    >>> run_a.path.copy(stats_path)
    >>> stat_fpath = stats_path / 'stats.json'
    >>> stats = kwutil.Json.loads(stat_fpath.read_text())
    >>> old_mean = float(stats[0].get('mean', 0.0))
    >>> stats[0]['mean'] = old_mean + 1.23
    >>> stat_fpath.write_text(kwutil.Json.dumps(stats))

    >>> run_b2 = HelmRun(stats_path)
    >>> rd2 = HelmRunDiff(run_a, run_b2, a_name='orig', b_name='stats+1.23')
    >>> info2 = rd2.summary_dict(level=10)
    >>> assert info2['value_agreement']['overall']['mismatched'] >= 1
    >>> rd.summary(level=1000)

    >>> # --- Case 3: perturb ONE per-instance stat mean ------------------
    >>> inst_path = dpath / (run_a.path.name + '_perinstmod')
    >>> run_a.path.copy(inst_path)
    >>> pi_fpath = inst_path / 'per_instance_stats.json'
    >>> if pi_fpath.exists():
    ...     perinst = kwutil.Json.loads(pi_fpath.read_text())
    ...     # deterministically modify the first mean-bearing stat for the first entry
    ...     ei, sj = 0, None
    ...     for j, s in enumerate(perinst[ei]['stats']):
    ...         if int(s.get('count', 0) or 0) and ('mean' in s):
    ...             sj = j
    ...             break
    ...     assert sj is not None
    ...     old = float(perinst[ei]['stats'][sj]['mean'])
    ...     perinst[ei]['stats'][sj]['mean'] = old + 9.0
    ...     pi_fpath.write_text(kwutil.Json.dumps(perinst))
    ...     run_bi = HelmRun(inst_path)
    ...     rd_i = HelmRunDiff(run_a, run_bi, a_name='orig', b_name='perinst+9')
    ...     inst_info = rd_i.instance_summary_dict(top_n=5)
    ...     assert inst_info['means']['mismatched'] >= 1
    ...     rd_i.summary(level=1000)

    >>> # --- Case 4: run spec diff  ----------------
    >>> new_dpath = dpath / (run_a.path.name + '_runspec_mod')
    >>> run_a.path.copy(new_dpath)
    >>> spec_fpath = new_dpath / 'run_spec.json'
    >>> run_spec = kwutil.Json.loads(spec_fpath.read_text())
    >>> run_spec['adapter_spec']['model_deployment'] = 'someotherdeploy/gpt2'
    >>> spec_fpath.write_text(kwutil.Json.dumps(run_spec))
    >>> run_b4 = HelmRun(new_dpath)
    >>> rd = HelmRunDiff(run_a, run_b4, a_name='orig', b_name='runspec_mod')
    >>> rd.summary(level=1000)
    >>> info = rd.summary_dict(level=10)
    >>> assert not info['run_spec_dict_ok']

"""

from __future__ import annotations

import ubelt as ub

from dataclasses import dataclass
from magnet.backends.helm import helm_hashers
from magnet.backends.helm.helm_metrics import classify_metric
from magnet.backends.helm.helm_run_analysis import HelmRunAnalysis
from typing import Any, Callable, Iterable


def _format_bool(ok: bool) -> str:
    return '✅' if ok else '❌'


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _walker_diff(a: Any, b: Any, *, max_paths: int = 12) -> list[str]:
    """

    Return a dict with formatted lines for:
      - unique1: paths only in a
      - unique2: paths only in b
      - faillist: differing values at same path

    Each list is independently truncated to `max_paths`, with a final
    "<N more not shown>" line if needed.

    Example:
        >>> a = {'foo': {'bar': [1], 'baz': 1}}
        >>> b = {'foo': {'bar': [2], 'biz': 2}}
        >>> _walker_diff(a, b)

        >>> a = {
        >>>     "shared": {"same": 0, "chg": 1, "deep": {"x": 1}},
        >>>     "only_a_top": True,
        >>>     "only_a": {"k0": 0, "k1": 1, "k2": 2},
        >>>     "arr": [0, 1],
        >>> }
        >>> b = {
        >>>     "shared": {"same": 0, "chg": 2, "deep": {"x": 9, "y": 10}},
        >>>     "only_b_top": True,
        >>>     "only_b": {"j0": 0, "j1": 1, "j2": 2},
        >>>     "arr": [0, 2, 3],
        >>> }
        >>> _walker_diff(a, b)
    """
    walker_a = ub.IndexableWalker(a)
    walker_b = ub.IndexableWalker(b)
    info = walker_a.diff(walker_b)
    info.pop("passlist", None)

    def _format_path(path: Iterable[Any]) -> str:
        return ".".join(map(str, path))

    def _truncate(lines: list[str], max_items: int) -> list[str]:
        """
        If truncation happens, append ONE final line: "<N more not shown>"
        where N is the correct remainder.
        """
        if max_items is None or max_items <= 0:
            return lines
        n = len(lines)
        if n <= max_items:
            return lines
        remain = n - max_items
        return lines[:max_items] + [f"<{remain} more not shown>"]

    unique1 = sorted(info.get("unique1", []))
    unique2 = sorted(info.get("unique2", []))
    faillist = sorted(info.get("faillist", []), key=lambda d: d.path)

    out = info | {
        "unique1": _truncate([_format_path(p) + ': ' + _smart_truncate(repr(walker_a[p]), 80) for p in unique1], max_paths),
        "unique2": _truncate([_format_path(p) + ': ' + _smart_truncate(repr(walker_b[p]), 80) for p in unique2], max_paths),
        "faillist": _truncate(
            [f"{_format_path(d.path)}: {_smart_truncate(repr(d.value1), 80)} != {_smart_truncate(repr(d.value2), 80)}" for d in faillist],
            max_paths,
        ),
    }
    return out


def _default_writer(writer=None) -> Callable[[str], Any]:
    if writer is not None:
        return writer
    try:
        from rich import print as rich_print  # type: ignore
    except Exception:  # nocover
        return print
    else:
        return rich_print


def _escape_rich(text: str) -> str:
    """Escape rich markup (mainly brackets) without losing readability."""
    try:
        from rich.markup import escape  # type: ignore
    except Exception:  # nocover
        return text
    else:
        return escape(text)


def _sanitize_text(text: Any) -> str:
    if text is None:
        return ''
    s = str(text)
    # Drop most control chars except newlines/tabs.
    s = ''.join(
        (ch if (ch == '\n' or ch == '\t' or ord(ch) >= 32) else ' ') for ch in s
    )
    return s


def _smart_truncate(text: Any, max_chars: int) -> str:
    """Truncate long prompts/completions with a stable hash tail."""
    s = _sanitize_text(text)
    if max_chars <= 0:
        return _escape_rich(s)
    try:
        from kwutil.slugify_ext import smart_truncate  # type: ignore
    except Exception:  # nocover
        # fallback: hard truncate
        s2 = (s[:max_chars] + '…') if len(s) > max_chars else s
        return _escape_rich(s2)
    else:
        s2 = smart_truncate(
            s,
            max_length=max_chars,
            trunc_loc=0.5,
            hash_len=8,
            head='~',
            tail='~',
        )
        return _escape_rich(s2)


def _short_urepr(obj: Any, max_chars: int = 140) -> str:
    """Compact repr for diffs; keeps it readable and bounded."""
    try:
        s = ub.urepr(obj, nl=0, sv=1)
    except Exception:
        s = repr(obj)
    return _smart_truncate(s, max_chars)


@dataclass(frozen=True)
class Coverage:
    """Coverage bookkeeping for two key-sets."""

    n_a: int
    n_b: int
    n_isect: int
    n_union: int
    only_a: int
    only_b: int

    @classmethod
    def from_sets(cls, a: set[Any], b: set[Any]) -> 'Coverage':
        isect = a & b
        union = a | b
        return cls(
            n_a=len(a),
            n_b=len(b),
            n_isect=len(isect),
            n_union=len(union),
            only_a=len(a - b),
            only_b=len(b - a),
        )


def _fmt(x: Any) -> str:
    if x is None:
        return 'None'
    if isinstance(x, float):
        return f'{x:.4g}'
    return str(x)


class HelmRunDiff(ub.NiceRepr):
    """Compare two HELM runs.

    Parameters
    ----------
    run_a, run_b:
        Either :class:`HelmRunAnalysis` or a ``HelmRun`` reader (coerced).
    a_name, b_name:
        Human-friendly labels for reports.
    short_hash:
        Controls readability of hashed ids used in stat keys.
    """

    def __init__(
        self,
        run_a,
        run_b,
        *,
        a_name: str = 'A',
        b_name: str = 'B',
        short_hash: int = 16,
    ):
        self.a = (
            run_a
            if isinstance(run_a, HelmRunAnalysis)
            else HelmRunAnalysis(run_a, name=a_name)
        )
        self.b = (
            run_b
            if isinstance(run_b, HelmRunAnalysis)
            else HelmRunAnalysis(run_b, name=b_name)
        )
        self.a_name = a_name
        self.b_name = b_name
        self.short_hash = short_hash
        self._cache: dict[Any, Any] = {}

    def __nice__(self):
        return f'{self.a_name} vs {self.b_name}'

    # ---------------------------------------------------------------------
    # Base summaries

    def summary_dict(self, *, level: int = 10) -> dict[str, Any]:
        """Programmatic run-to-run summary.

        This is meant to be stable enough to power Sankey bucketing and
        higher-level dashboards.

        Key fields
        ----------
        run_spec_name_ok:
            Whether ``run_spec['name']`` matches.
        run_spec_dict_ok:
            Whether the entire run_spec.json matches (hash equality).
        scenario_ok:
            True/False if both scenario.json exist and match (hash equality),
            None if scenario is missing in one/both runs.
        stats_coverage_by_name:
            Coverage of stat names only (ignores count/values).
        stats_coverage_by_name_count:
            Coverage of stat name + count (still ignores values).
        value_agreement:
            Mean agreement on intersecting run-level stats, split by
            metric class (core/bookkeeping/untracked).

        Notes
        -----
        ``level`` mainly controls optional extras. For now, the dict always
        includes the L1 checks above.
        """
        cache_key = ('summary_dict', level)
        if cache_key in self._cache:
            return self._cache[cache_key]

        a_spec = self.a.run_spec() or {}
        b_spec = self.b.run_spec() or {}
        a_scen = self.a.scenario() or {}
        b_scen = self.b.scenario() or {}

        # 1) run spec name
        a_run_name = a_spec.get('name', None)
        b_run_name = b_spec.get('name', None)
        run_spec_name_ok = (a_run_name == b_run_name) and (
            a_run_name is not None
        )

        # 2) run spec dict hash
        spec_hash_a = helm_hashers.stable_hash36(
            helm_hashers.canonicalize_for_hashing(a_spec)
        )
        spec_hash_b = helm_hashers.stable_hash36(
            helm_hashers.canonicalize_for_hashing(b_spec)
        )
        run_spec_dict_ok = spec_hash_a == spec_hash_b
        if level == 0:
            spec_diff_paths = None
        else:
            spec_diff_paths = (
                {} if run_spec_dict_ok else _walker_diff(a_spec, b_spec)
            )

        # 3) scenario check with unknown semantics
        scen_known = bool(a_scen) and bool(b_scen)
        if not scen_known:
            scenario_ok: bool | None = None
            scenario_hash_a = None
            scenario_hash_b = None
            scen_diff_paths: list[str] = []
        else:
            scenario_hash_a = helm_hashers.stable_hash36(
                helm_hashers.canonicalize_for_hashing(a_scen)
            )
            scenario_hash_b = helm_hashers.stable_hash36(
                helm_hashers.canonicalize_for_hashing(b_scen)
            )
            scenario_ok = scenario_hash_a == scenario_hash_b
            if level == 0:
                scen_diff_paths = None
            else:
                scen_diff_paths = (
                    {} if scenario_ok else _walker_diff(a_scen, b_scen)
                )

        # 4/5) stats coverage
        a_stats = self.a.stats() or []
        b_stats = self.b.stats() or []
        a_name_keys = {
            helm_hashers.stat_key(
                s.get('name', None), short_hash=self.short_hash
            )
            for s in a_stats
        }
        b_name_keys = {
            helm_hashers.stat_key(
                s.get('name', None), short_hash=self.short_hash
            )
            for s in b_stats
        }
        cov_name = Coverage.from_sets(a_name_keys, b_name_keys)

        a_name_count_keys = {
            helm_hashers.stat_key(
                s.get('name', None),
                count=s.get('count', None),
                short_hash=self.short_hash,
            )
            for s in a_stats
        }
        b_name_count_keys = {
            helm_hashers.stat_key(
                s.get('name', None),
                count=s.get('count', None),
                short_hash=self.short_hash,
            )
            for s in b_stats
        }
        cov_name_count = Coverage.from_sets(
            a_name_count_keys, b_name_count_keys
        )

        # 6) value agreement (means) on intersecting keys
        value_summary = self._value_agreement_summary()

        out: dict[str, Any] = {
            'a': self._lite_run_dict(self.a),
            'b': self._lite_run_dict(self.b),
            'run_spec_name_ok': run_spec_name_ok,
            'run_spec_name_a': a_run_name,
            'run_spec_name_b': b_run_name,
            'run_spec_dict_ok': run_spec_dict_ok,
            'run_spec_hash_a': spec_hash_a,
            'run_spec_hash_b': spec_hash_b,
            'run_spec_diff_paths': spec_diff_paths,
            'scenario_ok': scenario_ok,
            'scenario_hash_a': scenario_hash_a,
            'scenario_hash_b': scenario_hash_b,
            'scenario_diff_paths': scen_diff_paths,
            'stats_coverage_by_name': cov_name.__dict__,
            'stats_coverage_by_name_count': cov_name_count.__dict__,
            'value_agreement': value_summary,
        }

        if level >= 20:
            # Optional: include instance-level summary in the dict.
            try:
                out['instance_value_agreement'] = self.instance_summary_dict(
                    top_n=10
                )
            except Exception as ex:  # nocover
                out['instance_value_agreement'] = {'error': repr(ex)}

        self._cache[cache_key] = out
        return out

    def summary_text(self, *, level: int = 0) -> str:
        """Return a text summary (built by calling :meth:`summary`)."""
        lines: list[str] = []
        self.summary(level=level, writer=lines.append)
        return '\n'.join(lines).rstrip()

    def summary(self, *, level: int = 10, writer=None) -> None:
        """Writer-style diff report.

        Levels
        ------
        * level <= 0: single line
        * level >= 10: one page
        * level >= 20: include top mismatches
        * level >= 30: include instance-level summary headline
        """
        writer = _default_writer(writer)
        info = self.summary_dict(level=level)

        ok = info['run_spec_dict_ok'] and (info['scenario_ok'] in {True, None})
        cov = info['stats_coverage_by_name']
        agree = info['value_agreement']['overall']['agree_ratio']

        if level <= 0:
            spec_name_a = info["run_spec_name_a"]
            spec_name_b = info["run_spec_name_b"]
            if spec_name_a == spec_name_b:
                line_name = spec_name_a
            else:
                line_name = '{spec_name_a} // {spec_name_b}'
            writer(
                f'{_format_bool(ok)} {self.a_name} vs {self.b_name} {line_name} '
                f'spec={_format_bool(info["run_spec_dict_ok"])} '
                f'stats={cov["n_isect"]}/{cov["n_union"]} '
                f'agree={agree:.3f}'
            )

        if level > 0:
            writer(f'HelmRunDiff: {self.a_name} vs {self.b_name}')

            # Side-by-side lite
            writer(
                f'  {self.a_name}: {self._analysis_summary_line(self.a, level=0)}'
            )
            writer(
                f'  {self.b_name}: {self._analysis_summary_line(self.b, level=0)}'
            )

            writer('')
            writer(
                f'Run spec name: {_format_bool(info["run_spec_name_ok"])}  '
                f'{info["run_spec_name_a"]}  vs  {info["run_spec_name_b"]}'
            )
            writer(
                f'Run spec dict: {_format_bool(info["run_spec_dict_ok"])}  '
                f'hashA={str(info["run_spec_hash_a"])[:10]}  hashB={str(info["run_spec_hash_b"])[:10]}'
            )

            if level >= 15:
                if (not info['run_spec_dict_ok']):
                    writer(
                        f'  diff: {ub.urepr(info["run_spec_diff_paths"])}'
                    )

            if info['scenario_ok'] is None:
                writer(
                    'Scenario: ⚠️  unknown (missing scenario.json in one or both runs)'
                )
            else:
                if level >= 15:
                    writer(f'Scenario: {_format_bool(bool(info["scenario_ok"]))}')
                    if (info['scenario_ok'] is False):
                        writer(
                            f'  diff: {ub.urepr(info["scenario_diff_paths"])}'
                        )

            writer('')
            cov2 = info['stats_coverage_by_name_count']
            writer('Stats coverage:')
            writer(
                f'  by name:       A={cov["n_a"]} B={cov["n_b"]} '
                f'isect={cov["n_isect"]} union={cov["n_union"]} onlyA={cov["only_a"]} onlyB={cov["only_b"]}'
            )
            writer(
                f'  by name+count: A={cov2["n_a"]} B={cov2["n_b"]} '
                f'isect={cov2["n_isect"]} union={cov2["n_union"]} onlyA={cov2["only_a"]} onlyB={cov2["only_b"]}'
            )

            writer('')
            writer('Value agreement (mean on intersecting run-level stats):')
            ov = info['value_agreement']['overall']
            writer(
                f'  overall: comparable={ov["comparable"]} mismatched={ov["mismatched"]} '
                f'agree_ratio={ov["agree_ratio"]:.3f}'
            )
            for cls in ('core', 'bookkeeping', 'untracked'):
                s = info['value_agreement']['by_class'][cls]
                writer(
                    f'  {cls:11s}: comparable={s["comparable"]} mismatched={s["mismatched"]} '
                    f'agree_ratio={s["agree_ratio"]:.3f}'
                )

            if level >= 20:
                top = info['value_agreement'].get('top_mismatches', [])
                if top:
                    writer('  top mismatches:')
                    for r in top:
                        writer(
                            f'    {r["key"]}  A={_fmt(r["a"])}  B={_fmt(r["b"])}  |Δ|={_fmt(r["abs_delta"])}'
                        )

            if level >= 30:
                writer('')
                try:
                    inst = self.instance_summary_dict(top_n=5)
                except Exception as ex:
                    writer(f'Instance-level diff: ⚠️  unable to compute: {ex!r}')
                else:
                    means = inst['means']
                    writer(
                        f'Instance-level means: comparable={means["comparable"]} mismatched={means["mismatched"]} '
                        f'agree={means["agree_ratio"]:.3f} (unpert={means["agree_ratio_unperturbed"]:.3f}, '
                        f'pert={means["agree_ratio_perturbed"]:.3f})'
                    )

    def _analysis_summary_line(
        self, ana: HelmRunAnalysis, *, level: int = 0
    ) -> str:
        """Best-effort one-liner per-run summary for side-by-side views."""
        if hasattr(ana, 'summary_text'):
            try:
                return ana.summary_text(level=level)  # type: ignore
            except Exception:
                pass
        if hasattr(ana, 'summary'):
            try:
                lines: list[str] = []
                ana.summary(level=level, writer=lines.append)  # type: ignore
                return ' '.join([ln.strip() for ln in lines if ln.strip()])
            except Exception:
                pass
        d = self._lite_run_dict(ana)
        name = d.get('run_spec_name', None)
        return str(name)

    def _lite_run_dict(self, ana: HelmRunAnalysis) -> dict[str, Any]:
        """Best-effort stable per-run dict used in diff summaries."""
        if hasattr(ana, 'summary_dict'):
            try:
                return ana.summary_dict(level=0)  # type: ignore
            except Exception:
                pass
        if hasattr(ana, 'summary_lite'):
            try:
                return ana.summary_lite()  # type: ignore
            except Exception:
                pass
        spec = ana.run_spec() or {}
        return {'run_spec_name': spec.get('name', None)}

    # ---------------------------------------------------------------------
    # Run-level mean agreement

    def _value_agreement_summary(
        self,
        *,
        abs_tol: float = 0.0,
        rel_tol: float = 0.0,
        top_n: int = 12,
    ) -> dict[str, Any]:
        """Compare mean values for intersecting run-level stats."""
        cache_key = (
            'value_agreement',
            abs_tol,
            rel_tol,
            top_n,
            self.short_hash,
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        idx_a = self.a.stat_index(
            drop_zero_count=True, require_mean=True, short_hash=self.short_hash
        )
        idx_b = self.b.stat_index(
            drop_zero_count=True, require_mean=True, short_hash=self.short_hash
        )
        keys = set(idx_a.keys()) & set(idx_b.keys())

        def agrees(x: float, y: float) -> bool:
            if abs_tol == 0.0 and rel_tol == 0.0:
                return x == y
            return abs(x - y) <= max(abs_tol, rel_tol * max(abs(x), abs(y)))

        by_class = {
            'core': {'comparable': 0, 'mismatched': 0},
            'bookkeeping': {'comparable': 0, 'mismatched': 0},
            'untracked': {'comparable': 0, 'mismatched': 0},
        }

        mismatches: list[dict[str, Any]] = []
        comparable = 0
        mismatched = 0
        for k in keys:
            a = idx_a[k]
            b = idx_b[k]
            if a.mean is None or b.mean is None:
                continue
            comparable += 1
            cls = a.metric_class
            by_class[cls]['comparable'] += 1
            if not agrees(a.mean, b.mean):
                mismatched += 1
                by_class[cls]['mismatched'] += 1
                mismatches.append(
                    {
                        'key': k,
                        'a': a.mean,
                        'b': b.mean,
                        'abs_delta': abs(a.mean - b.mean),
                    }
                )

        mismatches.sort(key=lambda r: r['abs_delta'], reverse=True)
        top = mismatches[:top_n]

        out = {
            'overall': {
                'comparable': comparable,
                'mismatched': mismatched,
                'agree_ratio': ratio(comparable, mismatched),
            },
            'by_class': {
                k: {
                    'comparable': v['comparable'],
                    'mismatched': v['mismatched'],
                    'agree_ratio': ratio(v['comparable'], v['mismatched']),
                }
                for k, v in by_class.items()
            },
            'top_mismatches': top,
        }

        self._cache[cache_key] = out
        return out

    # ---------------------------------------------------------------------
    # Instance-level agreement / drilldowns

    def instance_summary_dict(
        self,
        *,
        top_n: int = 10,
        abs_tol: float = 0.0,
        rel_tol: float = 0.0,
    ) -> dict[str, Any]:
        """Programmatic summary of per-instance stat agreement.

        This summarizes agreement on *mean* for joined per-instance stats.

        Returns
        -------
        dict with keys:

        * coverage: overlap on joined-row keys
        * means:
            - comparable: number of comparable rows (mean present in both)
            - mismatched: number of rows failing tolerance check
            - agree_ratio: 1 - mismatched / comparable
            - agree_ratio_unperturbed / agree_ratio_perturbed
        * top_mismatches_by_group:
            Mapping from ``(metric_class, metric_name)`` to a list of mismatch
            items (sorted by |Δ|), each containing:
                key, a, b, abs_delta, signed_delta

        Notes
        -----
        * Metric class is computed via :func:`classify_metric`.
        * "Perturbed" is determined by whether the joined key contains a
          non-None perturbation id / perturbation descriptor.
        """
        cache_key = (
            'instance_summary_dict',
            top_n,
            abs_tol,
            rel_tol,
            self.short_hash,
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        joined_a = self.a.joined_instance_stat_table(
            assert_assumptions=False, short_hash=self.short_hash
        )
        joined_b = self.b.joined_instance_stat_table(
            assert_assumptions=False, short_hash=self.short_hash
        )

        # Try to use the table's own key->row mapping if present
        map_a = getattr(joined_a, 'row_by_key', None)
        map_b = getattr(joined_b, 'row_by_key', None)

        def _iter_rows(joined) -> Iterable[Any]:
            if map_a is not None and joined is joined_a:
                return map_a.values()
            if map_b is not None and joined is joined_b:
                return map_b.values()
            if isinstance(joined, dict):
                return joined.values()
            if hasattr(joined, '__iter__'):
                return joined
            return []

        # Fallback: build row maps from iteration
        def _row_key(row: Any) -> Any:
            return (
                getattr(row, 'key', None)
                or getattr(row, 'stat_key', None)
                or getattr(row, 'row_key', None)
                or row
            )

        if map_a is None:
            map_a = {_row_key(r): r for r in _iter_rows(joined_a)}
        if map_b is None:
            map_b = {_row_key(r): r for r in _iter_rows(joined_b)}

        set_a = set(map_a)
        set_b = set(map_b)
        cov = Coverage.from_sets(set_a, set_b)

        def agrees(x: float, y: float) -> bool:
            if abs_tol == 0.0 and rel_tol == 0.0:
                return x == y
            return abs(x - y) <= max(abs_tol, rel_tol * max(abs(x), abs(y)))

        comparable = 0
        mismatched = 0
        # overall perturbed/unperturbed bookkeeping
        var_stats = {
            'unperturbed': {'comparable': 0, 'mismatched': 0},
            'perturbed': {'comparable': 0, 'mismatched': 0},
        }

        grouped: dict[tuple[str, str | None], list[dict[str, Any]]] = {}

        for k in set_a & set_b:
            ra = map_a[k]
            rb = map_b[k]
            sa = (
                getattr(ra, 'stat', None)
                if hasattr(ra, 'stat')
                else (ra.get('stat', None) if isinstance(ra, dict) else None)
            )
            sb = (
                getattr(rb, 'stat', None)
                if hasattr(rb, 'stat')
                else (rb.get('stat', None) if isinstance(rb, dict) else None)
            )

            ma = _safe_float(
                (sa or {}).get('mean', None)
                if isinstance(sa, dict)
                else getattr(sa, 'mean', None)
            )
            mb = _safe_float(
                (sb or {}).get('mean', None)
                if isinstance(sb, dict)
                else getattr(sb, 'mean', None)
            )
            ca = (
                int((sa or {}).get('count', 0) or 0)
                if isinstance(sa, dict)
                else int(getattr(sa, 'count', 0) or 0)
            )
            cb = (
                int((sb or {}).get('count', 0) or 0)
                if isinstance(sb, dict)
                else int(getattr(sb, 'count', 0) or 0)
            )

            # Only compare mean-bearing rows with support
            if ma is None or mb is None:
                continue
            if ca == 0 or cb == 0:
                continue

            comparable += 1

            # Determine metric name
            name_obj = (
                (sa or {}).get('name', None)
                if isinstance(sa, dict)
                else getattr(sa, 'name_obj', None)
            )
            metric = (
                name_obj.get('name', None)
                if isinstance(name_obj, dict)
                else None
            )
            if metric is None and sa is not None and not isinstance(sa, dict):
                metric = getattr(sa, 'metric', None)
            metric_class, _ = classify_metric(metric)
            gkey = (metric_class, metric)

            # Determine perturbed vs unperturbed (best-effort)
            perturbed = False
            if hasattr(k, 'perturbation_id'):
                perturbed = getattr(k, 'perturbation_id', None) is not None
            elif isinstance(k, tuple) and len(k) >= 3:
                # historical tuple layout: (instance_id, tti, perturbation_id, ...)
                perturbed = k[2] is not None
            variant = 'perturbed' if perturbed else 'unperturbed'
            var_stats[variant]['comparable'] += 1

            if not agrees(ma, mb):
                mismatched += 1
                var_stats[variant]['mismatched'] += 1
                item = {
                    'key': k,
                    'a': ma,
                    'b': mb,
                    'abs_delta': abs(ma - mb),
                    'signed_delta': (mb - ma),
                }
                grouped.setdefault(gkey, []).append(item)

        # Sort each group and cap
        for gk, items in grouped.items():
            items.sort(key=lambda r: r['abs_delta'], reverse=True)
            grouped[gk] = items[:top_n]

        means = {
            'comparable': comparable,
            'mismatched': mismatched,
            'agree_ratio': ratio(comparable, mismatched),
            'agree_ratio_unperturbed': ratio(
                var_stats['unperturbed']['comparable'],
                var_stats['unperturbed']['mismatched'],
            ),
            'agree_ratio_perturbed': ratio(
                var_stats['perturbed']['comparable'],
                var_stats['perturbed']['mismatched'],
            ),
        }

        out = {
            'coverage': cov.__dict__,
            'means': means,
            'top_mismatches_by_group': grouped,
        }

        self._cache[cache_key] = out
        return out

    def summarize_instances(
        self,
        *,
        level: int = 10,
        top_n: int = 5,
        show_details: int = 5,
        prompt_chars: int = 220,
        completion_chars: int = 140,
        input_chars: int = 200,
        diff_max_items: int = 7,
        writer=None,
    ) -> None:
        """Writer-style instance-level report.

        This prints:
        * coverage + agreement ratios
        * top mismatches for core metrics and for bookkeeping metrics
        * for the first `show_details` mismatches per (metric_class, metric)
          group: prompt/input/completion excerpts and a compact request_state diff.

        The large texts are smart-truncated (hash preserved) and escaped so
        rich doesn't interpret markup.
        """
        writer = _default_writer(writer)
        info = self.instance_summary_dict(top_n=top_n)
        cov = info['coverage']
        means = info['means']

        writer(f'Instance-level diff: {self.a_name} vs {self.b_name}')
        writer(
            f'  coverage: A={cov["n_a"]} B={cov["n_b"]} isect={cov["n_isect"]} '
            f'union={cov["n_union"]} onlyA={cov["only_a"]} onlyB={cov["only_b"]}'
        )
        writer(
            f'  means: comparable={means["comparable"]} mismatched={means["mismatched"]} '
            f'agree_ratio={means["agree_ratio"]:.3f} (unpert={means["agree_ratio_unperturbed"]:.3f}, '
            f'pert={means["agree_ratio_perturbed"]:.3f})'
        )

        grouped: dict[tuple[str, str | None], list[dict[str, Any]]] = (
            info.get('top_mismatches_by_group', {}) or {}
        )

        # Choose groups to show: core first, bookkeeping second
        def _group_rank(
            item: tuple[tuple[str, str | None], list[dict[str, Any]]],
        ) -> tuple[int, float]:
            (cls, _metric), items = item
            cls_rank = {'core': 0, 'bookkeeping': 1, 'untracked': 2}.get(cls, 9)
            max_abs = items[0]['abs_delta'] if items else 0.0
            return (cls_rank, -max_abs)

        groups_sorted = sorted(grouped.items(), key=_group_rank)

        # Decide which metric classes are eligible at this level
        allowed_classes = {'core'}
        if level >= 20:
            allowed_classes |= {'untracked'}
        if level >= 30:
            allowed_classes |= {'bookkeeping'}

        # Filter groups by allowed class
        filtered = [g for g in groups_sorted if g[0][0] in allowed_classes]

        # If we filtered everything out (e.g. no core diffs), fall back to showing *something*
        if not filtered and groups_sorted:
            # Prefer untracked, then bookkeeping, then whatever exists
            pref_order = ['untracked', 'bookkeeping', 'core']
            for cls in pref_order:
                filtered = [g for g in groups_sorted if g[0][0] == cls]
                if filtered:
                    break
            if not filtered:
                filtered = groups_sorted

        groups_sorted = filtered

        # If level is low, avoid dumping too many groups
        max_groups = None
        if level < 20:
            max_groups = 6
        if max_groups is not None:
            # Ensure we show at least some core and some bookkeeping groups when possible.
            core = [g for g in groups_sorted if g[0][0] == 'core']
            book = [g for g in groups_sorted if g[0][0] == 'bookkeeping']
            other = [
                g
                for g in groups_sorted
                if g[0][0] not in {'core', 'bookkeeping'}
            ]
            keep = []
            keep.extend(core[: max_groups // 2])
            keep.extend(book[: max_groups - len(keep)])
            if len(keep) < max_groups:
                keep.extend(other[: max_groups - len(keep)])
            # Preserve original ordering among kept groups
            keep_set = {id(x) for x in keep}
            groups_sorted = [g for g in groups_sorted if id(g) in keep_set]

        if groups_sorted:
            writer('  top mismatches:')

        # Build joined lookup tables for details
        A_join = None
        B_join = None
        A_map = None
        B_map = None

        if show_details and level >= 10:
            A_join = self.a.joined_instance_stat_table(
                assert_assumptions=False, short_hash=self.short_hash
            )
            B_join = self.b.joined_instance_stat_table(
                assert_assumptions=False, short_hash=self.short_hash
            )
            A_map = getattr(A_join, 'row_by_key', None)
            B_map = getattr(B_join, 'row_by_key', None)

        for (cls, metric), items in groups_sorted:
            writer(f'  [bold]top mismatches ({(cls, metric)!r}):[/bold]')
            for rank, item in enumerate(items[:top_n], start=1):
                k = item['key']
                a = float(item['a'])
                b = float(item['b'])
                abs_d = float(item['abs_delta'])
                signed_d = float(item['signed_delta'])

                # Try to extract split/sub_split info from key if it is tuple-like
                split = None
                if isinstance(k, tuple) and len(k) >= 5:
                    # (id, tti, pert_id, metric, split, ...)
                    split = k[4]

                metric_label = (
                    metric if metric is not None else 'unknown_metric'
                )
                if split is not None:
                    metric_label = f'{metric_label}, split={split}'

                writer(f'   {rank:2d}. metric: {metric_label}')
                writer(f'      key: {k}')
                writer(
                    f'      A={_fmt(a)}  B={_fmt(b)}  Δ(B-A)={_fmt(signed_d)}  |Δ|={_fmt(abs_d)}'
                )

                if (
                    show_details
                    and level >= 10
                    and rank <= show_details
                    and A_map is not None
                    and B_map is not None
                ):
                    ra = A_map.get(k, None)
                    rb = B_map.get(k, None)

                    rs_a = (
                        getattr(ra, 'request_state', None)
                        if ra is not None
                        else None
                    )
                    rs_b = (
                        getattr(rb, 'request_state', None)
                        if rb is not None
                        else None
                    )
                    if rs_a is None and isinstance(ra, dict):
                        rs_a = ra.get('request_state', None)
                    if rs_b is None and isinstance(rb, dict):
                        rs_b = rb.get('request_state', None)

                    # important: use repr, to avoid rendering newline chars.
                    pa = (
                        _smart_truncate(
                            repr(((rs_a or {}).get('request') or {}).get(
                                'prompt', None
                            )),
                            prompt_chars,
                        )
                        if isinstance(rs_a, dict)
                        else ''
                    )
                    pb = (
                        _smart_truncate(
                            repr(((rs_b or {}).get('request') or {}).get(
                                'prompt', None
                            )),
                            prompt_chars,
                        )
                        if isinstance(rs_b, dict)
                        else ''
                    )
                    prompts_equal = pa == pb
                    writer(f'      prompts_equal={prompts_equal}')

                    def _inst_input(rs: Any) -> str:
                        if not isinstance(rs, dict):
                            return ''
                        inst = rs.get('instance') or {}
                        inp = inst.get('input') or {}
                        if isinstance(inp, dict) and 'text' in inp:
                            return _smart_truncate(
                                repr(inp.get('text', None)), input_chars
                            )
                        # important: use repr, to avoid rendering newline chars.
                        return _smart_truncate(repr(inp), input_chars)

                    def _completion(rs: Any) -> str:
                        if not isinstance(rs, dict):
                            return ''
                        comps = (rs.get('result') or {}).get(
                            'completions'
                        ) or []
                        txt = comps[0].get('text', None) if comps else None
                        # important: use repr, to avoid rendering newline chars.
                        return _smart_truncate(repr(txt), completion_chars)

                    # --- Inputs / completions (avoid duplicates) ---
                    input_a = _inst_input(rs_a) if rs_a is not None else None
                    input_b = _inst_input(rs_b) if rs_b is not None else None
                    comp_a  = _completion(rs_a) if rs_a is not None else None
                    comp_b  = _completion(rs_b) if rs_b is not None else None

                    inputs_equal = (input_a == input_b) and (input_a is not None)
                    comps_equal  = (comp_a == comp_b) and (comp_a is not None)

                    # Input
                    if inputs_equal:
                        writer(f'      input (same): {input_a}')
                    else:
                        if input_a is not None:
                            writer(f'      [{self.a_name}] input: {input_a}')
                        if input_b is not None:
                            writer(f'      [{self.b_name}] input: {input_b}')

                    # Completion
                    if comps_equal:
                        writer(f'      completion (same): {comp_a}')
                    else:
                        if comp_a is not None:
                            writer(f'      [{self.a_name}] completion: {comp_a}')
                        if comp_b is not None:
                            writer(f'      [{self.b_name}] completion: {comp_b}')

                    if level >= 20:
                        if isinstance(rs_a, dict) and isinstance(rs_b, dict):
                            diffs = _walker_diff(
                                rs_a, rs_b, max_paths=diff_max_items
                            )
                            writer(
                                f'      request_state_diff: {ub.urepr(diffs)}'
                            )

                writer('')


def ratio(c: int, m: int) -> float:
    return 1.0 - (m / c) if c else float('nan')
