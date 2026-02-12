"""magnet.backends.helm.helm_run_diff

Run-to-run comparison built on :class:`~magnet.backends.helm.helm_run_analysis.HelmRunAnalysis`.

This refactor has two goals:

1) Make comparisons easier to write by reusing the same cached indices /
   canonicalization logic.
2) Support multiple report granularities (one-line, one-page, deeper dives)
   without a config system.

The public API is intentionally small:

* :class:`RunDiff` - wrap two runs, cache expensive computations
* :meth:`RunDiff.summary_dict` / :meth:`RunDiff.summary_text`

Everything else is an implementation detail and can be promoted later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import ubelt as ub

# (metric classification happens inside HelmRunAnalysis.stat_index)
from magnet.backends.helm.helm_run_analysis import HelmRunAnalysis
from magnet.backends.helm.import helm_hashers


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
    """Return a compact list of changed paths using ubelt's walker."""
    try:
        w = ub.IndexableWalker(a, list_cls=tuple)
        diff = w.diff(b)
    except Exception:
        return []
    paths = []
    for d in diff:
        # d.path is a tuple of keys
        paths.append('/'.join(map(str, d.path)))
        if len(paths) >= max_paths:
            break
    return paths


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


class RunDiff(ub.NiceRepr):
    """Compare two HELM runs.

    Parameters
    ----------
    run_a, run_b:
        Either :class:`HelmRunAnalysis` or a ``HelmRun`` reader (coerced).
    a_name, b_name:
        Human-friendly labels for display.
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
        self.a = run_a if isinstance(run_a, HelmRunAnalysis) else HelmRunAnalysis(run_a, name=a_name)
        self.b = run_b if isinstance(run_b, HelmRunAnalysis) else HelmRunAnalysis(run_b, name=b_name)
        self.a_name = a_name
        self.b_name = b_name
        self.short_hash = short_hash
        self._cache: dict[Any, Any] = {}

    def __nice__(self):
        return f"{self.a_name} vs {self.b_name}"

    # --- Core summary -------------------------------------------------

    def summary_dict(self, *, level: str = 'l1') -> dict[str, Any]:
        """Programmatic comparison summary.

        level='l1' corresponds to the original "base-level" checks:

        1. run_spec_name equality
        2. run_spec dict hash equality + diff paths
        3. scenario dict hash equality (with "unknown" semantics)
        4. stats coverage by name (ignore count/value)
        5. stats coverage by (name+count)
        6. value agreement on intersecting keys (split by core/bookkeeping/untracked)
        """
        cache_key = ('summary_dict', level)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if level != 'l1':
            raise KeyError(level)

        a_spec = self.a.run_spec() or {}
        b_spec = self.b.run_spec() or {}
        a_scen = self.a.scenario() or {}
        b_scen = self.b.scenario() or {}

        # 1) run spec name
        a_name = a_spec.get('name', None)
        b_name = b_spec.get('name', None)
        run_spec_name_ok = (a_name == b_name) and (a_name is not None)

        # 2) run spec dict check
        spec_hash_a = helm_hashers.stable_hash36(helm_hashers.canonicalize_for_hashing(a_spec))
        spec_hash_b = helm_hashers.stable_hash36(helm_hashers.canonicalize_for_hashing(b_spec))
        run_spec_dict_ok = spec_hash_a == spec_hash_b
        spec_diff_paths = [] if run_spec_dict_ok else _walker_diff(a_spec, b_spec)

        # 3) scenario check (treat both None/missing as unknown)
        scen_known = bool(a_scen) and bool(b_scen)
        if not scen_known:
            scenario_ok = None
            scenario_hash_a = None
            scenario_hash_b = None
            scen_diff_paths = []
        else:
            scenario_hash_a = helm_hashers.stable_hash36(helm_hashers.canonicalize_for_hashing(a_scen))
            scenario_hash_b = helm_hashers.stable_hash36(helm_hashers.canonicalize_for_hashing(b_scen))
            scenario_ok = scenario_hash_a == scenario_hash_b
            scen_diff_paths = [] if scenario_ok else _walker_diff(a_scen, b_scen)

        # 4/5) stats coverage
        a_stats = self.a.stats() or []
        b_stats = self.b.stats() or []
        a_name_keys = {helm_hashers.stat_key(s.get('name', None), short_hash=self.short_hash) for s in a_stats}
        b_name_keys = {helm_hashers.stat_key(s.get('name', None), short_hash=self.short_hash) for s in b_stats}
        cov_name = Coverage.from_sets(a_name_keys, b_name_keys)

        a_name_count_keys = {
            helm_hashers.stat_key(s.get('name', None), count=s.get('count', None), short_hash=self.short_hash)
            for s in a_stats
        }
        b_name_count_keys = {
            helm_hashers.stat_key(s.get('name', None), count=s.get('count', None), short_hash=self.short_hash)
            for s in b_stats
        }
        cov_name_count = Coverage.from_sets(a_name_count_keys, b_name_count_keys)

        # 6) value agreement (means) on intersecting keys
        value_summary = self._value_agreement_summary()

        out = {
            'a': self.a.summary_dict(level='lite'),
            'b': self.b.summary_dict(level='lite'),
            'run_spec_name_ok': run_spec_name_ok,
            'run_spec_name_a': a_name,
            'run_spec_name_b': b_name,
            'run_spec_dict_ok': run_spec_dict_ok,
            'run_spec_hash_a': spec_hash_a,
            'run_spec_hash_b': spec_hash_b,
            'run_spec_diff_paths': spec_diff_paths,
            'scenario_ok': scenario_ok,  # None => unknown
            'scenario_hash_a': scenario_hash_a,
            'scenario_hash_b': scenario_hash_b,
            'scenario_diff_paths': scen_diff_paths,
            'stats_coverage_by_name': cov_name.__dict__,
            'stats_coverage_by_name_count': cov_name_count.__dict__,
            'value_agreement': value_summary,
        }

        self._cache[cache_key] = out
        return out

    def summary_text(self, *, level: str = 'line') -> str:
        """Human-readable report.

        level='line' is intended for tables.
        level='page' is a compact multi-line report.
        """
        info = self.summary_dict(level='l1')

        if level == 'line':
            ok = info['run_spec_dict_ok'] and (info['scenario_ok'] in {True, None})
            cov = info['stats_coverage_by_name']
            agree = info['value_agreement']['overall']['agree_ratio']
            return (
                f"{_format_bool(ok)} {self.a_name} vs {self.b_name} "
                f"spec={_format_bool(info['run_spec_dict_ok'])} "
                f"stats={cov['n_isect']}/{cov['n_union']} "
                f"agree={agree:.3f}"
            )

        if level == 'page':
            lines: list[str] = []
            lines.append(f"RunDiff: {self.a_name} vs {self.b_name}")
            lines.append(f"  {self.a_name}: {self.a.summary_text(level='line')}")
            lines.append(f"  {self.b_name}: {self.b.summary_text(level='line')}")

            lines.append("")
            lines.append(f"Run spec name: {_format_bool(info['run_spec_name_ok'])}  {info['run_spec_name_a']}  vs  {info['run_spec_name_b']}")
            lines.append(f"Run spec dict: {_format_bool(info['run_spec_dict_ok'])}  hashA={info['run_spec_hash_a'][:10]}  hashB={info['run_spec_hash_b'][:10]}")
            if not info['run_spec_dict_ok'] and info['run_spec_diff_paths']:
                lines.append(f"  diff paths: {', '.join(info['run_spec_diff_paths'])}")

            if info['scenario_ok'] is None:
                lines.append("Scenario: ⚠️  unknown (missing scenario.json in one or both runs)")
            else:
                lines.append(f"Scenario: {_format_bool(bool(info['scenario_ok']))}")
                if info['scenario_ok'] is False and info['scenario_diff_paths']:
                    lines.append(f"  diff paths: {', '.join(info['scenario_diff_paths'])}")

            cov = info['stats_coverage_by_name']
            cov2 = info['stats_coverage_by_name_count']
            lines.append("")
            lines.append("Stats coverage:")
            lines.append(
                f"  by name:       A={cov['n_a']} B={cov['n_b']} isect={cov['n_isect']} union={cov['n_union']} onlyA={cov['only_a']} onlyB={cov['only_b']}"
            )
            lines.append(
                f"  by name+count: A={cov2['n_a']} B={cov2['n_b']} isect={cov2['n_isect']} union={cov2['n_union']} onlyA={cov2['only_a']} onlyB={cov2['only_b']}"
            )

            lines.append("")
            lines.append("Value agreement (mean on intersecting stats):")
            ov = info['value_agreement']['overall']
            lines.append(f"  overall: comparable={ov['comparable']} mismatched={ov['mismatched']} agree_ratio={ov['agree_ratio']:.3f}")
            for cls in ('core', 'bookkeeping', 'untracked'):
                s = info['value_agreement']['by_class'][cls]
                lines.append(
                    f"  {cls:11s}: comparable={s['comparable']} mismatched={s['mismatched']} agree_ratio={s['agree_ratio']:.3f}"
                )

            top = info['value_agreement'].get('top_mismatches', [])
            if top:
                lines.append("  top mismatches:")
                for r in top:
                    lines.append(
                        f"    {r['key']}  A={_fmt(r['a'])}  B={_fmt(r['b'])}  Δ={_fmt(r['abs_delta'])}"
                    )
            return "\n".join(lines)

        raise KeyError(level)

    # --- Implementation helpers ---------------------------------------

    def _value_agreement_summary(
        self,
        *,
        abs_tol: float = 0.0,
        rel_tol: float = 0.0,
        top_n: int = 12,
    ) -> dict[str, Any]:
        """Compare mean values for intersecting stats.

        Uses the readable stat keys produced by :meth:`HelmRunAnalysis.stat_index`.
        """
        cache_key = ('value_agreement', abs_tol, rel_tol, top_n, self.short_hash)
        if cache_key in self._cache:
            return self._cache[cache_key]

        idx_a = self.a.stat_index(drop_zero_count=True, require_mean=True, short_hash=self.short_hash)
        idx_b = self.b.stat_index(drop_zero_count=True, require_mean=True, short_hash=self.short_hash)
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
            cls = a.metric_class  # should match b.metric_class for same metric name
            by_class[cls]['comparable'] += 1
            if not agrees(a.mean, b.mean):
                mismatched += 1
                by_class[cls]['mismatched'] += 1
                mismatches.append({'key': k, 'a': a.mean, 'b': b.mean, 'abs_delta': abs(a.mean - b.mean)})

        mismatches.sort(key=lambda r: r['abs_delta'], reverse=True)
        top = mismatches[:top_n]

        def ratio(c: int, m: int) -> float:
            return 1.0 - (m / c) if c else 1.0

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


def _fmt(x: Any) -> str:
    if x is None:
        return 'None'
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)
