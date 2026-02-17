"""magnet.backends.helm.helm_run_diff

Run-to-run comparison built on :class:`~magnet.backends.helm.helm_run_analysis.HelmRunAnalysis`.

This refactor has two goals:

1) Make comparisons easier to write by reusing the same cached indices /
   canonicalization logic.
2) Support multiple report granularities (one-line, one-page, deeper dives)
   without a config system.

The public API is intentionally small:

* :class:`HelmRunDiff` - wrap two runs, cache expensive computations
* :meth:`HelmRunDiff.summary_dict` / :meth:`HelmRunDiff.summary_text`

Everything else is an implementation detail and can be promoted later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import ubelt as ub

# (metric classification happens inside HelmRunAnalysis.stat_index)
from magnet.backends.helm.helm_run_analysis import HelmRunAnalysis
from magnet.backends.helm import helm_hashers


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


class HelmRunDiff(ub.NiceRepr):
    """Compare two HELM runs.

    Parameters
    ----------
    run_a, run_b:
        Either :class:`HelmRunAnalysis` or a ``HelmRun`` reader (coerced).
    a_name, b_name:
        Human-friendly labels for display.

    Example:
        >>> import ubelt as ub
        >>> import kwutil
        >>> from magnet.backends.helm.helm_outputs import HelmRun
        >>> from magnet.backends.helm.helm_run_diff import HelmRunDiff
        >>> run_a = HelmRun.demo()
        >>> dpath = ub.Path.appdir('magnet/tests/helm/rundiff').delete().ensuredir()

        >>> # --- Case 1: identical copy -> perfect agreement -----------------
        >>> same_path = dpath / (run_a.path.name + '_same')
        >>> run_a.path.copy(same_path)
        >>> run_b = HelmRun(same_path)
        >>> rd = HelmRunDiff(run_a, run_b, a_name='orig', b_name='same')
        >>> info = rd.summary_dict(level='l1')
        >>> assert info['run_spec_dict_ok'] is True
        >>> assert info['scenario_ok'] in {True, None}
        >>> assert info['value_agreement']['overall']['mismatched'] == 0
        >>> assert info['value_agreement']['overall']['agree_ratio'] == 1.0
        >>> print(rd.summary_text(level='line'))  # xdoctest: +ELLIPSIS
        ✅ orig vs same ...

        >>> # --- Case 2: perturb a single RUN-level stat mean ----------------
        >>> stats_path = dpath / (run_a.path.name + '_statsmod')
        >>> run_a.path.copy(stats_path)
        >>> stat_fpath = stats_path / 'stats.json'
        >>> stats = kwutil.Json.load(stat_fpath)
        >>> old_mean = float(stats[0].get('mean', 0.0))
        >>> stats[0]['mean'] = old_mean + 1.23
        >>> stat_fpath.write_text(kwutil.Json.dumps(stats))
        >>> run_b2 = HelmRun(stats_path)
        >>> rd2 = HelmRunDiff(run_a, run_b2, a_name='orig', b_name='stats+1.23')
        >>> info2 = rd2.summary_dict(level='l1')
        >>> assert info2['value_agreement']['overall']['mismatched'] >= 1
        >>> top = info2['value_agreement']['top_mismatches'][0]
        >>> assert abs(float(top['abs_delta']) - 1.23) < 1e-9
        >>> print(rd2.summary_text(level='line'))  # xdoctest: +ELLIPSIS
        ✅ orig vs stats+1.23 ...

        >>> # --- Dive into instance-level diffs (per_instance_stats) ------------
        >>> # Make a copy and perturb ONE per-instance stat mean.
        >>> inst_path = dpath / (run_a.path.name + '_perinstmod')
        >>> run_a.path.copy(inst_path)
        >>> pi_fpath = inst_path / 'per_instance_stats.json'
        >>> perinst = kwutil.Json.load(pi_fpath)

        >>> # Deterministically modify the first mean-bearing stat for the first entry.
        >>> ei = 0
        >>> sj = None
        >>> for j, s in enumerate(perinst[ei]['stats']):
        ...     if s.get('count', 0) and ('mean' in s):
        ...         sj = j
        ...         break
        >>> assert sj is not None
        >>> old_mean = float(perinst[ei]['stats'][sj]['mean'])
        >>> perinst[ei]['stats'][sj]['mean'] = old_mean + 9.0
        >>> pi_fpath.write_text(kwutil.Json.dumps(perinst))
        >>> run_bi = HelmRun(inst_path)

        >>> # HelmRunDiff should own the keying + join logic and produce a report.
        >>> rd_i = HelmRunDiff(run_a, run_bi, a_name='orig', b_name='perinst+9')

        >>> inst_info = rd_i.instance_summary_dict(top_n=5)
        >>> assert inst_info['means']['mismatched'] >= 1
        >>> assert inst_info['means']['agree_ratio'] < 1.0
        >>> top = inst_info['top_mismatches'][0]
        >>> assert abs(float(top['abs_delta']) - 9.0) < 1e-9

        >>> # Human-readable one-liner + page report
        >>> print(rd_i.instance_summary_text(level='line'))  # xdoctest: +ELLIPSIS
        InstanceDiff orig vs perinst+9: isect=... agree=...
        >>> print(rd_i.instance_summary_text(level='page'))  # xdoctest: +IGNORE_WANT
        ...
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

    def summary_text(self, *, level: str = 'line', writer=None) -> str:
        """Human-readable report.

        level='line' is intended for tables.
        level='page' is a compact multi-line report.
        """
        info = self.summary_dict(level='l1')

        if writer is None:
            from rich.console import Console
            _console = Console()
            writer = _console.print

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
            writer(f"HelmRunDiff: {self.a_name} vs {self.b_name}")
            writer(f"  {self.a_name}: {self.a.summary_text(level='line')}")
            writer(f"  {self.b_name}: {self.b.summary_text(level='line')}")

            writer("")
            writer(f"Run spec name: {_format_bool(info['run_spec_name_ok'])}  {info['run_spec_name_a']}  vs  {info['run_spec_name_b']}")
            writer(f"Run spec dict: {_format_bool(info['run_spec_dict_ok'])}  hashA={info['run_spec_hash_a'][:10]}  hashB={info['run_spec_hash_b'][:10]}")
            if not info['run_spec_dict_ok'] and info['run_spec_diff_paths']:
                writer(f"  diff paths: {', '.join(info['run_spec_diff_paths'])}")

            if info['scenario_ok'] is None:
                writer("Scenario: ⚠️  unknown (missing scenario.json in one or both runs)")
            else:
                writer(f"Scenario: {_format_bool(bool(info['scenario_ok']))}")
                if info['scenario_ok'] is False and info['scenario_diff_paths']:
                    writer(f"  diff paths: {', '.join(info['scenario_diff_paths'])}")

            cov = info['stats_coverage_by_name']
            cov2 = info['stats_coverage_by_name_count']
            writer("")
            writer("Stats coverage:")
            writer(
                f"  by name:       A={cov['n_a']} B={cov['n_b']} isect={cov['n_isect']} union={cov['n_union']} onlyA={cov['only_a']} onlyB={cov['only_b']}"
            )
            writer(
                f"  by name+count: A={cov2['n_a']} B={cov2['n_b']} isect={cov2['n_isect']} union={cov2['n_union']} onlyA={cov2['only_a']} onlyB={cov2['only_b']}"
            )

            writer("")
            writer("Value agreement (mean on intersecting stats):")
            ov = info['value_agreement']['overall']
            writer(f"  overall: comparable={ov['comparable']} mismatched={ov['mismatched']} agree_ratio={ov['agree_ratio']:.3f}")
            for cls in ('core', 'bookkeeping', 'untracked'):
                s = info['value_agreement']['by_class'][cls]
                writer(
                    f"  {cls:11s}: comparable={s['comparable']} mismatched={s['mismatched']} agree_ratio={s['agree_ratio']:.3f}"
                )

            top = info['value_agreement'].get('top_mismatches', [])
            if top:
                writer("  top mismatches:")
                for r in top:
                    writer(
                        f"    {r['key']}  A={_fmt(r['a'])}  B={_fmt(r['b'])}  Δ={_fmt(r['abs_delta'])}"
                    )
            return "\n".join(lines)

        raise KeyError(level)

    def instance_summary_dict(self, *, top_n: int = 10):
        """
        Programmatic summary of per-instance stat agreement.

        Returns a dict with:
          - coverage of (instance, trial, perturbation, metric) keys
          - mean agreement stats
          - top mismatches by abs delta
          - perturbed vs unperturbed breakdown
        """
        return instance_summary_dict(self, top_n=top_n)

    def instance_summary_text(self, *args, **kwargs) -> None:
        """
        Write a line-oriented instance-level diff report.

        Uses `writer` for immediate output. If writer is None, defaults to rich.print.

        Keeps long prompt / completion / input excerpts readable via kwutil smart_truncate.
        Request-state diffs are summarized as a small set of highlighted lines.
        """
        return summarize_instances(self, *args, **kwargs)

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


def _diff_request_states(
    rs_a: dict,
    rs_b: dict,
    *,
    max_diffs: int = 40,
    context_chars: int = 180,
):
    """
    Compute a compact walker diff between two request_state dicts.

    This strips high-churn fields (timestamps, runtimes, full token logprobs, etc.)
    so the diff focuses on prompt/instance/references/completions.

    Returns
    -------
    dict with:
        - ok: bool
        - n_diffs: int
        - diffs: List[dict(path=..., a=..., b=...)]
    """
    import ubelt as ub

    def _clip(v):
        # Keep output readable
        if isinstance(v, str):
            v2 = v.replace('\r\n', '\n')
            return (v2[:context_chars] + '…') if len(v2) > context_chars else v2
        return v

    # Shallow recursive prune helper
    def _prune(obj, path=()):
        # Remove/normalize high-churn keys
        if isinstance(obj, dict):
            drop = {
                # result churn
                'request_time', 'request_datetime', 'cached',
                # huge / noisy
                'tokens', 'logprob', 'top_k_per_token',
                # sometimes noisy (depends on run)
                'embedding',
            }
            out = {}
            for k, v in obj.items():
                if k in drop:
                    continue
                # also drop deep token lists if present
                if k == 'completions' and isinstance(v, list):
                    # Keep only completion texts (and maybe finish reason if present)
                    v2 = []
                    for c in v:
                        if isinstance(c, dict):
                            v2.append({kk: vv for kk, vv in c.items() if kk in {'text', 'finish_reason'}})
                        else:
                            v2.append(c)
                    out[k] = _prune(v2, path + (k,))
                else:
                    out[k] = _prune(v, path + (k,))
            return out
        elif isinstance(obj, list):
            return [_prune(v, path + (i,)) for i, v in enumerate(obj)]
        else:
            return obj

    A = _prune(rs_a)
    B = _prune(rs_b)

    wa = ub.IndexableWalker(A, list_cls=(list,))
    wb = ub.IndexableWalker(B, list_cls=(list,))

    # ubelt returns a structured diff object; we just need changed paths
    diff = wa.diff(wb)

    # Try to be robust to ubelt version differences
    changes = []
    for item in getattr(diff, 'changed', []) or []:
        # item usually has .path, .value1, .value2
        path = getattr(item, 'path', None)
        a_val = getattr(item, 'value1', None)
        b_val = getattr(item, 'value2', None)
        changes.append((path, a_val, b_val))

    # Added/removed can also be useful
    for item in getattr(diff, 'added', []) or []:
        path = getattr(item, 'path', None)
        a_val = None
        b_val = getattr(item, 'value', None)
        changes.append((path, a_val, b_val))
    for item in getattr(diff, 'removed', []) or []:
        path = getattr(item, 'path', None)
        a_val = getattr(item, 'value', None)
        b_val = None
        changes.append((path, a_val, b_val))

    # If ubelt diff API differs, fall back to brute path compare
    if not changes:
        # conservative fallback: compare leaf reprs
        leaves_a = {tuple(p): v for p, v in wa}
        leaves_b = {tuple(p): v for p, v in wb}
        all_paths = set(leaves_a) | set(leaves_b)
        for p in all_paths:
            if leaves_a.get(p) != leaves_b.get(p):
                changes.append((p, leaves_a.get(p), leaves_b.get(p)))

    # Sort by path for deterministic output
    changes = sorted(changes, key=lambda t: (t[0] is None, t[0]))

    diffs = []
    for path, av, bv in changes[:max_diffs]:
        diffs.append({
            'path': path,
            'a': _clip(av),
            'b': _clip(bv),
        })

    return {
        'ok': len(changes) == 0,
        'n_diffs': len(changes),
        'diffs': diffs,
    }


def _slug(text, max_len: int, *, hash_len: int = 8, trunc_loc: float = 0.5, word_boundary: bool = True):
    """
    Smart-truncate long text into a readable slug with a short hash so different
    long strings don't collapse to the same preview.

    Notes:
    - Converts newlines/tabs to spaces to keep tables readable.
    - Uses smart_truncate's internal hashing (hash_len) for stability.
    """
    if text is None:
        return None
    import kwutil
    s = str(text)
    # normalize whitespace for table-friendly display
    s = s.replace('\r\n', '\n').replace('\r', '\n')
    s = " ".join(s.split())
    if max_len and len(s) > max_len:
        return kwutil.slugify_ext.smart_truncate(
            s,
            max_length=max_len,
            word_boundary=word_boundary,
            separator=' ',
            trunc_loc=trunc_loc,
            hash_len=hash_len,
            head='~',
            tail='~',
        )
    return s


def instance_summary_dict(self, *, top_n: int = 10):
    """
    Programmatic summary of per-instance stat agreement.

    Adds:
      * means_by_class: agreement stats per metric class
      * top_mismatches_by_class: top-N mismatches per class (core/bookkeeping/untracked/...)
      * top_mismatches_all: global top-N mismatches (for quick triage)
    """
    from magnet.backends.helm.helm_metrics import classify_metric

    def _analysis(run_or_analysis, name: str | None = None):
        try:
            if hasattr(run_or_analysis, "joined_instance_stat_table"):
                return run_or_analysis
        except Exception:
            pass
        from magnet.backends.helm.helm_run_analysis import HelmRunAnalysis
        return HelmRunAnalysis(run_or_analysis, name=name)

    a_name = getattr(self, "a_name", "A")
    b_name = getattr(self, "b_name", "B")
    A = _analysis(getattr(self, "run_a", getattr(self, "a", None)), name=a_name)
    B = _analysis(getattr(self, "run_b", getattr(self, "b", None)), name=b_name)

    jt_a = A.joined_instance_stat_table(assert_assumptions=True)
    jt_b = B.joined_instance_stat_table(assert_assumptions=True)

    def _row_key(r):
        n = (r["stat"].get("name") or {})
        stat_pert = n.get("perturbation", None)
        stat_pert_name = stat_pert.get("name", None) if isinstance(stat_pert, dict) else None
        return (
            r["instance_id"],
            r["train_trial_index"],
            r.get("perturbation_id", None),
            n.get("name", None),
            n.get("split", None),
            n.get("sub_split", None),
            stat_pert_name,
        )

    def _mean_map(joined):
        out = {}
        for r in joined:
            s = r["stat"]
            if s.get("count", 0) and ("mean" in s):
                out[_row_key(r)] = float(s["mean"])
        return out

    map_a = _mean_map(jt_a)
    map_b = _mean_map(jt_b)

    ka, kb = set(map_a), set(map_b)
    isect = ka & kb
    only_a = ka - kb
    only_b = kb - ka

    mism = []
    for k in isect:
        a = map_a[k]
        b = map_b[k]
        if a != b:
            mism.append((abs(a - b), k, a, b))
    mism.sort(reverse=True)

    comparable = len(isect)
    mismatched = len(mism)
    agree_ratio = 1.0 if comparable == 0 else (comparable - mismatched) / comparable

    def _is_perturbed_key(k):
        return (k[2] is not None)

    isect_pert = [k for k in isect if _is_perturbed_key(k)]
    isect_base = [k for k in isect if not _is_perturbed_key(k)]

    def _agree_ratio_subset(keys):
        if not keys:
            return 1.0
        mm = 0
        for k in keys:
            if map_a[k] != map_b[k]:
                mm += 1
        return (len(keys) - mm) / len(keys)

    # by metric class
    by_class = {}
    for k in isect:
        metric = k[3] or ""
        cls = classify_metric(metric)
        d = by_class.setdefault(cls, {"comparable": 0, "mismatched": 0})
        d["comparable"] += 1
        if map_a[k] != map_b[k]:
            d["mismatched"] += 1
    for cls, d in by_class.items():
        comp = d["comparable"]
        mm = d["mismatched"]
        d["agree_ratio"] = 1.0 if comp == 0 else (comp - mm) / comp

    # top mismatches by class
    top_by_class = {}
    for abs_d, k, a, b in mism:
        cls = classify_metric(k[3] or "")
        top_by_class.setdefault(cls, []).append((abs_d, k, a, b))

    top_mismatches_by_class = {}
    for cls, items in top_by_class.items():
        items.sort(reverse=True)
        top_mismatches_by_class[cls] = [
            {
                "abs_delta": abs_d,
                "signed_delta": (b - a),
                "key": k,
                "a": a,
                "b": b,
                "metric_class": cls,
            }
            for (abs_d, k, a, b) in items[:top_n]
        ]

    out = {
        "coverage": {
            "n_a": len(ka),
            "n_b": len(kb),
            "n_isect": len(isect),
            "n_union": len(ka | kb),
            "only_a": len(only_a),
            "only_b": len(only_b),
        },
        "means": {
            "comparable": comparable,
            "mismatched": mismatched,
            "agree_ratio": agree_ratio,
            "agree_ratio_unperturbed": _agree_ratio_subset(isect_base),
            "agree_ratio_perturbed": _agree_ratio_subset(isect_pert),
        },
        "means_by_class": by_class,
        "top_mismatches_all": [
            {
                "abs_delta": abs_d,
                "signed_delta": (b - a),
                "key": k,
                "a": a,
                "b": b,
                "metric_class": classify_metric(k[3] or ""),
            }
            for (abs_d, k, a, b) in mism[:top_n]
        ],
        "top_mismatches_by_class": top_mismatches_by_class,
    }
    return out


def summarize_instances(
    self,
    *,
    level: str = "page",
    top_n: int = 10,
    show_details: int = 3,
    prompt_chars: int = 220,
    completion_chars: int = 120,
    input_chars: int = 200,
    diff_max: int = 80,
    diff_show: int = 12,
    writer=None,
) -> None:
    """
    Write a line-oriented instance-level diff report.

    - Uses `writer` for output. If None, defaults to `rich.print`.
    - Uses kwutil smart_truncate to shorten prompts/completions/inputs.
    - Prints top mismatches for BOTH core and bookkeeping (and untracked if present).
    - Avoids early returns except at the start for level='line' (OK for ipython copy/paste).
    """
    # from magnet.backends.helm.helm_metrics import classify_metric
    import kwutil

    if writer is None:
        try:
            from rich import print as rich_print
            writer = rich_print
        except Exception:
            writer = print

    # --- safe rich text helpers ---
    def _safe_rich(s: str | None) -> str | None:
        if s is None:
            return None
        s = str(s)
        s = "".join(ch for ch in s if ord(ch) >= 32)
        try:
            from rich.markup import escape
            s = escape(s)
        except Exception:
            s = s.replace("[", r"\[").replace("]", r"\]")
        return s

    def _slug(text, max_len: int, *, hash_len: int = 8, trunc_loc: float = 0.6):
        if text is None:
            return None
        s = str(text)
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        s = " ".join(s.split())
        if max_len and len(s) > max_len:
            s = kwutil.slugify_ext.smart_truncate(
                s,
                max_length=max_len,
                word_boundary=True,
                separator=" ",
                trunc_loc=trunc_loc,
                hash_len=hash_len,
                head="~",
                tail="~",
            )
        return _safe_rich(s)

    info = instance_summary_dict(self, top_n=top_n)
    cov = info["coverage"]
    means = info["means"]
    by_class = info.get("means_by_class") or {}
    top_by_class = info.get("top_mismatches_by_class") or {}

    a_name = getattr(self, "a_name", "A")
    b_name = getattr(self, "b_name", "B")

    if level == "line":
        writer(
            f"[bold]InstanceDiff[/bold] {a_name} vs {b_name}: "
            f"isect={cov['n_isect']}/{cov['n_union']} "
            f"agree={means['agree_ratio']:.3f} "
            f"base={means['agree_ratio_unperturbed']:.3f} "
            f"pert={means['agree_ratio_perturbed']:.3f}"
        )
        return None

    # --- analysis adapters ---
    def _analysis(run_or_analysis, name: str | None = None):
        try:
            if hasattr(run_or_analysis, "joined_instance_stat_table"):
                return run_or_analysis
        except Exception:
            pass
        from magnet.backends.helm.helm_run_analysis import HelmRunAnalysis
        return HelmRunAnalysis(run_or_analysis, name=name)

    A = _analysis(getattr(self, "run_a", getattr(self, "a", None)), name=a_name)
    B = _analysis(getattr(self, "run_b", getattr(self, "b", None)), name=b_name)

    jt_a = A.joined_instance_stat_table(assert_assumptions=True)
    jt_b = B.joined_instance_stat_table(assert_assumptions=True)

    # Must match instance_summary_dict() keying
    def _row_key(r):
        n = (r["stat"].get("name") or {})
        stat_pert = n.get("perturbation", None)
        stat_pert_name = stat_pert.get("name", None) if isinstance(stat_pert, dict) else None
        return (
            r["instance_id"],
            r["train_trial_index"],
            r.get("perturbation_id", None),
            n.get("name", None),
            n.get("split", None),
            n.get("sub_split", None),
            stat_pert_name,
        )

    lut_a = {_row_key(r): r for r in jt_a}
    lut_b = {_row_key(r): r for r in jt_b}

    def _prompt(rs):
        if rs is None:
            return None
        return _slug(((rs.get("request") or {}).get("prompt", None)), prompt_chars, hash_len=10, trunc_loc=0.65)

    def _completion(rs):
        if rs is None:
            return None
        comps = ((rs.get("result") or {}).get("completions", None)) or []
        txt = comps[0].get("text", None) if comps else None
        return _slug(txt, completion_chars, hash_len=8, trunc_loc=0.5)

    def _instance_input(rs):
        if rs is None:
            return None
        inst = rs.get("instance") or {}
        inp = inst.get("input") or {}
        if isinstance(inp, dict) and "text" in inp:
            return _slug(inp.get("text", None), input_chars, hash_len=8, trunc_loc=0.5)
        return _slug(inp, input_chars, hash_len=8, trunc_loc=0.5)

    def _metric_label_from_key(k):
        metric = k[3]
        split = k[4]
        sub = k[5]
        stat_pert = k[6]
        parts = [str(metric)]
        if split is not None:
            parts.append(f"split={split}")
        if sub is not None:
            parts.append(f"sub={sub}")
        if stat_pert is not None:
            parts.append(f"statPert={stat_pert}")
        return ", ".join(parts)

    # --- header ---
    writer(f"[bold]Instance-level diff[/bold]: {a_name} vs {b_name}")
    writer(
        f"  coverage: A={cov['n_a']} B={cov['n_b']} "
        f"isect={cov['n_isect']} union={cov['n_union']} onlyA={cov['only_a']} onlyB={cov['only_b']}"
    )
    writer(
        f"  means: comparable={means['comparable']} mismatched={means['mismatched']} "
        f"agree_ratio={means['agree_ratio']:.3f} "
        f"(unpert={means['agree_ratio_unperturbed']:.3f}, pert={means['agree_ratio_perturbed']:.3f})"
    )
    if by_class:
        order = ["core", "bookkeeping", "untracked"]
        keys = [k for k in order if k in by_class] + [k for k in sorted(by_class) if k not in order]
        parts = []
        for cls in keys:
            d = by_class[cls]
            parts.append(f"{cls}={d['agree_ratio']:.3f}({d['mismatched']}/{d['comparable']})")
        writer("  by_class: ")
        for p in parts:
            writer("   ", p)

    # We want to show both core and bookkeeping (and untracked if present)
    show_classes = []
    for cls in ["core", "bookkeeping", "untracked"]:
        if cls in top_by_class and top_by_class[cls]:
            show_classes.append(cls)
    for cls in sorted(top_by_class):
        if cls not in show_classes and top_by_class[cls]:
            show_classes.append(cls)

    if not show_classes:
        writer("  [green]No instance-level mismatches detected.[/green]")
    else:
        for cls in show_classes:
            tops = top_by_class.get(cls, [])[:top_n]
            writer(f"  [bold]top mismatches ({cls}):[/bold]")
            for idx, item in enumerate(tops, start=1):
                k = item["key"]
                a = float(item["a"])
                b = float(item["b"])
                abs_d = float(item["abs_delta"])
                signed_d = float(item.get("signed_delta", b - a))
                writer(f"    {idx:>2}. metric: {_metric_label_from_key(k)}")
                writer(f"        key: {k}")
                writer(f"        A={a:.4g}  B={b:.4g}  Δ(B-A)={signed_d:.4g}  |Δ|={abs_d:.4g}")

                if idx <= show_details:
                    ra = lut_a.get(k, None)
                    rb = lut_b.get(k, None)
                    rs_a = ra["request_state"] if ra is not None else None
                    rs_b = rb["request_state"] if rb is not None else None

                    pa = _prompt(rs_a)
                    pb = _prompt(rs_b)
                    prompts_equal = (pa == pb)
                    writer(f"        prompts_equal={prompts_equal}")

                    if rs_a is not None:
                        writer(f"        [{a_name}] input: {_instance_input(rs_a)}")
                        writer(f"        [{a_name}] completion: {_completion(rs_a)}")
                    if rs_b is not None:
                        writer(f"        [{b_name}] input: {_instance_input(rs_b)}")
                        writer(f"        [{b_name}] completion: {_completion(rs_b)}")

                    if prompts_equal:
                        if pb is not None:
                            writer(f"        prompt: {pb}")
                    else:
                        if pa is not None:
                            writer(f"        prompt[{a_name}]: {pa}")
                        if pb is not None:
                            writer(f"        prompt[{b_name}]: {pb}")

                    if rs_a is not None and rs_b is not None and hasattr(self, "_diff_request_states"):
                        rd = self._diff_request_states(rs_a, rs_b, max_diffs=diff_max)
                        diffs = list(rd.get("diffs", []) or [])

                        def _path_tuple(p):
                            if isinstance(p, (list, tuple)):
                                return tuple(p)
                            return (p,)

                        # Dedup redundant prefix diffs
                        paths = [_path_tuple(d.get("path")) for d in diffs]
                        path_set = set(paths)
                        keep = []
                        for d in diffs:
                            p = _path_tuple(d.get("path"))
                            is_prefix = any((q != p and len(q) > len(p) and q[: len(p)] == p) for q in path_set)
                            if not is_prefix:
                                keep.append(d)
                        diffs = keep

                        IMPORTANT_PREFIXES = (
                            ("request", "prompt"),
                            ("result", "completions"),
                            ("instance", "input"),
                            ("instance", "references"),
                            ("instance", "perturbation"),
                            ("request", "max_tokens"),
                            ("request", "model_deployment"),
                            ("request", "model"),
                        )

                        def _priority(path):
                            path = _path_tuple(path)
                            for rank, pref in enumerate(IMPORTANT_PREFIXES):
                                if path[: len(pref)] == pref:
                                    return rank
                            return 999

                        diffs.sort(key=lambda d: (_priority(d.get("path")), str(d.get("path"))))
                        n_diffs = rd.get("n_diffs", len(diffs))
                        show = min(diff_show, len(diffs))
                        writer(f"        request_state_diff: n_diffs={n_diffs} (showing {show})")

                        for d in diffs[:show]:
                            p = _path_tuple(d.get("path"))
                            path_s = "/".join(map(str, p))
                            av = d.get("a")
                            bv = d.get("b")
                            if p == ("request", "prompt") or (p[:3] == ("result", "completions", 0) and p[-1] == "text"):
                                avs = _slug(av, 120, hash_len=10)
                                bvs = _slug(bv, 120, hash_len=10)
                            else:
                                avs = _slug(av, 70, hash_len=6)
                                bvs = _slug(bv, 70, hash_len=6)
                            writer(f"          - {path_s}: {avs}  ->  {bvs}")

                writer("")
            writer("")
    return None

