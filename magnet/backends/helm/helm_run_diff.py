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


class RunDiff(ub.NiceRepr):
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
        >>> from magnet.backends.helm.helm_run_diff import RunDiff
        >>> run_a = HelmRun.demo()
        >>> dpath = ub.Path.appdir('magnet/tests/helm/rundiff').delete().ensuredir()

        >>> # --- Case 1: identical copy -> perfect agreement -----------------
        >>> same_path = dpath / (run_a.path.name + '_same')
        >>> run_a.path.copy(same_path)
        >>> run_b = HelmRun(same_path)
        >>> rd = RunDiff(run_a, run_b, a_name='orig', b_name='same')
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
        >>> rd2 = RunDiff(run_a, run_b2, a_name='orig', b_name='stats+1.23')
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

        >>> # RunDiff should own the keying + join logic and produce a report.
        >>> rd_i = RunDiff(run_a, run_bi, a_name='orig', b_name='perinst+9')

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

    def instance_summary_dict(self, *, top_n: int = 10):
        """
        Programmatic summary of per-instance stat agreement.

        Returns a dict with:
          - coverage of (instance, trial, perturbation, metric) keys
          - mean agreement stats
          - top mismatches by abs delta
          - perturbed vs unperturbed breakdown
        """
        A = self.a
        B = self.b

        jt_a = A.joined_instance_stat_table(assert_assumptions=True)
        jt_b = B.joined_instance_stat_table(assert_assumptions=True)

        def row_key(r):
            n = (r['stat'].get('name') or {})
            pert = n.get('perturbation', None)
            pert_name = pert.get('name', None) if isinstance(pert, dict) else None
            # Include stat perturbation name (if present) because those are distinct metrics.
            return (
                r['instance_id'],
                r['train_trial_index'],
                # request/instance perturbation id (variant selector)
                r.get('perturbation_id', None),
                # stat identity
                n.get('name', None),
                n.get('split', None),
                n.get('sub_split', None),
                pert_name,
            )

        def mean_map(jt):
            out = {}
            for r in jt:
                s = r['stat']
                if s.get('count', 0) and ('mean' in s):
                    out[row_key(r)] = float(s['mean'])
            return out

        map_a = mean_map(jt_a)
        map_b = mean_map(jt_b)

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

        # Perturbed/unperturbed breakdown using *variant* perturbation_id
        def is_perturbed_key(k):
            # k[2] is request/instance perturbation_id (None for base)
            return (k[2] is not None)

        isect_pert = [k for k in isect if is_perturbed_key(k)]
        isect_base = [k for k in isect if not is_perturbed_key(k)]

        def agree_ratio_subset(keys):
            if not keys:
                return 1.0
            mm = 0
            for k in keys:
                if map_a[k] != map_b[k]:
                    mm += 1
            return (len(keys) - mm) / len(keys)

        out = {
            'coverage': {
                'n_a': len(ka),
                'n_b': len(kb),
                'n_isect': len(isect),
                'n_union': len(ka | kb),
                'only_a': len(only_a),
                'only_b': len(only_b),
            },
            'means': {
                'comparable': comparable,
                'mismatched': mismatched,
                'agree_ratio': agree_ratio,
                'agree_ratio_unperturbed': agree_ratio_subset(isect_base),
                'agree_ratio_perturbed': agree_ratio_subset(isect_pert),
            },
            'top_mismatches': [
                {
                    'abs_delta': d,
                    'key': k,
                    'a': a,
                    'b': b,
                }
                for (d, k, a, b) in mism[:top_n]
            ],
        }
        return out


    def instance_summary_text(
        self,
        *,
        level: str = 'page',
        top_n: int = 10,
        show_details: int = 3,
        prompt_chars: int = 220,
        completion_chars: int = 120,
        input_chars: int = 200,
    ):
        """
        Human-readable instance-level summary.

        - level='line': compact one-liner
        - level='page': multi-line report with top mismatches and drilldown details.

        For the top `show_details` mismatches, includes:
          * metric label + signed delta (B - A)
          * whether prompts differ (and prints both excerpts if they do)
          * input + completion excerpts from A and B
        """
        info = self.instance_summary_dict(top_n=top_n)
        cov = info['coverage']
        means = info['means']

        a_name = getattr(self, 'a_name', 'A')
        b_name = getattr(self, 'b_name', 'B')

        if level == 'line':
            return (
                f"InstanceDiff {a_name} vs {b_name}: "
                f"isect={cov['n_isect']}/{cov['n_union']} "
                f"agree={means['agree_ratio']:.3f} "
                f"base={means['agree_ratio_unperturbed']:.3f} "
                f"pert={means['agree_ratio_perturbed']:.3f}"
            )

        # --- get analysis wrappers ---
        def _analysis(run_or_analysis, name=None):
            try:
                if hasattr(run_or_analysis, 'joined_instance_stat_table'):
                    return run_or_analysis
            except Exception:
                pass
            from magnet.backends.helm.helm_run_analysis import HelmRunAnalysis
            return HelmRunAnalysis(run_or_analysis, name=name)

        A = _analysis(getattr(self, 'run_a', getattr(self, 'a', None)), name=a_name)
        B = _analysis(getattr(self, 'run_b', getattr(self, 'b', None)), name=b_name)

        jt_a = A.joined_instance_stat_table(assert_assumptions=True)
        jt_b = B.joined_instance_stat_table(assert_assumptions=True)

        # Must match instance_summary_dict()'s internal keying
        def _row_key(r):
            n = (r['stat'].get('name') or {})
            pert = n.get('perturbation', None)
            pert_name = pert.get('name', None) if isinstance(pert, dict) else None
            return (
                r['instance_id'],
                r['train_trial_index'],
                r.get('perturbation_id', None),
                n.get('name', None),
                n.get('split', None),
                n.get('sub_split', None),
                pert_name,
            )

        lut_a = {_row_key(r): r for r in jt_a}
        lut_b = {_row_key(r): r for r in jt_b}

        def _clip(text, n):
            if text is None:
                return None
            text = str(text).replace('\r\n', '\n')
            return (text[:n] + '…') if len(text) > n else text

        def _prompt(rs):
            return _clip(((rs.get('request') or {}).get('prompt', None)), prompt_chars)

        def _completion(rs):
            comps = ((rs.get('result') or {}).get('completions', None)) or []
            txt = comps[0].get('text', None) if comps else None
            return _clip(txt, completion_chars)

        def _instance_input(rs):
            inst = rs.get('instance') or {}
            inp = inst.get('input') or {}
            if isinstance(inp, dict) and 'text' in inp:
                return _clip(inp.get('text', None), input_chars)
            return _clip(inp, input_chars)

        def _metric_label_from_key(k):
            # k layout:
            # (instance_id, tti, variant_pert_id, metric_name, split, sub_split, stat_pert_name)
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

        lines = []
        lines.append(f"Instance-level diff: {a_name} vs {b_name}")
        lines.append(
            f"  coverage: A={cov['n_a']} B={cov['n_b']} "
            f"isect={cov['n_isect']} union={cov['n_union']} onlyA={cov['only_a']} onlyB={cov['only_b']}"
        )
        lines.append(
            f"  means: comparable={means['comparable']} mismatched={means['mismatched']} "
            f"agree_ratio={means['agree_ratio']:.3f}"
        )
        lines.append(
            f"        unperturbed={means['agree_ratio_unperturbed']:.3f} "
            f"perturbed={means['agree_ratio_perturbed']:.3f}"
        )

        tops = info.get('top_mismatches', []) or []
        if tops:
            lines.append("  top mismatches:")
            for idx, item in enumerate(tops[:top_n]):
                k = item['key']
                a = float(item['a'])
                b = float(item['b'])
                abs_d = float(item['abs_delta'])
                signed_d = b - a  # signed delta, B - A

                metric_label = _metric_label_from_key(k)
                lines.append(f"    metric: {metric_label}")
                lines.append(f"    key: {k}")
                lines.append(f"    A={a:.4g}  B={b:.4g}  Δ(B-A)={signed_d:.4g}  |Δ|={abs_d:.4g}")

                if show_details and idx < show_details:
                    ra = lut_a.get(k, None)
                    rb = lut_b.get(k, None)

                    rs_a = ra['request_state'] if ra is not None else None
                    rs_b = rb['request_state'] if rb is not None else None

                    pa = _prompt(rs_a) if rs_a is not None else None
                    pb = _prompt(rs_b) if rs_b is not None else None
                    prompts_equal = (pa == pb)

                    lines.append(f"    prompts_equal={prompts_equal}")

                    # Inputs & completions
                    if rs_a is not None:
                        lines.append(f"      [{a_name}] input: {_instance_input(rs_a)}")
                        lines.append(f"      [{a_name}] completion: {_completion(rs_a)}")
                    if rs_b is not None:
                        lines.append(f"      [{b_name}] input: {_instance_input(rs_b)}")
                        lines.append(f"      [{b_name}] completion: {_completion(rs_b)}")

                    # Prompt(s)
                    if prompts_equal:
                        if pa is not None:
                            lines.append("      prompt:")
                            for pline in pa.splitlines():
                                lines.append(f"        {pline}")
                    else:
                        if pa is not None:
                            lines.append(f"      prompt [{a_name}]:")
                            for pline in pa.splitlines():
                                lines.append(f"        {pline}")
                        if pb is not None:
                            lines.append(f"      prompt [{b_name}]:")
                            for pline in pb.splitlines():
                                lines.append(f"        {pline}")

                # spacer between entries
                lines.append("")

        return "\n".join(lines).rstrip()


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
