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
    >>> import json
    >>> from magnet.backends.helm.helm_run_analysis import HelmRunAnalysis
    >>> from magnet.backends.helm.helm_run_diff import HelmRunDiff
    >>> class _DummyJoined:
    ...     def __init__(self):
    ...         self.row_by_key = {}
    ...     def __iter__(self):
    ...         return iter(self.row_by_key.values())
    >>> def _ana(run_spec, stats, request_states):
    ...     a = HelmRunAnalysis.__new__(HelmRunAnalysis)
    ...     a._raw_cache = {}
    ...     a._cache = {}
    ...     a.run = None
    ...     a.name = None
    ...     a.run_spec = lambda: run_spec
    ...     a.scenario = lambda: {'class_name': 'ToyScenario', 'output_path': 'tmp/a'}
    ...     a.scenario_state = lambda: {'request_states': request_states}
    ...     a.stats = lambda: stats
    ...     a.joined_instance_stat_table = lambda *args, **kwargs: _DummyJoined()
    ...     return a
    >>> rs = [{'instance': {'id': 'id1', 'split': 'test', 'input': {'text': 'Q'}}, 'train_trial_index': 0, 'request': {'prompt': 'P'}, 'result': {'completions': [{'text': 'A'}]}}]
    >>> stats_a = [{'name': {'name': 'exact_match', 'split': 'test'}, 'count': 1, 'mean': 1.0}]
    >>> stats_b = [{'name': {'name': 'exact_match', 'split': 'test'}, 'count': 1, 'mean': 0.0}]
    >>> spec_a = {'name': 'toy', 'adapter_spec': {'model': 'm'}, 'metric_specs': [{'class_name': 'M0', 'args': {}}]}
    >>> spec_b = {'name': 'toy', 'adapter_spec': {'model': 'm', 'model_deployment': 'huggingface/m'}, 'metric_specs': [{'class_name': 'M1', 'args': {}}]}
    >>> rd = HelmRunDiff(_ana(spec_a, stats_a, rs), _ana(spec_b, stats_b, rs), a_name='A', b_name='B')
    >>> info = rd.summary_dict(level=20)
    >>> assert info['run_spec_name_ok'] is True
    >>> assert info['dataset_overlap']['base_iou'] == 1.0
    >>> assert info['value_agreement']['overall']['mismatched'] == 1
    >>> assert isinstance(info['diagnosis']['label'], str)
    >>> _ = json.dumps(info, allow_nan=False)

"""

from __future__ import annotations

import math
import ubelt as ub

from collections import Counter
from dataclasses import dataclass
from magnet.backends.helm.util import helm_hashers
from magnet.backends.helm.util import helm_metrics
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


def _walker_diff(a: Any, b: Any, *, max_paths: int = 12) -> dict[str, Any]:
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
    info.pop('passlist', None)

    def _format_path(path: Iterable[Any]) -> str:
        return '.'.join(map(str, path))

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
        return lines[:max_items] + [f'<{remain} more not shown>']

    unique1 = sorted(info.get('unique1', []))
    unique2 = sorted(info.get('unique2', []))
    faillist = sorted(info.get('faillist', []), key=lambda d: d.path)

    out = info | {
        'unique1': _truncate(
            [
                _format_path(p) + ': ' + _smart_truncate(repr(walker_a[p]), 80)
                for p in unique1
            ],
            max_paths,
        ),
        'unique2': _truncate(
            [
                _format_path(p) + ': ' + _smart_truncate(repr(walker_b[p]), 80)
                for p in unique2
            ],
            max_paths,
        ),
        'faillist': _truncate(
            [
                f'{_format_path(d.path)}: {_smart_truncate(repr(d.value1), 80)} != {_smart_truncate(repr(d.value2), 80)}'
                for d in faillist
            ],
            max_paths,
        ),
    }
    return out


def _walker_diff_paths(a: Any, b: Any) -> dict[str, list[str]]:
    """Return full path-level differences (untruncated), path-only.

    The output is intentionally JSON-friendly and stable for diagnostics.
    """
    walker_a = ub.IndexableWalker(a)
    walker_b = ub.IndexableWalker(b)
    info = walker_a.diff(walker_b)

    def _format_path(path: Iterable[Any]) -> str:
        return '.'.join(map(str, path))

    unique1 = sorted(_format_path(p) for p in info.get('unique1', []))
    unique2 = sorted(_format_path(p) for p in info.get('unique2', []))
    faillist = sorted(
        _format_path(d.path) for d in info.get('faillist', [])
    )
    return {
        'unique1': unique1,
        'unique2': unique2,
        'faillist': faillist,
    }


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


def _coerce_path_token(tok: str) -> str | int:
    if tok.isdigit():
        try:
            return int(tok)
        except Exception:
            return tok
    return tok


def _path_get(obj: Any, path: str) -> tuple[Any, bool]:
    """Best-effort dotted-path getter supporting dict/list traversal."""
    cur = obj
    for raw_tok in path.split('.'):
        tok = _coerce_path_token(raw_tok)
        if isinstance(cur, dict):
            if tok in cur:
                cur = cur[tok]
            elif isinstance(tok, int) and str(tok) in cur:
                cur = cur[str(tok)]
            else:
                return None, False
        elif isinstance(cur, (list, tuple)):
            if isinstance(tok, int) and 0 <= tok < len(cur):
                cur = cur[tok]
            else:
                return None, False
        else:
            return None, False
    return cur, True


def _path_value_examples(
    a_obj: Any,
    b_obj: Any,
    paths: list[str],
    *,
    max_items: int = 20,
) -> list[dict[str, Any]]:
    """Return path-level value pairs for selected diff paths."""
    examples: list[dict[str, Any]] = []
    for p in sorted(paths):
        rec: dict[str, Any] = {'path': p}
        va, oka = _path_get(a_obj, p)
        vb, okb = _path_get(b_obj, p)
        rec['a'] = va if oka else None
        rec['b'] = vb if okb else None
        rec['a_found'] = bool(oka)
        rec['b_found'] = bool(okb)
        examples.append(rec)
        if len(examples) >= max_items:
            break
    return _json_compatible(examples)


def _json_compatible(obj: Any) -> Any:
    """Recursively coerce to strict JSON-compatible types.

    Notably:
    - tuples/sets -> lists
    - non-finite floats -> None
    - unknown objects -> string repr
    """
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _json_compatible(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_compatible(v) for v in obj]
    try:
        # common dataclass / custom key cases
        if hasattr(obj, 'as_tuple') and callable(getattr(obj, 'as_tuple')):
            return _json_compatible(list(obj.as_tuple()))
    except Exception:
        pass
    try:
        return ub.urepr(obj, nl=0, compact=1)
    except Exception:
        return str(obj)


def _preview_list(items: list[str], *, limit: int = 20) -> list[str]:
    """Return a stable preview list with an optional '<N more>' suffix."""
    if limit <= 0 or len(items) <= limit:
        return items
    remain = len(items) - limit
    return items[:limit] + [f'<{remain} more not shown>']


_RUNSPEC_EXEC_ADAPTER_NOISE_FIELDS = {
    # Added in newer HELM formats; often default/no-op in practice.
    'chain_of_thought_prefix',
    'chain_of_thought_suffix',
    'global_suffix',
    'num_trials',
}


def _classify_run_spec_path(path: str) -> str:
    """Classify run-spec diff paths into semantic buckets."""
    if path.startswith('metric_specs') or path.startswith('groups'):
        return 'evaluation'
    if path.startswith('adapter_spec.'):
        parts = path.split('.')
        field = parts[1] if len(parts) > 1 else ''
        if field in _RUNSPEC_EXEC_ADAPTER_NOISE_FIELDS:
            return 'nonsemantic'
        return 'execution'
    if path.startswith('scenario_spec') or path.startswith('data_augmenter_spec'):
        return 'execution'
    if path in {'name'}:
        return 'nonsemantic'
    return 'other'


def _classify_scenario_path(path: str) -> str:
    """Classify scenario diff paths into semantic buckets."""
    # scenario.output_path is environment-local and should not affect content
    if path == 'output_path' or path.endswith('.output_path'):
        return 'nonsemantic'
    return 'semantic'


def _canonicalize_metric_spec_for_semantic_diff(metric_spec: Any) -> Any:
    """Normalize one metric spec for order-insensitive semantic comparison."""
    if not isinstance(metric_spec, dict):
        return helm_hashers.canonicalize_for_hashing(metric_spec)
    out = {
        'class_name': metric_spec.get('class_name', None),
        'args': helm_hashers.canonicalize_for_hashing(
            metric_spec.get('args', None)
        ),
    }
    return out


def _canonicalize_run_spec_for_semantic_diff(run_spec: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize run_spec with order-insensitive handling for select lists."""
    spec = helm_hashers.canonicalize_for_hashing(run_spec)
    if not isinstance(spec, dict):
        return {'_invalid_spec': spec}
    spec = dict(spec)

    metric_specs = spec.get('metric_specs', None)
    if isinstance(metric_specs, list):
        canon_items = [
            _canonicalize_metric_spec_for_semantic_diff(ms)
            for ms in metric_specs
        ]
        canon_items = sorted(
            canon_items, key=lambda x: helm_hashers.stable_hash36(x)
        )
        spec['metric_specs'] = canon_items

    groups = spec.get('groups', None)
    if isinstance(groups, list):
        spec['groups'] = sorted(groups, key=lambda x: str(x))

    return spec


def _metric_specs_multiset_delta(
    metric_specs_a: Any,
    metric_specs_b: Any,
    *,
    short_hash: int = 12,
    max_items: int = 20,
) -> dict[str, Any]:
    """Order-insensitive multiset delta for run_spec.metric_specs."""
    specs_a = metric_specs_a if isinstance(metric_specs_a, list) else []
    specs_b = metric_specs_b if isinstance(metric_specs_b, list) else []

    def _make_id(ms: Any) -> tuple[str, dict[str, Any]]:
        canon = _canonicalize_metric_spec_for_semantic_diff(ms)
        sid = helm_hashers.stable_hash36(canon)[:short_hash]
        if isinstance(canon, dict):
            class_name = canon.get('class_name', None)
            args = canon.get('args', None)
        else:
            class_name = None
            args = canon
        rec = {
            'id': sid,
            'class_name': class_name,
            'args': args,
            'preview': _short_urepr(canon, max_chars=160),
        }
        return sid, rec

    id_to_rec: dict[str, dict[str, Any]] = {}
    a_ids: list[str] = []
    b_ids: list[str] = []
    for ms in specs_a:
        sid, rec = _make_id(ms)
        id_to_rec.setdefault(sid, rec)
        a_ids.append(sid)
    for ms in specs_b:
        sid, rec = _make_id(ms)
        id_to_rec.setdefault(sid, rec)
        b_ids.append(sid)

    a_counter = Counter(a_ids)
    b_counter = Counter(b_ids)
    keys = sorted(set(a_counter) | set(b_counter))
    added = []
    removed = []
    for sid in keys:
        ca = a_counter.get(sid, 0)
        cb = b_counter.get(sid, 0)
        if cb > ca:
            added.append(id_to_rec[sid] | {'count': cb - ca})
        if ca > cb:
            removed.append(id_to_rec[sid] | {'count': ca - cb})

    added = sorted(added, key=lambda r: (str(r.get('class_name')), r['id']))
    removed = sorted(removed, key=lambda r: (str(r.get('class_name')), r['id']))
    return _json_compatible(
        {
            'n_a': len(specs_a),
            'n_b': len(specs_b),
            'n_added': sum(r['count'] for r in added),
            'n_removed': sum(r['count'] for r in removed),
            'added': _preview_list(
                [ub.urepr(r, nl=0, compact=1) for r in added], limit=max_items
            ),
            'removed': _preview_list(
                [ub.urepr(r, nl=0, compact=1) for r in removed], limit=max_items
            ),
            'added_structured': added[:max_items],
            'removed_structured': removed[:max_items],
            'equal_as_multiset': (len(added) == 0 and len(removed) == 0),
        }
    )


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


def _key_to_serializable(key: Any) -> Any:
    """Convert various key types (dataclasses, tuples) into JSON-friendly types.

    - If object has ``as_tuple()``, use that and return a list.
    - If it's a tuple, return a list (JSON will accept either but list is explicit).
    - Otherwise fallback to string repr.
    """
    # dataclass-like keys (InstanceStatKey, InstanceVariantKey) implement as_tuple
    try:
        if hasattr(key, 'as_tuple') and callable(getattr(key, 'as_tuple')):
            return list(key.as_tuple())
    except Exception:
        pass
    if isinstance(key, tuple):
        return list(key)
    # lists are already JSON-safe
    if isinstance(key, list):
        return key
    # fallback: use a stable repr
    try:
        return ub.urepr(key, nl=0, compact=1)
    except Exception:
        return str(key)


def dataset_overlap_from_request_states(
    request_states_a: list[dict[str, Any]],
    request_states_b: list[dict[str, Any]],
    *,
    short_hash: int = 16,
    max_examples: int = 5,
) -> dict[str, Any]:
    """Compare two request_state lists at dataset/prompt/completion level.

    This is a pure function used by :meth:`HelmRunDiff.dataset_overlap_summary`.

    Example:
        >>> rs_a = [
        ...     {
        ...         'instance': {'id': 'id1', 'split': 'test', 'input': {'text': 'Q1'}},
        ...         'train_trial_index': 0,
        ...         'request': {'prompt': 'P1'},
        ...         'result': {'completions': [{'text': 'A1'}]},
        ...     },
        ...     {
        ...         'instance': {
        ...             'id': 'id1', 'split': 'test', 'input': {'text': 'Q1'},
        ...             'perturbation': {'name': 'dialect', 'prob': 1.0},
        ...         },
        ...         'train_trial_index': 0,
        ...         'request': {'prompt': 'P1-d'},
        ...         'result': {'completions': [{'text': 'A1d'}]},
        ...     },
        ... ]
        >>> rs_b = [
        ...     {
        ...         'instance': {'id': 'id1', 'split': 'test', 'input': {'text': 'Q1'}},
        ...         'train_trial_index': 0,
        ...         'request': {'prompt': 'P1x'},
        ...         'result': {'completions': [{'text': 'A1'}]},
        ...     },
        ... ]
        >>> info = dataset_overlap_from_request_states(rs_a, rs_b, max_examples=2)
        >>> assert info['base_coverage']['n_isect'] == 1
        >>> assert info['variant_coverage']['only_a'] == 1
        >>> assert info['content_equality']['prompt']['equal_ratio'] == 0.0
        >>> assert isinstance(info['mismatch_examples']['prompt'], list)
    """

    def _coerce_int(x: Any) -> int | None:
        try:
            if x is None:
                return None
            if isinstance(x, bool):
                return int(x)
            if isinstance(x, int):
                return x
            if isinstance(x, float) and x.is_integer():
                return int(x)
            if isinstance(x, str) and x.isdigit():
                return int(x)
        except Exception:
            pass
        return None

    def _base_key(rs: dict[str, Any]) -> tuple[Any, ...]:
        inst = rs.get('instance') or {}
        return (
            inst.get('id', None),
            _coerce_int(rs.get('train_trial_index', None)),
            inst.get('split', None),
        )

    def _variant_key(rs: dict[str, Any]) -> tuple[Any, ...]:
        inst = rs.get('instance') or {}
        pid = helm_hashers.perturbation_id(
            inst.get('perturbation', None), short_hash=short_hash
        )
        return _base_key(rs) + (pid,)

    def _index_unique(
        rows: list[dict[str, Any]], key_fn
    ) -> tuple[dict[tuple[Any, ...], dict[str, Any]], int]:
        out: dict[tuple[Any, ...], dict[str, Any]] = {}
        duplicates = 0
        for rs in rows:
            k = key_fn(rs)
            if k in out:
                duplicates += 1
                continue
            out[k] = rs
        return out, duplicates

    def _extract_input(rs: dict[str, Any]) -> Any:
        inst = rs.get('instance') or {}
        inp = inst.get('input', None)
        if isinstance(inp, dict) and 'text' in inp:
            return inp.get('text', None)
        return inp

    def _extract_prompt(rs: dict[str, Any]) -> Any:
        req = rs.get('request') or {}
        return req.get('prompt', None)

    def _extract_completion(rs: dict[str, Any]) -> Any:
        res = rs.get('result') or {}
        comps = res.get('completions') or []
        if not comps:
            return None
        first = comps[0]
        if isinstance(first, dict):
            return first.get('text', None)
        return first

    def _summarize(
        map_a: dict[tuple[Any, ...], dict[str, Any]],
        map_b: dict[tuple[Any, ...], dict[str, Any]],
        keys: set[tuple[Any, ...]],
        *,
        extractor,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        def _keysort(k: tuple[Any, ...]) -> str:
            try:
                return ub.urepr(k, nl=0, compact=1)
            except Exception:
                return str(k)

        comparable = 0
        mismatched = 0
        examples: list[dict[str, Any]] = []
        for k in sorted(keys, key=_keysort):
            va = extractor(map_a[k])
            vb = extractor(map_b[k])
            comparable += 1
            if va != vb:
                mismatched += 1
                if len(examples) < max_examples:
                    examples.append(
                        {
                            'key': _key_to_serializable(k),
                            'a': _short_urepr(va, max_chars=180),
                            'b': _short_urepr(vb, max_chars=180),
                        }
                    )
        return (
            {
                'comparable': comparable,
                'mismatched': mismatched,
                'equal_ratio': ratio(comparable, mismatched),
            },
            examples,
        )

    base_a, dup_base_a = _index_unique(request_states_a, _base_key)
    base_b, dup_base_b = _index_unique(request_states_b, _base_key)
    var_a, dup_var_a = _index_unique(request_states_a, _variant_key)
    var_b, dup_var_b = _index_unique(request_states_b, _variant_key)

    cov_base = Coverage.from_sets(set(base_a), set(base_b))
    cov_var = Coverage.from_sets(set(var_a), set(var_b))
    isect_base = set(base_a) & set(base_b)
    isect_var = set(var_a) & set(var_b)

    base_iou = (
        cov_base.n_isect / cov_base.n_union if cov_base.n_union else None
    )
    variant_iou = (
        cov_var.n_isect / cov_var.n_union if cov_var.n_union else None
    )

    input_eq, ex_input = _summarize(
        base_a, base_b, isect_base, extractor=_extract_input
    )
    prompt_eq, ex_prompt = _summarize(
        var_a, var_b, isect_var, extractor=_extract_prompt
    )
    completion_eq, ex_completion = _summarize(
        var_a, var_b, isect_var, extractor=_extract_completion
    )

    out = {
        'base_coverage': cov_base.__dict__,
        'variant_coverage': cov_var.__dict__,
        'base_iou': base_iou,
        'variant_iou': variant_iou,
        'content_equality': {
            'input': input_eq,
            'prompt': prompt_eq,
            'completion': completion_eq,
        },
        'duplicates': {
            'a': {'base': dup_base_a, 'variant': dup_var_a},
            'b': {'base': dup_base_b, 'variant': dup_var_b},
        },
        'mismatch_examples': {
            'input': ex_input,
            'prompt': ex_prompt,
            'completion': ex_completion,
        },
    }
    return _json_compatible(out)


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

        # 2) run spec dict hash (strict)
        spec_hash_a = helm_hashers.stable_hash36(
            helm_hashers.canonicalize_for_hashing(a_spec)
        )
        spec_hash_b = helm_hashers.stable_hash36(
            helm_hashers.canonicalize_for_hashing(b_spec)
        )
        run_spec_dict_ok = spec_hash_a == spec_hash_b
        if run_spec_dict_ok:
            spec_path_info: dict[str, list[str]] = {
                'unique1': [],
                'unique2': [],
                'faillist': [],
            }
        else:
            spec_path_info = _walker_diff_paths(a_spec, b_spec)
        if level == 0:
            spec_diff_paths = None
        else:
            spec_diff_paths = (
                {} if run_spec_dict_ok else _walker_diff(a_spec, b_spec)
            )

        # 2b) run spec semantic hash (order-insensitive for metric lists)
        a_spec_sem = _canonicalize_run_spec_for_semantic_diff(a_spec)
        b_spec_sem = _canonicalize_run_spec_for_semantic_diff(b_spec)
        spec_sem_hash_a = helm_hashers.stable_hash36(a_spec_sem)
        spec_sem_hash_b = helm_hashers.stable_hash36(b_spec_sem)
        run_spec_semantic_dict_ok = spec_sem_hash_a == spec_sem_hash_b
        if run_spec_semantic_dict_ok:
            spec_sem_path_info: dict[str, list[str]] = {
                'unique1': [],
                'unique2': [],
                'faillist': [],
            }
        else:
            spec_sem_path_info = _walker_diff_paths(a_spec_sem, b_spec_sem)
        if level == 0:
            spec_sem_diff_paths = None
        else:
            spec_sem_diff_paths = (
                {}
                if run_spec_semantic_dict_ok
                else _walker_diff(a_spec_sem, b_spec_sem)
            )
        run_spec_semantic = self._run_spec_semantic_summary(
            path_info=spec_sem_path_info,
            a_spec=a_spec,
            b_spec=b_spec,
        )

        # 3) scenario check with unknown semantics
        scen_known = bool(a_scen) and bool(b_scen)
        if not scen_known:
            scenario_ok: bool | None = None
            scenario_hash_a = None
            scenario_hash_b = None
            scen_diff_paths = []
            scen_path_info: dict[str, list[str]] | None = None
        else:
            scenario_hash_a = helm_hashers.stable_hash36(
                helm_hashers.canonicalize_for_hashing(a_scen)
            )
            scenario_hash_b = helm_hashers.stable_hash36(
                helm_hashers.canonicalize_for_hashing(b_scen)
            )
            scenario_ok = scenario_hash_a == scenario_hash_b
            if scenario_ok:
                scen_path_info = {'unique1': [], 'unique2': [], 'faillist': []}
            else:
                scen_path_info = _walker_diff_paths(a_scen, b_scen)
            if level == 0:
                scen_diff_paths = None
            else:
                scen_diff_paths = (
                    {} if scenario_ok else _walker_diff(a_scen, b_scen)
                )
        scenario_semantic = self._scenario_semantic_summary(
            scenario_ok=scenario_ok, path_info=scen_path_info
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
        dataset_summary: dict[str, Any] | None = None
        if level >= 5:
            try:
                dataset_summary = self.dataset_overlap_summary(max_examples=5)
            except Exception as ex:  # nocover
                dataset_summary = {'error': repr(ex)}
        diagnosis = self._diagnose_repro(
            run_spec_name_ok=run_spec_name_ok,
            run_spec_semantic=run_spec_semantic,
            scenario_semantic=scenario_semantic,
            dataset_overlap=dataset_summary,
            value_summary=value_summary,
        )

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
            'run_spec_semantic_dict_ok': run_spec_semantic_dict_ok,
            'run_spec_semantic_hash_a': spec_sem_hash_a,
            'run_spec_semantic_hash_b': spec_sem_hash_b,
            'run_spec_diff_paths_semantic': spec_sem_diff_paths,
            'run_spec_semantic': run_spec_semantic,
            'scenario_ok': scenario_ok,
            'scenario_hash_a': scenario_hash_a,
            'scenario_hash_b': scenario_hash_b,
            'scenario_diff_paths': scen_diff_paths,
            'scenario_semantic': scenario_semantic,
            'stats_coverage_by_name': cov_name.__dict__,
            'stats_coverage_by_name_count': cov_name_count.__dict__,
            'value_agreement': value_summary,
            'dataset_overlap': dataset_summary,
            'diagnosis': diagnosis,
        }

        if level >= 20:
            # Optional: include instance-level summary in the dict.
            try:
                out['instance_value_agreement'] = self.instance_summary_dict(
                    top_n=10
                )
            except Exception as ex:  # nocover
                out['instance_value_agreement'] = {'error': repr(ex)}

        out = _json_compatible(out)
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
            spec_name_a = info['run_spec_name_a']
            spec_name_b = info['run_spec_name_b']
            if spec_name_a == spec_name_b:
                line_name = spec_name_a
            else:
                line_name = '{spec_name_a} // {spec_name_b}'
            writer(
                f'{_format_bool(ok)} {self.a_name} vs {self.b_name} {line_name} '
                f'spec={_format_bool(info["run_spec_dict_ok"])} '
                f'stats={cov["n_isect"]}/{cov["n_union"]} '
                f'agree={_fmt(agree)}'
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
                if not info.get('run_spec_semantic_dict_ok', True):
                    writer(
                        f'  semantic diff: {ub.urepr(info["run_spec_diff_paths_semantic"])}'
                    )
                if (not info['run_spec_dict_ok']) and level >= 20:
                    writer(
                        f'  strict diff: {ub.urepr(info["run_spec_diff_paths"])}'
                    )
                rs_sem = info.get('run_spec_semantic', {}) or {}
                dep = rs_sem.get('deployment', {}) or {}
                if dep.get('changed', False):
                    writer(
                        f'  deployment: A={_short_urepr(dep.get("a", None), 80)} '
                        f'B={_short_urepr(dep.get("b", None), 80)}'
                    )

            if info['scenario_ok'] is None:
                writer(
                    'Scenario: ⚠️  unknown (missing scenario.json in one or both runs)'
                )
            else:
                if level >= 15:
                    writer(
                        f'Scenario: {_format_bool(bool(info["scenario_ok"]))}'
                    )
                    if info['scenario_ok'] is False:
                        writer(
                            f'  diff: {ub.urepr(info["scenario_diff_paths"])}'
                        )
            scen_sem = info.get('scenario_semantic', {}) or {}
            if scen_sem.get('known', False) and level >= 15:
                writer(
                    f'Scenario semantic: {_format_bool(bool(scen_sem.get("semantic_ok", False)))}'
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
                f'agree_ratio={_fmt(ov["agree_ratio"])}'
            )
            for cls in ('core', 'bookkeeping', 'untracked'):
                s = info['value_agreement']['by_class'][cls]
                writer(
                    f'  {cls:11s}: comparable={s["comparable"]} mismatched={s["mismatched"]} '
                    f'agree_ratio={_fmt(s["agree_ratio"])}'
                )

            if level >= 20:
                top = info['value_agreement'].get('top_mismatches', [])
                if top:
                    writer('  top mismatches:')
                    for r in top:
                        writer(
                            f'    {r["key"]}  A={_fmt(r["a"])}  B={_fmt(r["b"])}  |Δ|={_fmt(r["abs_delta"])}'
                        )

            if level >= 15:
                ds = info.get('dataset_overlap', None)
                if isinstance(ds, dict) and 'error' not in ds:
                    writer('')
                    writer('Dataset overlap:')
                    writer(
                        f'  base_iou={_fmt(ds.get("base_iou"))} '
                        f'variant_iou={_fmt(ds.get("variant_iou"))}'
                    )
                    ce = ds.get('content_equality', {}) or {}
                    for field in ('input', 'prompt', 'completion'):
                        row = ce.get(field, {}) or {}
                        writer(
                            f'  {field:10s}: comparable={row.get("comparable")} '
                            f'mismatched={row.get("mismatched")} equal_ratio={_fmt(row.get("equal_ratio"))}'
                        )
                elif isinstance(ds, dict) and 'error' in ds:
                    writer(f'Dataset overlap: ⚠️  {ds["error"]}')

            if level >= 10:
                diag = info.get('diagnosis', {}) or {}
                writer('')
                writer(f'Diagnosis: {diag.get("label", "unknown")}')
                if level >= 20:
                    reasons = diag.get('reasons', [])
                    if reasons:
                        writer(f'  reasons: {ub.urepr(reasons, nl=0)}')

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
                        f'agree={_fmt(means["agree_ratio"])} (unpert={_fmt(means["agree_ratio_unperturbed"])}, '
                        f'pert={_fmt(means["agree_ratio_perturbed"])})'
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

    def _run_spec_semantic_summary(
        self,
        *,
        path_info: dict[str, list[str]],
        a_spec: dict[str, Any],
        b_spec: dict[str, Any],
    ) -> dict[str, Any]:
        """Classify run-spec differences into semantic buckets."""
        all_paths = sorted(
            set(path_info.get('unique1', []))
            | set(path_info.get('unique2', []))
            | set(path_info.get('faillist', []))
        )
        by_class: dict[str, list[str]] = {
            'execution': [],
            'evaluation': [],
            'nonsemantic': [],
            'other': [],
        }
        for p in all_paths:
            by_class[_classify_run_spec_path(p)].append(p)

        deployment_paths = [
            p
            for p in all_paths
            if p.startswith('adapter_spec.model_deployment')
        ]
        deployment_a = (
            (a_spec.get('adapter_spec', {}) or {}).get(
                'model_deployment', None
            )
            if isinstance(a_spec, dict)
            else None
        )
        deployment_b = (
            (b_spec.get('adapter_spec', {}) or {}).get(
                'model_deployment', None
            )
            if isinstance(b_spec, dict)
            else None
        )
        deployment_changed = (deployment_a != deployment_b) or any(
            p.startswith('adapter_spec.model_deployment') for p in all_paths
        )
        metric_specs_delta = _metric_specs_multiset_delta(
            (a_spec or {}).get('metric_specs', None),
            (b_spec or {}).get('metric_specs', None),
            short_hash=self.short_hash,
            max_items=20,
        )
        evaluation_changed = bool(by_class['evaluation']) or (
            not bool(metric_specs_delta.get('equal_as_multiset', True))
        )
        execution_ok = len(by_class['execution']) == 0
        evaluation_only = (
            (len(all_paths) > 0)
            and execution_ok
            and (evaluation_changed or len(by_class['nonsemantic']) > 0)
        )
        return _json_compatible(
            {
                'n_total_paths': len(all_paths),
                'execution_ok': execution_ok,
                'evaluation_only': evaluation_only,
                'evaluation_changed': evaluation_changed,
                'deployment_changed': deployment_changed,
                'deployment': {
                    'a': deployment_a,
                    'b': deployment_b,
                    'changed': deployment_changed,
                },
                'counts': {
                    k: len(v)
                    for k, v in by_class.items()
                },
                'deployment_paths': _preview_list(deployment_paths, limit=20),
                'execution_paths': _preview_list(
                    by_class['execution'], limit=20
                ),
                'execution_value_examples': _path_value_examples(
                    a_spec, b_spec, by_class['execution'], max_items=20
                ),
                'evaluation_paths': _preview_list(
                    by_class['evaluation'], limit=20
                ),
                'metric_specs_multiset_delta': metric_specs_delta,
                'nonsemantic_paths': _preview_list(
                    by_class['nonsemantic'], limit=20
                ),
                'other_paths': _preview_list(by_class['other'], limit=20),
            }
        )

    def _scenario_semantic_summary(
        self,
        *,
        scenario_ok: bool | None,
        path_info: dict[str, list[str]] | None,
    ) -> dict[str, Any]:
        """Classify scenario differences into semantic/nonsemantic buckets."""
        if scenario_ok is None:
            return {
                'known': False,
                'strict_ok': None,
                'semantic_ok': None,
                'counts': {'semantic': 0, 'nonsemantic': 0},
                'semantic_paths': [],
                'nonsemantic_paths': [],
            }

        path_info = path_info or {'unique1': [], 'unique2': [], 'faillist': []}
        all_paths = sorted(
            set(path_info.get('unique1', []))
            | set(path_info.get('unique2', []))
            | set(path_info.get('faillist', []))
        )
        semantic_paths = [
            p for p in all_paths if _classify_scenario_path(p) == 'semantic'
        ]
        nonsemantic_paths = [
            p for p in all_paths if _classify_scenario_path(p) == 'nonsemantic'
        ]
        semantic_ok = bool(scenario_ok) or (len(semantic_paths) == 0)
        return _json_compatible(
            {
                'known': True,
                'strict_ok': bool(scenario_ok),
                'semantic_ok': semantic_ok,
                'counts': {
                    'semantic': len(semantic_paths),
                    'nonsemantic': len(nonsemantic_paths),
                },
                'semantic_paths': _preview_list(semantic_paths, limit=20),
                'nonsemantic_paths': _preview_list(
                    nonsemantic_paths, limit=20
                ),
            }
        )

    def dataset_overlap_summary(self, *, max_examples: int = 5) -> dict[str, Any]:
        """Compare scenario_state request datasets between runs.

        Example:
            >>> from magnet.backends.helm.helm_run_analysis import HelmRunAnalysis
            >>> ana = HelmRunAnalysis.__new__(HelmRunAnalysis)
            >>> ana._raw_cache = {}
            >>> ana._cache = {}
            >>> ana.run = None
            >>> ana.name = None
            >>> ana.scenario_state = lambda: {'request_states': [
            ...     {'instance': {'id': 'id1', 'split': 'test', 'input': {'text': 'Q1'}},
            ...      'train_trial_index': 0,
            ...      'request': {'prompt': 'P1'},
            ...      'result': {'completions': [{'text': 'A1'}]}},
            ... ]}
            >>> rd = HelmRunDiff(ana, ana)
            >>> ds = rd.dataset_overlap_summary(max_examples=2)
            >>> assert ds['base_iou'] == 1.0
            >>> assert ds['variant_iou'] == 1.0
            >>> assert ds['content_equality']['input']['equal_ratio'] == 1.0
        """
        cache_key = ('dataset_overlap_summary', max_examples, self.short_hash)
        if cache_key in self._cache:
            return self._cache[cache_key]
        rs_a = (self.a.scenario_state() or {}).get('request_states', []) or []
        rs_b = (self.b.scenario_state() or {}).get('request_states', []) or []
        out = dataset_overlap_from_request_states(
            rs_a,
            rs_b,
            short_hash=self.short_hash,
            max_examples=max_examples,
        )
        out = _json_compatible(out)
        self._cache[cache_key] = out
        return out

    def _diagnose_repro(
        self,
        *,
        run_spec_name_ok: bool,
        run_spec_semantic: dict[str, Any],
        scenario_semantic: dict[str, Any],
        dataset_overlap: dict[str, Any] | None,
        value_summary: dict[str, Any],
    ) -> dict[str, Any]:
        """High-level diagnosis for reproducibility triage.

        Returns a primary label plus a full list of contributing reasons.
        Lower ``priority`` is earlier / more significant in the pipeline.
        """
        reasons: list[dict[str, Any]] = []

        def add_reason(name: str, priority: int, details: dict[str, Any]) -> None:
            reasons.append(
                {
                    'name': name,
                    'priority': int(priority),
                    'details': _json_compatible(details),
                }
            )

        # Priority 0: run pairing / spec-level execution blockers
        if not run_spec_name_ok:
            add_reason(
                'wrong_run_pair',
                0,
                {'run_spec_name_ok': False},
            )

        execution_ok = bool(run_spec_semantic.get('execution_ok', False))
        if not execution_ok:
            add_reason(
                'execution_spec_drift',
                0,
                {
                    'execution_paths': run_spec_semantic.get(
                        'execution_paths', []
                    ),
                    'execution_value_examples': run_spec_semantic.get(
                        'execution_value_examples', []
                    ),
                    'counts': run_spec_semantic.get('counts', {}),
                },
            )

        if bool(run_spec_semantic.get('deployment_changed', False)):
            dep = run_spec_semantic.get('deployment', {}) or {}
            add_reason(
                'deployment_drift',
                0,
                {
                    'a_value': dep.get('a', None),
                    'b_value': dep.get('b', None),
                    'execution_paths': [
                        p
                        for p in (
                            run_spec_semantic.get('execution_paths', []) or []
                        )
                        if str(p).startswith('adapter_spec.model_deployment')
                    ]
                    or (run_spec_semantic.get('deployment_paths', []) or []),
                },
            )

        scen_known = bool(scenario_semantic.get('known', False))
        scen_semantic_ok = scenario_semantic.get('semantic_ok', None)
        if scen_known and not bool(scen_semantic_ok):
            add_reason(
                'scenario_spec_drift',
                0,
                {
                    'semantic_paths': scenario_semantic.get(
                        'semantic_paths', []
                    ),
                    'counts': scenario_semantic.get('counts', {}),
                },
            )

        # Priority 1: dataset/request-state drift
        if isinstance(dataset_overlap, dict):
            if 'error' in dataset_overlap:
                add_reason(
                    'dataset_overlap_error',
                    1,
                    {'error': dataset_overlap.get('error', None)},
                )
            else:
                base_iou = dataset_overlap.get('base_iou', None)
                variant_iou = dataset_overlap.get('variant_iou', None)
                if base_iou is not None and base_iou < 1.0:
                    add_reason(
                        'dataset_instance_drift',
                        1,
                        {
                            'base_iou': base_iou,
                            'base_coverage': dataset_overlap.get(
                                'base_coverage', {}
                            ),
                        },
                    )
                if variant_iou is not None and variant_iou < 1.0:
                    add_reason(
                        'dataset_variant_drift',
                        1,
                        {
                            'variant_iou': variant_iou,
                            'variant_coverage': dataset_overlap.get(
                                'variant_coverage', {}
                            ),
                        },
                    )

                ce = dataset_overlap.get('content_equality', {}) or {}
                mex = dataset_overlap.get('mismatch_examples', {}) or {}
                for field, reason_name, pr in [
                    ('input', 'dataset_input_drift', 1),
                    ('prompt', 'request_prompt_drift', 1),
                    ('completion', 'completion_content_drift', 2),
                ]:
                    row = ce.get(field, {}) or {}
                    eq = row.get('equal_ratio', None)
                    if eq is not None and eq < 1.0:
                        details = dict(row)
                        examples = mex.get(field, None)
                        if examples:
                            details['examples'] = examples
                        add_reason(reason_name, pr, details)

        # Priority 2: evaluation schema / metric set drift
        metric_specs_delta = (
            run_spec_semantic.get('metric_specs_multiset_delta', {}) or {}
        )
        eval_paths = run_spec_semantic.get('evaluation_paths', []) or []
        evaluation_changed = bool(eval_paths) or (
            not bool(metric_specs_delta.get('equal_as_multiset', True))
        )
        if evaluation_changed:
            details = {'evaluation_paths': eval_paths}
            if not bool(metric_specs_delta.get('equal_as_multiset', True)):
                details['metric_specs_multiset_delta'] = metric_specs_delta
            add_reason(
                'evaluation_spec_drift',
                2,
                details,
            )

        # Priority 3: value-level drift (may be downstream effect)
        core = ((value_summary.get('by_class') or {}).get('core') or {})
        book = ((value_summary.get('by_class') or {}).get('bookkeeping') or {})
        core_ratio = core.get('agree_ratio', None)
        book_ratio = book.get('agree_ratio', None)

        if core_ratio is None:
            add_reason(
                'no_comparable_core_metrics',
                3,
                {'core': core},
            )
        else:
            if core_ratio < 0.995:
                add_reason(
                    'core_metric_drift',
                    3,
                    {
                        'core_agree_ratio': core_ratio,
                        'core': core,
                    },
                )
            elif (book_ratio is not None) and (book_ratio < 0.95):
                add_reason(
                    'bookkeeping_metric_drift',
                    3,
                    {
                        'core_agree_ratio': core_ratio,
                        'bookkeeping_agree_ratio': book_ratio,
                        'bookkeeping': book,
                    },
                )

        if not reasons:
            add_reason(
                'no_detected_drift',
                0,
                {
                    'core_agree_ratio': core_ratio,
                    'bookkeeping_agree_ratio': book_ratio,
                },
            )

        reasons = sorted(
            reasons,
            key=lambda r: (
                int(r.get('priority', 999)),
                str(r.get('name', '')),
            ),
        )
        min_priority = min(int(r['priority']) for r in reasons)
        primary_reason_names = [
            r['name'] for r in reasons if int(r['priority']) == min_priority
        ]
        if primary_reason_names == ['no_detected_drift']:
            label = 'reproduced'
        elif len(primary_reason_names) == 1:
            label = primary_reason_names[0]
        else:
            label = 'multiple_primary_reasons'

        return {
            'label': label,
            'primary_priority': min_priority,
            'primary_reason_names': primary_reason_names,
            'reasons': reasons,
        }

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

        out = _json_compatible(out)
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
            List of objects; each element has fields ``metric_class`` and
            ``metric`` and an ``items`` list of mismatches (sorted by |Δ|).
            Each mismatch item contains: key, a, b, abs_delta, signed_delta.

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

        # use a temporary map for bookkeeping, but the final result will be a
        # list of objects so the summary dict is JSON-serializable.
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
            metric_class, _ = helm_metrics.classify_metric(metric)
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
                    'key': _key_to_serializable(k),
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

        # convert to a JSON-friendly structure: list of group objects
        group_list: list[dict[str, Any]] = []
        for (mclass, metric), items in grouped.items():
            group_list.append(
                {
                    'metric_class': mclass,
                    'metric': metric,
                    'items': items,
                }
            )
        out = {
            'coverage': cov.__dict__,
            'means': means,
            'top_mismatches_by_group': group_list,
        }

        out = _json_compatible(out)
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
            f'agree_ratio={_fmt(means["agree_ratio"])} (unpert={_fmt(means["agree_ratio_unperturbed"])}, '
            f'pert={_fmt(means["agree_ratio_perturbed"])})'
        )

        grouped: list[dict[str, Any]] = info.get('top_mismatches_by_group', []) or []
        # grouped is now a list of group objects; convert to list for sorting

        # Choose groups to show: core first, bookkeeping second
        def _group_rank(group: dict[str, Any]) -> tuple[int, float]:
            cls = group.get('metric_class')
            items = group.get('items', [])
            cls_rank = {'core': 0, 'bookkeeping': 1, 'untracked': 2}.get(cls, 9)
            max_abs = items[0]['abs_delta'] if items else 0.0
            return (cls_rank, -max_abs)

        groups_sorted = sorted(grouped, key=_group_rank)

        # Decide which metric classes are eligible at this level
        allowed_classes = {'core'}
        if level >= 20:
            allowed_classes |= {'untracked'}
        if level >= 30:
            allowed_classes |= {'bookkeeping'}

        # Filter groups by allowed class
        filtered = [g for g in groups_sorted if g.get('metric_class') in allowed_classes]

        # If we filtered everything out (e.g. no core diffs), fall back to showing *something*
        if not filtered and groups_sorted:
            # Prefer untracked, then bookkeeping, then whatever exists
            pref_order = ['untracked', 'bookkeeping', 'core']
            for cls in pref_order:
                filtered = [g for g in groups_sorted if g.get('metric_class') == cls]
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
            core = [g for g in groups_sorted if g.get('metric_class') == 'core']
            book = [g for g in groups_sorted if g.get('metric_class') == 'bookkeeping']
            other = [
                g
                for g in groups_sorted
                if g.get('metric_class') not in {'core', 'bookkeeping'}
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
            A_getrow = getattr(A_join, 'get_row', None)
            B_getrow = getattr(B_join, 'get_row', None)

        for group in groups_sorted:
            cls = group.get('metric_class')
            metric = group.get('metric')
            items = group.get('items', [])
            writer(f'  [bold]top mismatches ({(cls, metric)!r}):[/bold]')
            for rank, item in enumerate(items[:top_n], start=1):
                k = item['key']
                a = float(item['a'])
                b = float(item['b'])
                abs_d = float(item['abs_delta'])
                signed_d = float(item['signed_delta'])

                # Try to extract split/sub_split info from key if it is tuple/list-like
                split = None
                if (isinstance(k, tuple) or isinstance(k, list)) and len(k) >= 5:
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
                    # Try to resolve row objects from the join tables.
                    # Item keys were serialized (lists); attempt to use the
                    # table's `get_row` API with a tuple form, which will
                    # reconstruct InstanceStatKey when appropriate.
                    ra = None
                    rb = None
                    if A_getrow is not None:
                        try:
                            lookup_key = tuple(k) if isinstance(k, list) else k
                            ra = A_getrow(lookup_key)
                        except Exception:
                            ra = None
                    if ra is None and A_map is not None:
                        ra = A_map.get(k, None)

                    if B_getrow is not None:
                        try:
                            lookup_key = tuple(k) if isinstance(k, list) else k
                            rb = B_getrow(lookup_key)
                        except Exception:
                            rb = None
                    if rb is None and B_map is not None:
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
                            repr(
                                ((rs_a or {}).get('request') or {}).get(
                                    'prompt', None
                                )
                            ),
                            prompt_chars,
                        )
                        if isinstance(rs_a, dict)
                        else ''
                    )
                    pb = (
                        _smart_truncate(
                            repr(
                                ((rs_b or {}).get('request') or {}).get(
                                    'prompt', None
                                )
                            ),
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
                    comp_a = _completion(rs_a) if rs_a is not None else None
                    comp_b = _completion(rs_b) if rs_b is not None else None

                    inputs_equal = (input_a == input_b) and (
                        input_a is not None
                    )
                    comps_equal = (comp_a == comp_b) and (comp_a is not None)

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
                            writer(
                                f'      [{self.a_name}] completion: {comp_a}'
                            )
                        if comp_b is not None:
                            writer(
                                f'      [{self.b_name}] completion: {comp_b}'
                            )

                    if level >= 20:
                        if isinstance(rs_a, dict) and isinstance(rs_b, dict):
                            diffs = _walker_diff(
                                rs_a, rs_b, max_paths=diff_max_items
                            )
                            writer(
                                f'      request_state_diff: {ub.urepr(diffs)}'
                            )

                writer('')


def ratio(c: int, m: int) -> float | None:
    return (1.0 - (m / c)) if c else None
