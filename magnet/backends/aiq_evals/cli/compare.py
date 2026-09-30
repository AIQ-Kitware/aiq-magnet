"""Explicit downstream comparison of two EvaluationNode results (plan M7).

Scores from different engines are not interchangeable by default. This node
compares two evidence views only when the recipe declares ``mapping``: a
statement of why the two selected measurements correspond. It also requires
both sides to be eligible. It writes ``comparison.json``; the recipe's result
node reads ``left``, ``right`` and ``difference`` from it.

Usage::

    python -m magnet.backends.aiq_evals.cli.compare \\
        --left_fpath helm/evaluation.json --right_fpath inspect/evaluation.json \\
        --mapping "both are exact-match accuracy on the same 10 items" \\
        --out_fpath comparison.json
"""
from __future__ import annotations

import json
from pathlib import Path

import kwconf


def _side(fpath: str) -> dict:
    # Recomputed from the validated run, never read back from the node file.
    from magnet.backends.aiq_evals.pipeline import (
        InvalidEvaluation,
        load_evidence,
    )

    try:
        _, evidence = load_evidence(fpath)
    except (InvalidEvaluation, KeyError, TypeError, ValueError) as ex:
        return {'fpath': str(fpath), 'eligible': False, 'ineligible_reasons': f'invalid evaluation: {ex}'}
    view = evidence.to_dict()
    return {
        'fpath': str(fpath),
        'engine': view['engine'],
        'measurement_identity': view['measurement_identity'],
        'eligible': view['eligible'],
        'ineligible_reasons': '; '.join(view.get('ineligible_reasons') or []),
        'task': (view.get('selected') or {}).get('task'),
        'metric': (view.get('selected') or {}).get('metric'),
        'value': (view.get('selected') or {}).get('value'),
    }


def compare(left_fpath: str, right_fpath: str, mapping: str) -> dict:
    left, right = _side(left_fpath), _side(right_fpath)
    comparable = bool(left['eligible'] and right['eligible'])
    return {
        'mapping': mapping,
        'comparable': comparable,
        'left': left,
        'right': right,
        'difference': (right['value'] - left['value']) if comparable else None,
    }


class CompareCLI(kwconf.Config):
    """Compare two EvaluationNode results under an explicit mapping."""

    __prog__ = 'python -m magnet.backends.aiq_evals.cli.compare'

    left_fpath = kwconf.Value(None, required=True, parser=str, help="left node's evaluation.json")
    right_fpath = kwconf.Value(None, required=True, parser=str, help="right node's evaluation.json")
    mapping = kwconf.Value(None, required=True, parser=str, help='why the two selections correspond')
    out_fpath = kwconf.Value('comparison.json', parser=str, help='comparison record to write')


def main(argv: list[str] | None = None) -> int:
    args = CompareCLI.cli(argv=True if argv is None else argv, strict=True, special_options=False)
    if not args.mapping.strip() or args.mapping.strip().lower() in {'none', 'null'}:
        raise SystemExit('a cross-engine comparison needs an explicit --mapping justification')
    comparison = compare(args.left_fpath, args.right_fpath, args.mapping)
    out = Path(args.out_fpath)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(comparison, indent=2, sort_keys=True) + '\n')
    return 0


COMPARE_MODULE = 'magnet.backends.aiq_evals.cli.compare'


def load_kwdagger_result(node, node_dpath):
    """One flat row: ``metrics.<node>.{comparable,difference,left.*,right.*}``.

    Recomputed on every load from both sides' validated evidence. Which sides
    are compared, and under which mapping, come from the command kwdagger
    scheduled for this directory (``invoke.sh``), never from comparison.json,
    which is only a record of what the comparison saw when it ran.
    """
    from kwdagger.utils import util_dotdict

    from magnet.backends.aiq_evals.scheduled import invocation_args, localize

    args = invocation_args(node_dpath).get(COMPARE_MODULE) or {}
    if not {'left_fpath', 'right_fpath', 'mapping', 'out_fpath'} <= set(args):
        flat = {'comparable': False, 'ineligible_reasons': 'invalid comparison: no scheduling record (invoke.sh)'}
    else:
        rendered_dpath = Path(args['out_fpath']).parent
        payload = compare(
            str(localize(args['left_fpath'], rendered_dpath, node_dpath)),
            str(localize(args['right_fpath'], rendered_dpath, node_dpath)),
            args['mapping'],
        )
        flat = {'comparable': payload['comparable'], 'mapping': payload['mapping']}
        if payload['comparable']:
            flat['difference'] = payload['difference']
        for side in ('left', 'right'):
            for key, value in payload[side].items():
                flat[f'{side}.{key}'] = value
    return util_dotdict.DotDict({f'metrics.{k}': v for k, v in flat.items()}).insert_prefix(node.name, index=1)


if __name__ == '__main__':
    raise SystemExit(main())
