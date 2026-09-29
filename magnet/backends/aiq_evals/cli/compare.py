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

import argparse
import json
from pathlib import Path


def _side(fpath: str) -> dict:
    # Recomputed from the validated run, never read back from the node file.
    from magnet.backends.aiq_evals.pipeline import load_evidence

    _, evidence = load_evidence(fpath)
    view = evidence.to_dict()
    return {
        'engine': view['engine'],
        'measurement_identity': view['measurement_identity'],
        'eligible': view['eligible'],
        'ineligible_reasons': '; '.join(view.get('ineligible_reasons') or []),
        'task': (view.get('selected') or {}).get('task'),
        'metric': (view.get('selected') or {}).get('metric'),
        'value': (view.get('selected') or {}).get('value'),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='python -m magnet.backends.aiq_evals.cli.compare')
    parser.add_argument('--left_fpath', required=True)
    parser.add_argument('--right_fpath', required=True)
    parser.add_argument('--mapping', required=True)
    parser.add_argument('--out_fpath', default='comparison.json')
    args = parser.parse_args(argv)
    if not args.mapping.strip() or args.mapping.strip().lower() in {'none', 'null'}:
        raise SystemExit('a cross-engine comparison needs an explicit --mapping justification')
    left, right = _side(args.left_fpath), _side(args.right_fpath)
    comparable = bool(left['eligible'] and right['eligible'])
    comparison = {
        'mapping': args.mapping,
        'comparable': comparable,
        'left': left,
        'right': right,
        'difference': (right['value'] - left['value']) if comparable else None,
    }
    out = Path(args.out_fpath)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(comparison, indent=2, sort_keys=True) + '\n')
    return 0


def load_kwdagger_result(node, node_dpath):
    """One flat row: ``metrics.<node>.{comparable,difference,left.*,right.*}``."""
    from kwdagger.utils import util_dotdict

    payload = json.loads((Path(node_dpath) / node.out_paths[node.primary_out_key]).read_text())
    flat = {'comparable': payload['comparable'], 'mapping': payload['mapping']}
    if payload['comparable']:
        flat['difference'] = payload['difference']
    for side in ('left', 'right'):
        for key, value in payload[side].items():
            flat[f'{side}.{key}'] = value
    return util_dotdict.DotDict({f'metrics.{k}': v for k, v in flat.items()}).insert_prefix(node.name, index=1)


if __name__ == '__main__':
    raise SystemExit(main())
