"""Exit 0 iff an EvaluationNode's primary output references a valid succeeded run.

kwdagger guards each job with a "done" test; a bare file-existence test would
let a stale ``evaluation.json`` hide a missing, failed, or tampered run.
``--expected`` (JSON) pins the selector, coverage policy, and identities the
node was scheduled with.
"""
import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    from magnet.backends.aiq_evals.pipeline import evaluation_is_valid

    parser = argparse.ArgumentParser(prog='python -m magnet.backends.aiq_evals.cli.check_done')
    parser.add_argument('fpath')
    parser.add_argument('--expected', default=None, help='JSON fields evaluation.json must record')
    args = parser.parse_args(argv)
    expected = json.loads(args.expected) if args.expected else None
    return 0 if evaluation_is_valid(Path(args.fpath), expected) else 1


if __name__ == '__main__':
    raise SystemExit(main())
