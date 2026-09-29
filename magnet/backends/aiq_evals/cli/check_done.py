"""Exit 0 iff an EvaluationNode's primary output references a valid succeeded run.

kwdagger guards each job with a "done" test; a bare file-existence test would
let a stale ``evaluation.json`` hide a missing, failed, or tampered run.
"""
import sys
from pathlib import Path


def main(argv=None) -> int:
    from magnet.backends.aiq_evals.pipeline import evaluation_is_valid

    (fpath,) = argv if argv is not None else sys.argv[1:]
    return 0 if evaluation_is_valid(Path(fpath)) else 1


if __name__ == '__main__':
    raise SystemExit(main())
