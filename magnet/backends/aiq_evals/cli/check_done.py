"""Exit 0 iff an EvaluationNode's primary output references a valid succeeded run.

kwdagger guards each job with a "done" test; a bare file-existence test would
let a stale ``evaluation.json`` hide a missing, failed, or tampered run.
``--expected`` (JSON) pins the selector, coverage policy, and identities the
node was scheduled with.
"""
import json
from pathlib import Path

import kwconf


class CheckDoneCLI(kwconf.Config):
    """Exit 0 iff evaluation.json references a valid run, as scheduled."""

    __prog__ = 'python -m magnet.backends.aiq_evals.cli.check_done'

    fpath = kwconf.Value(None, position=1, required=True, parser=str, help='evaluation.json')
    expected = kwconf.Value(None, parser=str, help='JSON fields evaluation.json must record')
    force_pending = kwconf.Value(False, isflag=True, help='require the host provenance gate')


def main(argv=None) -> int:
    from magnet.backends.aiq_evals.pipeline import evaluation_is_valid

    args = CheckDoneCLI.cli(argv=True if argv is None else argv, strict=True, special_options=False)
    expected = json.loads(args.expected) if args.expected else None
    if str(args.force_pending).lower() in {'true', '1', 'yes'}:
        return 1
    return 0 if evaluation_is_valid(Path(args.fpath), expected) else 1


if __name__ == '__main__':
    raise SystemExit(main())
