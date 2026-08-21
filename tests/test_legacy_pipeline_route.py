"""
The soft-deprecated ``pipeline:`` route shares the DAG root with the new one.

It asks each configured instance for its own artifact, so a root holding other
card versions' instances does not leak them into this card's cells.
"""

import json
import textwrap

import pytest
import ubelt as ub

from magnet.evaluation import EvaluationCard

CARD = """
title: probe
description: probe
version: '1'
organizations: [Kitware]
submitter: {{name: t, email: t@example.com}}
links: []
tags: [test]
claim:
  python: |
    assert score < 100
symbols:
  score:
    type: float
    metadata:
      define_metric: {{objective: maximize, aggregation_strategy: {{type: mean}}}}
pipeline:
  emit:
    executable: python {script}
    algo_params:
      seed: {seeds}
    out_paths:
      results_fpath: ./results.json
"""

SCRIPT = """
import json, sys, pathlib
args = dict(a.lstrip('-').split('=', 1) for a in sys.argv[1:])
out = pathlib.Path(args['results_fpath'])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'result': {'score': float(args['seed'])}}))
"""


def _card(tmp_path, seeds):
    script = tmp_path / 'emit.py'
    script.write_text(textwrap.dedent(SCRIPT))
    fpath = tmp_path / 'card.yaml'
    fpath.write_text(CARD.format(script=script, seeds=list(seeds)))
    with pytest.warns(DeprecationWarning, match='soft-deprecated'):
        card = EvaluationCard(fpath, tmp_path / 'out')
        card.evaluate()
    return card


def test_a_rerun_does_not_collect_the_previous_cards_cells(tmp_path):
    tmp_path = ub.Path(tmp_path)
    first = _card(tmp_path, [1, 2])
    assert len(first.evaluations) == 2

    second = _card(tmp_path, [4, 5])
    scores = sorted(
        e.symbols.simple_view()['score'] for e in second.evaluations)
    assert scores == [4.0, 5.0]


def test_the_old_route_still_binds_bare_names(tmp_path):
    tmp_path = ub.Path(tmp_path)
    card = _card(tmp_path, [3])
    assert card.evaluations[0].symbols.simple_view()['score'] == 3.0


def test_the_run_shares_the_dag_root_and_links_to_it(tmp_path):
    tmp_path = ub.Path(tmp_path)
    card = _card(tmp_path, [1])
    run_dpath = card.output_path / card._run_hash

    assert (card.output_path / '_kwdagger').is_dir()
    # Consumers glob a run for its artifacts and figures.
    assert (run_dpath / 'kwdagger').is_symlink()
    assert (run_dpath / 'kwdagger').resolve() == (
        card.output_path / '_kwdagger').resolve()

    written = json.loads((run_dpath / 'verdict.json').read_text())
    assert written['result'] in {'VERIFIED', 'FALSIFIED', 'INCONCLUSIVE'}


def test_an_unchanged_cell_is_not_recomputed(tmp_path):
    # What the shared root buys the old route: editing the card leaves the
    # cells it did not change alone.
    tmp_path = ub.Path(tmp_path)
    _card(tmp_path, [1, 2])

    root = ub.Path(tmp_path) / 'out' / '_kwdagger'
    before = {
        p.parent.name: p.stat().st_mtime
        for p in root.glob('**/results.json')
    }
    assert len(before) == 2

    _card(tmp_path, [1, 2, 3])
    after = {
        p.parent.name: p.stat().st_mtime
        for p in root.glob('**/results.json')
    }

    assert len(after) == 3
    for node_id, mtime in before.items():
        assert after[node_id] == mtime, f'{node_id} was recomputed'
