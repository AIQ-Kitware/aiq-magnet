"""
The soft-deprecated ``pipeline:`` route keeps its own per-run DAG root.

It finds its results by globbing, so a root shared across card versions would
hand it artifacts from a card it is not running.
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


def test_the_run_keeps_its_own_dag_root(tmp_path):
    tmp_path = ub.Path(tmp_path)
    card = _card(tmp_path, [1])
    run_dpath = card.output_path / card._run_hash
    assert (run_dpath / 'kwdagger').is_dir()
    assert not (card.output_path / '_kwdagger').exists()
    written = json.loads(
        (run_dpath / 'verdict.json').read_text())
    assert written['result'] in {'VERIFIED', 'FALSIFIED', 'INCONCLUSIVE'}
