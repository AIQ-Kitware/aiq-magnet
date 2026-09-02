"""
Reduce every endpoint's answers to the numbers the claim is about.

Receives a kwdagger gather manifest: one line per upstream ``ask`` cell, so
the set of endpoints a verdict rests on is declared by the matrix and resolved
when the pipeline is compiled. Nothing scans a directory.

CommandLine:
    python -m smollm_example.cli.compare_answers \
        --answer_fpaths=manifest.txt --out_fpath=comparison.json
"""

import json
from collections import defaultdict

import kwconf
import ubelt as ub

__all__ = ['read_gathered_answers', 'compare', 'CompareAnswersCLI']


def read_gathered_answers(manifest_fpath) -> list:
    """
    Resolve a gather manifest into the answer payloads it names.

    Args:
        manifest_fpath (str | PathLike): newline-delimited paths from kwdagger.

    Returns:
        list: one parsed ``answers.json`` per upstream cell, in manifest order.
    """
    payloads = []
    for line in ub.Path(manifest_fpath).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        payloads.append(json.loads(ub.Path(line).read_text()))
    return payloads


def compare(payloads: list) -> dict:
    """
    Coverage and agreement across the gathered endpoints.

    ``coverage`` is the weakest endpoint's answered rate: the claim is that
    every endpoint answered every question, so the worst one is the one that
    decides. ``agreement`` is the fraction of questions where every endpoint
    that answered gave the same normalized value -- reported because it is the
    interesting number, claimed about by nothing, because against a simulator
    the text is random.

    Args:
        payloads (list): parsed ``answers.json`` documents.

    Returns:
        dict: scalar metrics only. A list-valued metric would arrive in the
            evidence row as a collection and has to be read differently, which
            is a lesson for another example.

    Example:
        >>> from smollm_example.cli.compare_answers import (
        ...     compare)
        >>> mk = lambda ep, vals: {
        ...     'result': {'metrics': {'endpoint': ep, 'answered_rate': 1.0}},
        ...     'answers': [{'id': i, 'answer': v, 'normalized': v,
        ...                  'expected': v} for i, v in enumerate(vals)]}
        >>> out = compare([mk('a', ['1', '2']), mk('b', ['1', '9'])])
        >>> out['n_endpoints'], out['agreement'], out['coverage']
        (2, 0.5, 1.0)
    """
    if not payloads:
        raise ValueError('gather produced no answers to compare')

    endpoints = []
    coverages = []
    by_item = defaultdict(list)
    for payload in payloads:
        metrics = payload.get('result', {}).get('metrics', {})
        endpoints.append(str(metrics.get('endpoint', '?')))
        coverages.append(float(metrics.get('answered_rate', 0.0)))
        for answer in payload['answers']:
            if answer['answer']:
                by_item[answer['id']].append(answer['normalized'])

    n_items = max((len(p['answers']) for p in payloads), default=0)
    agreed = sum(
        1 for values in by_item.values()
        if len(values) == len(payloads) and len(set(values)) == 1
    )
    return {
        'n_endpoints': len(payloads),
        'endpoints': ','.join(sorted(endpoints)),
        'n_items': n_items,
        # The claim's subject: did every endpoint answer everything it was
        # asked? That is a statement about leasing and the container, and it
        # is true or false regardless of what the models said.
        'coverage': min(coverages) if coverages else 0.0,
        'agreement': (agreed / n_items) if n_items else 0.0,
    }


class CompareAnswersCLI(kwconf.Config):
    """Reduce the gathered endpoints to coverage and agreement."""

    answer_fpaths: str = kwconf.Value(
        None, help='kwdagger gather manifest of answers.json paths',
        tags=['in_path'])
    out_fpath: str = kwconf.Value(
        'comparison.json', help='where to write the comparison',
        tags=['out_path', 'primary'])

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True, verbose='auto')
        payloads = read_gathered_answers(config['answer_fpaths'])
        payload = {'result': {'metrics': compare(payloads)}}
        out_fpath = ub.Path(config['out_fpath'])
        out_fpath.parent.ensuredir()
        out_fpath.write_text(json.dumps(payload, indent=2))


if __name__ == '__main__':
    CompareAnswersCLI.main()
