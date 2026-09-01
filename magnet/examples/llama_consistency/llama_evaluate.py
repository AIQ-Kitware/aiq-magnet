r"""
Compare two Llama models over the MMLU runs this pipeline materialized.

Reads a kwdagger gather manifest -- one materialized run directory per line --
rather than scanning a HELM cache directory. That is the whole point of the
node: what this comparison rests on is a set of declared upstream artifacts, so
the result cannot change because something else appeared in a shared folder.

CommandLine:
    python -m magnet.examples.llama_consistency.llama_evaluate \
        --run_dpaths=./gathered.txt \
        --base_model=meta/llama-2-70b --comp_model=meta/llama-2-7b \
        --out_fpath=comparison.json
"""
import json

import kwconf
import ubelt as ub

from magnet.backends.helm.helm_outputs import HelmSuiteRuns


def read_gathered_run_dpaths(manifest_fpath):
    """
    Resolve a gather manifest into the HELM run directories it names.

    Each line is one materialize node's output directory, which holds its run
    under ``benchmark_output/runs/<suite>/<run_name>``. Returns those run
    directories.

    Args:
        manifest_fpath (str | PathLike): newline-delimited paths from kwdagger.

    Returns:
        list: HELM run directories, in manifest order.
    """
    manifest_fpath = ub.Path(manifest_fpath)
    run_dpaths = []
    for line in manifest_fpath.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        runs_root = ub.Path(line) / 'benchmark_output' / 'runs'
        if not runs_root.exists():
            raise FileNotFoundError(
                f'materialized run {line!r} has no benchmark_output/runs'
            )
        for suite_dpath in sorted(runs_root.iterdir()):
            run_dpaths.extend(sorted(suite_dpath.iterdir()))
    return run_dpaths


def mean_exact_match_by_model(run_dpaths):
    """
    Average the test-split exact_match score across subjects, per model.

    Args:
        run_dpaths (list): HELM run directories.

    Returns:
        dict: model common name -> mean exact_match.
    """
    run_specs = HelmSuiteRuns.coerce([ub.Path(p) for p in run_dpaths])
    run_stats = run_specs.stats()
    # Benchmark stats only, per https://github.com/stanford-crfm/helm/issues/2362
    run_stats = run_stats[
        (run_stats['stats.name.name'] == 'exact_match')
        & (run_stats['stats.name.perturbation.computed_on'].isna())
        & (run_stats['stats.name.split'] == 'test')
    ]
    helm_models = (
        run_specs.run_spec()
        .set_index('run_spec.name')['run_spec.adapter_spec.model']
        .to_dict()
    )
    run_stats['model'] = run_stats['run_spec.name'].map(helm_models)
    return run_stats.groupby('model')['stats.mean'].mean().to_dict()


class LlamaEvaluateCLI(kwconf.Config):
    """Score two models over the gathered runs and report the gap."""

    run_dpaths: str = kwconf.Value(
        None, required=True, tags=['in_path'],
        help='kwdagger gather manifest of materialized run directories')
    base_model: str = kwconf.Value(None, required=True, help='reference model')
    comp_model: str = kwconf.Value(None, required=True, help='model compared')
    threshold: float = kwconf.Value(
        0.1, help='largest score gap consistent with one model family')
    out_fpath: str = kwconf.Value(
        'comparison.json', help='where to write the result',
        tags=['out_path', 'primary'])

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True, verbose='auto')

        run_dpaths = read_gathered_run_dpaths(config['run_dpaths'])
        scores = mean_exact_match_by_model(run_dpaths)
        missing = {config['base_model'], config['comp_model']} - set(scores)
        if missing:
            raise KeyError(
                f'the gathered runs contain no scores for {sorted(missing)}; '
                f'available: {sorted(scores)}'
            )

        base_score = scores[config['base_model']]
        comp_score = scores[config['comp_model']]
        gap = abs(base_score - comp_score)
        data = {
            'result': {
                'metrics': {
                    'base_model': config['base_model'],
                    'base_score': base_score,
                    'comp_model': config['comp_model'],
                    'comp_score': comp_score,
                    'threshold': config['threshold'],
                    'gap': gap,
                    'within_tolerance': bool(gap < config['threshold']),
                    'gathered_runs': len(run_dpaths),
                },
            },
        }
        out_fpath = ub.Path(config['out_fpath'])
        out_fpath.parent.ensuredir()
        out_fpath.write_text(json.dumps(data, indent=2))


__cli__ = LlamaEvaluateCLI

if __name__ == '__main__':
    __cli__.main()
