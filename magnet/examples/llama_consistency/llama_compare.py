"""
Reduce one model pair's scores to the quantity the card is about.
"""
import json

import kwconf
import ubelt as ub


def load_kwdagger_result(node, node_dpath):
    """Load the existing flat comparison JSON into kwdagger's result namespace."""
    from kwdagger.utils import util_dotdict

    node_dpath = ub.Path(node_dpath)
    output_fpath = node_dpath / node.out_paths[node.primary_out_key]
    payload = json.loads(output_fpath.read_text())
    metrics = {
        key: value for key, value in payload.items() if not key.startswith('_')
    }
    flat = util_dotdict.DotDict.from_nested({'metrics': metrics})
    return flat.insert_prefix(node.name, index=1)


class ExampleLlamaConsistencyCompareCLI(kwconf.Config):
    """
    Turn a pair of HELM scores into their gap.

    The node reads what `llama_predict` wrote and emits the comparison, so the
    card can state its claim against a number instead of recomputing it.

    This problem does not require two pipeline stages: ``llama_predict`` could
    compute and emit the gap itself. The separate comparison node is retained
    for now to demonstrate a real kwdagger artifact edge and result-node
    handoff. A future example should replace this with a case where the second
    stage is computationally necessary.
    """

    scores_fpath: str | None = kwconf.Value(
        None, required=True, help='scores written by llama_predict',
        tags=['in_path'])

    out_fpath: str = kwconf.Value(
        'comparison.json', help='where to write the comparison',
        tags=['out_path', 'primary'])

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True, verbose='auto')

        scores = json.loads(ub.Path(config['scores_fpath']).read_text())
        gap = abs(scores['comp_score'] - scores['base_score'])

        comparison = {
            'base_model': scores['base_model'],
            'comp_model': scores['comp_model'],
            'base_score': scores['base_score'],
            'comp_score': scores['comp_score'],
            'threshold': scores['threshold'],
            'gap': gap,
            'within_tolerance': gap < scores['threshold'],
        }

        dst_fpath = ub.Path(config['out_fpath'])
        dst_fpath.parent.ensuredir()
        # TODO: use safer for writing result files.
        dst_fpath.write_text(json.dumps(comparison, indent=2))


if __name__ == '__main__':
    ExampleLlamaConsistencyCompareCLI.main()
