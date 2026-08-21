"""
Reduce one model pair's scores to the quantity the card is about.
"""
import json

import kwconf
import ubelt as ub


class ConsistencyCompareCLI(kwconf.Config):
    """
    Turn a pair of HELM scores into their gap.

    The node reads what `llama_predict` wrote and emits the comparison, so the
    card can state its claim against a number instead of recomputing it.
    """

    __command__ = 'llama_compare'

    scores_fpath: str = kwconf.Value(
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
        dst_fpath.write_text(json.dumps(comparison, indent=2))


__cli__ = ConsistencyCompareCLI

if __name__ == '__main__':
    __cli__.main()
