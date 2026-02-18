import json

import ubelt as ub
import scriptconfig as scfg
import rich
from rich.markup import escape

class ConsistencyClaimCLI(scfg.DataConfig):
    """
    Llama consistency example claim representation.

    In lieu of the Claim definition in evaluation.py, this offers a more flexible injest -> evaluate -> write option.
    """

    symbols_fpath = scfg.Value('./data/llama-consistency/results.json', required=True, help=ub.paragraph(
        '''
        Default path to resolved symbol values.
        '''
        ),
        tags=['in_path'])

    verdict_fpath = scfg.Value('./data/llama-consistency/verdict.json', help=ub.paragraph(
        '''
        Output path for claim verdict. 
        '''
        ),
        tags=['out_path', 'primary'])

    @classmethod
    def main(cls, argv=None, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        rich.print('config = ' + escape(ub.urepr(config, nl=1)))

        verdict_json = {
            'result': None,
        }

        claim_str = "assert abs(comp_score - base_score) < threshold, f'{comp_model} score ({comp_score:.2f}) exceeds consistency bound on {base_model} ({base_score:.2f})'"
        status = "UNVERIFIED"
        out_msg = ""

        model_scores = json.loads(ub.Path(config.symbols_fpath).read_text())['result']

        # Copied from magnet.evaluation.Claim evaluate
        try:
            exec(claim_str, model_scores)
            status = "VERIFIED"
        except AssertionError as e:
            status = "FALSIFIED"
            out_msg = f"Assertion does not hold: {e}"
        except NameError as e:
            status = "INCONCLUSIVE"
            # This doesn't guarantee the missing variable is a symbol
            out_msg = f"SymbolNotResolved: {e}"
        except Exception as e:
            status = "INCONCLUSIVE"
            out_msg = f"ERROR evaluating claim: {e}"

        verdict_json['result'] = {
            'status': status,
            'output': out_msg,
        }

        dst_fpath = ub.Path(config.verdict_fpath)
        dst_fpath.parent.ensuredir()
        dst_fpath.write_text(json.dumps(verdict_json, indent=2))
        print(f'Wrote results to: {dst_fpath=}')


__cli__ = ConsistencyClaimCLI

if __name__ == '__main__':
    __cli__.main()

    r"""
    CommandLine:
        python ./cards/llama_consistency/cli/claim.py \
            --symbols_fpath ./data/llama-example-runs/results.json \
            --verdict_fpath ./data/llama-example-runs/verdict.json
    """