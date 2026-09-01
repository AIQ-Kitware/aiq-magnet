"""
Materialize one HELM run for this example, addressed by model and subject.

:mod:`magnet.backends.helm.cli.materialize_helm_run` takes a single
``run_entry`` string. That is the right surface for the general tool, but it
makes a poor matrix axis: a card would have to list one pre-composed string per
``(model, subject)`` pair and repeat the model list it already declares
elsewhere.

This wrapper takes the two axes separately and composes the run entry, so the
card's matrix says what it actually sweeps. It adds no materialization logic of
its own.

CommandLine:
    python -m magnet.examples.llama_consistency.materialize_run \
        --model=meta/llama-2-13b --subject=anatomy \
        --precomputed_root=./data/crfm-helm-public \
        --out_dpath=./materialized/llama-2-13b-anatomy
"""
import kwconf
import ubelt as ub

from magnet.backends.helm.cli.materialize_helm_run import (
    MaterializeHelmRunConfig)

#: HELM addresses an MMLU run by subject, adaptation method and model. The
#: method is fixed for this example, so only two of the three are axes.
RUN_ENTRY_TEMPLATE = 'mmlu:subject={subject},method={method},model={model}'


def load_kwdagger_result(node, node_dpath):
    """
    Report no result columns for this node.

    A materializer produces an artifact, not a measurement, and its primary
    output is a `DONE` sentinel rather than JSON. KWDagger aggregate loads a
    result node's predecessors as well as the node itself, and the generic YAML
    loader would try to parse that sentinel. A Python node signals "nothing to
    read here" by not defining `load_result` at all; a declarative node has the
    generic one whether it wants it or not, so it says so explicitly instead.
    """
    from kwdagger.utils import util_dotdict
    return util_dotdict.DotDict({})


class MaterializeLlamaRunCLI(kwconf.Config):
    """Reuse or compute one MMLU run for one model."""

    __command__ = 'materialize_llama_run'

    model: str = kwconf.Value(
        None, required=True, help='HELM model common name, e.g. meta/llama-2-13b')
    subject: str = kwconf.Value(
        None, required=True, help='MMLU subject, e.g. anatomy')
    method: str = kwconf.Value(
        'multiple_choice_joint', help='HELM adaptation method')
    precomputed_root: str = kwconf.Value(
        None, required=True, help='root of the downloaded HELM cache')
    suite: str = kwconf.Value(
        'llama-consistency', help='suite name for the materialized run')
    mode: str = kwconf.Value(
        'reuse_only',
        help='reuse_only | compute_if_missing | force_recompute')
    materialize: str = kwconf.Value('symlink', help='symlink | copy')
    out_dpath: str = kwconf.Value(
        None, required=True, help='where to materialize the run',
        tags=['out_path'])
    # kwdagger checks completion by the sentinel, so it owns that path and
    # passes it in absolutely. The underlying tool joins it onto out_dpath,
    # where an absolute right-hand side simply wins.
    done_fname: str = kwconf.Value(
        'DONE', help='completion sentinel', tags=['out_path', 'primary'])
    manifest_fname: str = kwconf.Value(
        'adapter_manifest.json', help='adapter manifest', tags=['out_path'])

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True, verbose='auto')
        run_entry = RUN_ENTRY_TEMPLATE.format(
            subject=config['subject'],
            method=config['method'],
            model=config['model'],
        )
        out_dpath = ub.Path(config['out_dpath'])
        out_dpath.ensuredir()
        return MaterializeHelmRunConfig.main(
            argv=False,
            run_entry=run_entry,
            suite=config['suite'],
            out_dpath=str(out_dpath),
            precomputed_root=config['precomputed_root'],
            mode=config['mode'],
            materialize=config['materialize'],
            done_fname=config['done_fname'],
            manifest_fname=config['manifest_fname'],
        )


__cli__ = MaterializeLlamaRunCLI

if __name__ == '__main__':
    __cli__.main()
