r"""
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


#: Passed to the materializer as given.
_FORWARDED = [
    'max_eval_instances',
    'enable_huggingface_models',
    'enable_local_huggingface_models',
    'model_deployments_fpath',
    'model_metadata_fpath',
    'tokenizer_configs_fpath',
    'require_per_instance_stats',
    'num_threads',
    'local_path',
]


def _sidecar_path(name):
    """Normalize a sidecar path so the materializer joins it correctly.

    :mod:`materialize_helm_run` treats ``done_fname`` and ``manifest_fname`` as
    filenames and computes ``out_dpath / name``. KWDagger owns those two paths
    and passes each one whole, relative to the working directory whenever
    ``--output_path`` was relative -- and a relative path with directories in it
    joins onto the node directory to give a second copy of itself nested inside.

    A bare filename is what that join expects, so leave it. Anything carrying
    directories is made absolute, which wins the join outright.
    """
    path = ub.Path(name)
    if str(path.parent) == '.':
        return str(path)
    return str(path.absolute())


class MaterializeLlamaRunCLI(kwconf.Config):
    """Reuse or compute one MMLU run for one model."""

    model: str = kwconf.Value(
        None, required=True, help='HELM model common name, e.g. meta/llama-2-13b')
    subject: str = kwconf.Value(
        None, required=True, help='MMLU subject, e.g. anatomy')
    method: str = kwconf.Value(
        'multiple_choice_joint', help='HELM adaptation method')
    precomputed_root: str = kwconf.Value(
        None, required=True, help='root of the downloaded HELM cache')

    # Forwarded to the materializer untouched. They are what computing a run
    # takes, as opposed to reusing one, and a wrapper that dropped them left
    # `compute_if_missing` reachable but unusable.
    max_eval_instances: int | None = kwconf.Value(
        None,
        help=(
            'instances to evaluate when computing. helm-run requires it, and '
            'it is identity-bearing: a computed run has to match the instance '
            'count of the precomputed runs it will be averaged with'
        ))
    enable_huggingface_models: str | None = kwconf.Value(
        None, help='HuggingFace model ids to make available to helm-run')
    enable_local_huggingface_models: str | None = kwconf.Value(
        None, help='local HuggingFace model paths to make available')
    model_deployments_fpath: str | None = kwconf.Value(
        None, help='HELM model deployment registrations')
    model_metadata_fpath: str | None = kwconf.Value(
        None, help='HELM model metadata registrations')
    tokenizer_configs_fpath: str | None = kwconf.Value(
        None, help='HELM tokenizer registrations')
    require_per_instance_stats: bool = kwconf.Value(
        True, help='require per_instance_stats.json when reusing')
    num_threads: int = kwconf.Value(1, help='helm-run threads')
    local_path: str = kwconf.Value('prod_env', help='helm-run local path')
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
        out_dpath = ub.Path(config['out_dpath']).absolute()
        out_dpath.ensuredir()
        return MaterializeHelmRunConfig.main(
            argv=False,
            run_entry=run_entry,
            suite=config['suite'],
            out_dpath=str(out_dpath),
            precomputed_root=config['precomputed_root'],
            mode=config['mode'],
            materialize=config['materialize'],
            done_fname=_sidecar_path(config['done_fname']),
            manifest_fname=_sidecar_path(config['manifest_fname']),
            **{key: config[key] for key in _FORWARDED},
        )


__cli__ = MaterializeLlamaRunCLI

if __name__ == '__main__':
    __cli__.main()
