r"""
kwdagger pipeline for materializing a single HELM run.

This pipeline currently contains one node:
    - materialize_helm_run: computes OR reuses a single HELM run-entry, writing
      results into the kwdagger job directory in a deterministic layout, and
      writing a DONE sentinel file last.

You can extend this later by adding downstream evaluation nodes that consume
the materialized HELM outputs.

.. code:: bash


helm-run --run-entries civil_comments:demographic=LGBTQ,model=meta/llama-7b,data_augmentation=canonical --suite my-suite --max-eval-instances 1000 --num-threads 1

    kwdagger schedule \
      --params="
        pipeline: 'magnet.backends.helm.pipeline.helm_single_run_pipeline()'
        matrix:
          helm.run_entry:
            - 'mmlu:subject=philosophy,model=openai/gpt2'
            - 'ewok:domain=physical_interactions,model=meta/llama-3-8b-chat'
          helm.max_eval_instances:
            - 10
          helm.precomputed_root: '/data/crfm-helm-public'
      " \
      --tmux_workers=4 \
      --root_dpath=$PWD/results \
      --backend=serial \
      --skip_existing=1 \
      --run=1

"""

from __future__ import annotations

import kwdagger


class MaterializeHelmRunNode(kwdagger.ProcessNode):
    """
    Wraps: python -m magnet.backends.helm.materialize_helm_run

    This script writes:
      <out_dpath>/benchmark_output/runs/<suite>/<run_name>/...
      <out_dpath>/adapter_manifest.json
      <out_dpath>/DONE

    We tell kwdagger that DONE is the primary output so skip_existing works.
    """
    name = 'helm'

    # If magnet is importable, this is the cleanest way to invoke the module.
    executable = "python -m magnet.backends.helm.materialize_helm_run"

    # We generally don't want kwdagger to try to "materialize" huge external
    # directories (like /data/crfm-helm-public) into each job dir, so we do NOT
    # put precomputed_roots in `in_paths`. We pass it as a normal CLI arg.
    in_paths = set()

    # Mark outputs as *paths* so kwdagger knows where to look for completion and
    # how to record/relocate outputs.
    #
    # IMPORTANT:
    # - out_dpath is the node working directory itself ('.')
    # - done_fname is treated by this pipeline as a "path" for kwdagger's
    #   completion checking. The script itself interprets it as a filename.
    out_paths = {
        "out_dpath": ".",
        "done_fname": "DONE",
        "manifest_fname": "adapter_manifest.json",
    }

    # kwdagger considers a node "complete" when the primary output exists.
    primary_out_key = "done_fname"

    # Parameters that *change the logical identity of the work*.
    # These participate in hashing, so a different run-entry => a new job folder.
    algo_params = {
        # NOTE: set these via kwdagger schedule matrix / params
        "run_entry": None,
        "suite": "my-suite",
        "max_eval_instances": None,
        "require_per_instance_stats": True,

        # Behavior toggles that change how/what we materialize
        "mode": "compute_if_missing",   # reuse_only | compute_if_missing | force_recompute
        "materialize": "symlink",       # symlink | copy
    }

    # Performance / environment parameters.
    # These are recorded, but ideally should not change the “meaning” of outputs.
    perf_params = {
        # Your shared precomputed root:
        "precomputed_root": "/data/crfm-helm-public",

        # helm-run perf knobs:
        "num_threads": 1,
    }

    # Optional: You can define load_result if you want kwdagger aggregate to read
    # something out of this node. Usually this node is an “adapter/materializer”
    # and a downstream eval node would implement load_result instead.
    #
    # def load_result(self, node_dpath) -> dict:
    #     return {}


def helm_single_run_pipeline():
    """
    Pipeline factory function used by kwdagger schedule / aggregate.

    Example usage:
        kwdagger schedule --params="pipeline: 'magnet.pipelines.helm_materialize_pipeline.helm_single_run_pipeline()' ..."
    """
    nodes = {
        "materialize_helm_run": MaterializeHelmRunNode(),
    }
    return kwdagger.Pipeline(nodes)
