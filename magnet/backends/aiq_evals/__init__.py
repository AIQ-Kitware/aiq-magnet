"""MAGNET integration with ``aiq_evals`` (see aiq-evals docs/planning/aiq-magnet-integration-plan.md).

``aiq_evals`` is an optional dependency (``pip install aiq-magnet[aiq-evals]``),
imported lazily when a node runs, resolves, or loads results. Importing this
package does not import it or any native evaluation engine.
"""
from magnet.backends.aiq_evals.pipeline import EvaluationNode, build_request_dict

__all__ = ['EvaluationNode', 'build_request_dict', 'apply_preflight']


def apply_preflight(pipeline, *, enabled: bool) -> None:
    """Enable preflight resolution on every EvaluationNode of a pipeline (M3/M9)."""
    for node in getattr(pipeline, 'node_dict', {}).values():
        if isinstance(node, EvaluationNode):
            node.preflight_resolution = enabled
