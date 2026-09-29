"""MAGNET integration with ``magnet_evals`` (see aiq-evals docs/planning/aiq-magnet-integration-plan.md).

``magnet_evals`` is an optional dependency (``pip install aiq-magnet[aiq-magnet-evals]``),
imported lazily when a node runs, resolves, or loads results. Importing this
package does not import it or any native evaluation engine.
"""
from magnet.backends.aiq_evals.pipeline import EvaluationNode, build_request_dict, preflight_scope

__all__ = ['EvaluationNode', 'build_request_dict', 'preflight_scope']
