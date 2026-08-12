"""
Loading statements from disk.

A **theorem index** is what a formalization repository exports for downstream
consumers. It is the only format carrying hypothesis binders, and therefore the
only one supporting hypothesis-level coverage. Its shape is the contract
between the Lean side and this package::

    version: 1
    project:
      name: ...
      repository: ...
      commit: <sha>          # what every reference is pinned to
      review: self-assessed
    theorems:
      - declaration: Some.Namespace.the_theorem
        informal: "..."
        conclusion: "MSE(candidate) <= MSE(baseline), eventually, w.h.p."
        file: Some/Namespace/File.lean
        line: 291
        axioms: [propext, Classical.choice, Quot.sound]
        hypotheses:
          - name: hgap
            informal: "the affine witness beats the baseline in population MSE"
            lean: "MSE Pf (yFull score Qstar) (affineModel theta (psi Qols)) < ..."
            structural: false

A statement loaded without binders reports ``hypotheses_enumerated == False``,
so a missing exporter cannot pass for a clean bill of health.
"""
import os
from typing import Any, Iterator, Sequence

from magnet.theory.model import (
    KERNEL_AXIOMS,
    Formalization,
    Hypothesis,
    Theorem,
)

#: The hygiene library shipped alongside this module.
HYGIENE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'hygiene.yaml')

__all__ = ['load', 'load_index', 'hygiene', 'HYGIENE_PATH']


def load_index(path, name: str | None = None) -> Formalization:
    """Load a theorem index (JSON or YAML, by extension)."""
    import ubelt as ub

    path = ub.Path(path)
    data = _read_structured(path)

    project = data.get('project') or {}
    formalization = Formalization(
        name=name or project.get('name') or path.parent.name,
        repository=project.get('repository'),
        commit=project.get('commit'),
        source=str(path),
        review=project.get('review', 'draft'),
        note=project.get('note', ''),
    )
    for spec in data.get('theorems') or []:
        formalization.add(
            Theorem(
                declaration=spec['declaration'],
                informal=spec.get('informal', ''),
                conclusion=spec.get('conclusion', ''),
                hypotheses=[
                    Hypothesis(
                        name=h['name'],
                        informal=h.get('informal', ''),
                        lean=h.get('lean'),
                        structural=bool(h.get('structural', False)),
                    )
                    for h in spec.get('hypotheses') or []
                ],
                axioms=spec.get('axioms'),
                file=spec.get('file'),
                line=spec.get('line'),
                review=spec.get('review', 'draft'),
                note=spec.get('note', ''),
            )
        )
    return formalization


def _read_structured(path) -> dict:
    text = path.read_text()
    if path.suffix == '.json':
        import json

        return json.loads(text) or {}
    import yaml

    return yaml.safe_load(text) or {}


def _walk_key(node: Any, key: str, ancestors: tuple = ()) -> Iterator[tuple[tuple, Any]]:
    """
    Yield ``(ancestor_mappings, value)`` for every occurrence of ``key``.

    Ancestors are nearest-first, so a caller can look for context -- an
    axiom-audit note, say -- on the enclosing node and then further up.
    """
    if isinstance(node, dict):
        chain = (node,) + ancestors
        for k, v in node.items():
            if k == key:
                yield (chain, v)
            yield from _walk_key(v, key, chain)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_key(item, key, ancestors)


def _axioms_from_notes(ancestors: Sequence[dict]) -> tuple[str, ...] | None:
    """
    Recover an axiom set from a manifest's prose note.

    Manifests record axiom audits as free text ("All capstones verified
    ``#print axioms`` = {propext, Classical.choice, Quot.sound}"), attached to
    whichever node groups the declarations it covers. Reading prose is a stopgap
    for a real exporter, so it is deliberately narrow: only the exact kernel
    axiom set is recognized, and anything else reads as unknown rather than
    being guessed at.

    Example:
        >>> note = {'note': 'verified #print axioms = {propext, Classical.choice, Quot.sound}'}
        >>> _axioms_from_notes([{}, note])
        ('Classical.choice', 'Quot.sound', 'propext')
        >>> _axioms_from_notes([{'note': 'looks fine to me'}]) is None
        True
    """
    import re

    for holder in ancestors:
        note = ' '.join(
            str(v) for k, v in holder.items() if k in {'note', 'notes'} and isinstance(v, str)
        )
        if not note or '#print axioms' not in note:
            continue
        braced = re.search(r'\{([^}]*)\}', note)
        if braced:
            named = {tok.strip() for tok in braced.group(1).split(',') if tok.strip()}
            if named == set(KERNEL_AXIOMS):
                return tuple(sorted(KERNEL_AXIOMS))
    return None


#: The only index format. Kept as an alias so callers read as intent.
load = load_index


def hygiene() -> Formalization:
    """The evaluation-hygiene statements shipped with this package."""
    return load_index(HYGIENE_PATH, name='Evaluation hygiene')
