"""
Loading statements from disk.

Two formats, for two different situations.

A **theorem index** is what a formalization repository exports for downstream
consumers: declaration, conclusion, hypothesis binders, axioms, file and line.
It is the only format that supports hypothesis-level coverage, because it is the
only one that carries binders. Its shape is the contract between the Lean side
and this package::

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

A **formalization manifest** (``formalization.yaml``, the mathlib-initiative
v0.3 schema) is metadata about a project rather than an export of its
statements: it names capstones and pairs some with informal descriptions, but it
carries no binders. Loading one gets you declaration names and, where a note
records an axiom audit, proof status -- enough to check that a reference names
something real, not enough to audit assumptions. Statements loaded this way
report ``hypotheses_enumerated == False`` so a missing exporter cannot pass for
a clean bill of health.
"""
import os
from typing import Any, Iterator, Sequence

from magnet.theory.model import KERNEL_AXIOMS, Formalization, Hypothesis, Theorem

#: The hygiene library shipped alongside this module.
HYGIENE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'hygiene.yaml')

__all__ = ['load_index', 'load_manifest', 'save_index', 'load', 'hygiene', 'HYGIENE_PATH']


def hygiene() -> Formalization:
    """
    The shared evaluation-hygiene statements shipped with MAGNET.

    Most cards have no theory of their own, but they are not assumption-free:
    they lean on the same premises repeatedly -- iid sampling, a scorer that
    tracks its construct, a sample big enough for the threshold asserted, a
    threshold not chosen after looking. Those are shared, so a card with no
    bespoke theorem still has somewhere to attach its assumptions.

    Example:
        >>> from magnet.theory import hygiene
        >>> library = hygiene()
        >>> statement = library['Hygiene.Concentration.mean_within_tolerance']
        >>> [h.name for h in statement.hypotheses]
        ['hiid', 'hbdd', 'hn']
        >>> statement.proof
        <ProofStatus.UNKNOWN: 'unknown'>
    """
    return load_index(HYGIENE_PATH)


def load(path, name: str | None = None) -> Formalization:
    """
    Load either format, choosing by filename.

    ``formalization.yaml`` is the manifest; anything else is an index.
    """
    import ubelt as ub

    path = ub.Path(path)
    if path.name == 'formalization.yaml':
        return load_manifest(path, name=name)
    return load_index(path, name=name)


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


def save_index(formalization: Formalization, path) -> None:
    """Write a formalization back out as a theorem index."""
    import ubelt as ub

    path = ub.Path(path)
    path.parent.ensuredir()
    data = formalization.to_dict()
    if path.suffix == '.json':
        import json

        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    else:
        import yaml

        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def load_manifest(path, name: str | None = None) -> Formalization:
    """Load a ``formalization.yaml`` (mathlib-initiative v0.3)."""
    import ubelt as ub

    path = ub.Path(path)
    data = _read_structured(path)
    project = data.get('project') or {}
    review = (data.get('review') or {}).get('status', 'draft')

    formalization = Formalization(
        name=name or project.get('name') or path.parent.name,
        repository=project.get('repository'),
        source=str(path),
        review=_coerce_review(review),
    )

    # Informal statements paired with declarations: the richest source, but it
    # only covers the subset someone bothered to align.
    for statement in (data.get('alignment') or {}).get('statements') or []:
        declaration = statement.get('lean')
        if declaration:
            formalization.add(
                Theorem(
                    declaration=declaration,
                    informal=statement.get('informal', ''),
                    note=statement.get('status', ''),
                )
            )

    # Capstone declarations appear under several groupings; rather than hardcode
    # one shape, take every 'capstones' list in the document.
    for ancestors, capstones in _walk_key(data, 'capstones'):
        axioms = _axioms_from_notes(ancestors)
        for declaration in capstones or []:
            if not isinstance(declaration, str):
                continue
            existing = formalization.theorems.get(declaration)
            if existing is None:
                formalization.add(Theorem(declaration=declaration, axioms=axioms))
            elif existing.axioms is None and axioms is not None:
                formalization.add(
                    Theorem(
                        declaration=declaration,
                        informal=existing.informal,
                        conclusion=existing.conclusion,
                        hypotheses=existing.hypotheses,
                        axioms=axioms,
                        file=existing.file,
                        line=existing.line,
                        review=existing.review,
                        note=existing.note,
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


def _coerce_review(value: str) -> str:
    from magnet.theory.model import ReviewStatus

    try:
        return str(ReviewStatus(value))
    except ValueError:
        return str(ReviewStatus.DRAFT)


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
