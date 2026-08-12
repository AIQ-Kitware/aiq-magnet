"""
The theory block of an evaluation card.

A card names the statements its claim is grounded on, and the ledger that says
how the code departs from them::

    theory:
      formalizations:
        - theory/indexes/conjectures.yaml
      ledger: theory/ledger.json
      grounds:
        - declaration: AIQ.Conjectures.Composition.linear_growth_of_nonexpansive
          informal: the card fits over per-node consistency scores

Relaxations are not listed in the card. A hypothesis is usually weakened in a
predictor rather than in the card, so it is annotated at the code that weakens
it -- code an evaluation never imports. ``magnet.theory.audit`` reads those
annotations out of source and writes the ledger; the card names it, and the
edges bearing on its own statements are selected from it.
"""
__all__ = ['basis_from_card', 'edges_from_ledger', 'CARD_KEY']

CARD_KEY = 'theory'


def edges_from_ledger(path, declarations):
    """
    Edges from an audit ledger that bear on the given statements.

    Args:
        path: a report written by ``magnet.theory.audit --format json``.
        declarations (set[str]): the statements to keep edges for.

    Returns:
        list: :class:`~magnet.theory.predicates.Edge`
    """
    import json

    import ubelt as ub

    from magnet.theory import Edge

    report = json.loads(ub.Path(path).read_text())
    edges = []
    for spec in report.get('basis', {}).get('edges', []):
        if spec['hypothesis'].split('::')[0] not in declarations:
            continue
        edges.append(
            Edge(
                spec['hypothesis'],
                relation=spec['relation'],
                severity=spec['severity'],
                informal=spec.get('informal', ''),
                note=spec.get('note', ''),
                evidence=spec.get('evidence'),
                id=spec.get('id'),
                review=spec.get('review', 'draft'),
                kind=spec.get('kind'),
                site=spec.get('site'),
                anchor=spec.get('anchor'),
            )
        )
    return edges


def basis_from_card(card, root=None, edges=None):
    """
    Build a :class:`~magnet.theory.TheoreticalBasis` from a card's ``theory``
    block, or return None if it has none.

    Args:
        card (dict): the card, as loaded from YAML.
        root: directory that relative paths in the block resolve against.
        edges: edges to use instead of reading the card's ``ledger``.

    Example:
        >>> from magnet.theory.cards import basis_from_card
        >>> basis_from_card({'title': 'no theory here'}) is None
        True
    """
    import ubelt as ub

    # `Grounding`, not the `grounds` predicate: these come from YAML, not from
    # a decorated code site. The static auditor reads `grounds(...)` calls and
    # would report this loop's non-literal reference as unresolvable.
    from magnet.theory import Grounding, TheoreticalBasis, load

    spec = card.get(CARD_KEY) or {}
    declared = spec.get('grounds') or []
    if not declared:
        return None

    root = ub.Path(root) if root is not None else ub.Path('.')
    formalizations = [load(root / p) for p in spec.get('formalizations') or []]

    groundings = [
        Grounding(
            entry['declaration'],
            informal=entry.get('informal', ''),
            note=entry.get('note', ''),
        )
        for entry in declared
    ]

    if edges is None:
        ledger = spec.get('ledger')
        declarations = {g.declaration for g in groundings}
        edges = edges_from_ledger(root / ledger, declarations) if ledger else []

    return TheoreticalBasis(
        groundings=tuple(groundings),
        edges=tuple(edges),
        formalizations=tuple(formalizations),
    )
