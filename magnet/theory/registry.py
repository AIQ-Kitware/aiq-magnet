"""
Where declared edges and groundings accumulate at run time.

A registry is needed because the relation usually is not declared in the card.
The assumption about a predictor's estimator is relaxed in the predictor's
source, one repository away from the YAML that claims the result. The predicate
marks it there; the registry is how a card assembled elsewhere finds it.

The registry is the *runtime* half. It only knows what has been imported, which
is why it is not the authority on what a codebase declares --
:mod:`magnet.theory.static` reads that from source without executing anything.
Use the registry for what a *run* did: which edges activated, with what
witnessed values, and how the checks came out.
"""
from typing import Iterator

__all__ = ['TheoryRegistry', 'REGISTRY']


class TheoryRegistry:
    """Collection of everything the predicates have constructed."""

    def __init__(self) -> None:
        self.edges: list = []
        self.groundings: list = []

    def add_edge(self, edge):
        self.edges.append(edge)
        return edge

    def add_grounding(self, grounding):
        self.groundings.append(grounding)
        return grounding

    def edges_for(self, declaration: str) -> list:
        """Every registered edge whose hypothesis belongs to ``declaration``."""
        return [e for e in self.edges if e.ref.declaration == declaration]

    def observed(self) -> list:
        """Edges that actually activated during this run."""
        return [e for e in self.edges if e.observations]

    def declared_but_unobserved(self) -> list:
        """
        Edges that never activated.

        A branch not taken. The ledger does not describe what executed, which is
        worth surfacing even though it is not an error.
        """
        return [e for e in self.edges if e.site is not None and not e.observations]

    def clear(self) -> None:
        self.edges.clear()
        self.groundings.clear()

    def __iter__(self) -> Iterator:
        return iter(self.edges)

    def __len__(self) -> int:
        return len(self.edges)


#: The default registry the predicates write to.
REGISTRY = TheoryRegistry()
