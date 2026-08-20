# Theory links: what this is, and what it deliberately is not

Notes from the design discussion, kept because the shape of this feature is
the result of throwing away a larger one, and the reasons will not be visible
from the code that survived.

The user-facing documentation is `docs/source/manual/theory_links.rst`. This
file is the argument behind it.

## The sentence the feature has to establish

> MAGNET can record how empirical code relates to a theoretical object.

Everything merged serves that. An earlier attempt tried to establish it *and*
build hypothesis-level assumption auditing at the same time, and came to ~3,490
added lines. Reviewing it meant understanding a coverage model, a severity
scale, proof status, freshness, review workflow and binder-level references
before you could judge whether the core idea was right.

## Three relations, one grammar

    practice tests theory
    practice approximates theory
    practice motivates theory

The three read as sentences with the same structure, which is what makes the
vocabulary learnable. They also form a progression:

| relation | the role theory plays |
|---|---|
| `tests` | theory says exactly what should happen; practice checks it |
| `approximates` | theory defines something exact; practice estimates it |
| `motivates` | practice finds something; theory is asked to explain it |

`motivates` is the one that took discussion to arrive at, and the one that
matters most for real evaluation work. An experiment that establishes a
phenomenon does not have to explain it. Naming the question is what lets an
explanation arrive later, and it is an honest connection where inventing a
theorem to point at would not be.

The loop this opens:

    empirical observation
        └── motivates → question
                           └── conjecture
                                  └── tests → new experiment

An entry keeps its id as it moves along that path, so code pointing at a
question keeps working when the question becomes a conjecture and then a
theorem.

## The examples are miniature methods

Each example directory is a small stand-in for what a team brings to the
program: a method, an idea it is evidence for, and a card that runs it.

    coin_flip/       card.yaml  experiment.py  CoinFlip.lean       theory.yaml
    monte_carlo/     card.yaml  experiment.py  Circle.lean         theory.yaml
    training_order/  card.yaml  experiment.py  TrainingOrder.lean  theory.yaml

`experiment.py` is the method. It carries the annotation, because the
relationship belongs next to the code that does the work rather than in a
separate registry that drifts from it. `card.yaml` is what MAGNET runs.
`theory.yaml` names what the method is evidence for. The `.lean` file is that
statement, formally.

Keeping them in one directory means an example can be read in one place and
copied out in one piece. It also means the card names its siblings
(`sources: [experiment.py]`) rather than reaching across the tree.

None of the three uses kwdagger. A card whose symbol imports a helper is
enough to exercise the mechanism, and it keeps DAG execution, leasing,
scheduling and result-cell machinery out of the review.

## Where the claim lives today, and where it is going

Right now each card decides its verdict with a `claim:` block of Python:

```yaml
claim:
  python: |
    assert deviation == 0
```

That is what `upstream/main` supports. The direction is a declarative
`evidence:` block — what was measured, the relation asserted, the scope it
holds over, what had to be relaxed — which exists on
`dev/evidence-and-per-cell-results` and is not part of this branch.

The two fit together without conflict: `evidence:` is how a card states its
finding, and a theory link is what that finding is evidence *for*. When the
branches meet, these three cards are the natural first users of both at once.
Wiring it in here would merge two reviews into one.

## Deliberately absent

Moved to a later pass, once reviewers understand the graph being enriched:

- `satisfies`, `substitutes`, `assumes`, `ignores`, `violates`, and
  hypothesis-level `approximates`
- `grounds`, whose theorem-level distinction the three relations replace
- coverage reports, assumption accounting, dangling-hypothesis bookkeeping
- severity, review status, freshness as modeled axes
- proof status read from the kernel; `#print axioms` and `KERNEL_AXIOMS`
- `Declaration::binder` references and hypothesis enumeration
- a generated ledger artifact separate from the card
- runtime attachment of annotations to the objects they decorate

The last one is worth a sentence: nothing consumed those attributes except
their own tests. Static extraction was already the declared source of truth,
so the runtime half was cost with no reader.

## Choices that are easy to get wrong

**One accepted spelling.** The extractor takes `<alias>.<relation>('literal')`
as a decorator or a with-item, and nothing else. Being liberal in what you
accept means nobody can predict whether an annotation counts. An earlier
version accepted object references, f-strings, concatenation, module-level
bindings and `typing.Annotated`, and needed 734 lines to do it.

**The relations are inert.** They return what they wrap. Annotated code behaves
identically whether or not anyone reads the annotations, which is what makes it
reasonable to ask a team to add them.

**`tests.__test__ = False`.** pytest collects any module-level callable named
`test*`, so importing the relation into a test file turns it into a collected
test that fails on its own argument. The verb is right; the collection
convention should not get to veto it.

**An entry without a `declaration` is valid.** It says the statement exists in
prose and nobody has formalized it, which is the state most entries start in.
Requiring formalization would push people toward pointing at whatever theorem
happens to exist.
