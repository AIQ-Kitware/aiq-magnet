# Monte Carlo: an approximation, with its premises accounted for

A quarter of the unit disc occupies `pi/4` of the unit square. Sampling points
and counting how many land inside estimates that ratio. Unlike `coin_flip`, the
estimate carries error the closed form does not, so the claim needs a tolerance.

The interesting part is not the estimate. It is that the source states, in
annotations, exactly how the implementation stands to the theorem's named
premises — including the one it does not satisfy.

## Run it

```bash
magnet evaluate_new magnet/examples/theory_links/monte_carlo/card.yaml \
    --output_path=./evaluation_runs --backend=serial
```

Five cells, one per seed. One seed is an estimate; five is a statement about
the estimator.

## What is where

| file | role |
|---|---|
| `card.yaml` | the recipe: pipeline, seed matrix, claim, theory block |
| `experiment.py` | the node, carrying the statement and premise annotations |
| `theory.yaml` | generated index, including the theorem's four named premises |
| `Circle.lean` | the exact area theorem and the asymptotic sampling theorem |

## The premise ledger

`Circle.lean`'s sampling theorem exposes four premises. `experiment.py`
accounts for three and leaves one alone:

| premise | relation | why |
|---|---|---|
| `hindicator` | `satisfies` | the hit predicate is exactly the formal region |
| `hmeas` | `assumes` | the measurability bridge is taken, not formalized |
| `huniform` | `substitutes` | a seeded LCG stands in for uniform draws |
| `hiid` | *unaccounted* | deliberately left open |

That gap is the point. `theory.json` reports `complete: false` with
`unaccounted: ["hiid"]`, which is how MAGNET states a static theory/practice
gap without claiming the Python has been verified.

## Reading the card

The claim asserts the error is *both* under tolerance and above zero. A finite
sample landing exactly on `pi/4` would mean the estimator is not doing what the
card says it does.

`tolerance` is a declared symbol rather than a pipeline output, because the
bound the claim tests against belongs to the card, not to the run.

The theorem is asymptotic and every cell here is finite, so a passing verdict is
evidence about the estimator at this sample size — not a test of the limit.
