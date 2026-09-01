# Coin flip: theory predicts the outcome exactly

Enumerating all `2^n` sequences of `n` fair flips gives the probability of each
head count. The binomial law states the same quantity in closed form. They
agree exactly, so the deviation is zero rather than small.

This is what `theory.tests` means: the theoretical object predicts the outcome,
and the code confirms it. No tolerance appears anywhere in the card.

## Run it

```bash
magnet evaluate_new magnet/examples/theory_links/coin_flip/card.yaml \
    --output_path=./evaluation_runs --backend=serial
```

Four cells, one per flip count in the matrix, each its own verdict.

## What is where

| file | role |
|---|---|
| `card.yaml` | the recipe: pipeline, matrix, claim, theory block |
| `experiment.py` | the kwdagger node, carrying the `@theory.tests` annotation |
| `theory.yaml` | generated index mapping the entry ID to the Lean declaration |
| `CoinFlip.lean` | the formal statement |

## Reading the card

The node is named `enumeration`, not `enumerate`. A claim can address a node by
its bare name, so a node named for a Python builtin would shadow it.

`evidence.scope: requested` keeps the verdict tied to the flip counts this
invocation asked for, rather than every enumeration cached in the shared store.

Enumeration is exponential in `n_flips`, which is why the matrix stops at 14.

## What to look at afterwards

`theory.json` in the run directory. The `tests` link is read out of
`experiment.py`, so it says which function was annotated and where:

```json
{"relation": "tests", "ref": "Examples.CoinFlip.Binomial",
 "qualname": "enumerated_head_counts", "file": "experiment.py"}
```

Compare with `monte_carlo`, where the same machinery reports a theory/practice
gap instead of an exact agreement.
