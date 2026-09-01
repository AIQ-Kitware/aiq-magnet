# Theory link examples

Three small cards that connect empirical code to theoretical objects. Each one
is a complete kwdagger recipe: a pipeline node that computes something, a claim
read off the result, and a `theory:` block that resolves static links from the
node's source into `theory.json`.

They exist to show the *link*, not the science. The computations are trivial on
purpose so the annotation is the only interesting part.

| example | relation it demonstrates | verdict rests on |
|---|---|---|
| [`coin_flip`](coin_flip) | `tests` — theory predicts the outcome exactly | an exact zero |
| [`monte_carlo`](monte_carlo) | `approximates` + premise accounting | a tolerance, over 5 seeds |
| [`fibonacci_performance`](fibonacci_performance) | `motivates` vs `approximates` | an observed phenomenon |

Run any of them:

```bash
magnet evaluate_new magnet/examples/theory_links/<name>/card.yaml \
    --output_path=./evaluation_runs --backend=serial
```

Each run directory holds `verdict.json` (the claim) and `theory.json` (the
links). The links are read from the source annotations, so `theory.json` is a
property of the card and its code, not of how the card was executed.

The Lean sources beside each card are checked separately by `check_lean.sh`;
MAGNET reads the generated `theory.yaml` index, never Lean itself.
