# Fibonacci performance: separating a question from an explanation

Two Python functions compute the same Fibonacci number. The naive recursion is
much slower than the loop. The card measures that, and then is careful about
what the measurement is evidence *for*.

## Run it

```bash
magnet evaluate_new \
    magnet/examples/theory_links/fibonacci_performance/card.yaml \
    --output_path=./evaluation_runs --backend=serial
```

One cell: there is one phenomenon to observe, and repeats are handled inside the
timing rather than by fanning out jobs.

## What is where

| file | role |
|---|---|
| `card.yaml` | the recipe: one benchmark node, claim, theory block |
| `experiment.py` | the node, carrying both annotations |
| `theory.yaml` | generated index: one question, one theorem |
| `FibonacciCost.lean` | the abstract operation-count model |

## Two links, on purpose

The benchmark carries different relations to two different objects:

* `motivates` → `Examples.FibonacciPerformance.Why`, an open **question**. The
  runtime observation is a reason to ask why, not an answer.
* `approximates` → `...RecursiveCallGapAt28`, a **theorem** about operation
  counts. Wall-clock time is a proxy for abstract work; Lean does not model
  CPython timing and the card does not pretend it does.

A `motivates` link creates no premise-coverage obligation — a question has no
premises to satisfy — which is why `theory.json` shows an empty
`premise_coverage` here and a populated one in `monte_carlo`.

## Reading the card

The claim asserts three separate things: the implementations still agree, the
recursive one was slower on this run, and the abstract call-count gap still
matches the formal `n=28` example. The middle one is a timing observation and
could in principle fail on a strange machine; that is what makes it empirical
rather than a restatement of the theorem.
