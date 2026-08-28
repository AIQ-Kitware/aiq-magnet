import Mathlib

/-!
# Counting coin-flip outcomes

The statement behind `theory_coin_flip_exact.yaml`. The card enumerates every
sequence of `n` fair flips and compares the head-count frequencies against the
binomial law; this is that law, as a counting fact about subsets.
-/

namespace MagnetExamples.CoinFlip

/-- Exactly `n.choose k` of the `2 ^ n` outcomes of `n` flips show `k` heads.

Reading a sequence of flips as the set of positions that came up heads, the
outcomes with `k` heads are the `k`-element subsets of `Fin n`. -/
theorem card_heads_eq_choose (n k : ℕ) :
    ((Finset.univ : Finset (Fin n)).powersetCard k).card = n.choose k := by
  simp [Finset.card_powersetCard]

/-- There are `2 ^ n` outcomes in total, so each has probability `1 / 2 ^ n`
and the head count `k` has probability `n.choose k / 2 ^ n`. -/
theorem card_outcomes (n : ℕ) :
    (Finset.univ : Finset (Fin n)).powerset.card = 2 ^ n := by
  simp

end MagnetExamples.CoinFlip
