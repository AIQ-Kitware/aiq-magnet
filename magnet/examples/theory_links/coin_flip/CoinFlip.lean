namespace MagnetExamples.CoinFlip

/-- Every sequence of `n` flips, in the order the Python enumeration produces
them. -/
def sequences : Nat → List (List Bool)
  | 0 => [[]]
  | n + 1 => (sequences n).flatMap fun rest => [true :: rest, false :: rest]

/-- Binomial coefficient. Defined here because core Lean has no `Nat.choose`
and this file deliberately does not depend on Mathlib. -/
def choose : Nat → Nat → Nat
  | _, 0 => 1
  | 0, _ + 1 => 0
  | n + 1, k + 1 => choose n k + choose n (k + 1)

/-- How many heads a sequence shows. -/
def headCount (flips : List Bool) : Nat := (flips.filter id).length

/-- There are `2 ^ n` sequences of `n` flips, so each is equally likely with
probability `1 / 2 ^ n`. -/
theorem length_sequences (n : Nat) : (sequences n).length = 2 ^ n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    have : ∀ l : List (List Bool),
        (List.map (fun _ => 2) l).sum = 2 * l.length := by
      intro l
      induction l with
      | nil => rfl
      | cons _ t iht => simp [iht, Nat.mul_succ, Nat.mul_comm, Nat.add_comm]
    simp [sequences, List.length_flatMap, this, ih, Nat.pow_succ,
      Nat.mul_comm]

/-- Exactly `n.choose k` of them show `k` heads. Together with the count above
this is the binomial probability the card checks by enumeration. -/
theorem count_headCount_eq (n k : Nat) :
    ((sequences n).filter (fun s => headCount s == k)).length = choose n k := by
  sorry

end MagnetExamples.CoinFlip
