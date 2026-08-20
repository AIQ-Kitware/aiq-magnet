import Mathlib

/-!
# Order sensitivity in online learning

The question behind `theory_training_order.yaml`. The card runs an online
perceptron over the same observations in two orders and gets two different
predictions. `OrderInvariant` names the property it fails.
-/

namespace MagnetExamples.TrainingOrder

/-- A learning rule is order-invariant when any reordering of the same
observations produces the same result. -/
def OrderInvariant {Obs Model : Type*} (train : List Obs → Model) : Prop :=
  ∀ xs ys : List Obs, xs.Perm ys → train xs = train ys

/-- A rule that keeps only the observation it saw first. -/
def firstSeen (xs : List ℤ) : ℤ := xs.headI

/-- Order invariance is a real restriction: some rules fail it.

The card exhibits a less contrived one. What no statement here answers is the
question the card points at -- which rules are order-invariant, and why. -/
theorem firstSeen_not_orderInvariant : ¬ OrderInvariant firstSeen := by
  intro h
  have perm : [(1 : ℤ), 2].Perm [2, 1] := List.Perm.swap 2 1 []
  have := h [1, 2] [2, 1] perm
  simp [firstSeen] at this

end MagnetExamples.TrainingOrder
