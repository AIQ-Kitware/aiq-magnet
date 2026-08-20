namespace MagnetExamples.TrainingOrder

/-- A learning rule is order-invariant when any reordering of the same
observations produces the same result. -/
def OrderInvariant {Obs Model : Type} (train : List Obs → Model) : Prop :=
  ∀ xs ys : List Obs, xs.Perm ys → train xs = train ys

/-- A rule that keeps only the first observation it sees. -/
def firstSeen (xs : List Nat) : Nat := xs.headD 0

/-- Some learning rules are not order-invariant. -/
theorem firstSeen_not_orderInvariant : ¬ OrderInvariant firstSeen := by
  intro h
  have perm : [1, 2].Perm [2, 1] := List.Perm.swap 2 1 []
  have := h [1, 2] [2, 1] perm
  simp [firstSeen] at this

end MagnetExamples.TrainingOrder
