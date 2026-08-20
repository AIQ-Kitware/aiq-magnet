import Mathlib

/-!
# The area a Monte Carlo sample estimates

The statement behind `theory_monte_carlo.yaml`. The card draws points in the
unit square and counts how many land inside the quarter disc; this is the
quantity that fraction converges to.
-/

namespace MagnetExamples.Circle

open MeasureTheory Metric

/-- The closed unit disc has area `π`. -/
theorem volume_unit_disc :
    volume (closedBall (0 : ℂ) 1) = (NNReal.pi : ENNReal) := by
  simp [Complex.volume_closedBall]

/-- The part of the closed unit disc in the first quadrant, which is the region
the card samples. -/
def quarterDisc : Set ℂ := {z | ‖z‖ ≤ 1 ∧ 0 ≤ z.re ∧ 0 ≤ z.im}

/-- The quarter disc has area `π / 4`. The unit square it is sampled from has
area 1, so this is also the fraction of uniform samples landing inside. -/
theorem volume_quarterDisc :
    volume quarterDisc = (NNReal.pi : ENNReal) / 4 := by
  sorry

end MagnetExamples.Circle
