import Mathlib

/-!
# Geometry and the sampling model behind the Monte Carlo example

The empirical example has two theoretical targets.

`volume_quarterDisc` is the exact geometric quantity being estimated.
`monteCarloEstimator_consistent` states the stronger asymptotic sampling model:
measurable, independent, uniformly distributed points in the unit square have a
quarter-disc hit rate converging almost surely to `π / 4`.

The second declaration deliberately exposes its scientific assumptions as named
binders. MAGNET's static premise annotations resolve against those names. The
example implementation uses a deterministic LCG rather than IID random draws,
so its source can describe that substitution without recording runtime values.
-/

namespace MagnetExamples.Circle

open Filter MeasureTheory Metric ProbabilityTheory
open scoped ENNReal

/-- The closed unit disc has area `π`. -/
theorem volume_unit_disc :
    volume (closedBall (0 : ℂ) 1) = (NNReal.pi : ENNReal) := by
  simp [Complex.volume_closedBall]

/-- The part of the closed unit disc in the first quadrant. -/
def quarterDisc : Set ℂ := {z | ‖z‖ ≤ 1 ∧ 0 ≤ z.re ∧ 0 ≤ z.im}

/-- The unit square from which the Monte Carlo model draws points. -/
def unitSquare : Set ℂ :=
  {z | 0 ≤ z.re ∧ z.re ≤ 1 ∧ 0 ≤ z.im ∧ z.im ≤ 1}

/-- The quarter disc has area `π / 4`. -/
theorem volume_quarterDisc :
    volume quarterDisc = (NNReal.pi : ENNReal) / 4 := by
  sorry

/--
A compact formal statement of the sampling model the empirical estimator is
trying to approximate.

The four named binders are intentionally part of the public theory/practice
interface:

* `hindicator`: the empirical hit predicate is exactly quarter-disc membership;
* `hmeas`: each sample point is measurable;
* `hiid`: the sample family is independent;
* `huniform`: every point has the uniform unit-square distribution.

A full proof can later discharge this from a strong law plus
`volume_quarterDisc`. The declaration is already useful before that proof is
finished because the empirical implementation can account for each premise by
name.
-/
theorem monteCarloEstimator_consistent
    {Ω : Type*}
    [MeasurableSpace Ω]
    (P : Measure Ω)
    [IsProbabilityMeasure P]
    (X : ℕ → Ω → ℂ)
    (hit : ℂ → Bool)
    (hindicator : ∀ z, hit z = decide (z ∈ quarterDisc))
    (hmeas : ∀ i, Measurable (X i))
    (hiid : ProbabilityTheory.iIndepFun X P)
    (huniform : ∀ i, Measure.map (X i) P = volume.restrict unitSquare) :
    ∀ᵐ ω ∂P,
      Tendsto
        (fun n : ℕ =>
          ((Finset.range n).sum fun i =>
            if hit (X i ω) = true then (1 : ℝ) else 0) / n)
        atTop
        (𝓝 (Real.pi / 4)) := by
  sorry

end MagnetExamples.Circle
