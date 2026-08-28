import Mathlib

/-!
# A structural explanation for a runtime observation

The Python example empirically observes that naive recursive Fibonacci is much
slower than an iterative implementation. This file deliberately does not model
Python or wall-clock time. It states a small abstract cost model that explains
why a large runtime gap is plausible: recursion repeats subproblems, while the
iterative implementation takes one loop step per input index.
-/

namespace MagnetExamples.FibonacciPerformance

/-- Number of function calls made by the usual naive recursive Fibonacci
implementation, counting the current call. -/
def recursiveCalls : Nat → Nat
  | 0 => 1
  | 1 => 1
  | n + 2 => 1 + recursiveCalls (n + 1) + recursiveCalls n

/-- Number of loop iterations in the corresponding iterative implementation. -/
def iterativeSteps (n : Nat) : Nat := n

/-- At the input used by the example card, the abstract recursive call count is
more than one thousand times the iterative step count. This is an operation
count statement, not a theorem about wall-clock execution time. -/
theorem recursiveCalls_28_costGap :
    1000 * iterativeSteps 28 < recursiveCalls 28 := by
  native_decide

end MagnetExamples.FibonacciPerformance
