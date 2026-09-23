"""
theorem_2_2_demo.py
--------------------
Chunk 3: constructive demonstration of Theorem 2.2 (undesirability of SMP
in multi-class cases).

STATUS: newly written. This is not present anywhere in the repo (the repo
has no theoretical/demo code, only the M2M-GNN layer). We reproduce the
paper's own proof strategy from Appendix B.3 computationally, using hand-set
signed weights (NOT a trained model) -- this is a proof-by-example, not a
learning experiment.

Paper's argument (Sec 2.2, Theorem 2.2, proof in Appendix B.3):
  Take a 3-node path v1 - v2 - v3, with 3 DISTINCT classes: y1=1, y2=2, y3=3.
  Every edge is heterophilic (endpoints have different labels), so a
  "desirable" SMP (Def 2.1) must assign NEGATIVE weight to both edges:
      A(1)_{21} < 0   (edge v1-v2, both single-hop matrices desirable)
      A(2)_{32} < 0   (edge v2-v3)
  The cumulative 2-hop matrix is T = A(2) @ A(1). Its (3,1) entry represents
  the two-hop path v1 -> v2 -> v3, and by matrix multiplication:
      T_31 = A(2)_{32} * A(1)_{21}
  sign(T_31) = sign(A(2)_{32}) * sign(A(1)_{21}) = (-1)*(-1) = +1
  But y1 != y3 (class 1 vs class 3), so Def 2.1 REQUIRES T_31 <= 0.
  We got T_31 > 0: a straightforward, direct violation, despite BOTH
  single-hop matrices being individually perfectly desirable.
"""

import numpy as np


def is_desirable(M: np.ndarray, y: np.ndarray, tol: float = 1e-9) -> tuple[bool, list]:
    """
    Check Definition 2.1: for all i,j -> M_ij >= 0 if y_i == y_j,
                                          M_ij <= 0 if y_i != y_j.
    Returns (is_desirable, list_of_violations).
    Only checks entries where M_ij != 0 (no edge => no constraint).
    """
    n = len(y)
    violations = []
    for i in range(n):
        for j in range(n):
            if abs(M[i, j]) < tol:
                continue  # no edge (or exactly the diagonal with 0), skip
            same_class = (y[i] == y[j])
            if same_class and M[i, j] < -tol:
                violations.append((i, j, M[i, j], "same-class but weight < 0"))
            elif (not same_class) and M[i, j] > tol:
                violations.append((i, j, M[i, j], "different-class but weight > 0"))
    return (len(violations) == 0), violations


def minimal_counterexample():
    """
    The paper's exact 3-node, 3-class counterexample (Appendix B.3).
    Node indices: 0 = v1 (class 1), 1 = v2 (class 2), 2 = v3 (class 3).
    Path: v1 -- v2 -- v3 (v2 is the middle/shared node).
    """
    y = np.array([1, 2, 3])  # three distinct classes -- this is the key requirement (C > 2)

    # A1: single-hop propagation matrix used to go from layer 0 -> layer 1.
    # Only the v1-v2 edge exists (and its symmetric counterpart + self-loops,
    # which we omit here for clarity since they don't affect the argument).
    A1 = np.zeros((3, 3))
    A1[1, 0] = -0.6   # message v1 -> v2 : heterophilic edge, correctly NEGATIVE
    A1[0, 1] = -0.6   # symmetric

    # A2: single-hop propagation matrix used to go from layer 1 -> layer 2.
    # Only the v2-v3 edge exists.
    A2 = np.zeros((3, 3))
    A2[2, 1] = -0.7   # message v2 -> v3 : heterophilic edge, correctly NEGATIVE
    A2[1, 2] = -0.7   # symmetric

    return y, A1, A2


if __name__ == "__main__":
    y, A1, A2 = minimal_counterexample()
    print("Classes:", y, " (v1=class1, v2=class2, v3=class3 -- all distinct, C=3)")
    print()

    ok1, viol1 = is_desirable(A1, y)
    ok2, viol2 = is_desirable(A2, y)
    print(f"A(1) desirable? {ok1}   (violations: {viol1})")
    print(f"A(2) desirable? {ok2}   (violations: {viol2})")
    print()
    print("A(1) =\n", A1)
    print("A(2) =\n", A2)

    # Cumulative 2-hop matrix, exactly as Eq. 2 defines it: T = A(K)...A(1)
    T = A2 @ A1
    print("\nT = A(2) @ A(1) =\n", T)

    okT, violT = is_desirable(T, y)
    print(f"\nT desirable? {okT}")
    if not okT:
        print("Violations found in T (the cumulative 2-hop matrix):")
        for (i, j, val, reason) in violT:
            print(f"  T[{i},{j}] = {val:+.4f}  -- {reason} (y[{i}]={y[i]}, y[{j}]={y[j]})")

    print("\n=== Conclusion ===")
    print(f"T[2,0] = {T[2,0]:+.4f}  (this is the v1 -> v3 two-hop entry)")
    print(f"y[0]={y[0]} (v1), y[2]={y[2]} (v3): DIFFERENT classes -> Def 2.1 requires T[2,0] <= 0")
    print(f"But T[2,0] = {T[2,0]:+.4f} > 0: DESIRABILITY VIOLATED at the 2-hop level,")
    print("despite BOTH single-hop matrices A(1) and A(2) being individually desirable.")