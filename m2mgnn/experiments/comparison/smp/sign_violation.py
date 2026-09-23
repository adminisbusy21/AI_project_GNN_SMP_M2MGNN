"""
sign_violation_study.py
------------------------
Chunk 3 (extension): is Theorem 2.2's failure a rare cherry-picked edge case,
or does it happen systematically as the number of classes C grows?

STATUS: newly written. Not in the repo or the paper itself -- this is our
own empirical extension of Theorem 2.2, designed to test the theorem's own
stated condition ("C > 2") directly: we expect the violation rate to be
exactly 0 when C=2 (binary case, matching the paper's remark that binary SMP
does NOT suffer this problem) and to be >0 once C>=3.

Method:
  1. Generate a random graph with N nodes, C classes (uniform random labels).
  2. Build a "perfectly desirable" 1-hop weighted adjacency matrix: for each
     edge, weight = +w if same class, -w if different class (w > 0 random).
     This is a BY-CONSTRUCTION desirable matrix (Def 2.1 holds exactly).
  3. Compute the 2-hop matrix T = A @ A (using the SAME desirable A twice,
     matching the "even if every A(k) is desirable" premise of Theorem 2.2).
  4. For every pair (i,j) with a nonzero 2-hop path (T_ij != 0), check
     whether sign(T_ij) matches what Def 2.1 requires given y_i, y_j.
  5. Report the fraction of violating pairs, for C = 2, 3, 4, 5, 6.
"""

import numpy as np


def build_desirable_matrix(y: np.ndarray, edge_prob: float, rng: np.random.Generator) -> np.ndarray:
    """
    Build a random graph's adjacency with weights assigned EXACTLY per
    Definition 2.1: positive for same-class pairs, negative for different-
    class pairs. This matrix is desirable by construction -- we are not
    learning it, we are directly testing the theorem's premise.
    """
    n = len(y)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < edge_prob:
                w = rng.uniform(0.3, 1.0)  # magnitude, sign decided below
                sign = 1.0 if y[i] == y[j] else -1.0
                A[i, j] = sign * w
                A[j, i] = sign * w  # symmetric
    return A


def violation_rate(T: np.ndarray, y: np.ndarray, tol: float = 1e-9) -> tuple[float, int, int]:
    """Fraction of nonzero T_ij entries that violate Def 2.1."""
    n = len(y)
    total_nonzero = 0
    violations = 0
    for i in range(n):
        for j in range(n):
            if i == j or abs(T[i, j]) < tol:
                continue
            total_nonzero += 1
            same = (y[i] == y[j])
            if same and T[i, j] < -tol:
                violations += 1
            elif (not same) and T[i, j] > tol:
                violations += 1
    rate = violations / total_nonzero if total_nonzero > 0 else float("nan")
    return rate, violations, total_nonzero


def run_study(n_nodes=60, edge_prob=0.15, n_trials=30, seed=0):
    rng = np.random.default_rng(seed)
    results = []
    for C in [2, 3, 4, 5, 6]:
        rates = []
        for trial in range(n_trials):
            y = rng.integers(0, C, size=n_nodes)
            A = build_desirable_matrix(y, edge_prob, rng)

            # sanity check: A itself must be desirable by construction
            # (spot check, not exhaustive, since we KNOW how we built it)

            T = A @ A  # 2-hop cumulative matrix, Eq. (2) with A(1)=A(2)=A
            rate, viol, total = violation_rate(T, y)
            if not np.isnan(rate):
                rates.append(rate)
        results.append({
            "C": C,
            "mean_violation_rate": round(float(np.mean(rates)), 4),
            "std_violation_rate": round(float(np.std(rates)), 4),
            "n_trials": len(rates),
        })
    return results


if __name__ == "__main__":
    print("Studying 2-hop desirability violation rate as C (number of classes) varies.")
    print("A(1)=A(2)=A is built to be PERFECTLY desirable by construction each trial.")
    print(f"Setup: 60 nodes, edge_prob=0.15, 30 random trials per C.\n")

    results = run_study()
    print(f"{'C':>3s}  {'mean violation rate':>20s}  {'std':>8s}")
    for r in results:
        print(f"{r['C']:>3d}  {r['mean_violation_rate']:>20.4f}  {r['std_violation_rate']:>8.4f}")

    print("\nExpectation from Theorem 2.2 / paper's remark on binary case:")
    print("  C=2 -> violation rate should be ~0 (binary SMP is theoretically safe)")
    print("  C>=3 -> violation rate should be > 0 and likely grow with C")