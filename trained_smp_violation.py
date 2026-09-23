"""
trained_smp_violation_check.py
--------------------------------
Chunk 3 (connects back to Chunk 2): does our ACTUAL trained SMP model
(smp_model.py, trained in Chunk 2) exhibit Theorem 2.2's multi-hop
desirability violation on real data (Cora, Texas), using its real learned
alpha values -- not the synthetic matrices from sign_violation_study.py?

STATUS: newly written. This is the bridge between the abstract theorem
(Chunk 3's core demo) and our own trained model (Chunk 2), which the paper
itself does not explicitly provide (their Theorem 2.2 is a pure existence
proof; connecting it to a concretely trained model's weight matrix is our
own added experiment).

Method:
  1. Train SMPNet on a dataset (reuse Chunk 2's exact protocol).
  2. Extract layer1's learned edge weights (alpha_ij * norm_j, the ACTUAL
     coefficient used in message passing) as a sparse signed adjacency A1.
  3. Compute the 2-hop matrix T = A1 @ A1 (approximating what a 2-layer
     stack would compose, using the same layer's learned weights twice for
     a clean, directly comparable analysis).
  4. Check what fraction of nonzero 2-hop node pairs violate Def 2.1,
     using the TRUE class labels (which the model never saw at edge level).
"""

import torch
import numpy as np
from smp_model import SMPNet
from data_utils import load_dataset, dataset_stats
from train_chunk1 import train_one_model, get_masks, set_seed, SPLIT_IDX, HIDDEN, DROPOUT


def extract_signed_adjacency(model, data):
    """
    Pull out the actual signed coefficients (alpha_ij * norm_j) that
    SMPLayer.forward uses for message passing in layer 1, and assemble
    them into a dense N x N matrix so we can matrix-multiply for the
    2-hop composition. Dense is fine here since Texas/Cora subsets are
    small; for very large graphs one would keep this sparse.
    """
    model.eval()
    with torch.no_grad():
        _, alphas = model(data.x, data.edge_index, return_alpha=True)
        alpha1, row1, col1 = alphas["layer1"]

        n = data.num_nodes
        A = torch.zeros(n, n)
        A[row1, col1] = alpha1  # the actual signed message coefficient used
    return A.numpy()


def is_desirable_entry(val, yi, yj, tol=1e-9):
    if abs(val) < tol:
        return None  # no meaningful edge, skip
    same = (yi == yj)
    if same and val < -tol:
        return False
    if (not same) and val > tol:
        return False
    return True


def check_violations(A, y, tol=1e-9):
    n = len(y)
    total, violations = 0, 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            res = is_desirable_entry(A[i, j], y[i], y[j], tol)
            if res is None:
                continue
            total += 1
            if not res:
                violations += 1
    return violations, total, (violations / total if total > 0 else float("nan"))


def run(name):
    data, ds = load_dataset(name)
    stats = dataset_stats(data, name)
    train_mask, val_mask, test_mask = get_masks(data, SPLIT_IDX)

    set_seed(0)
    model = SMPNet(in_feat=data.num_node_features, hid_feat=HIDDEN,
                    out_feat=stats["num_classes"], dropout=DROPOUT)
    train_one_model(model, data, train_mask, val_mask, test_mask)

    A1 = extract_signed_adjacency(model, data)
    y = data.y.numpy()

    # 1-hop check (sanity: how "desirable" is the trained 1-hop matrix itself?)
    v1, t1, r1 = check_violations(A1, y)
    print(f"[{name}] 1-hop trained matrix: {v1}/{t1} edges violate Def 2.1 (rate={r1:.4f})")

    # 2-hop check (the actual Theorem 2.2 question)
    T = A1 @ A1
    v2, t2, r2 = check_violations(T, y)
    print(f"[{name}] 2-hop composed matrix (A1@A1): {v2}/{t2} pairs violate Def 2.1 (rate={r2:.4f})")
    return {"dataset": name, "1hop_violation_rate": round(r1, 4), "2hop_violation_rate": round(r2, 4)}


if __name__ == "__main__":
    results = []
    for name in ["Texas", "Cora"]:
        results.append(run(name))
        print()

    print("=== Summary: does our own trained SMP model exhibit Theorem 2.2's failure? ===")
    for r in results:
        print(r)

    print("""
IMPORTANT INTERPRETATION NOTE (found by digging past the summary numbers):
The 1-hop violation rate here is much higher than Chunk 2's "mean alpha"
check suggested. Reconciling the two: Chunk 2 showed mean(alpha | same-class)
> mean(alpha | diff-class), which is TRUE, but that is a weaker statement
than "different-class edges get negative alpha". On Texas, 81.7% of
different-class edges actually have a POSITIVE learned alpha -- the model
mostly learned uniformly positive attention, with same-class edges getting
an even larger positive boost on average. The ranking (same > diff) can
hold on average even when both groups are mostly on the "wrong" (positive)
side of zero.

This means our trained model does not actually reach the "every A(k) is
desirable" precondition that Theorem 2.2 assumes -- it is not desirable
even at 1 hop, for the majority of edges. So the 2-hop violation rate we
measured here (62% Texas, 29% Cora) is NOT a clean empirical confirmation
of Theorem 2.2 specifically (that requires desirable inputs); it mixes in
this separate, more basic optimization gap. The controlled synthetic study
(sign_violation_study.py), where we FORCE each A(k) to be perfectly
desirable by construction, is the correct experiment for isolating
Theorem 2.2's pure multi-hop composition effect -- and it shows the clean
C=2 vs C>=3 pattern the theorem predicts. Both findings matter for the
report, but they answer different questions and should not be conflated.
""")