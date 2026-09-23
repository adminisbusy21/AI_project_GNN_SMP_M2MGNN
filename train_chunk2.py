"""
train_chunk2.py
Chunk 2: train MLP, GCN, and SMP on Texas (heterophilic) and Cora (homophilic),
using the same geom-gcn splits and training protocol as Chunk 1.
STATUS: newly written; reuses Chunk 1's train_one_model loop unchanged
(same optimizer, same model-selection-by-val-accuracy rule) so that adding
SMP to the comparison is a fair, apples-to-apples extension rather than a
different experiment.
Usage:
    python3 train_chunk2.py
"""

import torch
from data_utils import load_dataset, dataset_stats
from models_baseline import MLP, GCN
from smp_model import SMPNet
from train_chunk1 import train_one_model, get_masks, set_seed, SEED, HIDDEN, DROPOUT, SPLIT_IDX


def run_dataset(name):
    data, ds = load_dataset(name)
    stats = dataset_stats(data, name)
    train_mask, val_mask, test_mask = get_masks(data, SPLIT_IDX)

    results = {"dataset": name, "edge_homophily": stats["edge_homophily_no_selfloop"]}

    models = [("MLP", MLP), ("GCN", GCN), ("SMP", SMPNet)]
    for model_name, ModelClass in models:
        set_seed(SEED)
        model = ModelClass(
            in_feat=data.num_node_features,
            hid_feat=HIDDEN,
            out_feat=stats["num_classes"],
            dropout=DROPOUT,
        )
        val_acc, test_acc = train_one_model(model, data, train_mask, val_mask, test_mask)
        results[f"{model_name}_val_acc"] = round(val_acc, 4)
        results[f"{model_name}_test_acc"] = round(test_acc, 4)

    return results


def inspect_learned_signs(name):
    """
    After training SMP, check whether it actually learned Def. 2.1's
    'desirable' pattern: alpha >= 0 for same-class edges, alpha <= 0 for
    different-class edges. This directly probes the research question,
    beyond just accuracy.
    """
    data, ds = load_dataset(name)
    stats = dataset_stats(data, name)
    train_mask, val_mask, test_mask = get_masks(data, SPLIT_IDX)

    set_seed(SEED)
    model = SMPNet(in_feat=data.num_node_features, hid_feat=HIDDEN,
                    out_feat=stats["num_classes"], dropout=DROPOUT)
    train_one_model(model, data, train_mask, val_mask, test_mask)

    model.eval()
    with torch.no_grad():
        _, alphas = model(data.x, data.edge_index, return_alpha=True)
        alpha1, row1, col1 = alphas["layer1"]

        same_class = (data.y[row1] == data.y[col1])
        # exclude self-loops (row==col) added inside SMPLayer -- trivially same-class
        not_self = row1 != col1
        same_class = same_class & not_self
        diff_class = (~same_class) & not_self

        same_mean = alpha1[same_class].mean().item() if same_class.any() else float("nan")
        diff_mean = alpha1[diff_class].mean().item() if diff_class.any() else float("nan")

    return {
        "dataset": name,
        "mean_alpha_same_class_edges": round(same_mean, 4),
        "mean_alpha_diff_class_edges": round(diff_mean, 4),
        "desirable_direction": same_mean > diff_mean,  # Def 2.1 expectation
    }


if __name__ == "__main__":
    all_results = []
    for name in ["Texas", "Cora"]:
        res = run_dataset(name)
        all_results.append(res)
        print(res)

    print("\n=== Summary (single split, split_idx=0): MLP vs GCN vs SMP ===")
    header = f"{'Dataset':10s} {'EdgeHom':>8s} {'MLP':>8s} {'GCN':>8s} {'SMP':>8s} {'SMP-GCN':>9s}"
    print(header)
    for r in all_results:
        diff = r["SMP_test_acc"] - r["GCN_test_acc"]
        print(f"{r['dataset']:10s} {r['edge_homophily']:8.3f} "
              f"{r['MLP_test_acc']:8.4f} {r['GCN_test_acc']:8.4f} {r['SMP_test_acc']:8.4f} {diff:+9.4f}")

    print("\n=== Learned sign check (Def 2.1: same-class alpha should be > diff-class alpha) ===")
    for name in ["Texas", "Cora"]:
        sign_res = inspect_learned_signs(name)
        print(sign_res)