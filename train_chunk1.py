"""
train_chunk1.py
----------------
Chunk 1: train MLP vs GCN on Texas (heterophilic) and Cora (homophilic),
using the geom-gcn splits shipped with each dataset.

STATUS: newly written. No training code exists in the official repo.

Usage:
    python3 train_chunk1.py
"""

import torch
import torch.nn.functional as F
from data_utils import load_dataset, dataset_stats
from models_baseline import MLP, GCN

DEVICE = torch.device("cpu")
SEED = 0
HIDDEN = 64
EPOCHS = 200
LR = 0.01
WEIGHT_DECAY = 5e-4
DROPOUT = 0.5
SPLIT_IDX = 0  # which of the 10 geom-gcn splits to use for this single-run demo


def set_seed(seed):
    torch.manual_seed(seed)


def get_masks(data, split_idx):
    # train_mask/val_mask/test_mask have shape [num_nodes, num_splits]
    return (data.train_mask[:, split_idx],
            data.val_mask[:, split_idx],
            data.test_mask[:, split_idx])


def train_one_model(model, data, train_mask, val_mask, test_mask,
                     epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = 0.0
    best_test_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            val_acc = (pred[val_mask] == data.y[val_mask]).float().mean().item()
            test_acc = (pred[test_mask] == data.y[test_mask]).float().mean().item()

        # model selection by validation accuracy, as is standard practice
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

    return best_val_acc, best_test_acc


def run_dataset(name):
    data, ds = load_dataset(name)
    stats = dataset_stats(data, name)
    train_mask, val_mask, test_mask = get_masks(data, SPLIT_IDX)

    results = {"dataset": name, "edge_homophily": stats["edge_homophily_no_selfloop"]}

    for model_name, ModelClass in [("MLP", MLP), ("GCN", GCN)]:
        set_seed(SEED)
        model = ModelClass(
            in_feat=data.num_node_features,
            hid_feat=HIDDEN,
            out_feat=stats["num_classes"],
            dropout=DROPOUT,
        ).to(DEVICE)
        val_acc, test_acc = train_one_model(model, data, train_mask, val_mask, test_mask)
        results[f"{model_name}_val_acc"] = round(val_acc, 4)
        results[f"{model_name}_test_acc"] = round(test_acc, 4)

    return results


if __name__ == "__main__":
    all_results = []
    for name in ["Texas", "Cora"]:
        res = run_dataset(name)
        all_results.append(res)
        print(res)

    print("\n=== Summary (single split, split_idx=0) ===")
    header = f"{'Dataset':10s} {'EdgeHom':>8s} {'MLP_test':>10s} {'GCN_test':>10s} {'GCN-MLP':>10s}"
    print(header)
    for r in all_results:
        diff = r["GCN_test_acc"] - r["MLP_test_acc"]
        print(f"{r['dataset']:10s} {r['edge_homophily']:8.3f} "
              f"{r['MLP_test_acc']:10.4f} {r['GCN_test_acc']:10.4f} {diff:+10.4f}")