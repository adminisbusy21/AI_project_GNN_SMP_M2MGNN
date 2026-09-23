
import os
import sys

import torch
import torch.nn as nn

# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ------------------------------------------------------------
# Existing project utilities
# ------------------------------------------------------------

from data_utils import load_dataset, dataset_stats

# IMPORTANT:
# This imports the ORIGINAL M2M-GNN implementation.
from model import M2MGNN


# ============================================================
# Experimental configuration
# ============================================================

SEED = 0
SPLIT_IDX = 0

HIDDEN = 64
NUM_LAYERS = 2

DROPOUT = 0.5
DROPOUT2 = 0.7

C = 5
BETA = 0.5
TEMPERATURE = 1.0

LR = 0.01
WD1 = 0.01
WD2 = 0.01

LAMBDA = 0.0

MAX_EPOCHS = 200

DATASETS = ["Cora", "Texas"]


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Train one M2M-GNN model
# ============================================================

def train_one_model(model, data, train_mask, val_mask, test_mask):

    optimizer = torch.optim.Adam(
        [
            {
                "params": model.params1,
                "weight_decay": WD1
            },
            {
                "params": model.params2,
                "weight_decay": WD2
            }
        ],
        lr=LR
    )

    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_test_acc = 0.0
    best_epoch = 0

    for epoch in range(1, MAX_EPOCHS + 1):

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        model.train()

        optimizer.zero_grad()

        out = model(
            data.x,
            data.edge_index
        )

        loss = (
            criterion(
                out[train_mask],
                data.y[train_mask]
            )
            + LAMBDA * model.reg
        )

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Validation + test
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            out = model(
                data.x,
                data.edge_index
            )

            pred = out.argmax(dim=1)

            val_acc = (
                pred[val_mask] == data.y[val_mask]
            ).float().mean().item()

            test_acc = (
                pred[test_mask] == data.y[test_mask]
            ).float().mean().item()

        # ----------------------------------------------------
        # Select model using validation accuracy
        # ----------------------------------------------------

        if val_acc > best_val_acc:

            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch

    return best_val_acc, best_test_acc, best_epoch


# ============================================================
# Run one dataset
# ============================================================

def run_dataset(dataset_name):

    print("\n" + "=" * 60)
    print(f"M2M-GNN BASELINE — {dataset_name}")
    print("=" * 60)

    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device : {device}")
    print(f"Seed   : {SEED}")
    print(f"Split  : {SPLIT_IDX}")

    # --------------------------------------------------------
    # Load dataset using the same project loader
    # --------------------------------------------------------

    data, dataset = load_dataset(
        dataset_name,
        root=os.path.join(PROJECT_ROOT, "data")
    )

    stats = dataset_stats(
        data,
        dataset_name
    )

    data = data.to(device)

    print(f"Nodes    : {data.num_nodes}")
    print(f"Features : {data.num_node_features}")
    print(f"Classes  : {stats['num_classes']}")

    # --------------------------------------------------------
    # Standard geom-GCN split
    # --------------------------------------------------------

    train_mask = data.train_mask[:, SPLIT_IDX]
    val_mask = data.val_mask[:, SPLIT_IDX]
    test_mask = data.test_mask[:, SPLIT_IDX]

    # --------------------------------------------------------
    # Create M2M-GNN
    # --------------------------------------------------------

    set_seed(SEED)

    model = M2MGNN(
        in_feat=data.num_node_features,
        hid_feat=HIDDEN,
        out_feat=stats["num_classes"],
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        c=C,
        beta=BETA,
        dropout2=DROPOUT2,
        temperature=TEMPERATURE,
        remove_self_loof=True
    ).to(device)

    print("\nM2M-GNN configuration:")
    print(f"hidden      = {HIDDEN}")
    print(f"layers      = {NUM_LAYERS}")
    print(f"dropout     = {DROPOUT}")
    print(f"dropout2    = {DROPOUT2}")
    print(f"c           = {C}")
    print(f"beta        = {BETA}")
    print(f"temperature = {TEMPERATURE}")
    print(f"lr          = {LR}")
    print(f"epochs      = {MAX_EPOCHS}")

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    val_acc, test_acc, best_epoch = train_one_model(
        model,
        data,
        train_mask,
        val_mask,
        test_mask
    )

    print("\nResult:")
    print(f"Best validation accuracy : {val_acc:.4f}")
    print(f"Test accuracy             : {test_acc:.4f}")
    print(f"Best epoch                : {best_epoch}")

    return {
        "dataset": dataset_name,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "best_epoch": best_epoch
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("M2M-GNN CHUNK 2 BASELINE EXPERIMENT")
    print("=" * 60)

    results = []

    # --------------------------------------------------------
    # FIRST RUN: Cora and Texas
    # --------------------------------------------------------

    for dataset_name in DATASETS:

        result = run_dataset(dataset_name)

        results.append(result)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("M2M-GNN SUMMARY")
    print("=" * 60)

    print(
        f"{'Dataset':<10}"
        f"{'Val Acc':>12}"
        f"{'Test Acc':>12}"
        f"{'Best Epoch':>14}"
    )

    for result in results:

        print(
            f"{result['dataset']:<10}"
            f"{result['val_accuracy']:>12.4f}"
            f"{result['test_accuracy']:>12.4f}"
            f"{result['best_epoch']:>14}"
        )

    print("=" * 60)
