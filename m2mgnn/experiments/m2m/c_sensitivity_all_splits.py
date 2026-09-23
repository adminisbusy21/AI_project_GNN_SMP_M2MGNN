"""
c_sensitivity.py
----------------
M2M-GNN experiment: effect of the number of message types (c).

The official M2M-GNN implementation in model.py is kept unchanged.
This experiment only varies the parameter c.

All other training parameters are fixed.

For each c value, the model is trained independently on all
10 standardized geom-gcn splits of the Texas dataset.

The experiment reports:
    - accuracy on each split
    - mean accuracy
    - standard deviation
    - mean number of epochs
"""

import os
import sys

import torch
import numpy as np
from torch import nn
from torch_geometric.datasets import WebKB


# ================================================================
# Project root
# ================================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the ORIGINAL M2M-GNN implementation.
from model import M2MGNN


# ================================================================
# Fixed experimental configuration
# ================================================================

DATASET_NAME = "texas"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 0

HIDDEN = 32
NUM_LAYERS = 2

DROPOUT = 0.8
DROPOUT2 = 0.3

BETA = 0.5

LR = 0.005
WD1 = 0.01
WD2 = 0.01

TEMPERATURE = 1.0
LAMBDA = 0.0

PATIENCE = 200
MAX_EPOCHS = 1000

# Only this parameter changes.
C_VALUES = [1, 2, 3, 5, 8]


# ================================================================
# Reproducibility
# ================================================================

def set_seed(seed):

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ================================================================
# Load dataset
# ================================================================

def load_data():

    dataset = WebKB(
        root=os.path.join(PROJECT_ROOT, "data"),
        name=DATASET_NAME
    )

    data = dataset[0].to(DEVICE)

    return dataset, data


# ================================================================
# Train one model for ONE split
# ================================================================

def train_one_split(dataset, data, c, split):

    # ------------------------------------------------------------
    # Fresh initialization for every split.
    # ------------------------------------------------------------

    set_seed(SEED)

    model = M2MGNN(
        in_feat=dataset.num_features,
        hid_feat=HIDDEN,
        out_feat=dataset.num_classes,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        c=c,
        beta=BETA,
        dropout2=DROPOUT2,
        temperature=TEMPERATURE,
        remove_self_loof=True
    ).to(DEVICE)

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

    # ------------------------------------------------------------
    # Select this split.
    # ------------------------------------------------------------

    train_mask = data.train_mask[:, split]
    val_mask = data.val_mask[:, split]
    test_mask = data.test_mask[:, split]

    # ------------------------------------------------------------
    # Early stopping variables.
    # ------------------------------------------------------------

    best_val_loss = float("inf")
    best_test_acc = 0.0

    patience_counter = 0

    for epoch in range(MAX_EPOCHS):

        # ========================================================
        # Training
        # ========================================================

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

        # ========================================================
        # Validation
        # ========================================================

        model.eval()

        with torch.no_grad():

            out = model(
                data.x,
                data.edge_index
            )

            val_loss = criterion(
                out[val_mask],
                data.y[val_mask]
            ).item()

            pred = out.argmax(dim=1)

            test_acc = (
                pred[test_mask] == data.y[test_mask]
            ).float().mean().item()

        # ========================================================
        # Early stopping
        # ========================================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_test_acc = test_acc

            patience_counter = 0

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:
                break

    return best_test_acc, epoch + 1


# ================================================================
# Run all 10 splits for one c value
# ================================================================

def train_one_c(dataset, data, c):

    split_accuracies = []
    split_epochs = []

    print(f"\nRunning c = {c} ...")

    for split in range(10):

        accuracy, epochs = train_one_split(
            dataset,
            data,
            c,
            split
        )

        split_accuracies.append(accuracy)
        split_epochs.append(epochs)

        print(
            f"    split {split}: "
            f"accuracy = {accuracy:.4f}, "
            f"epochs = {epochs}"
        )

    # ------------------------------------------------------------
    # Statistics across the 10 splits.
    # ------------------------------------------------------------

    mean_accuracy = float(
        np.mean(split_accuracies)
    )

    std_accuracy = float(
        np.std(split_accuracies)
    )

    mean_epochs = float(
        np.mean(split_epochs)
    )

    return (
        mean_accuracy,
        std_accuracy,
        mean_epochs,
        split_accuracies
    )


# ================================================================
# Main experiment
# ================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("M2M-GNN C SENSITIVITY EXPERIMENT")
    print("=" * 60)

    print(f"Dataset : {DATASET_NAME}")
    print(f"Device  : {DEVICE}")
    print(f"Seed    : {SEED}")
    print(f"c values: {C_VALUES}")

    print("\nFixed parameters:")

    print(f"hidden      = {HIDDEN}")
    print(f"num_layers  = {NUM_LAYERS}")
    print(f"dropout     = {DROPOUT}")
    print(f"dropout2    = {DROPOUT2}")
    print(f"beta        = {BETA}")
    print(f"lr          = {LR}")
    print(f"wd1         = {WD1}")
    print(f"wd2         = {WD2}")
    print(f"temperature = {TEMPERATURE}")
    print(f"patience    = {PATIENCE}")
    print(f"max_epochs  = {MAX_EPOCHS}")

    dataset, data = load_data()

    results = []

    print("\n" + "-" * 60)

    # ============================================================
    # Run every c value
    # ============================================================

    for c in C_VALUES:

        (
            mean_accuracy,
            std_accuracy,
            mean_epochs,
            split_accuracies
        ) = train_one_c(
            dataset,
            data,
            c
        )

        results.append(
            {
                "c": c,
                "mean_accuracy": mean_accuracy,
                "std_accuracy": std_accuracy,
                "mean_epochs": mean_epochs,
                "split_accuracies": split_accuracies
            }
        )

        print(
            f"\nSummary for c = {c}:"
        )

        print(
            f"    mean accuracy = {mean_accuracy:.4f}"
        )

        print(
            f"    std           = {std_accuracy:.4f}"
        )

        print(
            f"    mean epochs   = {mean_epochs:.1f}"
        )

    # ============================================================
    # Final results
    # ============================================================

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(
        f"{'c':>5} "
        f"{'Mean Accuracy':>18} "
        f"{'Std':>10} "
        f"{'Mean Epochs':>12}"
    )

    for result in results:

        print(
            f"{result['c']:>5} "
            f"{result['mean_accuracy']:>18.4f} "
            f"{result['std_accuracy']:>10.4f} "
            f"{result['mean_epochs']:>12.1f}"
        )

    print("=" * 60)

    # ============================================================
    # Individual split results
    # ============================================================

    print("\nSPLIT-WISE ACCURACIES")
    print("=" * 60)

    for result in results:

        print(
            f"\nc = {result['c']}"
        )

        for split, accuracy in enumerate(
            result["split_accuracies"]
        ):

            print(
                f"    split {split}: {accuracy:.4f}"
            )

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)