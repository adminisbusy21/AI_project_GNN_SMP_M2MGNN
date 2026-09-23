"""
beta_sensitivity.py
-------------------
M2M-GNN experiment: effect of the ego-feature strength (beta).

The official M2M-GNN implementation in model.py is kept unchanged.

Only beta is varied.
All other parameters are fixed.

For each beta value, the model is trained independently on all
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
# Make project root importable
# ================================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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

# c is fixed because we are studying beta now.
C = 5

LR = 0.005
WD1 = 0.01
WD2 = 0.01

TEMPERATURE = 1.0
LAMBDA = 0.0

PATIENCE = 200
MAX_EPOCHS = 1000

# Only beta changes.
BETA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]


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
# Train and evaluate one beta value
# ================================================================

def train_one_beta(dataset, data, beta):

    split_accuracies = []
    split_epochs = []

    # ------------------------------------------------------------
    # Run this beta value on all 10 standardized geom-gcn splits.
    # ------------------------------------------------------------

    for split in range(10):

        train_mask = data.train_mask[:, split]
        val_mask = data.val_mask[:, split]
        test_mask = data.test_mask[:, split]

        # --------------------------------------------------------
        # Fresh model for every split
        # --------------------------------------------------------

        set_seed(SEED)

        model = M2MGNN(
            in_feat=dataset.num_features,
            hid_feat=HIDDEN,
            out_feat=dataset.num_classes,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            c=C,
            beta=beta,
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

        best_val_loss = float("inf")
        best_test_acc = 0.0

        patience_counter = 0
        best_state = None

        # --------------------------------------------------------
        # Training
        # --------------------------------------------------------

        for epoch in range(MAX_EPOCHS):

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
            # Validation
            # ----------------------------------------------------

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

            # ----------------------------------------------------
            # Early stopping
            # ----------------------------------------------------

            if val_loss < best_val_loss:

                best_val_loss = val_loss
                best_test_acc = test_acc

                patience_counter = 0

                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

            else:

                patience_counter += 1

                if patience_counter >= PATIENCE:
                    break

        # --------------------------------------------------------
        # Restore best model
        # --------------------------------------------------------

        if best_state is not None:
            model.load_state_dict(best_state)

        split_accuracies.append(best_test_acc)
        split_epochs.append(epoch + 1)

        print(
            f"    split {split}: "
            f"accuracy = {best_test_acc:.4f}, "
            f"epochs = {epoch + 1}"
        )

    # ------------------------------------------------------------
    # Statistics across the 10 splits
    # ------------------------------------------------------------

    mean_accuracy = float(np.mean(split_accuracies))
    std_accuracy = float(np.std(split_accuracies))
    mean_epochs = float(np.mean(split_epochs))

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
    print("M2M-GNN BETA SENSITIVITY EXPERIMENT")
    print("=" * 60)

    print(f"Dataset : {DATASET_NAME}")
    print(f"Device  : {DEVICE}")
    print(f"Seed    : {SEED}")
    print(f"c       : {C}")
    print(f"beta values: {BETA_VALUES}")

    print("\nFixed parameters:")

    print(f"hidden      = {HIDDEN}")
    print(f"num_layers  = {NUM_LAYERS}")
    print(f"dropout     = {DROPOUT}")
    print(f"dropout2    = {DROPOUT2}")
    print(f"c           = {C}")
    print(f"lr          = {LR}")
    print(f"temperature = {TEMPERATURE}")

    dataset, data = load_data()

    results = []

    print("\n" + "-" * 60)

    # ------------------------------------------------------------
    # Run every beta value
    # ------------------------------------------------------------

    for beta in BETA_VALUES:

        print(f"\nRunning beta = {beta} ...")

        (
            mean_accuracy,
            std_accuracy,
            mean_epochs,
            split_accuracies
        ) = train_one_beta(
            dataset,
            data,
            beta
        )

        results.append(
            {
                "beta": beta,
                "mean_accuracy": mean_accuracy,
                "std_accuracy": std_accuracy,
                "mean_epochs": mean_epochs,
                "split_accuracies": split_accuracies
            }
        )

        print(
            f"beta = {beta:.2f} | "
            f"mean accuracy = {mean_accuracy:.4f} | "
            f"std = {std_accuracy:.4f} | "
            f"mean epochs = {mean_epochs:.1f}"
        )

    # ============================================================
    # Final results
    # ============================================================

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(
        f"{'Beta':>8} "
        f"{'Mean Accuracy':>18} "
        f"{'Std':>10} "
        f"{'Mean Epochs':>12}"
    )

    for result in results:

        print(
            f"{result['beta']:>8.2f} "
            f"{result['mean_accuracy']:>18.4f} "
            f"{result['std_accuracy']:>10.4f} "
            f"{result['mean_epochs']:>12.1f}"
        )

    print("=" * 60)
    import csv

RESULTS_FILE = os.path.join(
    PROJECT_ROOT,
    "experiments",
    "m2m",
    "results.csv"
)

file_exists = os.path.exists(RESULTS_FILE)

with open(RESULTS_FILE, "a", newline="") as f:

    writer = csv.writer(f)

    if not file_exists or os.path.getsize(RESULTS_FILE) == 0:
        writer.writerow([
            "experiment",
            "parameter",
            "value",
            "mean_accuracy",
            "std_accuracy",
            "mean_epochs"
        ])

    for result in results:
        writer.writerow([
            "beta_sensitivity",
            "beta",
            result["beta"],
            result["mean_accuracy"],
            result["std_accuracy"],
            result["mean_epochs"]
        ])

print(f"\nResults saved to: {RESULTS_FILE}")