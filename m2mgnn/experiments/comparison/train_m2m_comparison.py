import os
import sys
import torch
import torch.nn.functional as F

# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import M2MGNN
from experiments.comparison.smp.data_utils import load_dataset, dataset_stats


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 0
HIDDEN = 32
NUM_LAYERS = 2
DROPOUT = 0.8
DROPOUT2 = 0.3
C = 5
BETA = 0.5
LR = 0.005
WD1 = 0.01
WD2 = 0.01
TEMPERATURE = 1.0

EPOCHS = 1000
PATIENCE = 200

SPLIT_IDX = 0


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# Get masks
# ------------------------------------------------------------
def get_masks(data, split_idx):
    train_mask = data.train_mask[:, split_idx].bool()
    val_mask = data.val_mask[:, split_idx].bool()
    test_mask = data.test_mask[:, split_idx].bool()

    return train_mask, val_mask, test_mask


# ------------------------------------------------------------
# Train M2M-GNN
# ------------------------------------------------------------
def train_m2m(data):

    set_seed(SEED)

    train_mask, val_mask, test_mask = get_masks(
        data,
        SPLIT_IDX
    )

    model = M2MGNN(
        in_feat=data.num_node_features,
        hid_feat=HIDDEN,
        out_feat=int(data.y.max().item()) + 1,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        c=C,
        beta=BETA,
        dropout2=DROPOUT2,
        temperature=TEMPERATURE,
        remove_self_loof=True
    ).to(DEVICE)

    data = data.to(DEVICE)

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

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):

        model.train()

        optimizer.zero_grad()

        out = model(
            data.x,
            data.edge_index
        )

        loss = (
            F.cross_entropy(
                out[train_mask],
                data.y[train_mask]
            )
            + 0.0 * model.reg
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

            pred = out.argmax(dim=1)

            val_acc = (
                pred[val_mask] == data.y[val_mask]
            ).float().mean().item()

            test_acc = (
                pred[test_mask] == data.y[test_mask]
            ).float().mean().item()

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------
        if val_acc >= best_val_acc:

            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch

            patience_counter = 0

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:
                break

    return best_val_acc, best_test_acc, best_epoch


# ------------------------------------------------------------
# Run dataset
# ------------------------------------------------------------
def run_dataset(name):

    data, dataset = load_dataset(name)

    stats = dataset_stats(
        data,
        name
    )

    val_acc, test_acc, epoch = train_m2m(data)

    return {
        "dataset": name,
        "edge_homophily": stats["edge_homophily_no_selfloop"],
        "M2MGNN_val_acc": round(val_acc, 4),
        "M2MGNN_test_acc": round(test_acc, 4),
        "M2MGNN_epoch": epoch
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":

    print("=" * 65)
    print("M2M-GNN COMPARISON EXPERIMENT")
    print("=" * 65)

    print(f"Device       : {DEVICE}")
    print(f"Seed         : {SEED}")
    print(f"Hidden       : {HIDDEN}")
    print(f"Layers       : {NUM_LAYERS}")
    print(f"Dropout      : {DROPOUT}")
    print(f"Dropout2     : {DROPOUT2}")
    print(f"c            : {C}")
    print(f"beta         : {BETA}")
    print(f"Learning rate: {LR}")
    print(f"Split        : {SPLIT_IDX}")

    print("\n" + "-" * 65)

    results = []

    for name in ["Texas", "Cora"]:

        print(f"\nRunning M2M-GNN on {name}...")

        result = run_dataset(name)

        results.append(result)

        print(result)

    print("\n" + "=" * 65)
    print("FINAL M2M-GNN RESULTS")
    print("=" * 65)

    print(
        f"{'Dataset':10s} "
        f"{'EdgeHom':>10s} "
        f"{'Val Acc':>12s} "
        f"{'Test Acc':>12s} "
        f"{'Epoch':>10s}"
    )

    for r in results:

        print(
            f"{r['dataset']:10s} "
            f"{r['edge_homophily']:10.4f} "
            f"{r['M2MGNN_val_acc']:12.4f} "
            f"{r['M2MGNN_test_acc']:12.4f} "
            f"{r['M2MGNN_epoch']:10d}"
        )

    print("=" * 65)