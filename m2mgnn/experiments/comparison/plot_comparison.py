import csv
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Load CSV without pandas
# ------------------------------------------------------------
file_path = r".\experiments\comparison\comparison_results.csv"

rows = []

with open(file_path, "r", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        rows.append(row)


# ------------------------------------------------------------
# Separate datasets
# ------------------------------------------------------------
datasets = ["Texas", "Cora"]
models = ["MLP", "GCN", "M2M-GNN"]

accuracy = {}

for dataset in datasets:

    accuracy[dataset] = {}

    for model in models:

        for row in rows:

            if (
                row["dataset"] == dataset
                and row["model"] == model
            ):
                accuracy[dataset][model] = float(
                    row["test_accuracy"]
                )


# ------------------------------------------------------------
# Create plot
# ------------------------------------------------------------
x = [0, 1]

width = 0.25

mlp_values = [
    accuracy["Texas"]["MLP"],
    accuracy["Cora"]["MLP"]
]

gcn_values = [
    accuracy["Texas"]["GCN"],
    accuracy["Cora"]["GCN"]
]

m2m_values = [
    accuracy["Texas"]["M2M-GNN"],
    accuracy["Cora"]["M2M-GNN"]
]


plt.figure(figsize=(9, 6))

plt.bar(
    [i - width for i in x],
    mlp_values,
    width=width,
    label="MLP"
)

plt.bar(
    x,
    gcn_values,
    width=width,
    label="GCN"
)

plt.bar(
    [i + width for i in x],
    m2m_values,
    width=width,
    label="M2M-GNN"
)


# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------
plt.xlabel("Dataset")
plt.ylabel("Test Accuracy")

plt.title(
    "MLP vs GCN vs M2M-GNN"
)

plt.xticks(
    x,
    datasets
)

plt.ylim(0, 1)

plt.legend()

plt.tight_layout()


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
output_file = (
    r".\experiments\comparison\model_comparison_accuracy.png"
)

plt.savefig(
    output_file,
    dpi=300
)

print("\nPlot saved successfully:")
print(output_file)

print("\n=== RESULTS ===")

for dataset in datasets:

    print(f"\n{dataset}")

    for model in models:

        print(
            f"{model}: "
            f"{accuracy[dataset][model]:.4f}"
        )

plt.show()