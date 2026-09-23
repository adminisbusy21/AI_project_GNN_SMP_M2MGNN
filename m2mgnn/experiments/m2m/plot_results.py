import csv
import matplotlib.pyplot as plt


RESULTS_FILE = "experiments/m2m/results.csv"


beta_values = []
beta_means = []
beta_stds = []

c_values = []
c_means = []
c_stds = []


with open(RESULTS_FILE, "r", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:

        if row["experiment"] == "beta_sensitivity":
            beta_values.append(float(row["value"]))
            beta_means.append(float(row["mean_accuracy"]))
            beta_stds.append(float(row["std_accuracy"]))

        elif row["experiment"] == "c_sensitivity":
            c_values.append(int(row["value"]))
            c_means.append(float(row["mean_accuracy"]))
            c_stds.append(float(row["std_accuracy"]))


# ------------------------------------------------------------
# Beta sensitivity
# ------------------------------------------------------------

plt.figure()

plt.errorbar(
    beta_values,
    beta_means,
    yerr=beta_stds,
    marker="o",
    capsize=5
)

plt.xlabel("Beta")
plt.ylabel("Mean Test Accuracy")
plt.title("M2M-GNN Beta Sensitivity")
plt.grid(True)

plt.savefig(
    "experiments/m2m/beta_sensitivity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# C sensitivity
# ------------------------------------------------------------

plt.figure()

plt.errorbar(
    c_values,
    c_means,
    yerr=c_stds,
    marker="o",
    capsize=5
)

plt.xlabel("Number of Message Types (c)")
plt.ylabel("Mean Test Accuracy")
plt.title("M2M-GNN C Sensitivity")
plt.grid(True)

plt.savefig(
    "experiments/m2m/c_sensitivity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()