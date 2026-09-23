"""
data_utils.py
-------------
Chunk 1: dataset loading + statistics.

STATUS: newly written (the official m2mgnn repo ships no dataset code at all).

We load datasets exactly the way the paper says it does (Sec 4.1 / Appendix C):
"we employ the setting provided by Pytorch Geometric (PyG) (Platonov et al., 2022)"
which in turn traces back to the geom-gcn 48/32/20 splits (Pei et al., 2020).

PyG's WebKB / Planetoid loaders download these splits directly from
raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn -- the same source the
heterophily literature (including this paper) standardizes on.
"""

import torch
from torch_geometric.datasets import WebKB, Planetoid


def load_dataset(name: str, root: str = "./data"):
    """
    Load a dataset by name and return its PyG Data object plus the dataset name.

    Supported in Chunk 1: 'Texas' (heterophilic), 'Cora' (homophilic).
    Both come with 10 precomputed geom-gcn splits: data.train_mask has shape
    [num_nodes, 10] -- one column per split. We will use split index 0 for
    Chunk 1's single-run experiment (later chunks can average over all 10,
    as the paper does, for the final comparison table).
    """
    name = name.lower()
    if name == "texas":
        ds = WebKB(root=f"{root}/Texas", name="Texas")
    elif name == "cora":
        ds = Planetoid(root=f"{root}/Cora", name="Cora", split="geom-gcn")
    else:
        raise ValueError(f"Dataset '{name}' not wired up in Chunk 1. "
                          f"Only 'Texas' and 'Cora' are loaded for now.")
    data = ds[0]
    return data, ds


def edge_homophily(data, exclude_self_loops: bool = True) -> float:
    """
    Compute the edge homophily ratio:
        h = |{(i,j) in E : y_i == y_j}| / |E|

    This is exactly Definition used for Table 1 ("Edge Hom.") in the paper:
    "the edge homophily ratio (Edge Hom.) is defined as
     |{(vi, vj) in E} : yi = yj| / |E|."

    NOTE on a discrepancy we found and verified (do not silently trust PyG counts):
    PyG's WebKB/Planetoid loaders store the graph as a DIRECTED edge_index and
    (for WebKB) include self-loops. Comparing against Table 1:
      - Cora: our edge count is exactly 2x the paper's (10556 vs 5278) because
        every undirected edge is stored as two directed entries.
      - Texas: our raw edge_index has 325 entries, including 16 self-loops;
        the paper reports 295 (undirected, presumably self-loop-free) edges.
    Self-loops trivially inflate the homophily ratio (a node always shares its
    own label), so we exclude them here by default to get a statistic that is
    comparable in spirit to Table 1, even if exact undirected deduplication
    is not reconstructed.
    """
    row, col = data.edge_index
    if exclude_self_loops:
        mask = row != col
        row, col = row[mask], col[mask]
    same_label = (data.y[row] == data.y[col]).float()
    return same_label.mean().item()


def dataset_stats(data, name: str) -> dict:
    """Collect the same statistics the paper reports in Table 1."""
    row, col = data.edge_index
    self_loops = int((row == col).sum().item())
    return {
        "name": name,
        "num_nodes": data.num_nodes,
        "num_edges_directed_raw": data.edge_index.size(1),
        "self_loops": self_loops,
        "num_features": data.num_node_features,
        "num_classes": int(data.y.max().item()) + 1,
        "edge_homophily_no_selfloop": round(edge_homophily(data, exclude_self_loops=True), 4),
    }


if __name__ == "__main__":
    rows = []
    for name in ["Texas", "Cora"]:
        data, ds = load_dataset(name)
        stats = dataset_stats(data, name)
        rows.append(stats)
        print(stats)