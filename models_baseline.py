"""
models_baseline.py
-------------------
Chunk 1: minimal MLP and GCN baselines.

STATUS: newly written. The official m2mgnn repo does not implement any
baseline (only the M2M-GNN layer in model.py). We use PyG's built-in
GCNConv, which implements exactly Eq. (1) of the paper:
    H^(k) = sigma( A_hat H^(k-1) W )
with A_hat = D^-0.5 A D^-0.5 (symmetric normalization, self-loops added).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MLP(nn.Module):
    """
    Graph-agnostic baseline: looks only at each node's own features.
    No edge_index is used anywhere in forward() -- this is the point.
    """
    def __init__(self, in_feat, hid_feat, out_feat, dropout=0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_feat, hid_feat)
        self.lin2 = nn.Linear(hid_feat, out_feat)
        self.dropout = dropout

    def forward(self, x, edge_index=None):  # edge_index accepted but ignored
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x


class GCN(nn.Module):
    """
    Standard 2-layer GCN (Kipf & Welling, 2016), i.e. Eq. (1) in the paper.
    Directly reuses torch_geometric.nn.GCNConv rather than reimplementing
    sparse normalization by hand -- this is standard practice and lets us
    focus on the *behavioral comparison*, not on re-deriving GCNConv's math.
    """
    def __init__(self, in_feat, hid_feat, out_feat, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_feat, hid_feat)
        self.conv2 = GCNConv(hid_feat, out_feat)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x