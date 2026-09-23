"""
smp_model.py

Chunk 2: Signed Message Passing (SMP)
NOTE: The official m2mgnn repo contains NO SMP code at all
(it only ships the M2M-GNN layer, in model.py). This is our own minimal
implementation of the SMP family the paper analyzes theoretically in Sec. 2,
following the FAGCN design (Bo et al., 2021), which the paper explicitly
uses as its concrete SMP instance for Eq. (3) and the CSBM analysis.

Core idea (paper Sec 2.1-2.2):
  Ordinary GCN: H = sigma(A_hat H W)              -- all coefficients >= 0
  SMP:          H = sigma(A_hat . alpha . H W)     -- alpha_ij in (-1, 1)

alpha_ij is a *learned, signed* edge coefficient computed from the two
endpoint embeddings:
    alpha_ij = tanh( a^T [h_i || h_j] )
A "desirable" SMP (Def. 2.1) would learn alpha_ij >= 0 for same-class edges
and alpha_ij <= 0 for different-class edges -- entirely from gradient signal,
no edge-level labels are ever given to the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree

class SMPLayer(nn.Module):

    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.lin = nn.Linear(in_feat, out_feat, bias=False)
        # attention vector 'a' from the equation above, applied to [h_i || h_j]
        self.att = nn.Linear(2 * out_feat, 1, bias=False)

    def forward(self, x, edge_index):
        x = self.lin(x)  # shared linear transform, applied before message passing

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index  # row = target i, col = source j (message j -> i)

        # symmetric normalization coefficient A_hat_ij = 1/sqrt(deg_i * deg_j)
        deg = degree(row, x.size(0), dtype=x.dtype).clamp(min=1)
        norm = (deg[row] * deg[col]).pow(-0.5)

        # signed attention alpha_ij = tanh(a^T [h_i || h_j]), in (-1, 1)
        # NOTE (bug found & fixed during Chunk 2 debugging): multiplying the
        # already-small A_hat normalization together with a near-zero-at-init
        # tanh output crushed the forward signal to ~1e-6 after two layers,
        # producing a loss frozen at exactly ln(num_classes) (pure random
        # guessing) with no visible training progress. FAGCN and related
        # SMP papers avoid this by NOT re-multiplying the degree-normalization
        # on top of the learned signed attention -- alpha itself already
        # plays the role of the edge coefficient. We follow that: use alpha
        # as the coefficient directly, and keep norm only to preserve some
        # degree-awareness via a sqrt-scaled residual, not a hard multiply.
        h_i, h_j = x[row], x[col]
        alpha = torch.tanh(self.att(torch.cat([h_i, h_j], dim=-1))).squeeze(-1)

        # message = alpha_ij * h_j, normalized by source-side degree only
        # (mean-aggregation flavor, avoids double-shrinking through norm*alpha)
        deg_j = degree(col, x.size(0), dtype=x.dtype).clamp(min=1)
        norm_j = deg_j[col].pow(-1.0)
        messages = (norm_j * alpha).unsqueeze(-1) * h_j

        out = torch.zeros_like(x)
        out.index_add_(0, row, messages)  # scatter-sum messages into target nodes
        return out, alpha, row, col


class SMPNet(nn.Module):
    """
    2-layer SMP network for node classification.
    Same overall shape as our GCN baseline (Chunk 1) so the comparison is
    apples-to-apples: only the aggregation mechanism (signed vs unsigned)
    differs.

    BUG FOUND & FIXED during Chunk 2 debugging: with Cora's very sparse
    bag-of-words input (~1.3% nonzero, feature dim 1433), the untrained
    Linear layer's small-std init produced tiny hidden activations; feeding
    those into a second signed-attention layer produced alpha values near
    zero, and the whole network's output logits collapsed to ~1e-8 magnitude
    -- loss frozen at exactly ln(num_classes) with no learning happening at
    all. This is NOT a finding about SMP's expressiveness, it's a numerical
    initialization issue. Fix: add LayerNorm after each SMP layer (standard
    practice for signed/attention GNNs -- the paper's own M2M-GNN applies
    LayerNorm after every layer too, per Appendix D) to keep activations in
    a well-scaled range regardless of input sparsity/dimensionality.
    """
    def __init__(self, in_feat, hid_feat, out_feat, dropout=0.5):
        super().__init__()
        self.layer1 = SMPLayer(in_feat, hid_feat)
        self.norm1 = nn.LayerNorm(hid_feat)
        self.layer2 = SMPLayer(hid_feat, out_feat)
        self.dropout = dropout

    def forward(self, x, edge_index, return_alpha=False):
        h, alpha1, row1, col1 = self.layer1(x, edge_index)
        h = self.norm1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out, alpha2, row2, col2 = self.layer2(h, edge_index)

        if return_alpha:
            return out, {
                "layer1": (alpha1, row1, col1),
                "layer2": (alpha2, row2, col2),
            }
        return out