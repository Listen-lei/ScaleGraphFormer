import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter, scatter_add, scatter_max
from torch_geometric.utils import degree, maybe_num_nodes
import opt_einsum as oe
from typing import Optional, List
import warnings

# --------------------------- DegreeScaler ---------------------------
class ScaleGraphFormerDegreeScaler(nn.Module):
    """
    Apply degree scaling to attention-aggregated node features.
    Inputs:
        wV: (N, H, D) - per-node, per-head aggregated features
        log_deg: (N,) or (N,1,1) - log(degree+1)
    Outputs:
        out: (N, H, D) - scaled and projected node features
    Supports scalers: 'identity', 'amplification', 'attenuation', 'linear'
    """
    def __init__(self, out_dim, scalers: Optional[List[str]] = None):
        super().__init__()
        if scalers is None:
            scalers = ['identity', 'amplification', 'attenuation', 'linear']
        self.scalers = scalers
        self.out_dim = out_dim
        self.proj = nn.Linear(self.out_dim * len(self.scalers), self.out_dim)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, wV, log_deg):
        N, H, D = wV.size()
        ld = log_deg.view(N, 1, 1) if log_deg.dim() == 1 else log_deg.view(N, 1, 1)
        scaled = []
        for s in self.scalers:
            if s == 'identity':
                scaled.append(wV)
            elif s == 'amplification':
                scaled.append(wV * ld)
            elif s == 'attenuation':
                scaled.append(wV / (1.0 + ld))
            elif s == 'linear':
                scaled.append(wV * (1.0 + 0.5 * ld))
            else:
                scaled.append(wV)
        cat = torch.cat(scaled, dim=-1).view(N * H, D * len(self.scalers))
        out = self.proj(cat).view(N, H, D)
        return out

# --------------------------- Sparse Softmax ---------------------------
def pyg_sparse_softmax(src, index, num_nodes=None):
    """Sparse softmax along first dimension grouped by index."""
    num_nodes = maybe_num_nodes(index, num_nodes)
    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)
    return out

# --------------------------- Multi-Head Attention Layer ---------------------------
class ScaleGraphFormerAttentionLayer(nn.Module):
    """
    Multi-Head Attention with optional degree scaling (PNA-inspired) and edge enhancement.
    """
    def __init__(self, in_dim, out_dim, num_heads, use_bias=True,
                 dropout=0., clamp=5., edge_enhance=True,
                 act=None, use_pna_scaler=True, pna_scalers: Optional[List[str]] = None):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.clamp = np.abs(clamp) if clamp is not None else None
        self.edge_enhance = edge_enhance

        # Linear projections
        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(in_dim, out_dim * num_heads * 2, bias=True)
        for lin in [self.Q, self.K, self.V, self.E]:
            nn.init.xavier_normal_(lin.weight)

        self.Aw = nn.Parameter(torch.zeros(out_dim, num_heads, 1))
        nn.init.xavier_normal_(self.Aw)

        self.act = nn.Identity() if act is None else act()

        if edge_enhance:
            self.VeRow = nn.Parameter(torch.zeros(out_dim, num_heads, out_dim))
            nn.init.xavier_normal_(self.VeRow)

        # Degree scaler
        self.use_pna_scaler = use_pna_scaler
        if self.use_pna_scaler:
            if pna_scalers is None:
                pna_scalers = ['identity', 'amplification', 'attenuation', 'linear']
            self.pna_scaler = ScaleGraphFormerDegreeScaler(out_dim, scalers=pna_scalers)

    def propagate_attention(self, batch):
        src = batch.K_h[batch.edge_index[0]]
        dest = batch.Q_h[batch.edge_index[1]]
        score = src + dest

        if batch.get("E", None) is not None:
            batch.E = batch.E.view(-1, self.num_heads, self.out_dim * 2)
            E_w, E_b = batch.E[:, :, :self.out_dim], batch.E[:, :, self.out_dim:]
            score = score * E_w
            score = torch.sqrt(torch.relu(score)) - torch.sqrt(torch.relu(-score))
            score = score + E_b

        score = self.act(score)
        e_t = score
        if batch.get("E", None) is not None:
            batch.wE = score.flatten(1)

        score = oe.contract("ehd, dhc->ehc", score, self.Aw, backend="torch")
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)

        score = pyg_sparse_softmax(score, batch.edge_index[1])
        score = self.dropout(score)
        batch.attn = score

        msg = batch.V_h[batch.edge_index[0]] * score
        batch.wV = torch.zeros_like(batch.V_h)
        scatter(msg, batch.edge_index[1], dim=0, out=batch.wV, reduce='add')

        if self.edge_enhance and batch.E is not None:
            rowV = scatter(e_t * score, batch.edge_index[1], dim=0, dim_size=batch.wV.size(0), reduce="add")
            rowV = oe.contract("nhd, dhc->nhc", rowV, self.VeRow, backend="torch")
            batch.wV = batch.wV + rowV

        # Apply degree scaler
        if self.use_pna_scaler:
            deg = degree(batch.edge_index[1], num_nodes=batch.x.size(0), dtype=batch.x.dtype).to(batch.x.device)
            log_deg = torch.log1p(deg)
            batch.wV = self.pna_scaler(batch.wV, log_deg)

    def forward(self, batch):
        batch.Q_h = self.Q(batch.x).view(-1, self.num_heads, self.out_dim)
        batch.K_h = self.K(batch.x).view(-1, self.num_heads, self.out_dim)
        batch.V_h = self.V(batch.x).view(-1, self.num_heads, self.out_dim)
        if batch.get("edge_attr", None) is not None:
            batch.E = self.E(batch.edge_attr)
        else:
            batch.E = None

        self.propagate_attention(batch)
        h_out = batch.wV
        e_out = batch.get('wE', None)
        return h_out, e_out

# --------------------------- Transformer Layer ---------------------------
class ScaleGraphFormerLayer(nn.Module):
    """
    ScaleGraphFormer Transformer layer with residual, LayerNorm, activation and degree-scaled attention.
    """
    def __init__(self, in_dim, out_dim, num_heads,
                 dropout=0.0, attn_dropout=0.0,
                 use_pna_scaler=True,
                 pna_scalers: Optional[List[str]] = None):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.act = nn.ReLU()

        self.attention = ScaleGraphFormerAttentionLayer(
            in_dim=in_dim,
            out_dim=out_dim // num_heads,
            num_heads=num_heads,
            dropout=attn_dropout,
            use_pna_scaler=use_pna_scaler,
            pna_scalers=pna_scalers
        )

        self.O_h = nn.Linear(out_dim // num_heads * num_heads, out_dim)

    def forward(self, batch):
        h_in = batch.x
        h_attn, _ = self.attention(batch)
        h_attn = h_attn.view(batch.num_nodes, -1)
        h_attn = self.O_h(h_attn)
        h_attn = self.dropout_layer(h_attn)

        h = h_in + h_attn
        h = self.layer_norm(h)
        h = self.act(h)
        batch.x = h
        return batch
