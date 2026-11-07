"""Teacher and student architectures.

MLP, SAGE, GCN, GAT and the Model wrapper are derived from GLNN (Copyright Snap
Inc. 2021, MIT License, see LICENSE).
"""
import dgl
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.nn import GATConv, GraphConv, SAGEConv
from dgl.ops import edge_softmax


class MLP(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        norm_type="none",
    ):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h_list, h


class SAGE(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        activation,
        norm_type="none",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.norm_type = norm_type
        self.activation = activation
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(SAGEConv(input_dim, output_dim, "gcn"))
        else:
            self.layers.append(SAGEConv(input_dim, hidden_dim, "gcn"))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(SAGEConv(hidden_dim, hidden_dim, "gcn"))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(SAGEConv(hidden_dim, output_dim, "gcn"))

    def forward(self, blocks, feats):
        h = feats
        h_list = []
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[: block.num_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h = layer(block, (h, h_dst))
            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = self.activation(h)
                h = self.dropout(h)
        return h_list, h

    def forward_full(self, g, feats):
        """Forward pass over a whole graph, with full neighborhoods."""
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = self.activation(h)
                h = self.dropout(h)
        return h_list, h


class GCN(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        activation,
        norm_type="none",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(GraphConv(input_dim, output_dim, activation=activation))
        else:
            self.layers.append(GraphConv(input_dim, hidden_dim, activation=activation))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(
                    GraphConv(hidden_dim, hidden_dim, activation=activation)
                )
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(GraphConv(hidden_dim, output_dim))

    def forward(self, g, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = self.dropout(h)
        return h_list, h


class GAT(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        activation,
        num_heads=8,
        attn_drop=0.3,
        negative_slope=0.2,
        residual=False,
    ):
        super(GAT, self).__init__()
        # For GAT, the number of layers is required to be > 1
        assert num_layers > 1

        hidden_dim //= num_heads
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.activation = activation

        heads = ([num_heads] * num_layers) + [1]
        # input (no residual)
        self.layers.append(
            GATConv(
                input_dim,
                hidden_dim,
                heads[0],
                dropout_ratio,
                attn_drop,
                negative_slope,
                False,
                self.activation,
            )
        )

        for l in range(1, num_layers - 1):
            # due to multi-head, the in_dim = hidden_dim * num_heads
            self.layers.append(
                GATConv(
                    hidden_dim * heads[l - 1],
                    hidden_dim,
                    heads[l],
                    dropout_ratio,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                )
            )

        self.layers.append(
            GATConv(
                hidden_dim * heads[-2],
                output_dim,
                heads[-1],
                dropout_ratio,
                attn_drop,
                negative_slope,
                residual,
                None,
            )
        )

    def forward(self, g, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            # [num_head, node_num, nclass] -> [num_head, node_num*nclass]
            h = layer(g, h)
            if l != self.num_layers - 1:
                h = h.flatten(1)
                h_list.append(h)
            else:
                h = h.mean(1)
        return h_list, h


def _compute_lap_pe(g, pe_dim):
    """Laplacian positional encoding: returns [N, pe_dim] float tensor of
    the smallest non-trivial eigenvectors of the symmetric normalized
    Laplacian.
    """
    N = g.number_of_nodes()
    src, dst = g.edges()
    src = src.cpu().numpy()
    dst = dst.cpu().numpy()
    # drop self-loops before Laplacian
    mask = src != dst
    src, dst = src[mask], dst[mask]
    data = np.ones(len(src), dtype=np.float64)
    A = sp.coo_matrix((data, (src, dst)), shape=(N, N)).tocsr()
    A = (A + A.T) * 0.5  # symmetrize for safety
    deg = np.asarray(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(np.clip(deg, 1e-12, None), -0.5)
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    L = sp.eye(N) - D_inv_sqrt @ A @ D_inv_sqrt

    k = pe_dim + 1  # +1 because first eigenvector is trivial (zero eigenvalue)
    if N <= 500 or k >= N - 1:
        eigvals, eigvecs = np.linalg.eigh(L.toarray())
        eigvals, eigvecs = eigvals[:k], eigvecs[:, :k]
    else:
        eigvals, eigvecs = spla.eigsh(L, k=k, which="SA", tol=1e-6)
    idx = np.argsort(eigvals)
    eigvecs = eigvecs[:, idx]
    pe = eigvecs[:, 1:pe_dim + 1].astype(np.float32)  # [N, pe_dim]
    return torch.from_numpy(pe)


class GTLayer(nn.Module):
    """Single Graph Transformer layer with sparse adjacency-restricted attention.
    Uses DGL edge_softmax: memory scales with |E|, not |V|^2.
    """
    def __init__(self, hidden_dim, num_heads, dropout_ratio, attn_dropout):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.drop = nn.Dropout(dropout_ratio)
        self.attn_drop = nn.Dropout(attn_dropout)

    def forward(self, g, h):
        N, D = h.shape
        H, Dh = self.num_heads, self.head_dim
        q = self.q_proj(h).view(N, H, Dh)
        k = self.k_proj(h).view(N, H, Dh)
        v = self.v_proj(h).view(N, H, Dh)

        with g.local_scope():
            g.ndata["q"] = q
            g.ndata["k"] = k
            g.ndata["v"] = v
            g.apply_edges(fn.u_dot_v("k", "q", "score"))   # [E, H, 1]
            score = g.edata["score"] * self.scale
            alpha = edge_softmax(g, score)                   # per-destination softmax
            alpha = self.attn_drop(alpha)
            g.edata["alpha"] = alpha
            g.update_all(fn.u_mul_e("v", "alpha", "m"), fn.sum("m", "agg"))
            out = g.ndata["agg"]                             # [N, H, Dh]

        out = out.reshape(N, D)
        out = self.drop(self.out_proj(out))

        h = self.ln1(h + out)
        h2 = self.drop(self.ffn(h))
        h = self.ln2(h + h2)
        return h


class GT(nn.Module):
    """Graph Transformer with Laplacian positional encoding.
    Sparse attention over graph edges (plus self-loops) via DGL edge_softmax.
    LapPE and self-loop-augmented graph are computed once on first forward.
    """
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim,
                 dropout_ratio, num_heads=4, pe_dim=8, attn_dropout=None):
        super().__init__()
        self.num_layers = num_layers
        self.pe_dim = pe_dim
        self.use_pe = pe_dim > 0
        attn_dropout = dropout_ratio if attn_dropout is None else attn_dropout

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        if self.use_pe:
            self.pe_proj = nn.Linear(pe_dim, hidden_dim)
        self.input_drop = nn.Dropout(dropout_ratio)

        self.layers = nn.ModuleList([
            GTLayer(hidden_dim, num_heads, dropout_ratio, attn_dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_dim, output_dim)

        # populated on first forward; keyed by (n_nodes, device) so that
        # re-calls with `g.to(device)` (which may return a new object) are cheap.
        self._g_cached = None
        self._cached_n = None
        self.register_buffer("pe", torch.empty(0), persistent=False)

    def _prepare(self, g, device):
        g_sl = dgl.remove_self_loop(g)
        g_sl = dgl.add_self_loop(g_sl)
        g_sl = g_sl.to(device)
        self._g_cached = g_sl
        self._cached_n = g.number_of_nodes()
        if self.use_pe:
            self.pe = _compute_lap_pe(g, self.pe_dim).to(device)

    def forward(self, g, feats):
        device = feats.device
        N = feats.shape[0]
        if (self._g_cached is None
                or self._cached_n != N
                or self._g_cached.device != device):
            self._prepare(g, device)
        g_sl = self._g_cached

        h = self.input_proj(feats)
        if self.use_pe:
            h = h + self.pe_proj(self.pe)
        h = self.input_drop(F.relu(h))

        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(g_sl, h)
            if l != self.num_layers - 1:
                h_list.append(h)
        logits = self.classifier(h)
        return h_list, logits


class GraphGPSLayer(nn.Module):
    """Single GraphGPS layer: parallel MPNN branch + global-attention branch,
    each wrapped in a residual + post-LN block, then combined and passed
    through an FFN with its own residual + LN.

    attn_mode="dense": full N x N self-attention (all nodes attend to all nodes).
    attn_mode="sparse": edge_softmax over adjacency+self-loops (for graphs where
                       N^2 is too large).
    """
    def __init__(self, hidden_dim, num_heads, dropout_ratio, attn_dropout,
                 attn_mode="dense"):
        super().__init__()
        self.attn_mode = attn_mode
        assert attn_mode in ("dense", "sparse")
        self.mpnn = GraphConv(hidden_dim, hidden_dim, norm="both",
                              weight=True, bias=True, activation=None)

        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_out = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(attn_dropout)

        self.mpnn_drop = nn.Dropout(dropout_ratio)
        self.attn_drop_res = nn.Dropout(dropout_ratio)
        self.ln_mpnn = nn.LayerNorm(hidden_dim)
        self.ln_attn = nn.LayerNorm(hidden_dim)
        self.ln_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Dropout(dropout_ratio),
        )

    def _attn_sparse(self, g, h):
        N, D = h.shape
        H, Dh = self.num_heads, self.head_dim
        q = self.q_proj(h).view(N, H, Dh)
        k = self.k_proj(h).view(N, H, Dh)
        v = self.v_proj(h).view(N, H, Dh)
        with g.local_scope():
            g.ndata["q"] = q
            g.ndata["k"] = k
            g.ndata["v"] = v
            g.apply_edges(fn.u_dot_v("k", "q", "score"))
            score = g.edata["score"] * self.scale
            alpha = edge_softmax(g, score)
            alpha = self.attn_drop(alpha)
            g.edata["alpha"] = alpha
            g.update_all(fn.u_mul_e("v", "alpha", "m"), fn.sum("m", "agg"))
            out = g.ndata["agg"]
        return self.attn_out(out.reshape(N, D))

    def _attn_dense(self, h):
        N, D = h.shape
        H, Dh = self.num_heads, self.head_dim
        q = self.q_proj(h).view(N, H, Dh).transpose(0, 1)  # [H, N, Dh]
        k = self.k_proj(h).view(N, H, Dh).transpose(0, 1)
        v = self.v_proj(h).view(N, H, Dh).transpose(0, 1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [H, N, N]
        alpha = F.softmax(scores, dim=-1)
        alpha = self.attn_drop(alpha)
        out = torch.matmul(alpha, v)                                 # [H, N, Dh]
        out = out.transpose(0, 1).reshape(N, D)
        return self.attn_out(out)

    def forward(self, g, h):
        h_in = h
        h_local = self.mpnn(g, h)
        h_local = self.mpnn_drop(h_local)
        h_local = self.ln_mpnn(h_in + h_local)

        if self.attn_mode == "dense":
            h_attn = self._attn_dense(h)
        else:
            h_attn = self._attn_sparse(g, h)
        h_attn = self.attn_drop_res(h_attn)
        h_attn = self.ln_attn(h_in + h_attn)

        h = h_local + h_attn
        h = self.ln_ffn(h + self.ffn(h))
        return h


class GraphGPS(nn.Module):
    """GraphGPS teacher: LapPE + L layers of (MPNN || Attn) hybrid.
    Shares LapPE + self-loop-augmented graph caching with GT.
    """
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim,
                 dropout_ratio, num_heads=4, pe_dim=8, attn_dropout=None,
                 attn_mode="dense"):
        super().__init__()
        self.num_layers = num_layers
        self.pe_dim = pe_dim
        self.use_pe = pe_dim > 0
        attn_dropout = dropout_ratio if attn_dropout is None else attn_dropout

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        if self.use_pe:
            self.pe_proj = nn.Linear(pe_dim, hidden_dim)
        self.input_drop = nn.Dropout(dropout_ratio)

        self.layers = nn.ModuleList([
            GraphGPSLayer(hidden_dim, num_heads, dropout_ratio, attn_dropout,
                          attn_mode=attn_mode)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_dim, output_dim)

        self._g_cached = None
        self._cached_n = None
        self.register_buffer("pe", torch.empty(0), persistent=False)

    def _prepare(self, g, device):
        g_sl = dgl.remove_self_loop(g)
        g_sl = dgl.add_self_loop(g_sl)
        g_sl = g_sl.to(device)
        self._g_cached = g_sl
        self._cached_n = g.number_of_nodes()
        if self.use_pe:
            self.pe = _compute_lap_pe(g, self.pe_dim).to(device)

    def forward(self, g, feats):
        device = feats.device
        N = feats.shape[0]
        if (self._g_cached is None
                or self._cached_n != N
                or self._g_cached.device != device):
            self._prepare(g, device)
        g_sl = self._g_cached

        h = self.input_proj(feats)
        if self.use_pe:
            h = h + self.pe_proj(self.pe)
        h = self.input_drop(F.relu(h))

        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(g_sl, h)
            if l != self.num_layers - 1:
                h_list.append(h)
        logits = self.classifier(h)
        return h_list, logits


def _compute_hop_tokens(g, feats, K):
    """Per-node hop-token sequence H^k = A_hat^k X for k=0..K.
    A_hat = D^{-1/2} (A + I) D^{-1/2}. Returns [N, K+1, F] on the device of feats.
    """
    device = feats.device
    g_sl = dgl.add_self_loop(dgl.remove_self_loop(g)).to(device)
    deg = g_sl.in_degrees().float().clamp(min=1.0).to(device)
    norm = deg.pow(-0.5).unsqueeze(-1)       # [N, 1]

    tokens = [feats]
    h = feats
    for _ in range(K):
        with g_sl.local_scope():
            g_sl.ndata["h"] = h * norm
            g_sl.update_all(fn.copy_u("h", "m"), fn.sum("m", "agg"))
            h = g_sl.ndata["agg"] * norm
        tokens.append(h)
    return torch.stack(tokens, dim=1)        # [N, K+1, F]


class NAGphormerLayer(nn.Module):
    """One transformer block self-attending over K+1 hop tokens *per node*.
    Sequence dim = K+1, batch dim = N; attention is O((K+1)^2) regardless of N.
    """
    def __init__(self, hidden_dim, num_heads, dropout_ratio, attn_dropout):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.drop = nn.Dropout(dropout_ratio)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

    def forward(self, x):
        # x: [N, T, D] with T = K+1
        N, T, D = x.shape
        H, Dh = self.num_heads, self.head_dim
        qkv = self.qkv(x).reshape(N, T, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]     # each [N, H, T, Dh]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale    # [N, H, T, T]
        alpha = F.softmax(scores, dim=-1)
        alpha = self.attn_drop(alpha)
        out = torch.matmul(alpha, v)         # [N, H, T, Dh]
        out = out.permute(0, 2, 1, 3).reshape(N, T, D)
        out = self.drop(self.out_proj(out))

        x = self.ln1(x + out)
        x = self.ln2(x + self.drop(self.ffn(x)))
        return x


class NAGphormer(nn.Module):
    """NAGphormer: hop tokenization + sequence attention.

    Precompute H^k = A_hat^k X for k=0..K once (buffer). Project each hop to
    hidden_dim, add a learnable per-hop positional embedding, then L layers
    of transformer encoder attending over the K+1-token sequence per-node.
    Readout by mean pooling over the tokens.
    """
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim,
                 dropout_ratio, num_heads=4, attn_dropout=None, K=10):
        super().__init__()
        self.num_layers = num_layers
        self.K = K
        attn_dropout = dropout_ratio if attn_dropout is None else attn_dropout

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hop_pe = nn.Parameter(torch.randn(K + 1, hidden_dim) * 0.02)
        self.input_drop = nn.Dropout(dropout_ratio)

        self.layers = nn.ModuleList([
            NAGphormerLayer(hidden_dim, num_heads, dropout_ratio, attn_dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_dim, output_dim)

        # buffer for per-node hop-token sequences [N, K+1, input_dim]
        self._cached_n = None
        self.register_buffer("hop_tokens", torch.empty(0), persistent=False)

    def _prepare(self, g, feats):
        self.hop_tokens = _compute_hop_tokens(g, feats, self.K)
        self._cached_n = feats.shape[0]

    def forward(self, g, feats):
        device = feats.device
        N = feats.shape[0]
        if (self._cached_n != N
                or self.hop_tokens.numel() == 0
                or self.hop_tokens.device != device):
            self._prepare(g, feats)

        x = self.input_proj(self.hop_tokens)         # [N, K+1, D]
        x = x + self.hop_pe.unsqueeze(0)
        x = self.input_drop(x)

        h_list = []
        for l, layer in enumerate(self.layers):
            x = layer(x)
            if l != self.num_layers - 1:
                h_list.append(x.mean(dim=1))

        h = x.mean(dim=1)
        logits = self.classifier(h)
        return h_list, logits


class CAWTransform(nn.Module):
    """Optional Linear + ReLU heads, used only for the feature-side W2 term.

    mode 'none' is the identity. 'student' adds a trainable head of width `dim`
    on the student's penultimate features. 'both' also maps the teacher's
    penultimate features through a Linear + ReLU head that stays fixed at its
    initialization. The heads exist only during training; the deployed student
    is the plain MLP.
    """

    def __init__(self, mode, dim):
        super().__init__()
        if mode not in ("none", "student", "both"):
            raise ValueError(f"Unknown caw_transform: {mode}")
        self.student_head = None
        self.teacher_head = None
        if mode in ("student", "both"):
            self.student_head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
        if mode == "both":
            self.teacher_head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
            for param in self.teacher_head.parameters():
                param.requires_grad = False

    def forward(self, h_s, h_t):
        if self.student_head is not None:
            h_s = self.student_head(h_s)
        if self.teacher_head is not None:
            h_t = self.teacher_head(h_t)
        return h_s, h_t


class Model(nn.Module):
    """Wrapper of different models, dispatched on conf["model_name"]."""

    def __init__(self, conf):
        super(Model, self).__init__()
        self.model_name = conf["model_name"]
        if "MLP" in conf["model_name"]:
            self.encoder = MLP(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                norm_type=conf["norm_type"],
            ).to(conf["device"])
        elif "SAGE" in conf["model_name"]:
            self.encoder = SAGE(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
            ).to(conf["device"])
        elif "GCN" in conf["model_name"]:
            self.encoder = GCN(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
            ).to(conf["device"])
        elif "GAT" in conf["model_name"]:
            self.encoder = GAT(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                num_heads=conf["num_heads"],
                attn_drop=conf["attn_dropout_ratio"],
            ).to(conf["device"])
        elif "NAG" in conf["model_name"]:
            self.encoder = NAGphormer(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                num_heads=conf.get("num_heads", 4),
                attn_dropout=conf.get("attn_dropout_ratio", None),
                K=conf.get("K", 10),
            ).to(conf["device"])
        elif "GPS" in conf["model_name"]:
            self.encoder = GraphGPS(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                num_heads=conf.get("num_heads", 4),
                pe_dim=conf.get("pe_dim", 8),
                attn_dropout=conf.get("attn_dropout_ratio", None),
                attn_mode=conf.get("attn_mode", "dense"),
            ).to(conf["device"])
        elif "GT" in conf["model_name"]:
            self.encoder = GT(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                num_heads=conf.get("num_heads", 4),
                pe_dim=conf.get("pe_dim", 8),
                attn_dropout=conf.get("attn_dropout_ratio", None),
            ).to(conf["device"])

    def forward(self, data, feats):
        """data: a graph `g`, a list of blocks (SAGE training) or None (MLP)."""
        return self.encode(data, feats)[1]

    def encode(self, data, feats):
        """Return (h_list, logits), with h_list the intermediate representations."""
        if "MLP" in self.model_name:
            return self.encoder(feats)
        return self.encoder(data, feats)

    def encode_full(self, g, feats):
        """Like `encode`, on a whole graph. SAGE skips neighbor sampling."""
        if "SAGE" in self.model_name:
            return self.encoder.forward_full(g, feats)
        return self.encode(g, feats)
