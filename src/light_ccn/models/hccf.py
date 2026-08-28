"""HCCF: Hypergraph Contrastive Collaborative Filtering.

Reference:
    Xia, Huang, Xu, Dai, Bo, Zhang, Chen, Pei, Huang.
    "Hypergraph Contrastive Collaborative Filtering." SIGIR 2022.

Architecture (verified against the authors' SSLRec PyTorch reference,
github.com/HKUDS/SSLRec, models/general_cf/hccf.py, + config/modelconf/hccf.yml):

Two complementary views, accumulated layer by layer:
  * **Local (graph) view** — LightGCN-style propagation on the normalized
    user-item adjacency:  G = Â · E.
  * **Global (hypergraph) view** — a *learnable* low-rank hyperedge structure.
    The node→hyperedge incidence is content-dependent:
        H_u = E_u^(0) · Wᵤ   (shape n_users × K),   H_i = E_i^(0) · Wᵢ
    and the hypergraph message passing folds nodes through the K hyperedges:
        HGNN(H, X) = act( H · act(Hᵀ · X) ).
  Each layer's embedding is the sum of the two views; the final embedding is
  the sum over all layers (incl. the layer-0 ego).

The **cross-view InfoNCE** contrasts the local (graph) embedding against the
global (hypergraph) embedding, per layer (verified verbatim against SSLRec's
``cal_infonce_loss_spec_nodes`` + the HCCF ``cal_loss``): anchors are the
batch's unique users / positive items, and the denominator runs over *all*
nodes of that type (``pckEmbeds1 @ embeds2.T``). The local (GCN) view is
**detached** in the contrastive term, so the InfoNCE pulls only the hypergraph
view toward the fixed GCN view. The model applies its own weight λ and exposes
the term via ``auxiliary_loss()`` (consumed by the Trainer after BPR).

The L2 regularization is the reference's ``reg_params``: weight decay over *all*
parameters (the two embedding tables + the two hyperedge matrices), scaled by
reg_weight in the Trainer. (The propagated embeddings are NOT regularized — they
blow up through the hypergraph matmul at init, which is inherent to HCCF and
also present in the reference; on large data many steps tame it.)

Default HPs follow the HCCF paper (Xia et al., SIGIR 2022, §4.1 / Fig. 6):
hyper_num=128, temperature τ=1.0 (the paper's best; it searches {0.1,0.3,1,3,10}),
cl_weight λ₂=1e-3 (the paper tunes the loss-balance weights in {1e-5,…}),
leaky=0.5 (the paper explicitly uses a LeakyReLU with 0.5 negative slope —
NOT 1.0, which would make the HGNN activation a pure identity and delete the
hypergraph nonlinearity), keep_rate=0.5 (dropout searched in {0.25,0.5,0.75}),
mult=1.0. embed_dim is left at our project default (64) rather than HCCF's 32,
so the comparison against our other 64-dim models is capacity-fair.

NOTE: an earlier run used τ=0.1, λ₂=1.0, leaky=1.0 — those three together let
the InfoNCE term (~35/step) swamp the BPR term (~0.5/step), starving the
ranking objective; the model under-trained badly (amazon-music still rising at
epoch 600; beidian/foursquare stuck at the noise floor). The defaults above fix
that.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from light_ccn.models.base import BaseCFModel
from light_ccn.models import register_model


@register_model("hccf")
class HCCF(BaseCFModel):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        n_layers: int = 2,
        adj_matrix: torch.Tensor | None = None,
        hyper_num: int = 128,      # K — number of (learnable) hyperedges
        cl_temp: float = 1.0,      # InfoNCE temperature τ (HCCF paper: best at τ=1.0)
        cl_weight: float = 1e-3,   # contrastive weight λ₂ (HCCF paper tunes in {1e-5,…})
        leaky: float = 0.5,        # LeakyReLU slope in the HGNN (HCCF paper: 0.5)
        keep_rate: float = 0.5,    # edge / incidence keep prob (1−dropout)
        mult: float = 1.0,         # incidence scaling
        **_unused,
    ):
        super().__init__(n_users, n_items, embed_dim)
        self.n_layers = n_layers
        self.adj_matrix = adj_matrix
        self.hyper_num = hyper_num
        self.cl_temp = cl_temp
        self.cl_weight = cl_weight
        self.keep_rate = keep_rate
        self.mult = mult
        self.act = nn.LeakyReLU(negative_slope=leaky)

        # Learnable hyperedge embedding matrices (d × K).
        self.user_hyper = nn.Parameter(torch.empty(embed_dim, hyper_num))
        self.item_hyper = nn.Parameter(torch.empty(embed_dim, hyper_num))
        nn.init.xavier_uniform_(self.user_hyper)
        nn.init.xavier_uniform_(self.item_hyper)

        self._cl_loss: torch.Tensor | None = None

    def set_adj_matrix(self, adj_matrix: torch.Tensor) -> None:
        self.adj_matrix = adj_matrix

    # ---- stochastic regularization (training only) ----
    def _edge_drop(self, adj: torch.Tensor) -> torch.Tensor:
        if not self.training or self.keep_rate >= 1.0:
            return adj
        adj = adj.coalesce()
        mask = torch.rand(adj._nnz(), device=adj.device) < self.keep_rate
        idx = adj.indices()[:, mask]
        val = adj.values()[mask] / self.keep_rate
        return torch.sparse_coo_tensor(idx, val, adj.shape).coalesce()

    # ---- hypergraph message passing: node → hyperedge → node ----
    def _hgnn(self, incidence: torch.Tensor, embeds: torch.Tensor) -> torch.Tensor:
        hids = self.act(incidence.t() @ embeds)   # (K, d)  aggregate nodes into hyperedges
        return self.act(incidence @ hids)          # (n, d)  scatter hyperedges back to nodes

    def _forward_views(self):
        """Return (final_user, final_item, gcn_views, hyper_views).
        gcn_views / hyper_views are the per-layer (N, d) embeddings used for
        the cross-view contrastive loss."""
        ego = self.get_ego_embeddings()                  # (N, d), N = n_users + n_items
        user_ego, item_ego = ego[: self.n_users], ego[self.n_users:]

        # Content-dependent, learnable node→hyperedge incidence.
        uu = (user_ego @ self.user_hyper) * self.mult    # (n_users, K)
        ii = (item_ego @ self.item_hyper) * self.mult    # (n_items, K)
        if self.training and self.keep_rate < 1.0:
            uu = F.dropout(uu, p=1.0 - self.keep_rate)
            ii = F.dropout(ii, p=1.0 - self.keep_rate)

        adj = self._edge_drop(self.adj_matrix)
        embeds_list = [ego]
        gcn_views, hyper_views = [], []
        for _ in range(self.n_layers):
            prev = embeds_list[-1]
            gcn = torch.sparse.mm(adj, prev)                          # local view
            hu = self._hgnn(uu, prev[: self.n_users])                # global view (users)
            hi = self._hgnn(ii, prev[self.n_users:])                 # global view (items)
            hyper = torch.cat([hu, hi], dim=0)
            embeds_list.append(gcn + hyper)
            gcn_views.append(gcn)
            hyper_views.append(hyper)
        final = torch.stack(embeds_list, dim=0).sum(dim=0)           # sum over layers (incl. ego)
        return final[: self.n_users], final[self.n_users:], gcn_views, hyper_views

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        u, i, _, _ = self._forward_views()
        return u, i

    # ---- cross-view InfoNCE (HCCF original: ALL nodes as negatives) ----
    def _infonce(self, z1_all: torch.Tensor, z2_all: torch.Tensor,
                 idx: torch.Tensor) -> torch.Tensor:
        """Contrast the batch's anchor nodes (idx) against the FULL node set of
        that type, matching the authors' HCCF `contrastLoss`: the numerator is
        the same-node alignment, the denominator sums over *all* nodes (not just
        the batch). The batch-only negatives we used before made the contrastive
        signal too weak — HCCF then trained slowly (still rising at ep600 vs the
        paper's ~80-epoch convergence) and λ₂ barely mattered."""
        z1 = F.normalize(z1_all, dim=-1)
        z2 = F.normalize(z2_all, dim=-1)
        a1 = z1[idx]                                  # (B, d) anchors
        a2 = z2[idx]
        pos = (a1 * a2).sum(dim=-1) / self.cl_temp    # (B,)
        logits = (a1 @ z2.t()) / self.cl_temp         # (B, N) — all nodes as negatives
        return (-pos + torch.logsumexp(logits, dim=1)).mean()

    def forward(self, users, pos_items, neg_items):
        fu, fi, gcn_views, hyper_views = self._forward_views()
        user_e = fu[users]
        pos_e = fi[pos_items]
        neg_e = fi[neg_items]

        # L2 weight decay over ALL parameters — the embedding tables + the two
        # hyperedge matrices — exactly the reference HCCF's `reg_params(self)`
        # (sum of ||W||² over model.parameters()); the Trainer multiplies by
        # reg_weight (1e-7). NOT the *propagated* embeddings (those explode
        # through the hypergraph matmul at init).
        reg_loss = sum(p.pow(2).sum() for p in self.parameters())

        # Cross-view contrastive: local (gcn) vs global (hypergraph), per layer,
        # over the batch's unique users and positive items.
        uu = torch.unique(users)
        ii = torch.unique(pos_items)
        cl = user_e.new_zeros(())
        for g, h in zip(gcn_views, hyper_views):
            # The reference DETACHES the local (GCN) view in the contrastive
            # term: the InfoNCE only pulls the hypergraph view toward the (fixed)
            # GCN view, instead of collapsing the two views into each other.
            g = g.detach()
            cl = cl + self._infonce(g[: self.n_users], h[: self.n_users], uu)
            cl = cl + self._infonce(g[self.n_users:], h[self.n_users:], ii)
        self._cl_loss = self.cl_weight * cl

        return user_e, pos_e, neg_e, reg_loss

    def auxiliary_loss(self) -> torch.Tensor | None:
        return self._cl_loss
