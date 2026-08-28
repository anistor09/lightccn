"""SimGCL: Simple Graph Contrastive Learning for Recommendation.

Reference:
    Yu, Yin, Xia, Chen, Cui, Nguyen. "Are Graph Augmentations Necessary?
    Simple Graph Contrastive Learning for Recommendation." SIGIR 2022.

Implementation follows the authors' SELFRec reference: a LightGCN backbone
where the two contrastive views are produced NOT by graph augmentation (as in
SGL) but by adding a small random noise to the embeddings at every propagation
layer:

    e_l := e_l + sign(e_l) ⊙ normalize(U(0,1)) · ε

The InfoNCE contrastive loss between the two noised views (over the batch's
users and positive items) is added to the BPR + L2 objective. The model
applies its own contrastive weight λ_cl and exposes the term through
``auxiliary_loss()``, which the Trainer adds after the BPR loss.

Backbone propagation matches SimGCL/SELFRec: the layer-combination is the mean
over the L *propagated* layers (the layer-0 ego embedding is not included),
used for both BPR scoring and evaluation.

Verified line-by-line against the authors' SELFRec reference
(github.com/Coder-Yu/SELFRec, model/graph/SimGCL.py): the noise perturbation,
the mean-over-L-layers encoder, the two perturbed views, the InfoNCE over the
batch's unique user/positive-item indices, and the L2 regularization on the
*propagated* user + positive-item embeddings all match. (The L2 reg therefore
follows SimGCL's convention, which differs from our LightGCN/NGCF baselines
that reg the layer-0 embeddings — each baseline follows its own paper.)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from light_ccn.models.base import BaseCFModel
from light_ccn.models import register_model


@register_model("simgcl")
class SimGCL(BaseCFModel):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        n_layers: int = 2,
        adj_matrix: torch.Tensor | None = None,
        eps: float = 0.1,          # noise magnitude (SimGCL default 0.1)
        cl_temp: float = 0.2,      # InfoNCE temperature τ (SimGCL default 0.2)
        cl_weight: float = 0.5,    # λ for the contrastive term (SimGCL: 0.1–1)
        **_unused,
    ):
        super().__init__(n_users, n_items, embed_dim)
        self.n_layers = n_layers
        self.adj_matrix = adj_matrix
        self.eps = eps
        self.cl_temp = cl_temp
        self.cl_weight = cl_weight
        self._cl_loss: torch.Tensor | None = None

    def set_adj_matrix(self, adj_matrix: torch.Tensor) -> None:
        self.adj_matrix = adj_matrix

    # ---- propagation ----
    def _propagate(self, perturbed: bool) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.get_ego_embeddings()
        layer_embs = []
        for _ in range(self.n_layers):
            emb = torch.sparse.mm(self.adj_matrix, emb)
            if perturbed:
                noise = torch.rand_like(emb)
                emb = emb + torch.sign(emb) * F.normalize(noise, dim=-1) * self.eps
            layer_embs.append(emb)
        combined = torch.stack(layer_embs, dim=0).mean(dim=0)  # mean over L layers (no ego)
        return combined[: self.n_users], combined[self.n_users:]

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Clean (non-perturbed) — used for BPR scoring and evaluation.
        return self._propagate(perturbed=False)

    # ---- InfoNCE ----
    def _infonce(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """InfoNCE over a set of nodes present in both views.
        positives = the same node across views; negatives = the other nodes."""
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        pos = (z1 * z2).sum(dim=-1) / self.cl_temp          # (n,)
        logits = (z1 @ z2.t()) / self.cl_temp               # (n, n)
        return (-pos + torch.logsumexp(logits, dim=1)).mean()

    def forward(self, users, pos_items, neg_items):
        # Clean embeddings for the BPR objective.
        user_all, item_all = self.propagate()
        user_e = user_all[users]
        pos_e = item_all[pos_items]
        neg_e = item_all[neg_items]

        # L2 regularization matches the SELFRec SimGCL reference: it penalizes
        # the *propagated* user + positive-item embeddings (not the ego/layer-0
        # rows, and not the negative). This differs from our LightGCN/NGCF
        # baselines (which reg layer-0) — by design: each baseline follows its
        # own paper's convention.
        reg_loss = (
            user_e.norm(2).pow(2)
            + pos_e.norm(2).pow(2)
        ) / (2 * users.shape[0])

        # Two independently-noised views; contrast over the batch's unique
        # users and positive items (keeps the InfoNCE matrix tractable).
        u1, i1 = self._propagate(perturbed=True)
        u2, i2 = self._propagate(perturbed=True)
        uu = torch.unique(users)
        ii = torch.unique(pos_items)
        cl = self._infonce(u1[uu], u2[uu]) + self._infonce(i1[ii], i2[ii])
        self._cl_loss = self.cl_weight * cl

        return user_e, pos_e, neg_e, reg_loss

    def auxiliary_loss(self) -> torch.Tensor | None:
        """λ_cl · InfoNCE for the last batch; consumed by the Trainer."""
        return self._cl_loss
