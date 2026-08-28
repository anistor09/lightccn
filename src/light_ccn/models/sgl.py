"""SGL: Self-supervised Graph Learning for Recommendation."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from light_ccn.models.base import BaseCFModel
from light_ccn.models import register_model
from light_ccn.utils.helpers import scipy_to_torch_sparse


@register_model("sgl")
class SGL(BaseCFModel):
    """Self-supervised Graph Learning (SGL-ED: edge dropout variant).

    Uses LightGCN as backbone with two augmented graph views
    for contrastive self-supervised learning.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        n_layers: int = 3,
        adj_matrix: torch.Tensor | None = None,
        bipartite_adj_scipy: sp.spmatrix | None = None,
        augment_ratio: float = 0.1,
        device: torch.device | str = "cpu",
    ):
        super().__init__(n_users, n_items, embed_dim)
        self.n_layers = n_layers
        self.adj_matrix = adj_matrix
        self.bipartite_adj_scipy = bipartite_adj_scipy
        self.augment_ratio = augment_ratio
        self._device = device if isinstance(device, torch.device) else torch.device(device)

        # Augmented view caches (rebuilt each epoch)
        self.adj_view1: torch.Tensor | None = None
        self.adj_view2: torch.Tensor | None = None

        # Cached contrastive views
        self._user_z1: torch.Tensor | None = None
        self._user_z2: torch.Tensor | None = None
        self._item_z1: torch.Tensor | None = None
        self._item_z2: torch.Tensor | None = None

    def set_adj_matrix(self, adj_matrix: torch.Tensor) -> None:
        self.adj_matrix = adj_matrix

    def _create_augmented_adj(self) -> torch.Tensor:
        """Create an augmented adjacency by edge dropout."""
        adj = self.bipartite_adj_scipy.tocoo()
        n_edges = adj.nnz
        keep_mask = np.random.random(n_edges) > self.augment_ratio
        rows = adj.row[keep_mask]
        cols = adj.col[keep_mask]
        data = adj.data[keep_mask]
        aug_adj = sp.csr_matrix((data, (rows, cols)), shape=adj.shape)

        # Symmetric normalization
        rowsum = np.array(aug_adj.sum(axis=1)).flatten()
        with np.errstate(divide="ignore"):
            d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        D = sp.diags(d_inv_sqrt)
        normed = D @ aug_adj @ D

        return scipy_to_torch_sparse(normed).to(self._device)

    def _lightgcn_propagate(self, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run LightGCN-style propagation with a given adjacency."""
        ego = self.get_ego_embeddings()
        all_embs = [ego]
        emb = ego
        for _ in range(self.n_layers):
            emb = torch.sparse.mm(adj, emb)
            all_embs.append(emb)
        combined = torch.stack(all_embs, dim=0).mean(dim=0)
        return combined[:self.n_users], combined[self.n_users:]

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Main propagation using the original (non-augmented) adj."""
        return self._lightgcn_propagate(self.adj_matrix)

    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Build augmented views for this forward pass
        self.adj_view1 = self._create_augmented_adj()
        self.adj_view2 = self._create_augmented_adj()

        # Main propagation
        user_all, item_all = self.propagate()
        user_e = user_all[users]
        pos_e = item_all[pos_items]
        neg_e = item_all[neg_items]

        # Augmented propagation for contrastive views
        user_z1, item_z1 = self._lightgcn_propagate(self.adj_view1)
        user_z2, item_z2 = self._lightgcn_propagate(self.adj_view2)
        self._user_z1, self._user_z2 = user_z1, user_z2
        self._item_z1, self._item_z2 = item_z1, item_z2

        # Ego regularization
        ego_user = self.user_embedding(users)
        ego_pos = self.item_embedding(pos_items)
        ego_neg = self.item_embedding(neg_items)
        reg_loss = (
            ego_user.norm(2).pow(2)
            + ego_pos.norm(2).pow(2)
            + ego_neg.norm(2).pow(2)
        ) / (2 * users.shape[0])

        return user_e, pos_e, neg_e, reg_loss

    def get_contrastive_views(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return cached contrastive embeddings from the last forward pass."""
        return self._user_z1, self._user_z2, self._item_z1, self._item_z2
