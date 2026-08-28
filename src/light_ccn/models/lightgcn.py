"""LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation."""

from __future__ import annotations

import torch

from light_ccn.models.base import BaseCFModel
from light_ccn.models import register_model


@register_model("lightgcn")
class LightGCN(BaseCFModel):
    """LightGCN model.

    Propagation: E^{(k+1)} = A_hat @ E^{(k)}
    Final embeddings: mean of [E^{(0)}, E^{(1)}, ..., E^{(K)}]
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        n_layers: int = 3,
        adj_matrix: torch.Tensor | None = None,
    ):
        super().__init__(n_users, n_items, embed_dim)
        self.n_layers = n_layers
        self.adj_matrix = adj_matrix

    def set_adj_matrix(self, adj_matrix: torch.Tensor) -> None:
        """Set or update the adjacency matrix (e.g., after moving to device)."""
        self.adj_matrix = adj_matrix

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        ego = self.get_ego_embeddings()
        all_embs = [ego]

        emb = ego
        for _ in range(self.n_layers):
            emb = torch.sparse.mm(self.adj_matrix, emb)
            all_embs.append(emb)

        combined = self.layer_combination(all_embs)
        user_embs = combined[:self.n_users]
        item_embs = combined[self.n_users:]
        return user_embs, item_embs
