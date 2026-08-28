"""LightCCN-Flat: LightGCN with augmented item-item adjacency from cell complex."""

from __future__ import annotations

import torch
import torch.nn as nn

from light_ccn.models.base import BaseCFModel
from light_ccn.models import register_model


@register_model("lightccn_flat")
class LightCCNFlat(BaseCFModel):
    """LightCCN-Flat model.

    Same propagation as LightGCN but uses an augmented adjacency:
    A_tilde = [[0, R], [R^T, gamma * S]]
    where S is the item-item adjacency derived from cell complex faces.

    Additive formulation: full user-item signal is preserved, gamma only
    controls how much item-item signal is added. gamma=0 = pure LightGCN.

    Weight modes (controlled by `signed_weights` flag):
    - Default (signed_weights=False): single pre-computed adjacency (gamma baked in)
    - Signed (signed_weights=True): two separate adjacency matrices with learnable
      unconstrained weights: emb = w_node * A_gcn @ emb + w_edge * A_ii @ emb
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        n_layers: int = 3,
        adj_matrix: torch.Tensor | None = None,
        adj_item_item: torch.Tensor | None = None,
        signed_weights: bool = False,
    ):
        super().__init__(n_users, n_items, embed_dim)
        self.n_layers = n_layers
        self.signed_weights = signed_weights
        self.adj_matrix = adj_matrix
        self.adj_item_item = adj_item_item

        if signed_weights:
            # Learnable unconstrained weights for node and edge signals
            # w_node: weight for base LightGCN adjacency (user-item)
            # w_edge: weight for item-item adjacency (from cell complex)
            self.flat_weights = nn.Parameter(torch.tensor([1.0, 0.5]))

    def set_adj_matrix(self, adj_matrix: torch.Tensor) -> None:
        self.adj_matrix = adj_matrix

    def get_attention_weights(self) -> dict[str, dict[str, float]]:
        """Return the learned mixing weights as plain floats.

        Only available in signed_weights mode.
        """
        if not self.signed_weights:
            return {}
        with torch.no_grad():
            w = self.flat_weights.cpu().tolist()
        return {
            "node": {"w_node": w[0]},
            "edge": {"w_edge": w[1]},
        }

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        ego = self.get_ego_embeddings()
        all_embs = [ego]

        emb = ego
        for _ in range(self.n_layers):
            if self.signed_weights and self.adj_item_item is not None:
                # Two-component propagation with learnable weights
                node_signal = torch.sparse.mm(self.adj_matrix, emb)
                edge_signal = torch.sparse.mm(self.adj_item_item, emb)
                emb = self.flat_weights[0] * node_signal + self.flat_weights[1] * edge_signal
            else:
                # Single pre-computed adjacency (gamma baked in)
                emb = torch.sparse.mm(self.adj_matrix, emb)
            all_embs.append(emb)

        combined = self.layer_combination(all_embs)
        user_embs = combined[:self.n_users]
        item_embs = combined[self.n_users:]
        return user_embs, item_embs
