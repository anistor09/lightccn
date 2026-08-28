"""NGCF: Neural Graph Collaborative Filtering."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from light_ccn.models.base import BaseCFModel
from light_ccn.models import register_model


@register_model("ngcf")
class NGCF(BaseCFModel):
    """Neural Graph Collaborative Filtering.

    Features vs LightGCN:
    - Weight matrices W1, W2 per layer
    - LeakyReLU activation
    - Element-wise product interaction term
    - Node and message dropout
    - Layer combination via concatenation

    ``faithful=True`` switches to the released TF code's exact semantics
    (expects the row-normalized D^-1(A+I) operator as ``adj_matrix``):
    LeakyReLU per path instead of after the sum, self-loop inside the
    operator instead of an explicit ego term, per-layer L2 normalization
    of the concatenated outputs, and Xavier init on the layer weights.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        n_layers: int = 3,
        adj_matrix: torch.Tensor | None = None,
        node_dropout: float = 0.1,
        message_dropout: float = 0.1,
        faithful: bool = False,
    ):
        super().__init__(n_users, n_items, embed_dim)
        self.n_layers = n_layers
        self.adj_matrix = adj_matrix
        self.node_dropout_rate = node_dropout
        self.message_dropout_rate = message_dropout
        self.faithful = faithful

        # Per-layer weight matrices
        layer_dims = [embed_dim] * (n_layers + 1)
        self.W1 = nn.ModuleList()
        self.W2 = nn.ModuleList()
        for i in range(n_layers):
            self.W1.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            self.W2.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        if faithful:
            # Official code Xavier-initializes every W and bias (biases as
            # [1, d] matrices, hence the fan-based bound).
            for lin in list(self.W1) + list(self.W2):
                nn.init.xavier_uniform_(lin.weight)
                bound = math.sqrt(6.0 / (1 + lin.bias.shape[0]))
                nn.init.uniform_(lin.bias, -bound, bound)

    def set_adj_matrix(self, adj_matrix: torch.Tensor) -> None:
        self.adj_matrix = adj_matrix

    def _node_dropout(self, adj: torch.Tensor) -> torch.Tensor:
        """Apply dropout to adjacency matrix (drop nodes)."""
        if not self.training or self.node_dropout_rate == 0:
            return adj
        adj = adj.coalesce()
        mask = torch.rand(adj._nnz(), device=adj.device) > self.node_dropout_rate
        indices = adj.indices()[:, mask]
        values = adj.values()[mask] / (1 - self.node_dropout_rate)
        return torch.sparse_coo_tensor(indices, values, adj.shape).coalesce()

    def _message_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout to messages."""
        if not self.training or self.message_dropout_rate == 0:
            return x
        return F.dropout(x, p=self.message_dropout_rate, training=True)

    def layer_combination(self, layer_embs: list[torch.Tensor]) -> torch.Tensor:
        """NGCF uses concatenation instead of mean."""
        return torch.cat(layer_embs, dim=1)

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        ego = self.get_ego_embeddings()
        all_embs = [ego]

        adj = self._node_dropout(self.adj_matrix)
        emb = ego

        for k in range(self.n_layers):
            # Neighborhood aggregation
            neighbor_emb = torch.sparse.mm(adj, emb)

            if self.faithful:
                # Official semantics: adj carries the self-loop, each path
                # gets its own LeakyReLU, and only the copy appended for
                # concatenation is L2-normalized — the next layer continues
                # from the unnormalized embeddings, as in the TF code.
                sum_emb = F.leaky_relu(self.W1[k](neighbor_emb), negative_slope=0.2)
                bi_emb = F.leaky_relu(self.W2[k](emb * neighbor_emb), negative_slope=0.2)
                emb = self._message_dropout(sum_emb + bi_emb)
                all_embs.append(F.normalize(emb, p=2, dim=1))
                continue

            # NGCF message: W1*(e_i + neighbor) + W2*(e_i * neighbor)
            sum_emb = self.W1[k](emb + neighbor_emb)
            bi_emb = self.W2[k](emb * neighbor_emb)
            emb = F.leaky_relu(sum_emb + bi_emb, negative_slope=0.2)
            emb = self._message_dropout(emb)
            all_embs.append(emb)

        combined = self.layer_combination(all_embs)
        user_embs = combined[:self.n_users]
        item_embs = combined[self.n_users:]
        return user_embs, item_embs

    @property
    def embed_dim_final(self) -> int:
        """NGCF concatenates all layers, so final dim is (n_layers+1)*embed_dim."""
        return self.embed_dim * (self.n_layers + 1)

    def get_all_ratings(self, users: torch.Tensor) -> torch.Tensor:
        user_all, item_all = self.propagate()
        user_e = user_all[users]
        scores = user_e @ item_all.T
        return scores

    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        user_all, item_all = self.propagate()

        user_e = user_all[users]
        pos_e = item_all[pos_items]
        neg_e = item_all[neg_items]

        ego_user = self.user_embedding(users)
        ego_pos = self.item_embedding(pos_items)
        ego_neg = self.item_embedding(neg_items)
        reg_loss = (
            ego_user.norm(2).pow(2)
            + ego_pos.norm(2).pow(2)
            + ego_neg.norm(2).pow(2)
        ) / (2 * users.shape[0])

        return user_e, pos_e, neg_e, reg_loss
