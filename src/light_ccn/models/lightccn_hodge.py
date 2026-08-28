"""LightCCN-Hodge: LightGCN + node-level Hodge Laplacian from cell complex."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from light_ccn.models.base import BaseCFModel
from light_ccn.models import register_model

_VALID_WEIGHT_MODES = ("softmax", "signed", "tanh", "softplus")


@register_model("lightccn_hodge")
class LightCCNHodge(BaseCFModel):
    """LightCCN-Hodge model with principled Hodge Laplacian propagation.

    Uses two node-level operators with learnable weights:
        x^{k+1} = w1 * A_hat_0 @ x^k + w2 * L_hat_0 @ x^k

    where:
        A_hat_0 = D^{-1/2} A D^{-1/2}  (bipartite CF adjacency, same as LightGCN)
        L_hat_0 = D^{-1/2} (B1 B1^T - diag) D^{-1/2}  (0-th Hodge Laplacian)

    Weight modes:
    - "softmax": [w1,w2] = softmax(logits), sum to 1 (default)
    - "tanh": bounded to [-1, 1]
    - "softplus": positive, unbounded
    - "signed": unconstrained independent scalars
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        n_layers: int = 3,
        operators: dict[str, torch.Tensor] | None = None,
        weight_mode: str = "softmax",
    ):
        super().__init__(n_users, n_items, embed_dim)
        self.n_layers = n_layers
        self.operators = operators or {}

        if weight_mode not in _VALID_WEIGHT_MODES:
            raise ValueError(f"weight_mode must be one of {_VALID_WEIGHT_MODES}")
        self.weight_mode = weight_mode

        if weight_mode == "signed":
            self.node_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        elif weight_mode == "tanh":
            self.node_logits = nn.Parameter(torch.tensor([math.atanh(0.5)] * 2))
        else:
            # softmax (zeros → uniform) and softplus (softplus(0)=ln2, equal)
            self.node_logits = nn.Parameter(torch.zeros(2))

    def _get_weights(self) -> torch.Tensor:
        if self.weight_mode == "signed":
            return self.node_weights
        elif self.weight_mode == "softmax":
            return F.softmax(self.node_logits, dim=0)
        elif self.weight_mode == "tanh":
            return torch.tanh(self.node_logits)
        elif self.weight_mode == "softplus":
            return F.softplus(self.node_logits)
        raise ValueError(f"Unknown weight_mode: {self.weight_mode}")

    def set_operators(self, operators: dict[str, torch.Tensor]) -> None:
        self.operators = operators

    def get_attention_weights(self) -> dict[str, dict[str, float]]:
        with torch.no_grad():
            w = self._get_weights().cpu().tolist()
        return {
            "node": {"w1_cf": w[0], "w2_hodge": w[1]},
        }

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        w = self._get_weights()

        x = self.get_ego_embeddings()
        all_embs = [x]

        for _ in range(self.n_layers):
            cf_signal = torch.sparse.mm(self.operators["A_hat_0"], x)
            hodge_signal = torch.sparse.mm(self.operators["L_hat_0"], x)
            x = w[0] * cf_signal + w[1] * hodge_signal
            all_embs.append(x)

        combined = self.layer_combination(all_embs)
        user_embs = combined[:self.n_users]
        item_embs = combined[self.n_users:]
        return user_embs, item_embs
