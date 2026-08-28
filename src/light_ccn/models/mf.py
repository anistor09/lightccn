"""MF: Matrix Factorization trained with BPR loss (the BPR-MF floor baseline).

Reference:
    Rendle, Freudenthaler, Gantner, Schmidt-Thieme.
    "BPR: Bayesian Personalized Ranking from Implicit Feedback." UAI 2009.

This is the standard no-graph collaborative-filtering floor used as a baseline
by both NGCF (Wang et al., SIGIR 2019) and HOUR (Wang et al., Neurocomputing
2024). There is no message passing: the learned user/item embedding tables ARE
the final representations, and the score is the inner product

    s(u, i) = <e_u, e_i>.

Equivalently, MF is a 0-layer LightGCN — so it slots straight into the shared
``BaseCFModel`` BPR + L2 training path with a trivial ``propagate``.
"""

from __future__ import annotations

from light_ccn.models.base import BaseCFModel
from light_ccn.models import register_model


@register_model("mf")
class MF(BaseCFModel):
    """BPR Matrix Factorization. ``propagate`` returns the raw embedding tables;
    the inherited ``forward`` supplies the BPR pairwise loss + L2 regularization,
    identical to the other models so results are directly comparable."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        # Accepted-and-ignored so the driver can pass the common graph kwargs
        # (n_layers / adj_matrix) uniformly; MF has neither.
        **_unused,
    ):
        super().__init__(n_users, n_items, embed_dim)

    def propagate(self):
        # No propagation: final user/item embeddings == the layer-0 tables.
        return self._compute_node_embeddings()
