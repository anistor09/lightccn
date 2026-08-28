"""Base collaborative filtering model with shared embedding logic."""

from __future__ import annotations

from abc import ABC, abstractmethod

import scipy.sparse as sp
import torch
import torch.nn as nn

_VALID_INIT_MODES = {"table", "fc"}


class BaseCFModel(nn.Module, ABC):
    """Abstract base class for collaborative filtering models.

    Provides shared user/item embeddings, BPR forward pass, and layer combination.
    Subclasses must implement `propagate()`.

    init_mode:
      - "table" (default): free nn.Embedding tables, randomly initialized.
        Layer-0 embedding for user u is a learnable parameter looked up by ID.
      - "fc": HOUR-style encoder. Layer-0 embedding is computed from the
        interaction matrix R every forward pass:
            x_u = ReLU(W_U R_u + b_U),   x_i = ReLU(W_I R_i + b_I)
        REPLACES the tables (same parameter count d*(|U|+|I|), not added on
        top). Requires the interaction matrix R to be passed via
        `set_interaction_matrix(R)` before the first forward pass.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        init_mode: str = "table",
        fc_normalize_R: bool = False,
    ):
        super().__init__()
        if init_mode not in _VALID_INIT_MODES:
            raise ValueError(
                f"init_mode must be one of {_VALID_INIT_MODES}, got {init_mode!r}"
            )
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.init_mode = init_mode
        # Whether to symmetric-degree-normalize R before the FC encode.
        # Mirrors LightGCN's adjacency normalization: R_tilde = D_U^-1/2 R D_I^-1/2.
        # Removes the gradient-magnitude bias toward power users / popular items
        # that destabilizes the unnormalized FC encoder on sparse implicit CF.
        self.fc_normalize_R = fc_normalize_R

        if init_mode == "table":
            self.user_embedding = nn.Embedding(n_users, embed_dim)
            self.item_embedding = nn.Embedding(n_items, embed_dim)
            self.W_U = None
            self.W_I = None
            self.b_U = None
            self.b_I = None
            self._init_weights()
        else:  # "fc"
            # FC encoder parameters. Same trainable-param count as table mode
            # (d * |I| + d * |U| = d * (|U| + |I|)).
            self.user_embedding = None
            self.item_embedding = None
            self.W_U = nn.Parameter(torch.empty(embed_dim, n_items))
            self.W_I = nn.Parameter(torch.empty(embed_dim, n_users))
            self.b_U = nn.Parameter(torch.zeros(embed_dim))
            self.b_I = nn.Parameter(torch.zeros(embed_dim))
            self._init_fc_weights()
            # R buffers populated lazily via set_interaction_matrix().
            self._R: torch.sparse.Tensor | None = None
            self._R_t: torch.sparse.Tensor | None = None

    def _init_weights(self) -> None:
        """Default (table) initialization, unchanged from prior versions.

        table_init="xavier" (set post-construction via init_table_xavier())
        matches SSLRec's reference init for HCCF parity runs; the default
        normal(0.1) is the harness-wide LightGCN convention."""
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def init_table_xavier(self) -> None:
        """Re-initialize embedding tables with xavier_uniform (SSLRec HCCF convention)."""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def _init_fc_weights(self) -> None:
        """FC encoder init. Small Gaussian so layer-0 magnitudes are comparable
        to the table case (std=0.1) — the linear sum of std=0.1 entries over
        an average user's history (~tens of items on Beidian/Gowalla) gives a
        layer-0 norm of similar order to the table init."""
        nn.init.normal_(self.W_U, std=0.01)
        nn.init.normal_(self.W_I, std=0.01)

    def set_interaction_matrix(self, R: sp.spmatrix) -> None:
        """Register the interaction matrix R as sparse tensors on the model's
        device. Required before the first forward pass when init_mode='fc'.

        For init_mode='table' this is a no-op (R is unused).

        If fc_normalize_R is True, R is replaced with the LightGCN-style
        symmetric degree-normalized form
            R_tilde[u, i] = R[u, i] / sqrt(deg_u * deg_i)
        which removes the per-update gradient bias toward power users / popular
        items that destabilizes the unnormalized FC encoder on sparse implicit CF.
        """
        if self.init_mode != "fc":
            return
        R_csr = R.tocsr().astype("float32")
        if self.fc_normalize_R:
            # Symmetric normalization: R_tilde = D_U^-1/2 R D_I^-1/2.
            # Same form LightGCN applies to the propagation adjacency.
            deg_u = sp.csr_matrix(R_csr.sum(axis=1)).toarray().flatten().clip(min=1.0)
            deg_i = sp.csr_matrix(R_csr.sum(axis=0)).toarray().flatten().clip(min=1.0)
            d_u_inv_sqrt = 1.0 / (deg_u ** 0.5)
            d_i_inv_sqrt = 1.0 / (deg_i ** 0.5)
            R_csr = (
                sp.diags(d_u_inv_sqrt) @ R_csr @ sp.diags(d_i_inv_sqrt)
            ).astype("float32").tocsr()
        R = R_csr.tocoo()
        device = self._fc_device()
        indices = torch.stack([
            torch.as_tensor(R.row, dtype=torch.long),
            torch.as_tensor(R.col, dtype=torch.long),
        ])
        values = torch.as_tensor(R.data, dtype=torch.float32)
        self._R = torch.sparse_coo_tensor(indices, values, size=R.shape).coalesce().to(device)
        # Transpose: items as rows for item encoder.
        Rt = R.T.tocoo()
        indices_t = torch.stack([
            torch.as_tensor(Rt.row, dtype=torch.long),
            torch.as_tensor(Rt.col, dtype=torch.long),
        ])
        values_t = torch.as_tensor(Rt.data, dtype=torch.float32)
        self._R_t = torch.sparse_coo_tensor(indices_t, values_t, size=Rt.shape).coalesce().to(device)

    def _fc_device(self) -> torch.device:
        """Return the device of the FC params (used to place R buffers)."""
        return self.W_U.device

    def snapshot_fc_to_table(self) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
        """Convert the FC encoder into free nn.Embedding tables, then switch
        the model to init_mode='table'.

        Implements the literal "FC is initialization" reading of HOUR §4.2:
        the FC produces a structured starting point, then each user/item gets
        its own free embedding row that can drift independently.

        Steps:
          1. Compute current FC outputs X_U^(0), X_I^(0) under no_grad.
          2. Create fresh nn.Embedding tables and copy those values in.
          3. Drop W_U, W_I, b_U, b_I (set to None — caller is responsible for
             rebuilding the optimizer over remaining parameters).
          4. Clear the sparse R buffers (no longer needed).
          5. Flip self.init_mode to 'table'.

        Returns the new (user_embedding, item_embedding) parameters so the
        caller can register them with the optimizer.

        Requires init_mode='fc' and set_interaction_matrix to have been called.
        """
        if self.init_mode != "fc":
            raise RuntimeError(
                f"snapshot_fc_to_table requires init_mode='fc', got {self.init_mode!r}"
            )
        if self._R is None:
            raise RuntimeError(
                "snapshot_fc_to_table requires the interaction matrix to be set."
            )

        # Step 1: compute the FC outputs once, on the model's device.
        with torch.no_grad():
            X_U, X_I = self._compute_node_embeddings()
            X_U = X_U.detach().clone()
            X_I = X_I.detach().clone()

        device = X_U.device

        # Step 2: create fresh tables and copy values in.
        new_user = nn.Embedding(self.n_users, self.embed_dim).to(device)
        new_item = nn.Embedding(self.n_items, self.embed_dim).to(device)
        with torch.no_grad():
            new_user.weight.copy_(X_U)
            new_item.weight.copy_(X_I)

        # Step 3: register them on the module and clear the FC params.
        # Register first so PyTorch sees them as parameters (so the caller's
        # `model.parameters()` will include them).
        self.user_embedding = new_user
        self.item_embedding = new_item
        # Detach W_U/W_I/b_U/b_I from the module so they aren't kept in
        # parameters() and don't accumulate grad.
        for name in ("W_U", "W_I", "b_U", "b_I"):
            if getattr(self, name, None) is not None:
                # Replace the Parameter with None and remove from _parameters.
                if name in self._parameters:
                    del self._parameters[name]
                setattr(self, name, None)

        # Step 4: drop the sparse R buffers.
        self._R = None
        self._R_t = None

        # Step 5: flip the source.
        self.init_mode = "table"

        return self.user_embedding.weight, self.item_embedding.weight

    def _compute_node_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (X_U, X_I) for the current init_mode.

        - table: lookup the learnable tables.
        - fc:    ReLU(W_U R_u + b_U) and ReLU(W_I R_i + b_I), recomputed each
                 forward pass so gradients flow through R into W_U, W_I.
        """
        if self.init_mode == "table":
            return self.user_embedding.weight, self.item_embedding.weight
        if self._R is None or self._R_t is None:
            raise RuntimeError(
                "FC init requires the interaction matrix. Call "
                "model.set_interaction_matrix(R) after moving the model to its "
                "target device."
            )
        # W_U has shape (d, |I|); want X_U of shape (|U|, d) = ReLU(R @ W_U.T + b)
        X_U = torch.relu(torch.sparse.mm(self._R, self.W_U.t()) + self.b_U)
        X_I = torch.relu(torch.sparse.mm(self._R_t, self.W_I.t()) + self.b_I)
        return X_U, X_I

    def get_ego_embeddings(self) -> torch.Tensor:
        """Concatenate user and item layer-0 embeddings: (n_users + n_items, embed_dim).

        For init_mode='table' this looks up the embedding tables. For
        init_mode='fc' this computes the embeddings from the interaction matrix.
        """
        X_U, X_I = self._compute_node_embeddings()
        return torch.cat([X_U, X_I], dim=0)

    def layer_combination(self, layer_embs: list[torch.Tensor]) -> torch.Tensor:
        """Combine embeddings from all layers. Default: uniform mean."""
        return torch.stack(layer_embs, dim=0).mean(dim=0)

    @abstractmethod
    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Run message passing and return final (user_embs, item_embs).

        Returns:
            user_embs: (n_users, embed_dim)
            item_embs: (n_items, embed_dim)
        """
        ...

    def _ego_user(self, users: torch.Tensor) -> torch.Tensor:
        """Layer-0 (pre-propagation) embedding for the batch's users.

        Used by the BPR reg term. Source matches init_mode:
          - table: lookup row of user_embedding.
          - fc:    ReLU(W_U R_u + b_U) — the FC output for those users.
        """
        if self.init_mode == "table":
            return self.user_embedding(users)
        X_U, _ = self._compute_node_embeddings()
        return X_U[users]

    def _ego_item(self, items: torch.Tensor) -> torch.Tensor:
        """Layer-0 (pre-propagation) embedding for the batch's items."""
        if self.init_mode == "table":
            return self.item_embedding(items)
        _, X_I = self._compute_node_embeddings()
        return X_I[items]

    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """BPR forward pass.

        Args:
            users: (batch,) user indices.
            pos_items: (batch,) positive item indices.
            neg_items: (batch,) negative item indices.

        Returns:
            user_e: (batch, embed_dim) final user embeddings.
            pos_e: (batch, embed_dim) final positive item embeddings.
            neg_e: (batch, embed_dim) final negative item embeddings.
            reg_loss: L2 regularization on the *initial* (ego/layer-0) embeddings.
                For init_mode='table' this is the looked-up table row; for
                init_mode='fc' this is the FC output — the same role in both,
                just sourced differently.
        """
        user_all, item_all = self.propagate()

        user_e = user_all[users]
        pos_e = item_all[pos_items]
        neg_e = item_all[neg_items]

        # Regularization on layer-0 embeddings. In FC mode the "ego" embedding
        # is no longer a free parameter row but the FC output for that user/item.
        ego_user = self._ego_user(users)
        ego_pos = self._ego_item(pos_items)
        ego_neg = self._ego_item(neg_items)
        reg_loss = (
            ego_user.norm(2).pow(2)
            + ego_pos.norm(2).pow(2)
            + ego_neg.norm(2).pow(2)
        ) / (2 * users.shape[0])

        return user_e, pos_e, neg_e, reg_loss

    def get_all_ratings(self, users: torch.Tensor) -> torch.Tensor:
        """Compute scores for all items for given users.

        Args:
            users: (batch,) user indices.

        Returns:
            scores: (batch, n_items) predicted preference scores.
        """
        user_all, item_all = self.propagate()
        user_e = user_all[users]
        scores = user_e @ item_all.T
        return scores
