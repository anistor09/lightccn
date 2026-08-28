"""LightCCN-Multi: Full 7-operator cell complex propagation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from light_ccn.models.base import BaseCFModel
from light_ccn.models import register_model

_VALID_WEIGHT_MODES = ("softmax", "signed", "tanh", "softplus")
_VALID_PROP_MODES = (
    "derived_e",      # rank-1 only, edges recomputed each layer (legacy nodes_only)
    "derived_ef",     # + within-layer face round-trip (cascade), parameter-free
    "stateful_e",     # edge state carried across layers (cascade), no faces
    "stateful_ef",    # edge+face state carried (cascade), parameter-free
    "full_multi",     # stateful_ef + learnable edge/face tables
)

# Per-granularity number of weight groups for each cell type.
# - 'global'    : single shared weight per channel
# - 'type'      : node weights split by user/item (2 node groups). Edges/faces unchanged.
# - 'freq'      : all cell types bucketed by degree into 3 buckets (tail/torso/head).
# - 'freq_type' : nodes split by user/item AND bucket (6 node groups). Edges/faces by bucket only.
# - 'per_cell'  : finest possible — ONE learnable weight per individual cell.
#                 n_groups == n_cells and the cell->group map is the identity.
#                 The group counts are dynamic (depend on the complex), so this
#                 granularity is resolved specially in __init__, not via the
#                 static dicts below.
_VALID_GRANULARITIES = ("global", "type", "freq", "freq_type", "per_cell")
_GRAN_NODE_GROUPS = {"global": 1, "type": 2, "freq": 3, "freq_type": 6}
_GRAN_EDGE_FACE_GROUPS = {"global": 1, "type": 1, "freq": 3, "freq_type": 3}

# Higher-order READOUT modes (applied AFTER propagation, on the final node
# embeddings, before scoring). They inject the cell-complex structure directly
# into the user/item vectors used for the dot product, instead of (or on top
# of) propagation.
#   - 'none'    : no readout (the model is exactly as before).
#   - 'ho_item' : (A) augment the ITEM tower with its pooled edge/face cells.
#   - 'ho_user' : (B) augment the USER tower with the pooled cells of the
#                 items the user interacted with (one A_hat_0 hop).
#   - 'ho_both' : (A+B) augment both towers.
# Channels are rank-matched to the backbone: '*_e' modes inject EDGES only
# (gate beta); '*_ef' / full_multi inject EDGES + FACES (gates beta, gamma).
_VALID_READOUT_MODES = ("none", "ho_item", "ho_user", "ho_both")

# Where the readout takes its cell embeddings from:
#   - 'combined': re-derive edge/face summaries from the layer-combined node
#                 embeddings (E = B1up @ combined, Fc = B2up @ E).
#   - 'state'   : use the carried edge/face states of the LAST propagation
#                 layer (stateful modes only; faces fall back to derivation
#                 when no face state is carried).
_VALID_READOUT_SOURCES = ("combined", "state")
# Propagation modes whose readout also gets the FACE channel (the "_ef" family).
_READOUT_FACE_MODES = ("derived_ef", "stateful_ef", "full_multi")


@register_model("lightccn_multi")
class LightCCNMulti(BaseCFModel):
    """LightCCN-Multi model with full 7-operator formulation.

    Three-level propagation on the cell complex:
    - Nodes (users + items): propagate via node-node adj and edge->node incidence
    - Edges (item-item edges): propagate via node->edge, edge-edge adj, face->edge
    - Faces (item triples): propagate via edge->face, face-face adj

    Per-layer update equations (from spec Eqs 45-47):
        x_nodes^{k+1} = w1 * A_hat_0 @ x_nodes^k + w2 * B_hat_1_down @ x_edges^k
        x_edges^{k+1} = w3 * B_hat_1_up @ x_nodes^k + w4 * A_hat_1 @ x_edges^k + w5 * B_hat_2_down @ x_faces^k
        x_faces^{k+1} = w6 * B_hat_2_up @ x_edges^k + w7 * A_hat_2 @ x_faces^k

    Weight modes:
    - "softmax": sum-to-1 per group via softmax (default)
    - "signed": unconstrained independent scalars
    - "tanh": bounded to [-1, 1]
    - "softplus": positive, unbounded

    nodes_only mode: edge/face states are deduced from node embeddings each layer
    (no trainable edge/face embeddings).
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        n_layers: int = 3,
        n_edges: int = 0,
        n_faces: int = 0,
        edge_embed_dim: int = 64,
        face_embed_dim: int = 64,
        operators: dict[str, torch.Tensor] | None = None,
        weight_mode: str = "softmax",
        nodes_only: bool = False,
        propagation_mode: str | None = None,
        edge_face_self_loop: bool = False,
        init_mode: str = "table",
        fc_normalize_R: bool = False,
        freeze_topology_channel: bool = False,
        weight_granularity: str = "global",
        readout_mode: str = "none",
        readout_source: str = "combined",
        # Backward compat
        signed_weights: bool = False,
    ):
        super().__init__(
            n_users, n_items, embed_dim,
            init_mode=init_mode, fc_normalize_R=fc_normalize_R,
        )
        self.n_layers = n_layers
        self.n_edges = n_edges
        self.n_faces = n_faces
        self.edge_embed_dim = edge_embed_dim
        self.face_embed_dim = face_embed_dim
        self.edge_face_self_loop = edge_face_self_loop
        self.weight_reg: float = 0.0

        # Resolve propagation mode (backward compat with the nodes_only flag).
        if propagation_mode is None:
            propagation_mode = "derived_e" if nodes_only else "full_multi"
        if propagation_mode not in _VALID_PROP_MODES:
            raise ValueError(
                f"propagation_mode must be one of {_VALID_PROP_MODES}, got {propagation_mode!r}"
            )
        self.propagation_mode = propagation_mode
        # Only full_multi carries learnable edge/face tables; every other mode
        # derives the higher-rank cochains from the nodes (parameter-free).
        self.nodes_only = propagation_mode != "full_multi"

        # Resolve weight mode (backward compat)
        if signed_weights:
            weight_mode = "signed"
        if weight_mode not in _VALID_WEIGHT_MODES:
            raise ValueError(f"weight_mode must be one of {_VALID_WEIGHT_MODES}, got {weight_mode!r}")
        self.weight_mode = weight_mode

        # Weight granularity: how the 7 mixing scalars are split into sub-groups.
        if weight_granularity not in _VALID_GRANULARITIES:
            raise ValueError(
                f"weight_granularity must be one of {_VALID_GRANULARITIES}, got {weight_granularity!r}"
            )
        self.weight_granularity = weight_granularity
        if weight_granularity == "per_cell":
            # Finest granularity: one weight per individual cell. n_groups equals
            # the number of cells of that rank; the cell->group map is identity.
            #
            # Dead-param guard: only the cell-types whose mixing weights actually
            # appear in THIS propagation mode's update are inflated to per-cell.
            # Cell-types whose weights are never read stay at a single (global)
            # scalar, so we don't allocate (and never train) thousands of dead
            # per-cell weights. Live cell-types per mode:
            #   derived_e            -> node only        (w1,w2)
            #   derived_ef/stateful_e-> node + edge      (faces never weighted)
            #   stateful_ef/full_multi-> node + edge + face
            # The guard is at cell-TYPE granularity (not per-channel), so the
            # hard-coded w_edge[i]/w_face[i] indexing in the step methods is
            # untouched.
            edge_w_live = propagation_mode != "derived_e"
            face_w_live = propagation_mode in ("stateful_ef", "full_multi")
            self._percell_active = {"node": True, "edge": edge_w_live, "face": face_w_live}
            self.n_node_groups: int = n_users + n_items
            self.n_edge_groups: int = max(n_edges, 1) if edge_w_live else 1
            self.n_face_groups: int = max(n_faces, 1) if face_w_live else 1
        else:
            self._percell_active = {"node": False, "edge": False, "face": False}
            self.n_node_groups = _GRAN_NODE_GROUPS[weight_granularity]
            self.n_edge_groups = _GRAN_EDGE_FACE_GROUPS[weight_granularity]
            self.n_face_groups = _GRAN_EDGE_FACE_GROUPS[weight_granularity]

        # Per-cell group index buffers (default: group 0 for every cell, which
        # exactly reproduces the global behavior). `set_cell_buckets` populates
        # these from raw degrees when granularity is a bucket scheme; for
        # 'per_cell' we fill them with the identity map right here.
        self.register_buffer(
            "node_group_idx", torch.zeros(n_users + n_items, dtype=torch.long)
        )
        self.register_buffer(
            "edge_group_idx", torch.zeros(max(n_edges, 1), dtype=torch.long)
        )
        self.register_buffer(
            "face_group_idx", torch.zeros(max(n_faces, 1), dtype=torch.long)
        )
        if weight_granularity == "per_cell":
            # identity map only for the cell-types that are inflated to per-cell;
            # inactive types keep n_groups == 1, so their group_idx is ignored
            # (_expand_to_cells short-circuits on n_groups == 1).
            self.node_group_idx.copy_(torch.arange(n_users + n_items, dtype=torch.long))
            if self._percell_active["edge"]:
                self.edge_group_idx.copy_(torch.arange(max(n_edges, 1), dtype=torch.long))
            if self._percell_active["face"]:
                self.face_group_idx.copy_(torch.arange(max(n_faces, 1), dtype=torch.long))

        # Edge/face embeddings: only full_multi has learnable tables.
        learnable_cochains = propagation_mode == "full_multi"
        if learnable_cochains and n_edges > 0:
            self.edge_embedding: nn.Embedding | None = nn.Embedding(n_edges, edge_embed_dim)
            nn.init.normal_(self.edge_embedding.weight, std=0.1)
        else:
            self.edge_embedding = None

        if learnable_cochains and n_faces > 0:
            self.face_embedding: nn.Embedding | None = nn.Embedding(n_faces, face_embed_dim)
            nn.init.normal_(self.face_embedding.weight, std=0.1)
        else:
            self.face_embedding = None

        # Cross-level projection layers (needed when dims differ)
        self.proj_edge_to_node: nn.Linear | None = None
        self.proj_node_to_edge: nn.Linear | None = None
        self.proj_face_to_edge: nn.Linear | None = None
        self.proj_edge_to_face: nn.Linear | None = None

        if edge_embed_dim != embed_dim:
            self.proj_edge_to_node = nn.Linear(edge_embed_dim, embed_dim, bias=False)
            self.proj_node_to_edge = nn.Linear(embed_dim, edge_embed_dim, bias=False)

        if face_embed_dim != edge_embed_dim:
            self.proj_face_to_edge = nn.Linear(face_embed_dim, edge_embed_dim, bias=False)
            self.proj_edge_to_face = nn.Linear(edge_embed_dim, face_embed_dim, bias=False)

        # Learnable mixing weights
        self._init_mixing_weights(weight_mode)

        # Ablation: freeze the topology channel (w_2 in the node update) to zero
        # so the node update reduces to w_1 * A_hat_0 @ X — i.e. pure LightGCN-
        # style propagation. Used to isolate the contribution of the rank-2
        # cell-complex topology from the contribution of node-level FC init.
        #
        # Only meaningful (and supported) under weight_mode='signed', where
        # weights ARE the parameters and can be pinned to exact values.
        # Other modes (softmax/tanh/softplus) compute weights from logits via
        # nonlinearities and cannot be cleanly fixed at zero.
        self.freeze_topology_channel = freeze_topology_channel
        if freeze_topology_channel:
            if weight_mode != "signed":
                raise ValueError(
                    "freeze_topology_channel=True requires weight_mode='signed' "
                    f"(got {weight_mode!r}). Other modes compute weights via "
                    "nonlinearities (softmax/tanh/softplus) and cannot pin w_2 "
                    "to exactly zero."
                )
            with torch.no_grad():
                # node_weights is (2, G_n): pin all groups to [1.0, 0.0].
                self.node_weights.data[0, :] = 1.0
                self.node_weights.data[1, :] = 0.0
            # Freeze so the optimizer never moves them. We freeze the whole
            # node_weights tensor; the topology contribution is permanently
            # disabled and the same-level term is held at 1.0 throughout.
            self.node_weights.requires_grad_(False)

        # Store operators
        self.operators = operators or {}

        # ---- Higher-order readout (A / B / A+B) ----------------------------
        # Applied after propagation, on the final node embeddings, before
        # scoring. Parameter-free cell pooling (reuses the B_hat operators) +
        # one or two learned scalar gates. Channels are rank-matched to the
        # backbone (edges always; faces only for the "_ef" family).
        if readout_mode not in _VALID_READOUT_MODES:
            raise ValueError(
                f"readout_mode must be one of {_VALID_READOUT_MODES}, got {readout_mode!r}"
            )
        if readout_source not in _VALID_READOUT_SOURCES:
            raise ValueError(
                f"readout_source must be one of {_VALID_READOUT_SOURCES}, got {readout_source!r}"
            )
        if (
            readout_source == "state"
            and readout_mode != "none"
            and propagation_mode in ("derived_e", "derived_ef")
        ):
            raise ValueError(
                "readout_source='state' needs carried cell states; "
                f"propagation_mode {propagation_mode!r} does not carry any"
            )
        self.readout_mode = readout_mode
        self.readout_source = readout_source
        self.readout_faces = (
            readout_mode != "none"
            and propagation_mode in _READOUT_FACE_MODES
            and n_faces > 0
        )
        if readout_mode != "none":
            # Gates initialise at 0 so the model starts exactly at the
            # no-readout baseline and learns to open the channel if useful.
            # Signed scalars (can go negative — the model may subtract topology).
            self.readout_beta = nn.Parameter(torch.zeros(1))   # edge channel
            self.readout_gamma = (
                nn.Parameter(torch.zeros(1)) if self.readout_faces else None
            )
            # Mask used to apply the user-side (B) augmentation to user rows only.
            user_mask = torch.zeros(n_users + n_items, 1)
            user_mask[:n_users] = 1.0
            self.register_buffer("_user_row_mask", user_mask)
        else:
            self.readout_beta = None
            self.readout_gamma = None

    def _init_mixing_weights(self, weight_mode: str) -> None:
        """Initialize weight parameters based on mode.

        When ``edge_face_self_loop`` is enabled, the edge group grows by one
        scalar (w8: edge<-edge self) and the face group by one (w9: face<-face
        self). For softmax these enter the same softmax simplex as the other
        channels in their group; for tanh/softplus/signed they're independent.

        Parameter tensors are always shape ``(n_channels, n_groups)`` where
        ``n_groups`` depends on ``weight_granularity``. ``n_groups == 1``
        recovers the original scalar-per-channel behavior.
        """
        n_edge = 4 if self.edge_face_self_loop else 3
        n_face = 3 if self.edge_face_self_loop else 2
        G_n, G_e, G_f = self.n_node_groups, self.n_edge_groups, self.n_face_groups
        if weight_mode == "signed":
            self.node_weights = nn.Parameter(torch.full((2, G_n), 0.5))
            self.edge_weights = nn.Parameter(torch.full((n_edge, G_e), 1.0 / n_edge))
            self.face_weights = nn.Parameter(torch.full((n_face, G_f), 1.0 / n_face))
        elif weight_mode == "tanh":
            # atanh(1/N) so initial effective weights are uniform positive
            self.node_logits = nn.Parameter(torch.full((2, G_n), math.atanh(0.5)))
            self.edge_logits = nn.Parameter(torch.full((n_edge, G_e), math.atanh(1.0 / n_edge)))
            self.face_logits = nn.Parameter(torch.full((n_face, G_f), math.atanh(1.0 / n_face)))
        else:
            # softmax (zeros → uniform per group) and softplus (softplus(0)=ln2, all equal)
            self.node_logits = nn.Parameter(torch.zeros(2, G_n))
            self.edge_logits = nn.Parameter(torch.zeros(n_edge, G_e))
            self.face_logits = nn.Parameter(torch.zeros(n_face, G_f))

    def _raw_per_group_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-group effective weights, shape ``(n_channels, n_groups)``.

        Softmax is applied along dim=0 → each group column is its own simplex.
        """
        if self.weight_mode == "signed":
            return self.node_weights, self.edge_weights, self.face_weights
        if self.weight_mode == "softmax":
            return (F.softmax(self.node_logits, dim=0),
                    F.softmax(self.edge_logits, dim=0),
                    F.softmax(self.face_logits, dim=0))
        if self.weight_mode == "tanh":
            return (torch.tanh(self.node_logits),
                    torch.tanh(self.edge_logits),
                    torch.tanh(self.face_logits))
        if self.weight_mode == "softplus":
            return (F.softplus(self.node_logits),
                    F.softplus(self.edge_logits),
                    F.softplus(self.face_logits))
        raise ValueError(f"Unknown weight_mode: {self.weight_mode}")

    @staticmethod
    def _expand_to_cells(w: torch.Tensor, idx: torch.Tensor, n_groups: int) -> torch.Tensor:
        """Expand per-group weights to per-cell.

        Returns ``(n_channels,)`` (0-d per-channel scalars after [i] indexing)
        when ``n_groups == 1`` and ``(n_channels, n_cells)`` otherwise.
        """
        if n_groups == 1:
            return w.squeeze(-1)
        return w.index_select(dim=1, index=idx)

    def _get_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute effective weights from raw parameters, expanded to per-cell.

        Returns ``(w_node, w_edge, w_face)``. Each is shape:
          - ``(n_channels,)`` when its granularity is global (``n_groups == 1``)
          - ``(n_channels, n_cells)`` otherwise

        Step methods consume ``w[i]`` (a 0-d scalar or 1-d per-cell vector)
        via :py:meth:`_mix`.
        """
        wn, we, wf = self._raw_per_group_weights()
        wn = self._expand_to_cells(wn, self.node_group_idx, self.n_node_groups)
        we = self._expand_to_cells(we, self.edge_group_idx, self.n_edge_groups)
        wf = self._expand_to_cells(wf, self.face_group_idx, self.n_face_groups)
        return wn, we, wf

    @staticmethod
    def _mix(w_ch: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Multiply a channel weight ``w_ch`` by an embedding tensor ``X``.

        ``w_ch`` is a 0-d scalar (global granularity) or a 1-d per-cell vector
        of shape ``(X.shape[0],)`` (granular). Broadcasts to ``X``'s feature
        dimension.
        """
        if w_ch.dim() == 0:
            return w_ch * X
        return w_ch.unsqueeze(-1) * X

    # ---- bucket assignment ----
    @staticmethod
    def _bucketize(degrees: torch.Tensor, n_buckets: int = 3,
                   boundaries: tuple[float, float] = (0.2, 0.8)) -> torch.Tensor:
        """Rank-bucket cells by degree: 0=tail (bottom 20%), 1=torso (middle 60%),
        2=head (top 20%). Returns int64 indices of shape ``(n,)``."""
        n = degrees.numel()
        if n == 0:
            return torch.zeros(0, dtype=torch.long)
        order = degrees.argsort()
        bucket = torch.zeros(n, dtype=torch.long)
        tail_end = int(boundaries[0] * n)
        head_start = int(boundaries[1] * n)
        bucket[order[:tail_end]] = 0
        bucket[order[tail_end:head_start]] = 1
        bucket[order[head_start:]] = 2
        return bucket

    def set_cell_buckets(
        self,
        node_degrees: torch.Tensor | None = None,
        edge_degrees: torch.Tensor | None = None,
        face_degrees: torch.Tensor | None = None,
    ) -> None:
        """Populate per-cell group indices for ``weight_granularity != 'global'``.

        Args:
            node_degrees: shape ``(n_users + n_items,)``, ordered [users, items].
                Required for ``freq`` and ``freq_type``. Ignored for ``type``.
            edge_degrees: shape ``(n_edges,)``. Required for ``freq`` and
                ``freq_type``.
            face_degrees: shape ``(n_faces,)``. Required for ``freq`` and
                ``freq_type`` when faces exist.
        """
        g = self.weight_granularity
        if g == "global":
            return  # zeros already
        if g == "per_cell":
            return  # identity map already filled in __init__ (no degrees needed)

        device = self.node_group_idx.device

        if g == "type":
            idx = torch.zeros(self.n_users + self.n_items, dtype=torch.long)
            idx[self.n_users:] = 1
            self.node_group_idx.copy_(idx.to(device))
            return

        if node_degrees is None or edge_degrees is None:
            raise ValueError(
                f"weight_granularity={g!r} requires node_degrees and edge_degrees"
            )

        if g == "freq":
            self.node_group_idx.copy_(
                self._bucketize(node_degrees, n_buckets=3).to(device)
            )
        elif g == "freq_type":
            u_deg = node_degrees[: self.n_users]
            i_deg = node_degrees[self.n_users :]
            b_u = self._bucketize(u_deg, n_buckets=3)          # 0..2 (user buckets)
            b_i = self._bucketize(i_deg, n_buckets=3) + 3      # 3..5 (item buckets)
            self.node_group_idx.copy_(torch.cat([b_u, b_i]).to(device))

        # Edges/faces use raw degree buckets in both 'freq' and 'freq_type'.
        self.edge_group_idx.copy_(
            self._bucketize(edge_degrees, n_buckets=3).to(self.edge_group_idx.device)
        )
        if face_degrees is not None and self.n_faces > 0:
            self.face_group_idx.copy_(
                self._bucketize(face_degrees, n_buckets=3).to(self.face_group_idx.device)
            )

    def set_operators(self, operators: dict[str, torch.Tensor]) -> None:
        self.operators = operators

    def bootstrap_higher_rank_init(self) -> None:
        """F2 fix: replace random init of H_1, H_2 with boundary aggregates.

        Sets:
            H_1^(0) = B_hat_1_up @ X^(0)
            H_2^(0) = B_hat_2_up @ H_1^(0)

        Requires operators to be set and edge/face embeddings to exist
        (nodes_only=False with n_edges > 0). No-op otherwise.
        """
        if self.nodes_only:
            return
        if "B_hat_1_up" not in self.operators:
            return

        with torch.no_grad():
            x_nodes = self.get_ego_embeddings()

            if self.edge_embedding is not None and self.n_edges > 0:
                x_edges = torch.sparse.mm(self.operators["B_hat_1_up"], x_nodes)
                if self.proj_node_to_edge is not None:
                    x_edges = self.proj_node_to_edge(x_edges)
                # Copy values into the learnable parameter
                self.edge_embedding.weight.data.copy_(x_edges)

                if self.face_embedding is not None and self.n_faces > 0 and "B_hat_2_up" in self.operators:
                    x_faces = torch.sparse.mm(self.operators["B_hat_2_up"], x_edges)
                    if self.proj_edge_to_face is not None:
                        x_faces = self.proj_edge_to_face(x_faces)
                    self.face_embedding.weight.data.copy_(x_faces)

    def get_attention_weights(self) -> dict[str, dict[str, float]]:
        """Legacy summary: scalar (group-mean) weight per channel.

        Under granular modes the per-group weights differ; use
        :py:meth:`get_weights_per_group` for the full breakdown.
        """
        with torch.no_grad():
            wn, we, wf = self._raw_per_group_weights()
            wn = wn.mean(dim=1).cpu().tolist()  # (n_channels,)
            we = we.mean(dim=1).cpu().tolist()
            wf = wf.mean(dim=1).cpu().tolist()
        out = {
            "node": {"w1_same_level": wn[0], "w2_from_edges": wn[1]},
            "edge": {"w3_from_nodes": we[0], "w4_same_level": we[1], "w5_from_faces": we[2]},
            "face": {"w6_from_edges": wf[0], "w7_same_level": wf[1]},
        }
        if self.edge_face_self_loop:
            out["edge"]["w8_self_loop"] = we[3]
            out["face"]["w9_self_loop"] = wf[2]
        if self.readout_mode != "none":
            out["readout"] = self.get_readout_gates()
        return out

    def get_weights_per_group(self) -> dict[str, object]:
        """Per-group weight matrix for plotting / observability.

        Returns a dict with raw lists shaped ``(n_channels, n_groups)`` for
        each cell level, plus the granularity string. For ``global`` each
        list has a single column equal to the scalar weights.
        """
        with torch.no_grad():
            wn, we, wf = self._raw_per_group_weights()
        return {
            "node": wn.cpu().tolist(),
            "edge": we.cpu().tolist(),
            "face": wf.cpu().tolist(),
            "granularity": self.weight_granularity,
            "n_node_groups": self.n_node_groups,
            "n_edge_groups": self.n_edge_groups,
            "n_face_groups": self.n_face_groups,
        }

    # ---- projection helpers (no-op when dims already match) ----
    def _to_edge(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_node_to_edge(x) if self.proj_node_to_edge is not None else x

    def _to_node(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_edge_to_node(x) if self.proj_edge_to_node is not None else x

    def _to_face(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_edge_to_face(x) if self.proj_edge_to_face is not None else x

    def _face_to_edge(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_face_to_edge(x) if self.proj_face_to_edge is not None else x

    def _mm(self, key: str, x: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(self.operators[key], x)

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        w_node, w_edge, w_face = self._get_weights()
        x_nodes = self.get_ego_embeddings()

        has_edges = self.n_edges > 0
        has_faces = self.n_faces > 0
        mode = self.propagation_mode

        # Seed carried edge/face states (stateful + full_multi only).
        x_edges = None
        x_faces = None
        if mode in ("stateful_e", "stateful_ef") and has_edges:
            x_edges = self._to_edge(self._mm("B_hat_1_up", x_nodes))
            if mode == "stateful_ef" and has_faces:
                x_faces = self._to_face(self._mm("B_hat_2_up", x_edges))
        elif mode == "full_multi":
            if self.edge_embedding is not None:
                x_edges = self.edge_embedding.weight
            if self.face_embedding is not None:
                x_faces = self.face_embedding.weight

        all_node_embs = [x_nodes]

        for _ in range(self.n_layers):
            if mode == "derived_e":
                x_nodes = self._step_derived_e(x_nodes, w_node, has_edges)
            elif mode == "derived_ef":
                x_nodes = self._step_derived_ef(x_nodes, w_node, w_edge, has_edges, has_faces)
            elif mode == "stateful_e":
                x_nodes, x_edges = self._step_stateful_e(x_nodes, x_edges, w_node, w_edge, has_edges)
            else:  # stateful_ef or full_multi (identical equations)
                x_nodes, x_edges, x_faces = self._step_stateful_ef(
                    x_nodes, x_edges, x_faces, w_node, w_edge, w_face, has_edges, has_faces
                )
            all_node_embs.append(x_nodes)

        combined = self.layer_combination(all_node_embs)
        combined = self._apply_readout(combined, x_edges, x_faces)
        user_embs = combined[:self.n_users]
        item_embs = combined[self.n_users:]
        return user_embs, item_embs

    def _apply_readout(
        self,
        combined: torch.Tensor,
        x_edges: torch.Tensor | None = None,
        x_faces: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Higher-order readout (A / B / A+B): inject pooled cell embeddings into
        the final node embeddings, before scoring.

        Returns ``combined`` unchanged when ``readout_mode == 'none'``.

        Cell pools (parameter-free, reuse the boundary operators). With
        ``readout_source == 'combined'`` (default) the cell summaries are
        re-derived from the layer-combined node embeddings:
            E       = B_hat_1_up   @ X        edge embeddings (node-mean of endpoints)
            p_E     = B_hat_1_down @ E        item-side edge pool  (user rows are 0)
            Fc      = B_hat_2_up   @ E        face embeddings (from edges)
            p_F     = B_hat_1_down @ B_hat_2_down @ Fc   item-side face pool
        With ``readout_source == 'state'`` the summaries are instead the carried
        edge/face states of the LAST propagation layer (``x_edges``/``x_faces``),
        pooled down through the same boundary operators; a missing face state
        falls back to derivation from the edge summaries.

        Item-tower (A): item rows += beta*p_E (+ gamma*p_F).  Because p_E/p_F are
        structurally zero on user rows, adding them to the full tensor only
        changes items.
        User-tower (B): bring the item-side pools to the users that engaged
        those items via one A_hat_0 hop, then add to USER rows only (mask).
        """
        if self.readout_mode == "none" or self.n_edges <= 0:
            return combined
        if "B_hat_1_up" not in self.operators:
            return combined

        use_state = self.readout_source == "state"
        if use_state and x_edges is not None:
            E = x_edges                                              # (n_edges, d) carried states
        else:
            E = self._to_edge(self._mm("B_hat_1_up", combined))      # (n_edges, d)
        p_E = self._to_node(self._mm("B_hat_1_down", E))             # (N, d), user rows = 0
        p_F = None
        if self.readout_faces and self.n_faces > 0 and "B_hat_2_up" in self.operators:
            if use_state and x_faces is not None:
                Fc = x_faces                                        # (n_faces, d) carried states
            else:
                Fc = self._to_face(self._mm("B_hat_2_up", E))       # (n_faces, d)
            e_from_f = self._face_to_edge(self._mm("B_hat_2_down", Fc))  # (n_edges, d)
            p_F = self._to_node(self._mm("B_hat_1_down", e_from_f)) # (N, d), user rows = 0

        out = combined

        # ---- (A) item-tower augmentation ----
        if self.readout_mode in ("ho_item", "ho_both"):
            add = self.readout_beta * p_E
            if p_F is not None:
                add = add + self.readout_gamma * p_F
            out = out + add  # only item rows are non-zero in p_E / p_F

        # ---- (B) user-tower augmentation ----
        if self.readout_mode in ("ho_user", "ho_both"):
            u_add = self.readout_beta * self._mm("A_hat_0", p_E)
            if p_F is not None:
                u_add = u_add + self.readout_gamma * self._mm("A_hat_0", p_F)
            out = out + u_add * self._user_row_mask  # restrict to user rows

        return out

    def get_readout_gates(self) -> dict[str, float | None]:
        """Current readout gate values (None when inactive), for logging."""
        if self.readout_mode == "none":
            return {"readout_mode": "none", "beta": None, "gamma": None}
        with torch.no_grad():
            beta = float(self.readout_beta.item())
            gamma = float(self.readout_gamma.item()) if self.readout_gamma is not None else None
        return {"readout_mode": self.readout_mode, "readout_source": self.readout_source,
                "beta": beta, "gamma": gamma}

    # ---- per-layer steps (all cascade: faces -> edges -> nodes) ----

    def _step_derived_e(self, x_nodes, w_node, has_edges):
        """X' = w1 A0 X + w2 B1down B1up X. Edges recomputed, faces unused."""
        node_from_node = self._mm("A_hat_0", x_nodes)
        if not has_edges:
            return node_from_node
        E = self._to_edge(self._mm("B_hat_1_up", x_nodes))
        node_from_edge = self._to_node(self._mm("B_hat_1_down", E))
        return self._mix(w_node[0], node_from_node) + self._mix(w_node[1], node_from_edge)

    def _step_derived_ef(self, x_nodes, w_node, w_edge, has_edges, has_faces):
        """Within-layer cascade: nodes -> edges -> faces -> edges -> nodes."""
        node_from_node = self._mm("A_hat_0", x_nodes)
        if not has_edges:
            return node_from_node
        e0 = self._to_edge(self._mm("B_hat_1_up", x_nodes))          # edges from nodes
        edge = self._mix(w_edge[0], e0)
        if has_faces:
            f = self._to_face(self._mm("B_hat_2_up", e0))            # faces from edges
            edge = edge + self._mix(
                w_edge[2], self._face_to_edge(self._mm("B_hat_2_down", f))
            )
        node_from_edge = self._to_node(self._mm("B_hat_1_down", edge))
        return self._mix(w_node[0], node_from_node) + self._mix(w_node[1], node_from_edge)

    def _step_stateful_e(self, x_nodes, x_edges, w_node, w_edge, has_edges):
        """Edge state carried; node reads the freshly updated edges (cascade).

        With ``edge_face_self_loop``, an additional w8*x_edges self term keeps
        each edge's previous embedding (zero-diagonal A_hat_1 otherwise drops it).
        """
        node_from_node = self._mm("A_hat_0", x_nodes)
        if not has_edges:
            return node_from_node, x_edges
        edge_from_node = self._to_edge(self._mm("B_hat_1_up", x_nodes))
        edge_from_edge = self._mm("A_hat_1", x_edges)
        x_edges_new = self._mix(w_edge[0], edge_from_node) + self._mix(w_edge[1], edge_from_edge)
        if self.edge_face_self_loop:
            x_edges_new = x_edges_new + self._mix(w_edge[3], x_edges)
        node_from_edge = self._to_node(self._mm("B_hat_1_down", x_edges_new))
        x_nodes_new = self._mix(w_node[0], node_from_node) + self._mix(w_node[1], node_from_edge)
        return x_nodes_new, x_edges_new

    def _step_stateful_ef(self, x_nodes, x_edges, x_faces, w_node, w_edge, w_face, has_edges, has_faces):
        """Cascade faces -> edges -> nodes; edge/face state carried across layers.

        With ``edge_face_self_loop``, the edge update adds w8*x_edges and the
        face update adds w9*x_faces (anchoring each cell's own embedding,
        which the zero-diagonal A_hat_1/A_hat_2 operators otherwise drop).
        """
        node_from_node = self._mm("A_hat_0", x_nodes)
        if not has_edges:
            return node_from_node, x_edges, x_faces

        # Faces first (from old edges, old faces)
        x_faces_new = x_faces
        if has_faces and x_faces is not None:
            face_from_edge = self._to_face(self._mm("B_hat_2_up", x_edges))
            face_from_face = self._mm("A_hat_2", x_faces)
            x_faces_new = self._mix(w_face[0], face_from_edge) + self._mix(w_face[1], face_from_face)
            if self.edge_face_self_loop:
                x_faces_new = x_faces_new + self._mix(w_face[2], x_faces)

        # Edges next (old nodes, old edges, FRESH faces, + optional self)
        edge_from_node = self._to_edge(self._mm("B_hat_1_up", x_nodes))
        edge_from_edge = self._mm("A_hat_1", x_edges)
        x_edges_new = self._mix(w_edge[0], edge_from_node) + self._mix(w_edge[1], edge_from_edge)
        if has_faces and x_faces_new is not None:
            edge_from_face = self._face_to_edge(self._mm("B_hat_2_down", x_faces_new))
            x_edges_new = x_edges_new + self._mix(w_edge[2], edge_from_face)
        if self.edge_face_self_loop:
            x_edges_new = x_edges_new + self._mix(w_edge[3], x_edges)

        # Nodes last (old nodes, FRESH edges)
        node_from_edge = self._to_node(self._mm("B_hat_1_down", x_edges_new))
        x_nodes_new = self._mix(w_node[0], node_from_node) + self._mix(w_node[1], node_from_edge)
        return x_nodes_new, x_edges_new, x_faces_new

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

        # Node embedding regularization. Source matches init_mode:
        #   - "table": looked-up table rows for the batch
        #   - "fc":    FC-encoder output for the batch (the same vectors x_u
        #             that feed propagation), penalized as in HOUR / SoTA
        #             implicit-feedback CF.
        ego_user = self._ego_user(users)
        ego_pos = self._ego_item(pos_items)
        ego_neg = self._ego_item(neg_items)
        reg_loss = (
            ego_user.norm(2).pow(2)
            + ego_pos.norm(2).pow(2)
            + ego_neg.norm(2).pow(2)
        ) / (2 * users.shape[0])

        # Edge/face embedding regularization (only when trainable)
        if not self.nodes_only:
            n_higher = self.n_edges + self.n_faces
            if n_higher > 0:
                higher_reg = torch.zeros(1, device=reg_loss.device, dtype=reg_loss.dtype)
                if self.edge_embedding is not None:
                    higher_reg = higher_reg + self.edge_embedding.weight.norm(2).pow(2)
                if self.face_embedding is not None:
                    higher_reg = higher_reg + self.face_embedding.weight.norm(2).pow(2)
                reg_loss = reg_loss + higher_reg.squeeze(0) / (2 * n_higher)

        # Weight regularization (for non-softmax modes). Apply to the raw
        # per-group parameters (cheaper than the per-cell expansion and
        # equivalent up to a constant factor — the per-cell view just repeats
        # each group's value across its cells).
        if self.weight_reg > 0 and self.weight_mode != "softmax":
            if self.weight_mode == "signed":
                params = (self.node_weights, self.edge_weights, self.face_weights)
            else:
                params = (self.node_logits, self.edge_logits, self.face_logits)
            w_reg = sum(p.norm(2).pow(2) for p in params)
            reg_loss = reg_loss + self.weight_reg * w_reg

        return user_e, pos_e, neg_e, reg_loss
