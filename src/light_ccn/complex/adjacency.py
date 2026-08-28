"""Adjacency matrix construction and normalization for all models."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def symmetric_norm(A: sp.spmatrix) -> sp.csr_matrix:
    """Compute D^{-1/2} A D^{-1/2} symmetric normalization.

    Args:
        A: Sparse adjacency matrix (should be symmetric).

    Returns:
        Symmetrically normalized sparse matrix.
    """
    A = A.tocsr().astype(np.float32)
    rowsum = np.array(A.sum(axis=1)).flatten()
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


def bipartite_norm(B: sp.spmatrix) -> sp.csr_matrix:
    """Row-normalize a bipartite incidence/adjacency matrix.

    Computes D_row^{-1/2} B D_col^{-1/2} where D_row and D_col
    are diagonal degree matrices for rows and columns respectively.

    Args:
        B: Sparse matrix (possibly non-square).

    Returns:
        Bipartite-normalized sparse matrix.
    """
    B = B.tocsr().astype(np.float32)
    rowsum = np.array(B.sum(axis=1)).flatten()
    colsum = np.array(B.sum(axis=0)).flatten()

    with np.errstate(divide="ignore"):
        d_row_inv_sqrt = np.power(rowsum, -0.5)
    d_row_inv_sqrt[np.isinf(d_row_inv_sqrt)] = 0.0

    with np.errstate(divide="ignore"):
        d_col_inv_sqrt = np.power(colsum, -0.5)
    d_col_inv_sqrt[np.isinf(d_col_inv_sqrt)] = 0.0

    D_row = sp.diags(d_row_inv_sqrt)
    D_col = sp.diags(d_col_inv_sqrt)
    return D_row @ B @ D_col


def build_lightgcn_adj(bipartite_adj: sp.spmatrix) -> sp.csr_matrix:
    """Build the symmetrically normalized adjacency for LightGCN.

    Args:
        bipartite_adj: The bipartite adjacency matrix
            [[0, R], [R^T, 0]] of shape (n_users+n_items, n_users+n_items).

    Returns:
        D^{-1/2} A D^{-1/2} normalized adjacency.
    """
    return symmetric_norm(bipartite_adj)


def build_lightccn_flat_adj(
    R: sp.spmatrix,
    S: sp.spmatrix,
    n_users: int,
    n_items: int,
    gamma: float = 0.5,
    S_user: sp.spmatrix | None = None,
    gamma_user: float | None = None,
) -> sp.csr_matrix:
    """Build the augmented adjacency for LightCCN-Flat.

    Additive formulation — keeps the full user-item signal and adds
    item-item and/or user-user edges:
        A_tilde = [[gamma_user * S_user, R], [R^T, gamma * S]]
        A_hat = D^{-1/2} A_tilde D^{-1/2}

    gamma=0.0 recovers pure LightGCN (no item-item edges).
    gamma>0 adds item-item signal from the cell complex.
    S_user adds user-user signal from user-side cell complex.

    Args:
        R: User-item interaction matrix (n_users, n_items).
        S: Item-item adjacency from face boundaries (n_items, n_items).
        n_users: Number of users.
        n_items: Number of items.
        gamma: Weight for item-item edges (>= 0).
        S_user: Optional user-user adjacency (n_users, n_users).
        gamma_user: Weight for user-user edges. Defaults to gamma if not set.

    Returns:
        Symmetrically normalized augmented adjacency.
    """
    R = R.tocsr().astype(np.float32)
    S = S.tocsr().astype(np.float32)
    if gamma < 0.0:
        raise ValueError(f"gamma must be >= 0, got {gamma}")

    if S_user is not None:
        if gamma_user is None:
            gamma_user = gamma
        S_user = S_user.tocsr().astype(np.float32)
        uu_block = gamma_user * S_user
    else:
        uu_block = sp.csr_matrix((n_users, n_users), dtype=np.float32)

    A_tilde = sp.bmat([
        [uu_block, R],
        [R.T, gamma * S],
    ], format="csr")

    return symmetric_norm(A_tilde)


def build_lightccn_flat_split_adj(
    R: sp.spmatrix,
    S: sp.spmatrix,
    n_users: int,
    n_items: int,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Build SEPARATE adjacency components for LightCCN-Flat with learnable weights.

    Returns two normalized matrices:
        1. A_gcn: symmetric_norm([[0, R], [R^T, 0]])  — base LightGCN adjacency
        2. A_ii:  [[0, 0], [0, symmetric_norm(S)]]     — item-item only (padded)

    Used with signed_weights mode: emb = w_node * A_gcn @ emb + w_edge * A_ii @ emb

    Args:
        R: User-item interaction matrix (n_users, n_items).
        S: Item-item adjacency from face boundaries (n_items, n_items).
        n_users: Number of users.
        n_items: Number of items.

    Returns:
        Tuple of (A_gcn, A_ii) both (n_users+n_items, n_users+n_items).
    """
    R = R.tocsr().astype(np.float32)
    S = S.tocsr().astype(np.float32)

    # Base LightGCN adjacency
    zero_uu = sp.csr_matrix((n_users, n_users), dtype=np.float32)
    zero_ii = sp.csr_matrix((n_items, n_items), dtype=np.float32)
    A_gcn = sp.bmat([
        [zero_uu, R],
        [R.T, zero_ii],
    ], format="csr")
    A_gcn = symmetric_norm(A_gcn)

    # Item-item adjacency (normalized separately, zero-padded for users)
    S_norm = symmetric_norm(S)
    zero_uu = sp.csr_matrix((n_users, n_users), dtype=np.float32)
    zero_ui = sp.csr_matrix((n_users, n_items), dtype=np.float32)
    zero_iu = sp.csr_matrix((n_items, n_users), dtype=np.float32)
    A_ii = sp.bmat([
        [zero_uu, zero_ui],
        [zero_iu, S_norm],
    ], format="csr")

    return A_gcn, A_ii


def build_hodge_operators(
    bipartite_adj: sp.spmatrix,
    B1: sp.spmatrix,
) -> dict[str, sp.csr_matrix]:
    """Build operators for LightCCN-Hodge (principled Hodge Laplacian).

    Uses the 0-th Hodge Laplacian L_0 = B1 @ B1^T as a node-level operator
    derived from the cell complex boundary structure. This is the theoretically
    correct way to propagate topological information at the node level.

    Returns two operators:
        A_hat_0: sym_norm(bipartite_adj)        -- CF signal (user-item)
        L_hat_0: sym_norm(B1 @ B1^T - diag)     -- topological signal (Hodge)

    Args:
        bipartite_adj: [[0, R], [R^T, 0]] shape (N, N) where N = n_users + n_items.
        B1: Node-edge incidence matrix (n_nodes, n_edges).

    Returns:
        Dict mapping operator names to normalized sparse matrices.
    """
    B1 = B1.tocsr().astype(np.float32)

    # CF signal: standard LightGCN adjacency
    A_hat_0 = symmetric_norm(bipartite_adj)

    # Hodge Laplacian L_0 = B1 @ B1^T (node-node adjacency from topology)
    # Two nodes are connected if they share an edge in the cell complex.
    # Remove self-loops (diagonal) before normalization.
    L0_raw = (B1 @ B1.T).tocsr()
    L0_raw.setdiag(0)
    L0_raw.eliminate_zeros()
    L_hat_0 = symmetric_norm(L0_raw)

    return {
        "A_hat_0": A_hat_0,
        "L_hat_0": L_hat_0,
    }


def build_multi_operators(
    bipartite_adj: sp.spmatrix,
    B1: sp.spmatrix,
    B2: sp.spmatrix,
    add_self_loops: bool = False,
) -> dict[str, sp.csr_matrix]:
    """Build all 7 normalized operators for LightCCN-Multi.

    Operators (from the spec Eqs 38-44):
        A_hat_0:      sym_norm(bipartite_adj)         -- node-node
        B_hat_1_down: bipartite_norm(B1)              -- edge->node
        B_hat_1_up:   bipartite_norm(B1^T)            -- node->edge
        A_hat_1:      sym_norm(B2 @ B2^T - diag)      -- edge-edge via faces
        B_hat_2_down: bipartite_norm(B2)              -- face->edge
        B_hat_2_up:   bipartite_norm(B2^T)            -- edge->face
        A_hat_2:      sym_norm(B2^T @ B2 - diag)      -- face-face

    Args:
        bipartite_adj: [[0, R], [R^T, 0]] shape (N, N) where N = n_users + n_items.
        B1: Node-edge incidence matrix (n_nodes, n_edges).
        B2: Edge-face incidence matrix (n_edges, n_faces).

    Returns:
        Dict mapping operator names to normalized sparse matrices.
    """
    B1 = B1.tocsr().astype(np.float32)
    B2 = B2.tocsr().astype(np.float32)

    # Eq 38: node-node adjacency
    A_hat_0 = symmetric_norm(bipartite_adj)

    # Eq 39: edge -> node (B1 maps edges to their boundary nodes)
    B_hat_1_down = bipartite_norm(B1)

    # Eq 40: node -> edge
    B_hat_1_up = bipartite_norm(B1.T)

    # Eq 41: edge-edge adjacency via shared faces.
    # When ``add_self_loops`` is True (Option A): keep the diagonal at 1 so
    # each edge propagates its own previous embedding into the next layer
    # (the self-loop is folded into the existing w4 channel, degree-normalised
    # together with the neighbours).
    A1_raw = (B2 @ B2.T).tocsr()
    A1_raw.setdiag(1 if add_self_loops else 0)
    A1_raw.eliminate_zeros()
    A_hat_1 = symmetric_norm(A1_raw)

    # Eq 42: face -> edge
    B_hat_2_down = bipartite_norm(B2)

    # Eq 43: edge -> face
    B_hat_2_up = bipartite_norm(B2.T)

    # Eq 44: face-face adjacency via shared edges (same self-loop treatment).
    A2_raw = (B2.T @ B2).tocsr()
    A2_raw.setdiag(1 if add_self_loops else 0)
    A2_raw.eliminate_zeros()
    A_hat_2 = symmetric_norm(A2_raw)

    return {
        "A_hat_0": A_hat_0,
        "B_hat_1_down": B_hat_1_down,
        "B_hat_1_up": B_hat_1_up,
        "A_hat_1": A_hat_1,
        "B_hat_2_down": B_hat_2_down,
        "B_hat_2_up": B_hat_2_up,
        "A_hat_2": A_hat_2,
    }
