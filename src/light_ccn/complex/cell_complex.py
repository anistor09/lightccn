"""Cell complex construction: face detection and incidence matrices."""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


# ── Module-level shared state for multiprocessing workers ─────────
# Initialized once per worker via _init_worker(), NOT pickled per task.
_shared_adj_indices: np.ndarray | None = None
_shared_adj_indptr: np.ndarray | None = None
_shared_user_indices: np.ndarray | None = None
_shared_user_indptr: np.ndarray | None = None
_shared_tau: int = 0


def _init_worker(adj_indices, adj_indptr, user_indices, user_indptr, tau):
    """Initialize shared arrays in each worker process (called once)."""
    global _shared_adj_indices, _shared_adj_indptr
    global _shared_user_indices, _shared_user_indptr, _shared_tau
    _shared_adj_indices = adj_indices
    _shared_adj_indptr = adj_indptr
    _shared_user_indices = user_indices
    _shared_user_indptr = user_indptr
    _shared_tau = tau


def _find_faces_for_item(args: tuple) -> list[tuple[int, int, int]]:
    """Find all faces (i, j, k) with i fixed. Runs in a worker process.

    Only receives (i, nbrs_i_arr) — large arrays come from shared state.
    """
    i, nbrs_i_arr = args
    adj_indices = _shared_adj_indices
    adj_indptr = _shared_adj_indptr
    user_indices = _shared_user_indices
    user_indptr = _shared_user_indptr
    tau = _shared_tau

    faces = []
    for j in nbrs_i_arr:
        # Get neighbors of j that are > j (from upper-triangular adj)
        j_start, j_end = adj_indptr[j], adj_indptr[j + 1]
        nbrs_j = adj_indices[j_start:j_end]
        nbrs_j_gt_j = nbrs_j[nbrs_j > j]

        # Candidates k: must be neighbor of both i (with k > j) and j (with k > j)
        # nbrs_i_arr contains only items > i; we need k > j specifically
        nbrs_i_gt_j = nbrs_i_arr[nbrs_i_arr > j]
        candidates_k = np.intersect1d(nbrs_i_gt_j, nbrs_j_gt_j, assume_unique=True)

        if len(candidates_k) == 0:
            continue

        # Preload users for i and j
        ui_start, ui_end = user_indptr[i], user_indptr[i + 1]
        uj_start, uj_end = user_indptr[j], user_indptr[j + 1]
        users_i = user_indices[ui_start:ui_end]
        users_j = user_indices[uj_start:uj_end]
        users_ij = np.intersect1d(users_i, users_j, assume_unique=True)

        if len(users_ij) < tau:
            continue

        for k in candidates_k:
            uk_start, uk_end = user_indptr[k], user_indptr[k + 1]
            users_k = user_indices[uk_start:uk_end]
            common = np.intersect1d(users_ij, users_k, assume_unique=True)
            if len(common) >= tau:
                faces.append((i, j, k))

    return faces


def filter_faces(
    faces: list[tuple[int, int, int]],
    tail_items: set[int],
    mode: str = "at_least_one",
) -> list[tuple[int, int, int]]:
    """Filter faces based on tail item membership.

    Args:
        faces: List of (i, j, k) face tuples (item indices).
        tail_items: Set of item indices considered tail.
        mode: "all_tail" keeps face only if all 3 items are tail.
              "at_least_one" keeps face if any item is tail.

    Returns:
        Filtered list of faces.
    """
    if mode == "all_tail":
        return [(i, j, k) for i, j, k in faces
                if i in tail_items and j in tail_items and k in tail_items]
    elif mode == "at_least_one":
        return [(i, j, k) for i, j, k in faces
                if i in tail_items or j in tail_items or k in tail_items]
    else:
        raise ValueError(f"Unknown filter mode: {mode!r}. Use 'all_tail' or 'at_least_one'.")


class UserCellComplexBuilder:
    """Build a cell complex on the USER side.

    Mirrors CellComplexBuilder but transposes the interaction matrix:
    - Co-occurrence: C = R @ R^T (users who share items)
    - Face (u1, u2, u3): user triple where ≥ tau items are shared by all 3
    - S_user: user-user adjacency where S_user[u,v] = # shared faces
    """

    def __init__(
        self,
        R: sp.spmatrix,
        tau: int = 20,
        cache_dir: str = "data/complex_cache",
        dataset_name: str = "dataset",
    ):
        self.R = R.tocsr()
        self.n_users, self.n_items = R.shape
        self.tau = tau
        self.cache_dir = Path(cache_dir)
        self.dataset_name = dataset_name
        self.cache_path = self.cache_dir / f"{dataset_name}_user_tau{tau}.npz"

    def find_faces(self) -> list[tuple[int, int, int]]:
        """Detect user-side triangular faces.

        A face (u1, u2, u3) exists if ≥ tau items were interacted with by
        ALL three users: |items(u1) ∩ items(u2) ∩ items(u3)| ≥ tau.
        """
        R_csr = self.R.tocsr()

        print("Computing user co-occurrence matrix C = R @ R^T...")
        C = (R_csr @ R_csr.T).tocsr()

        print(f"Thresholding at tau={self.tau}...")
        C_thresh = C.copy()
        C_thresh.data[C_thresh.data < self.tau] = 0
        C_thresh.eliminate_zeros()
        C_upper = sp.triu(C_thresh, k=1, format="csr")

        C_sym = C_upper + C_upper.T
        C_sym = C_sym.tocsr()
        C_sym.sort_indices()

        # items-per-user: R_csr row u = items of user u
        R_csr.sort_indices()

        users_with_nbrs = np.unique(C_upper.tocoo().row)
        print(f"Finding user faces ({len(users_with_nbrs)} users with neighbors)...")

        # Share arrays via module-level state for parallel workers
        adj_indices = C_sym.indices
        adj_indptr = C_sym.indptr
        item_indices = R_csr.indices
        item_indptr = R_csr.indptr
        tau = self.tau

        work_items = []
        for u in users_with_nbrs:
            u_start, u_end = C_upper.indptr[u], C_upper.indptr[u + 1]
            nbrs_u = C_upper.indices[u_start:u_end]
            if len(nbrs_u) > 0:
                work_items.append((u, nbrs_u))

        n_workers = min(os.cpu_count() or 1, len(work_items))

        # Reuse the same worker function — it works for any entity type
        # Just need to set the shared state to user-side arrays
        if n_workers <= 1 or len(work_items) < 50:
            _init_worker(adj_indices, adj_indptr, item_indices, item_indptr, tau)
            all_faces = []
            for args in tqdm(work_items, desc="User face detection"):
                all_faces.extend(_find_faces_for_item(args))
        else:
            print(f"  Using {n_workers} worker processes")
            all_faces = []
            with mp.Pool(
                processes=n_workers,
                initializer=_init_worker,
                initargs=(adj_indices, adj_indptr, item_indices, item_indptr, tau),
            ) as pool:
                for result in tqdm(
                    pool.imap_unordered(_find_faces_for_item, work_items),
                    total=len(work_items),
                    desc="User face detection",
                ):
                    all_faces.extend(result)

        all_faces.sort()
        print(f"Found {len(all_faces)} user faces")
        return all_faces

    @staticmethod
    def build_user_user_adjacency(
        faces: list[tuple[int, int, int]], n_users: int,
    ) -> sp.csr_matrix:
        """Build user-user adjacency S_user where S_user[u,v] = # shared faces."""
        if not faces:
            return sp.csr_matrix((n_users, n_users), dtype=np.float32)

        face_arr = np.array(faces, dtype=np.int64)
        rows = np.concatenate([face_arr[:, 0], face_arr[:, 1],
                               face_arr[:, 0], face_arr[:, 2],
                               face_arr[:, 1], face_arr[:, 2]])
        cols = np.concatenate([face_arr[:, 1], face_arr[:, 0],
                               face_arr[:, 2], face_arr[:, 0],
                               face_arr[:, 2], face_arr[:, 1]])
        vals = np.ones(len(rows), dtype=np.float32)

        S = sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_users))
        return S

    def build_and_cache(self) -> dict:
        """Build user cell complex and cache to disk."""
        if self.cache_path.exists():
            print(f"Loading cached user complex from {self.cache_path}")
            data = np.load(self.cache_path, allow_pickle=True)
            faces = [tuple(f) for f in data["faces"]]
            S_user = sp.csr_matrix(
                (data["S_data"], data["S_indices"], data["S_indptr"]),
                shape=tuple(data["S_shape"]),
            )
            print(f"  {len(faces)} user faces")
            return {"faces": faces, "S_user": S_user}

        faces = self.find_faces()
        S_user = self.build_user_user_adjacency(faces, self.n_users)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.cache_path,
            faces=np.array(faces) if faces else np.empty((0, 3), dtype=np.int64),
            S_data=S_user.data,
            S_indices=S_user.indices,
            S_indptr=S_user.indptr,
            S_shape=S_user.shape,
        )
        print(f"User complex cached to {self.cache_path}")
        return {"faces": faces, "S_user": S_user}


class CellComplexBuilder:
    """Build a cell complex from a user-item interaction matrix.

    Detects 2-cells (faces = item triples) using binary co-occurrence:
    a face (i, j, k) exists if >= tau users interacted with ALL three items.

    Then constructs:
    - S: item-item adjacency where S[i,j] = # shared faces
    - B1: node-edge incidence matrix (unsigned)
    - B2: edge-face incidence matrix (unsigned)
    """

    def __init__(
        self,
        R: sp.spmatrix,
        tau: int = 20,
        cache_dir: str = "data/complex_cache",
        dataset_name: str = "dataset",
    ):
        """
        Args:
            R: User-item interaction matrix (n_users, n_items), binary.
            tau: Minimum co-occurrence threshold for face detection.
            cache_dir: Directory for caching computed matrices.
            dataset_name: Name used in cache filenames.
        """
        self.R = R.tocsc()  # CSC for fast column (item) operations
        self.n_users, self.n_items = R.shape
        self.tau = tau
        self.cache_dir = Path(cache_dir)
        self.dataset_name = dataset_name
        self.cache_path = self.cache_dir / f"{dataset_name}_tau{tau}.npz"

    def find_faces(self) -> list[tuple[int, int, int]]:
        """Detect triangular faces via binary co-occurrence.

        Parallelized algorithm:
        1. C = R^T @ R (item co-occurrence, sparse)
        2. Threshold: keep pairs (i,j) where C[i,j] >= tau
        3. Build upper-triangular adjacency (i < j only)
        4. For each item i (in parallel), enumerate neighbors j > i,
           find candidates k > j in nbrs(i) ∩ nbrs(j),
           verify |users(i) ∩ users(j) ∩ users(k)| >= tau
        5. Return sorted list of (i, j, k) with i < j < k

        Returns:
            List of (i, j, k) tuples with i < j < k.
        """
        R_csc = self.R.tocsc()
        R_csr = self.R.tocsr()

        print("Computing item co-occurrence matrix C = R^T @ R...")
        C = (R_csr.T @ R_csr).tocsr()

        # Threshold and keep upper triangle (i < j)
        print(f"Thresholding at tau={self.tau}...")
        C_thresh = C.copy()
        C_thresh.data[C_thresh.data < self.tau] = 0
        C_thresh.eliminate_zeros()
        C_upper = sp.triu(C_thresh, k=1, format="csr")

        # Symmetric version for neighbor lookup (need nbrs(j) including k > j)
        C_sym = C_upper + C_upper.T
        C_sym = C_sym.tocsr()
        C_sym.sort_indices()

        # Precompute users-per-item in CSC format for fast column access
        # R_csc[:, item].indices gives sorted user indices
        R_item_csc = R_csc.tocsc()
        # Convert to CSR of R^T for contiguous row access: row i = users of item i
        R_T_csr = R_item_csc.T.tocsr()
        R_T_csr.sort_indices()

        # Items that have at least one upper-triangular neighbor
        items_with_nbrs = np.unique(C_upper.tocoo().row)
        print(f"Finding triangular faces ({len(items_with_nbrs)} items with neighbors)...")

        # Prepare work items for parallel execution
        # Each work item: (i, nbrs_i_array, adj_indices, adj_indptr, user_indices, user_indptr, tau)
        adj_indices = C_sym.indices
        adj_indptr = C_sym.indptr
        user_indices = R_T_csr.indices
        user_indptr = R_T_csr.indptr
        tau = self.tau

        work_items = []
        for i in items_with_nbrs:
            # Upper-triangular neighbors of i (j > i)
            i_start, i_end = C_upper.indptr[i], C_upper.indptr[i + 1]
            nbrs_i = C_upper.indices[i_start:i_end]
            if len(nbrs_i) > 0:
                # Only pass small per-item data; large arrays go via initializer
                work_items.append((i, nbrs_i))

        # Determine worker count (Colab L4 typically has 2-4 vCPUs)
        n_workers = min(os.cpu_count() or 1, len(work_items))

        if n_workers <= 1 or len(work_items) < 50:
            # Serial execution for small workloads
            # Set shared state directly (no subprocess)
            _init_worker(adj_indices, adj_indptr, user_indices, user_indptr, tau)
            all_faces = []
            for args in tqdm(work_items, desc="Face detection"):
                all_faces.extend(_find_faces_for_item(args))
        else:
            # Parallel execution using Pool with initializer
            # Large arrays are shared once per worker (not pickled per task)
            print(f"  Using {n_workers} worker processes")
            all_faces = []
            with mp.Pool(
                processes=n_workers,
                initializer=_init_worker,
                initargs=(adj_indices, adj_indptr, user_indices, user_indptr, tau),
            ) as pool:
                for result in tqdm(
                    pool.imap_unordered(_find_faces_for_item, work_items),
                    total=len(work_items),
                    desc="Face detection",
                ):
                    all_faces.extend(result)

        # Sort for deterministic ordering
        all_faces.sort()
        print(f"Found {len(all_faces)} faces")
        return all_faces

    @staticmethod
    def build_item_item_adjacency(faces: list[tuple[int, int, int]], n_items: int) -> sp.csr_matrix:
        """Build item-item adjacency S where S[i,j] = number of shared faces.

        Each face (i,j,k) contributes edges (i,j), (i,k), (j,k).
        """
        if not faces:
            return sp.csr_matrix((n_items, n_items), dtype=np.float32)

        face_arr = np.array(faces, dtype=np.int64)
        # Each face contributes 3 edges × 2 directions = 6 entries
        rows = np.concatenate([face_arr[:, 0], face_arr[:, 1],
                               face_arr[:, 0], face_arr[:, 2],
                               face_arr[:, 1], face_arr[:, 2]])
        cols = np.concatenate([face_arr[:, 1], face_arr[:, 0],
                               face_arr[:, 2], face_arr[:, 0],
                               face_arr[:, 2], face_arr[:, 1]])
        vals = np.ones(len(rows), dtype=np.float32)

        S = sp.csr_matrix((vals, (rows, cols)), shape=(n_items, n_items))
        return S

    @staticmethod
    def build_incidence_B1(
        edges: list[tuple[int, int]],
        n_nodes: int,
    ) -> sp.csr_matrix:
        """Build unsigned node-edge incidence matrix B1.

        B1[node, edge] = 1 if node is a boundary of edge.
        Shape: (n_nodes, n_edges)
        """
        n_edges = len(edges)
        if n_edges == 0:
            return sp.csr_matrix((n_nodes, 0), dtype=np.float32)

        edge_arr = np.array(edges, dtype=np.int64)
        rows = np.concatenate([edge_arr[:, 0], edge_arr[:, 1]])
        cols = np.concatenate([np.arange(n_edges), np.arange(n_edges)])
        vals = np.ones(len(rows), dtype=np.float32)

        B1 = sp.csr_matrix((vals, (rows, cols)), shape=(n_nodes, n_edges))
        return B1

    @staticmethod
    def build_incidence_B2(
        edges: list[tuple[int, int]],
        faces: list[tuple[int, int, int]],
    ) -> sp.csr_matrix:
        """Build unsigned edge-face incidence matrix B2.

        B2[edge, face] = 1 if edge is a boundary of face.
        Shape: (n_edges, n_faces)
        """
        n_edges = len(edges)
        n_faces = len(faces)
        if n_edges == 0 or n_faces == 0:
            return sp.csr_matrix((max(n_edges, 1), max(n_faces, 1)), dtype=np.float32)

        edge_to_idx = {e: idx for idx, e in enumerate(edges)}

        rows, cols = [], []
        for f_idx, (i, j, k) in enumerate(faces):
            for a, b in [(i, j), (i, k), (j, k)]:
                edge = (min(a, b), max(a, b))
                if edge in edge_to_idx:
                    rows.append(edge_to_idx[edge])
                    cols.append(f_idx)

        vals = np.ones(len(rows), dtype=np.float32)
        B2 = sp.csr_matrix((vals, (rows, cols)), shape=(n_edges, n_faces))
        return B2

    def build_and_cache(self) -> dict[str, sp.spmatrix | list]:
        """Build cell complex and cache to disk (or load from cache).

        Returns:
            Dict with keys: 'faces', 'edges', 'S', 'B1', 'B2'.
        """
        if self.cache_path.exists():
            print(f"Loading cached cell complex from {self.cache_path}")
            data = np.load(self.cache_path, allow_pickle=True)
            faces = [tuple(f) for f in data["faces"]]
            edges = [tuple(e) for e in data["edges"]]
            S = sp.csr_matrix(
                (data["S_data"], data["S_indices"], data["S_indptr"]),
                shape=tuple(data["S_shape"]),
            )
            B1 = sp.csr_matrix(
                (data["B1_data"], data["B1_indices"], data["B1_indptr"]),
                shape=tuple(data["B1_shape"]),
            )
            B2 = sp.csr_matrix(
                (data["B2_data"], data["B2_indices"], data["B2_indptr"]),
                shape=tuple(data["B2_shape"]),
            )
            print(f"  {len(faces)} faces, {len(edges)} edges")
            return {"faces": faces, "edges": edges, "S": S, "B1": B1, "B2": B2}

        # Build from scratch
        faces = self.find_faces()

        # Extract unique edges from faces
        edge_set: set[tuple[int, int]] = set()
        for i, j, k in faces:
            edge_set.add((min(i, j), max(i, j)))
            edge_set.add((min(i, k), max(i, k)))
            edge_set.add((min(j, k), max(j, k)))
        edges = sorted(edge_set)

        print(f"Building matrices: {len(edges)} unique edges from {len(faces)} faces")

        # n_nodes for B1 = n_users + n_items (bipartite node space)
        # Edges are item-item, so offset by n_users
        n_nodes = self.n_users + self.n_items
        edges_bipartite = [(self.n_users + a, self.n_users + b) for a, b in edges]

        S = self.build_item_item_adjacency(faces, self.n_items)
        B1 = self.build_incidence_B1(edges_bipartite, n_nodes)
        B2 = self.build_incidence_B2(edges, faces)

        # Cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.cache_path,
            faces=np.array(faces) if faces else np.empty((0, 3), dtype=np.int64),
            edges=np.array(edges) if edges else np.empty((0, 2), dtype=np.int64),
            S_data=S.data,
            S_indices=S.indices,
            S_indptr=S.indptr,
            S_shape=S.shape,
            B1_data=B1.data,
            B1_indices=B1.indices,
            B1_indptr=B1.indptr,
            B1_shape=B1.shape,
            B2_data=B2.data,
            B2_indices=B2.indices,
            B2_indptr=B2.indptr,
            B2_shape=B2.shape,
        )
        print(f"Cell complex cached to {self.cache_path}")

        return {"faces": faces, "edges": edges, "S": S, "B1": B1, "B2": B2}


def augment_with_ui_edges(
    complex_data: dict,
    R: sp.spmatrix,
    n_users: int,
    n_items: int,
) -> dict:
    """Feature flag: add USER-ITEM interaction edges to the rank-1 edge set.

    By default the cell complex has item-item edges only (users sit on no
    rank-1 cell, so B1 has zero user rows and edge embeddings exist only for
    item-item edges). With this augmentation, every interaction (u, i) in R
    becomes an additional rank-1 edge whose boundary nodes are {u, n_users+i}.
    Effect: B1 gains nonzero user rows, so

      * propagation: the w2 (B_hat_1_down) channel now delivers edge messages
        to USER nodes, and B_hat_1_up lifts user+item node embeddings into the
        new U-I edges -> U-I edges get embeddings in every propagation mode
        (derived/stateful recompute them; full_multi's learnable table grows).
      * readout: p_E = B_hat_1_down @ E becomes NONZERO on user rows.

    Item-item edges and all faces are unchanged. U-I edges belong to no face,
    so B2 gets all-zero rows for them (they are isolated in A_hat_1 = B2 B2^T).
    New edges are appended AFTER the item-item edges so existing indices and
    the face incidence stay valid; n_edges = B2.shape[0] stays consistent with
    B1.shape[1].

    Returns a shallow copy of complex_data with B1, B2 extended and the extra
    keys 'n_ui_edges' and 'n_item_edges'.
    """
    B1 = complex_data["B1"].tocsc().astype(np.float32)   # (n_nodes, n_item_edges)
    B2 = complex_data["B2"].tocsr().astype(np.float32)   # (n_item_edges, n_faces)
    n_item_edges = B1.shape[1]
    n_nodes = n_users + n_items
    assert B1.shape[0] == n_nodes, (B1.shape, n_nodes)

    Rcoo = R.tocoo()
    u = Rcoo.row.astype(np.int64)
    i = Rcoo.col.astype(np.int64) + n_users          # item -> bipartite node index
    n_ui = int(u.shape[0])
    if n_ui == 0:
        out = dict(complex_data)
        out["n_ui_edges"] = 0
        out["n_item_edges"] = n_item_edges
        return out

    # New B1 columns: unsigned incidence, 1 at the user node and 1 at the item node.
    rows = np.concatenate([u, i])
    cols = np.concatenate([np.arange(n_ui), np.arange(n_ui)])
    vals = np.ones(2 * n_ui, dtype=np.float32)
    B1_ui = sp.csc_matrix((vals, (rows, cols)), shape=(n_nodes, n_ui))
    B1_new = sp.hstack([B1, B1_ui]).tocsr()

    # New B2 rows: U-I edges are in no face -> all-zero rows.
    n_faces = B2.shape[1]
    B2_new = sp.vstack(
        [B2, sp.csr_matrix((n_ui, n_faces), dtype=np.float32)]
    ).tocsr()

    out = dict(complex_data)
    out["B1"] = B1_new
    out["B2"] = B2_new
    out["n_ui_edges"] = n_ui
    out["n_item_edges"] = n_item_edges
    return out
