"""NMF-derived face construction (HOUR-style).

E1: top-3 items per factor → 1 triangular face per factor
E2: top-5 items per factor → all C(5,3) = 10 triangular faces per factor

Caches NMF decompositions to data/nmf_cache/ so repeated experiments with
the same (dataset, n_factors) reuse the same Z_U, Z_I.
"""

from __future__ import annotations

import os
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def _fit_or_load_nmf(
    R: sp.spmatrix,
    n_factors: int,
    cache_dir: str,
    dataset_name: str,
    max_iter: int = 30,
    seed: int = 2020,
    regularization: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit factor model on R (or load from cache). Returns (Z_U, Z_I).

    Uses implicit.als (Hu et al. 2008 Weighted Matrix Factorization) as a
    FAST APPROXIMATION to the HOUR paper's masked-Frobenius NMF
    (Wang et al. §4.1, Equation 2):

        min_{Z_U, Z_I >= 0} ||Omega(R - Z_U Z_I^T)||_F^2 + lambda * (...)

    Three documented differences from the paper's spec:

      1. Non-negativity: HOUR enforces Z_U, Z_I >= 0; ALS does not enforce
         it. For our top-K hyperedge selection per factor, the largest
         positive values still dominate, so the semantics are preserved
         in practice — but a strict reader should know we are not solving
         the constrained problem.

      2. Mask Omega(·): HOUR treats zero entries as MISSING (mask = 0).
         ALS treats them as observed with low confidence (c_ui = 1, vs
         c_ui = 1 + alpha for observed). So ALS still pulls toward zero
         for unobserved pairs, just weakly.

      3. Target: HOUR fits real-valued R_ui; ALS fits binary preference
         p_ui = 1[r_ui > 0]. Equivalent here since R is binary CF data.

    Justification for the approximation: per-iteration cost drops from
    O(|U|·|I|·k) [sklearn full-Frobenius] to O(nnz·k) [ALS], which is a
    ~1500x reduction in per-iter FLOPs on Gowalla. The qualitative shape
    of the latent factors and the top-K selection that follows are
    expected to be close to what a true masked-NMF would produce.
    """
    # Cache key includes "_als" so old sklearn caches are not confused with new ALS caches
    cache_path = Path(cache_dir) / f"nmf_{dataset_name}_E{n_factors}_seed{seed}_als.npz"
    if cache_path.exists():
        print(f"Loading cached factor model from {cache_path}")
        data = np.load(cache_path)
        return data["Z_U"], data["Z_I"]

    print(f"Fitting implicit.als (HOUR-style WMF) with n_factors={n_factors}...")
    import os as _os
    _os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # avoid BLAS thread contention with implicit's own threads
    from implicit.als import AlternatingLeastSquares

    if sp.issparse(R):
        R_in = R.astype(np.float32).tocsr()
    else:
        R_in = sp.csr_matrix(R.astype(np.float32))

    als = AlternatingLeastSquares(
        factors=n_factors,
        regularization=regularization,
        iterations=max_iter,
        use_gpu=False,
        dtype=np.float32,
        random_state=seed,
    )
    als.fit(R_in, show_progress=False)

    # implicit returns matrices as either numpy arrays or GPU tensors;
    # use np.asarray to normalise.
    Z_U = np.asarray(als.user_factors).astype(np.float32)   # (|U|, |E|)
    Z_I = np.asarray(als.item_factors).astype(np.float32)   # (|I|, |E|)

    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache_path, Z_U=Z_U, Z_I=Z_I)
    print(f"Factor model cached to {cache_path}  (Z_U: {Z_U.shape}, Z_I: {Z_I.shape})")
    return Z_U, Z_I


def find_faces_nmf_top_n(
    R: sp.spmatrix,
    n_factors: int,
    top_n: int,
    cache_dir: str = "data/nmf_cache",
    dataset_name: str = "dataset",
    seed: int = 2020,
) -> list[tuple[int, int, int]]:
    """Generic NMF-derived face builder.

    For each factor e in {0, ..., n_factors - 1}:
        - Take the top-N items by Z_I[:, e].
        - If top_n == 3: one triangle per factor (E1).
        - If top_n  > 3: all C(top_n, 3) triangles per factor (E2-style).

    Args:
        R: User-item interaction matrix (n_users, n_items).
        n_factors: Number of NMF latent factors (|E|).
        top_n: Items per factor (3 = E1, 5 = E2).
        cache_dir: Cache directory for NMF outputs.
        dataset_name: Used in cache filename.
        seed: NMF random seed.

    Returns:
        Sorted list of (i, j, k) tuples with i < j < k.
    """
    Z_U, Z_I = _fit_or_load_nmf(R, n_factors, cache_dir, dataset_name, seed=seed)

    faces: set[tuple[int, int, int]] = set()
    for e in range(n_factors):
        col = Z_I[:, e]
        if col.sum() == 0:
            continue
        # Top-N items by factor weight
        top_items = np.argpartition(-col, min(top_n, len(col) - 1))[:top_n].tolist()
        # Skip items with zero loading
        top_items = [i for i in top_items if col[i] > 0]
        if len(top_items) < 3:
            continue
        top_items = sorted(top_items)
        for triple in combinations(top_items, 3):
            faces.add(triple)
    out = sorted(faces)
    print(f"NMF face builder produced {len(out)} faces "
          f"(n_factors={n_factors}, top_n={top_n})")
    return out


def find_faces_e1(
    R: sp.spmatrix,
    n_factors: int,
    cache_dir: str = "data/nmf_cache",
    dataset_name: str = "dataset",
    seed: int = 2020,
) -> list[tuple[int, int, int]]:
    """E1: top-3 items per factor → 1 triangular face per factor."""
    return find_faces_nmf_top_n(R, n_factors, top_n=3,
                                cache_dir=cache_dir, dataset_name=dataset_name, seed=seed)


def find_faces_e2(
    R: sp.spmatrix,
    n_factors: int,
    cache_dir: str = "data/nmf_cache",
    dataset_name: str = "dataset",
    seed: int = 2020,
) -> list[tuple[int, int, int]]:
    """E2: top-5 items per factor → all C(5, 3) = 10 triangular faces per factor."""
    return find_faces_nmf_top_n(R, n_factors, top_n=5,
                                cache_dir=cache_dir, dataset_name=dataset_name, seed=seed)


def build_complex_from_faces(
    faces: list[tuple[int, int, int]],
    n_users: int,
    n_items: int,
) -> dict[str, sp.spmatrix | list]:
    """Build the {faces, edges, S, B1, B2} dict from a triangular face list.

    Mirrors the postprocess in `CellComplexBuilder.build_and_cache()` but
    without re-running face detection. Use this when the face list comes
    from an alternative source (NMF, similarity augmentation, etc.).
    """
    from light_ccn.complex.cell_complex import CellComplexBuilder

    # Extract unique edges from faces
    edge_set: set[tuple[int, int]] = set()
    for i, j, k in faces:
        edge_set.add((min(i, j), max(i, j)))
        edge_set.add((min(i, k), max(i, k)))
        edge_set.add((min(j, k), max(j, k)))
    edges = sorted(edge_set)

    print(f"Building matrices: {len(edges)} unique edges from {len(faces)} faces")

    n_nodes = n_users + n_items
    edges_bipartite = [(n_users + a, n_users + b) for a, b in edges]

    S = CellComplexBuilder.build_item_item_adjacency(faces, n_items)
    B1 = CellComplexBuilder.build_incidence_B1(edges_bipartite, n_nodes)
    B2 = CellComplexBuilder.build_incidence_B2(edges, faces)

    return {"faces": faces, "edges": edges, "S": S, "B1": B1, "B2": B2}


# ───────────────────── E3 & E4 helpers (polygonal faces) ─────────────────────

def _polygon_boundary_edges(face: tuple) -> list[tuple[int, int]]:
    """Boundary cycle of a k-gon face.

    Given ordered vertices (v_0, v_1, ..., v_{k-1}), returns the k cycle
    edges as canonical (min, max) tuples:
        {(v_0, v_1), (v_1, v_2), ..., (v_{k-1}, v_0)}
    """
    k = len(face)
    edges = []
    for j in range(k):
        a, b = face[j], face[(j + 1) % k]
        edges.append((min(a, b), max(a, b)))
    return edges


def _build_incidence_B2_polygonal(
    edges: list[tuple[int, int]],
    faces: list[tuple],
) -> sp.csr_matrix:
    """Variable-arity B_2: each face column has |face| ones, one per cycle edge."""
    n_edges = len(edges)
    n_faces = len(faces)
    if n_edges == 0 or n_faces == 0:
        return sp.csr_matrix((max(n_edges, 1), max(n_faces, 1)), dtype=np.float32)

    edge_to_idx = {e: idx for idx, e in enumerate(edges)}
    rows, cols = [], []
    for f_idx, face in enumerate(faces):
        for e in _polygon_boundary_edges(face):
            if e in edge_to_idx:
                rows.append(edge_to_idx[e])
                cols.append(f_idx)
    vals = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_edges, n_faces))


def _build_item_item_adjacency_polygonal(
    faces: list[tuple],
    n_items: int,
) -> sp.csr_matrix:
    """Item-item adjacency from polygonal faces.

    Only counts edges where both endpoints are items; user-containing
    edges are excluded from this rank-0 item-item summary (they live in
    the bipartite block of A_hat_0 instead).
    """
    rows, cols = [], []
    for face in faces:
        for e in _polygon_boundary_edges(face):
            a, b = e
            # Items in raw (no-offset) space are 0..n_items-1; but for E4
            # faces store bipartite indices (users 0..M-1, items M..M+N-1).
            # We can't distinguish here without M; only include pairs where
            # both indices are < n_items (item-item in raw space) for E3.
            # For E4 we just return an empty S — irrelevant to the model.
            if 0 <= a < n_items and 0 <= b < n_items:
                rows.append(a); cols.append(b)
                rows.append(b); cols.append(a)
    if not rows:
        return sp.csr_matrix((n_items, n_items), dtype=np.float32)
    vals = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_items, n_items))


def find_faces_e3(
    R: sp.spmatrix,
    n_factors: int,
    cache_dir: str = "data/nmf_cache",
    dataset_name: str = "dataset",
    seed: int = 2020,
) -> list[tuple[int, ...]]:
    """E3: top-5 items per factor → one pentagon face per factor.

    Vertices are ordered by descending Z_I weight, so the polygon cycle is
    (top1, top2, top3, top4, top5).
    """
    Z_U, Z_I = _fit_or_load_nmf(R, n_factors, cache_dir, dataset_name, seed=seed)
    faces: list[tuple[int, ...]] = []
    for e in range(n_factors):
        col = Z_I[:, e]
        if col.sum() == 0:
            continue
        idx = np.argpartition(-col, min(5, len(col) - 1))[:5]
        # Sort by descending weight for a stable cyclic ordering
        idx = idx[np.argsort(-col[idx])]
        idx = [int(i) for i in idx if col[i] > 0]
        if len(idx) < 3:
            continue
        faces.append(tuple(idx))
    print(f"E3 face builder produced {len(faces)} pentagon faces "
          f"(n_factors={n_factors})")
    return faces


def find_faces_e4(
    R: sp.spmatrix,
    n_factors: int,
    n_users: int,
    cache_dir: str = "data/nmf_cache",
    dataset_name: str = "dataset",
    seed: int = 2020,
    top_n: int = 5,
) -> list[tuple[int, ...]]:
    """E4: HOUR mixed — top-N users + top-N items per factor → one
    (2N)-gon face per factor.

    Vertices are bipartite-indexed (users 0..M-1, items M..M+N-1) so the
    cell complex's edges naturally include user-user, user-item, and
    item-item edges as required.

    Cyclic ordering: top-N users (by Z_U) followed by top-N items (by Z_I),
    each in descending-weight order.
    """
    Z_U, Z_I = _fit_or_load_nmf(R, n_factors, cache_dir, dataset_name, seed=seed)
    faces: list[tuple[int, ...]] = []
    for e in range(n_factors):
        col_u = Z_U[:, e]
        col_i = Z_I[:, e]
        if col_u.sum() == 0 or col_i.sum() == 0:
            continue
        top_u = np.argpartition(-col_u, min(top_n, len(col_u) - 1))[:top_n]
        top_u = top_u[np.argsort(-col_u[top_u])]
        top_u = [int(u) for u in top_u if col_u[u] > 0]
        top_i = np.argpartition(-col_i, min(top_n, len(col_i) - 1))[:top_n]
        top_i = top_i[np.argsort(-col_i[top_i])]
        top_i = [int(i) for i in top_i if col_i[i] > 0]
        if len(top_u) < 2 or len(top_i) < 2:
            continue
        # Bipartite indices: items get offset by n_users
        bipartite = top_u + [n_users + i for i in top_i]
        faces.append(tuple(bipartite))
    print(f"E4 face builder produced {len(faces)} mixed user+item faces "
          f"(n_factors={n_factors})")
    return faces


def build_complex_from_polygonal_faces(
    faces: list[tuple],
    n_users: int,
    n_items: int,
    bipartite_indexed: bool = False,
) -> dict[str, sp.spmatrix | list]:
    """Build the {faces, edges, S, B1, B2} dict from a polygonal face list.

    Args:
        faces: List of tuples of any arity (3 = triangle, 5 = pentagon,
            2N = HOUR mixed). Each tuple is a cyclic vertex ordering.
        n_users: Number of users.
        n_items: Number of items.
        bipartite_indexed: If True, face vertices are already bipartite
            indices (users 0..M-1, items M..M+N-1; used for E4).
            If False, vertices are item-only indices 0..N-1 (used for E3).
    """
    from light_ccn.complex.cell_complex import CellComplexBuilder

    # Collect cycle edges
    edge_set: set[tuple[int, int]] = set()
    for face in faces:
        for e in _polygon_boundary_edges(face):
            edge_set.add(e)
    edges = sorted(edge_set)

    print(f"Building matrices: {len(edges)} unique edges from {len(faces)} polygonal faces")

    n_nodes = n_users + n_items

    # B1 takes bipartite edges. For E3 (item-only), offset by n_users; for
    # E4 (bipartite already), pass through.
    if bipartite_indexed:
        edges_bipartite = edges
    else:
        edges_bipartite = [(n_users + a, n_users + b) for a, b in edges]

    # S (item-item adjacency) only meaningful for item-only faces (E3)
    if bipartite_indexed:
        S = sp.csr_matrix((n_items, n_items), dtype=np.float32)
    else:
        S = _build_item_item_adjacency_polygonal(faces, n_items)

    B1 = CellComplexBuilder.build_incidence_B1(edges_bipartite, n_nodes)
    B2 = _build_incidence_B2_polygonal(edges_bipartite if bipartite_indexed else edges, faces)

    return {"faces": faces, "edges": edges, "S": S, "B1": B1, "B2": B2}
