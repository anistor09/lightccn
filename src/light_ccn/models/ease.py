"""EASE^R (Steck, WWW 2019) — closed-form linear item-item model.

B = P / (-diag(P)) with P = (X^T X + lam*I)^{-1}, diag(B) = 0;
scores(u) = X_u @ B. No training epochs: the lambda grid plays the role of
the epoch axis downstream (one eval row per lambda), so both selection
conventions apply unchanged (biased frame selects by test metric over rows,
unbiased frame by validation metric).

GPU-first: the Gram inversion runs on CUDA in float32 for large item counts
(well-conditioned thanks to lam) and float64 elsewhere.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


def fit_ease(R: sp.csr_matrix, lam: float, device) -> torch.Tensor:
    """Return the dense item-item weight matrix B (on GPU when it fits, else CPU).

    Memory-safe: casts the Gram matrix to its final dtype in numpy BEFORE any
    device move, uses Cholesky inverse (peak ~2x matrix size instead of ~3x),
    frees aggressively, and falls back to a CPU float32 path for item counts
    whose factorization cannot fit in GPU memory."""
    import gc
    n_items = R.shape[1]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    np_dtype = np.float64 if n_items <= 15000 else np.float32
    if n_items > 50000:
        # Heavy-tailed raters make the SPARSE Gram intermediate explode
        # (epinions: nnz(R^T R) past 2^31 forces int64 indices, ~50+ GB before
        # the dense array exists — killed every runtime shape). Build the Gram
        # densely in user-chunks instead: GPU when available (G + one chunk
        # ~28 GB VRAM), else CPU.
        gdev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        Rc = R.tocsr()
        Gt_acc = torch.zeros((n_items, n_items), dtype=torch.float32, device=gdev)
        chunk = 4096
        for s0 in range(0, R.shape[0], chunk):
            rows = np.asarray(Rc[s0:s0 + chunk].todense(), dtype=np.float32)
            x = torch.from_numpy(rows).to(gdev)
            Gt_acc.addmm_(x.T, x)
            del x
        G = Gt_acc.cpu().numpy()
        del Gt_acc
        if gdev.type == "cuda":
            torch.cuda.empty_cache()
    else:
        # Cast the SPARSE matrix before the product (R often arrives float64;
        # a float64 toarray would materialize an n^2 float64 intermediate).
        R32 = R.astype(np_dtype)
        G = (R32.T @ R32).toarray()
        del R32
        if G.dtype != np_dtype:
            G = G.astype(np_dtype)
    use_gpu = getattr(device, "type", str(device)) == "cuda" and torch.cuda.is_available()
    if use_gpu:
        free, _total = torch.cuda.mem_get_info()
        # torch.linalg.cholesky peak ~= input + output copy + cuSOLVER workspace
        # (~3x the matrix, observed on tky 62k: 2x held + 1x failed alloc).
        if 3.3 * G.nbytes > 0.92 * free:
            print(f"    [ease] n_items={n_items}: needs ~{3.3*G.nbytes/1e9:.1f} GB "
                  f"> GPU free {free/1e9:.1f} GB -> CPU fallback")
            use_gpu = False
    if use_gpu:
        Gt = torch.from_numpy(G).to(device)
        del G
        idx = torch.arange(n_items, device=device)
        Gt[idx, idx] += lam
        L = torch.linalg.cholesky(Gt)
        del Gt
        P = torch.cholesky_inverse(L)
        del L
        diag = torch.diagonal(P)
        B = P / (-diag.unsqueeze(0))
        B.fill_diagonal_(0.0)
        out = B.contiguous().to(torch.float32)
        del P, B
        torch.cuda.empty_cache()
        return out
    # CPU path for very large item counts: LAPACK potrf+potri (inverse from the
    # Cholesky factor, in-place, ~n^3*2/3 flops total vs ~n^3*4/3 for cho_solve(eye);
    # no n x n identity allocation).
    from scipy.linalg import lapack as ll
    G[np.arange(n_items), np.arange(n_items)] += lam
    potrf, potri = (ll.dpotrf, ll.dpotri) if np_dtype == np.float64 else (ll.spotrf, ll.spotri)
    cf, info = potrf(G, lower=1, clean=0, overwrite_a=1)
    if info != 0:
        raise RuntimeError(f"potrf failed (info={info})")
    P, info = potri(cf, lower=1, overwrite_c=1)
    if info != 0:
        raise RuntimeError(f"potri failed (info={info})")
    del G, cf
    # potri fills only the lower triangle; mirror it blockwise (no big temp):
    # off-diagonal column blocks, then the strict-upper part of each diagonal block.
    step = 4096
    for i0 in range(0, n_items, step):
        i1 = min(i0 + step, n_items)
        P[:i0, i0:i1] = P[i0:i1, :i0].T
        blk = P[i0:i1, i0:i1]
        il = np.tril_indices(i1 - i0, -1)
        blk.T[il] = blk[il]
    diag = np.diag(P).copy()
    P /= (-diag)[None, :]          # in-place: no second n x n allocation
    np.fill_diagonal(P, 0.0)
    if P.dtype != np.float32:      # only the small-n float64 path copies
        P = P.astype(np.float32)
    return torch.from_numpy(P)


class EASEScorer:
    """Duck-typed shim for the Evaluator's ``full_scores_for_users`` hook."""

    def __init__(self, R: sp.csr_matrix, B: torch.Tensor, device):
        self._R = R.tocsr().astype(np.float32)
        self._B = B
        self._device = device

    def eval(self):  # Evaluator calls model.eval()
        return self

    @torch.no_grad()
    def full_scores_for_users(self, batch_users: np.ndarray, device) -> torch.Tensor:
        rows = np.asarray(self._R[np.asarray(batch_users, dtype=np.int64)].todense(),
                          dtype=np.float32)
        x = torch.from_numpy(rows).to(self._B.device)
        return (x @ self._B).to(device)
