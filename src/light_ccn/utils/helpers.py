"""Utility functions: seeding, sparse conversions, timing."""

import time
import random
from contextlib import contextmanager

import numpy as np
import torch
import scipy.sparse as sp


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed.
        deterministic: If True, disable cudnn.benchmark and TF32 for exact
            reproducibility (slower). Default False for speed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        # TF32: ~2x faster float32 matmul on Ampere+ GPUs (A100, L4, etc.)
        # with negligible precision loss. Safe for CF training.
        torch.backends.cuda.matmul.allow_tf32 = not deterministic
        torch.backends.cudnn.allow_tf32 = not deterministic


def scipy_to_torch_sparse(mat: sp.spmatrix) -> torch.Tensor:
    """Convert a scipy sparse matrix to a PyTorch sparse COO tensor."""
    mat = mat.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
    values = torch.from_numpy(mat.data)
    shape = torch.Size(mat.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


@contextmanager
def timer(label: str = ""):
    """Context manager that prints elapsed time."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"[{label}] {elapsed:.2f}s" if label else f"Elapsed: {elapsed:.2f}s")
