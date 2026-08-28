"""Evaluation metrics with full-ranking protocol.

Rank-based evaluator: for each test user every item is scored, training items
are masked, and the *ranks of the user's test items* are extracted (stable
sort — ties broken by item id, so results are deterministic across runs and
frameworks). Every reported metric is then a function of those ranks:

  recall@K  = |test items in top-K| / n_rel            (standard denominator)
  ndcg@K    = DCG@K / IDCG@K (IDCG over min(n_rel, K) ideal positions)
  hr@K      = 1 if any test item in top-K
  mrr@K     = 1 / rank of FIRST test item, 0 if beyond K
              (equals RecWalk's ARHR when each user has exactly one test item)
  arhr@K    = mean over the user's test items of 1/rank (0 beyond K)

The ranks themselves can be returned (``return_ranks=True``) so the caller
can persist them: they are a sufficient statistic for every rank-based metric
at any cutoff and for per-user significance tests.

Also logged per evaluation (full eval only): mean pairwise cosine similarity
of the propagated embeddings — cos_uu (user-user), cos_ii (item-item),
cos_ui (user-item) — the MAD-style over-smoothing measurement, computed
exactly in O(N·d) via the identity  E[cos] = (||Σ û||² − n) / (n(n−1)).
"""

from __future__ import annotations

import numpy as np
import torch


# Precompute log2 discount factors for NDCG (positions 1..max_k)
_LOG2_DISCOUNT_CACHE: np.ndarray | None = None


def _get_discount(max_k: int) -> np.ndarray:
    """Get 1/log2(i+2) discount factors, cached."""
    global _LOG2_DISCOUNT_CACHE
    if _LOG2_DISCOUNT_CACHE is None or len(_LOG2_DISCOUNT_CACHE) < max_k:
        _LOG2_DISCOUNT_CACHE = 1.0 / np.log2(np.arange(2, max_k + 2, dtype=np.float64))
    return _LOG2_DISCOUNT_CACHE[:max_k]


def _mean_pairwise_cos(emb: torch.Tensor) -> float:
    """Exact mean pairwise cosine similarity over ALL pairs in O(N·d)."""
    n = emb.shape[0]
    if n < 2:
        return 0.0
    unit = torch.nn.functional.normalize(emb.float(), dim=1)
    s = unit.sum(dim=0)
    total = (s @ s).item()  # = n + 2*sum_{i<j} cos_ij
    return float((total - n) / (n * (n - 1)))


def _mean_cross_cos(emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
    """Exact mean cosine similarity between all (a, b) pairs in O(N·d)."""
    if emb_a.shape[0] == 0 or emb_b.shape[0] == 0:
        return 0.0
    ua = torch.nn.functional.normalize(emb_a.float(), dim=1).sum(dim=0)
    ub = torch.nn.functional.normalize(emb_b.float(), dim=1).sum(dim=0)
    return float((ua @ ub).item() / (emb_a.shape[0] * emb_b.shape[0]))


def _lightgcn_smoothness(emb: torch.Tensor, R, d_own: np.ndarray,
                         d_other: np.ndarray) -> float:
    """LightGCN Eq. 17 smoothness, exact, via the bipartite factorization.

    S = sum_{u,v} c_{v->u} * ||e_u/||e_u|| - e_v/||e_v||||^2  with (Eq. 14)
    c_{v->u} = (1/sqrt(|N_u||N_v|)) * sum_{i in N_u & N_v} 1/|N_i|.

    c factors through the interaction graph, so the double sum collapses:
    with X = unit_emb / sqrt(d_own)[:, None], Y = R^T X, q = R^T (1/sqrt d_own):
    S = 2*sum_i q_i^2/d_other_i - 2*sum_i ||Y_i||^2/d_other_i.
    O(nnz(R)*d), no pairwise matrix. For S_I pass R^T with degrees swapped.
    Zero-degree rows contribute nothing (their c weights are all zero).
    """
    unit = torch.nn.functional.normalize(emb.float(), dim=1).cpu().numpy()
    inv_sqrt = np.zeros_like(d_own)
    nz = d_own > 0
    inv_sqrt[nz] = 1.0 / np.sqrt(d_own[nz])
    X = unit * inv_sqrt[:, None]
    Y = R.T @ X                                   # (n_other, d)
    q = R.T @ inv_sqrt                            # (n_other,)
    inv_other = np.zeros_like(d_other)
    nz_o = d_other > 0
    inv_other[nz_o] = 1.0 / d_other[nz_o]
    term_c = float((q * q * inv_other).sum())
    term_e = float(((Y * Y).sum(axis=1) * inv_other).sum())
    return 2.0 * term_c - 2.0 * term_e


class Evaluator:
    """Full-ranking evaluation for collaborative filtering.

    For each test user, ranks ALL items (excluding training interactions),
    extracts the ranks of the held-out test items, and computes
    Recall/NDCG/HR/MRR/ARHR@K from them.
    """

    def __init__(self, topk: list[int] | None = None):
        self.topk = topk or [3, 5, 10, 20, 50]

    def _empty(self) -> dict[str, float]:
        return {f"{m}@{k}": 0.0
                for m in ["recall", "ndcg", "hr", "mrr", "arhr"]
                for k in self.topk}

    @torch.no_grad()
    def evaluate(
        self,
        model,
        dataset,
        eval_batch_size: int = 4096,
        device: torch.device | str = "cpu",
        item_filter: set[int] | None = None,
        return_ranks: bool = False,
    ):
        """Run full-ranking evaluation.

        Args:
            model: A BaseCFModel (must have propagate method).
            dataset: A CFDataset with train_dict and test_dict.
            eval_batch_size: Batch size for scoring users.
            device: Device for computation.
            item_filter: If set, only these test items count as relevant hits.
                Used for tail-item evaluation.
            return_ranks: If True, return ``(results, rank_data)`` where
                rank_data = {'user_ids', 'n_rel', 'ranks'} — ranks is the
                flat uint32 array of 1-based test-item ranks, user-major, in
                each user's test_dict item order (stable across epochs).

        Returns:
            Dict of aggregate metrics (plus cos_uu/cos_ii/cos_ui on full
            evals), or ``(dict, rank_data)`` when return_ranks=True.
        """
        model.eval()
        max_k = max(self.topk)

        test_users = [u for u in dataset.test_dict if len(dataset.test_dict[u]) > 0]
        if not test_users:
            out = self._empty()
            return (out, None) if return_ranks else out

        test_users = np.array(sorted(test_users), dtype=np.int64)

        # ── Propagate ONCE and cache embeddings ──
        # Models without factorized embeddings (Mult-VAE, EASE) expose
        # ``full_scores_for_users(batch_users, device) -> (bsz, n_items)``
        # and skip the embedding path entirely.
        score_hook = getattr(model, "full_scores_for_users", None)
        if score_hook is None:
            user_all, item_all = model.propagate()
        else:
            user_all = item_all = None

        # Test items per user (optionally filtered for tail eval)
        filter_arr = None
        if item_filter is not None:
            filter_arr = np.array(sorted(item_filter), dtype=np.int64)
        test_item_arrays = {}
        for u in test_users:
            items = np.array(dataset.test_dict[u], dtype=np.int64)
            if filter_arr is not None:
                items = items[np.isin(items, filter_arr)]
            test_item_arrays[u] = items
        if item_filter is not None:
            test_users = np.array(
                [u for u in test_users if len(test_item_arrays[u]) > 0],
                dtype=np.int64)
            if len(test_users) == 0:
                out = self._empty()
                return (out, None) if return_ranks else out

        n_rel = np.array([len(test_item_arrays[u]) for u in test_users],
                         dtype=np.int64)
        n_users_eval = len(test_users)

        # ── Batched ranking: stable ranks of each user's test items ──
        flat_user_parts: list[np.ndarray] = []
        flat_rank_parts: list[np.ndarray] = []

        for start in range(0, n_users_eval, eval_batch_size):
            end = min(start + eval_batch_size, n_users_eval)
            batch_users = test_users[start:end]
            bsz = len(batch_users)

            if score_hook is None:
                users_tensor = torch.from_numpy(batch_users).to(device)
                user_e = user_all[users_tensor]
                scores = user_e @ item_all.T  # (bsz, n_items)
            else:
                scores = score_hook(batch_users, device)  # (bsz, n_items)

            # Mask training items
            for i, u in enumerate(batch_users):
                train_items = dataset.train_dict.get(u, [])
                if train_items:
                    scores[i, train_items] = -float("inf")

            # Stable descending sort: ties broken by item id (deterministic).
            order = torch.argsort(scores, dim=1, descending=True, stable=True)
            ranks_row = torch.empty_like(order, dtype=torch.int32)
            arange = torch.arange(scores.shape[1], dtype=torch.int32,
                                  device=order.device).expand(bsz, -1)
            ranks_row.scatter_(1, order, arange)  # 0-based rank of every item
            del order, scores

            # Gather the test items' ranks in one flat indexed read
            rows_l, cols_l = [], []
            for i, u in enumerate(batch_users):
                t = test_item_arrays[u]
                rows_l.append(np.full(len(t), i, dtype=np.int64))
                cols_l.append(t)
            rows = torch.from_numpy(np.concatenate(rows_l)).to(ranks_row.device)
            cols = torch.from_numpy(np.concatenate(cols_l)).to(ranks_row.device)
            batch_ranks = ranks_row[rows, cols].cpu().numpy().astype(np.int64) + 1
            del ranks_row

            user_idx = np.concatenate(
                [np.full(len(test_item_arrays[u]), start + i, dtype=np.int64)
                 for i, u in enumerate(batch_users)])
            flat_user_parts.append(user_idx)
            flat_rank_parts.append(batch_ranks)

        fu = np.concatenate(flat_user_parts)   # user index (0..U-1) per test item
        fr = np.concatenate(flat_rank_parts)   # 1-based rank per test item

        # ── All metrics from ranks (vectorized, no per-user loops) ──
        cum_discount = np.cumsum(_get_discount(max_k))
        min_rank = np.full(n_users_eval, np.inf)
        np.minimum.at(min_rank, fu, fr)

        results: dict[str, float] = {}
        for k in self.topk:
            in_k = fr <= k
            fu_k, fr_k = fu[in_k], fr[in_k]
            hit_counts = np.bincount(fu_k, minlength=n_users_eval).astype(np.float64)
            results[f"recall@{k}"] = float(np.mean(hit_counts / n_rel))
            dcg = np.bincount(fu_k, weights=1.0 / np.log2(fr_k + 1.0),
                              minlength=n_users_eval)
            idcg = cum_discount[np.minimum(n_rel, k) - 1]
            results[f"ndcg@{k}"] = float(np.mean(dcg / idcg))
            results[f"hr@{k}"] = float(np.mean(hit_counts > 0))
            results[f"mrr@{k}"] = float(np.mean(
                np.where(min_rank <= k, 1.0 / np.maximum(min_rank, 1.0), 0.0)))
            arhr = np.bincount(fu_k, weights=1.0 / fr_k, minlength=n_users_eval)
            results[f"arhr@{k}"] = float(np.mean(arhr / n_rel))

        # ── Over-smoothing measurement (full eval only) ──
        if item_filter is None and user_all is not None:
            try:
                results["cos_uu"] = _mean_pairwise_cos(user_all)
                results["cos_ii"] = _mean_pairwise_cos(item_all)
                results["cos_ui"] = _mean_cross_cos(user_all, item_all)
            except Exception:
                pass  # never let diagnostics kill an eval
            try:
                # LightGCN §4.4.3 smoothness (S_U / S_I), graph cached per
                # dataset object (train_dict = the graph the model trained on).
                key = id(dataset)
                if getattr(self, "_smooth_key", None) != key:
                    import scipy.sparse as _sp
                    n_u = int(user_all.shape[0])
                    n_i = int(item_all.shape[0])
                    rows, cols = [], []
                    for u, its in dataset.train_dict.items():
                        rows.extend([u] * len(its))
                        cols.extend(its)
                    R = _sp.csr_matrix(
                        (np.ones(len(rows), dtype=np.float64), (rows, cols)),
                        shape=(n_u, n_i))
                    self._smooth_key = key
                    self._smooth_R = R
                    self._smooth_du = np.asarray(R.sum(axis=1)).ravel()
                    self._smooth_di = np.asarray(R.sum(axis=0)).ravel()
                results["s_u"] = _lightgcn_smoothness(
                    user_all, self._smooth_R, self._smooth_du, self._smooth_di)
                results["s_i"] = _lightgcn_smoothness(
                    item_all, self._smooth_R.T.tocsr(),
                    self._smooth_di, self._smooth_du)
            except Exception:
                pass  # never let diagnostics kill an eval

        if return_ranks:
            rank_data = {
                "user_ids": test_users,
                "n_rel": n_rel.astype(np.int32),
                "ranks": fr.astype(np.uint32),
            }
            return results, rank_data
        return results
