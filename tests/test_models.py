"""Tests for all model forward passes on tiny synthetic data."""

import numpy as np
import scipy.sparse as sp
import torch
import pytest

from light_ccn.complex.adjacency import (
    symmetric_norm,
    build_lightgcn_adj,
    build_lightccn_flat_adj,
    build_multi_operators,
)
from light_ccn.utils.helpers import scipy_to_torch_sparse

# Import to register models
import light_ccn.models.lightgcn
import light_ccn.models.ngcf
import light_ccn.models.sgl
import light_ccn.models.lightccn_flat
import light_ccn.models.lightccn_multi

from light_ccn.models import build_model


N_USERS = 10
N_ITEMS = 8
EMBED_DIM = 16
BATCH = 4


def _make_bipartite_adj():
    """Create a tiny bipartite adjacency matrix."""
    n = N_USERS + N_ITEMS
    rows = list(range(N_USERS)) + list(range(N_USERS, n))
    cols = [N_USERS + (u % N_ITEMS) for u in range(N_USERS)] + \
           [(i - N_USERS) for i in range(N_USERS, n)]
    vals = np.ones(len(rows), dtype=np.float32)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return A


def _make_interaction_matrix():
    """Create a tiny R matrix."""
    rows = list(range(N_USERS))
    cols = [u % N_ITEMS for u in range(N_USERS)]
    vals = np.ones(len(rows), dtype=np.float32)
    R = sp.csr_matrix((vals, (rows, cols)), shape=(N_USERS, N_ITEMS))
    return R


def _make_sample_batch():
    users = torch.randint(0, N_USERS, (BATCH,))
    pos_items = torch.randint(0, N_ITEMS, (BATCH,))
    neg_items = torch.randint(0, N_ITEMS, (BATCH,))
    return users, pos_items, neg_items


class TestLightGCN:
    def test_forward_shapes(self):
        A = _make_bipartite_adj()
        adj = build_lightgcn_adj(A)
        adj_t = scipy_to_torch_sparse(adj)

        model = build_model(
            "lightgcn",
            n_users=N_USERS, n_items=N_ITEMS,
            embed_dim=EMBED_DIM, n_layers=2,
            adj_matrix=adj_t,
        )

        users, pos, neg = _make_sample_batch()
        user_e, pos_e, neg_e, reg = model(users, pos, neg)

        assert user_e.shape == (BATCH, EMBED_DIM)
        assert pos_e.shape == (BATCH, EMBED_DIM)
        assert neg_e.shape == (BATCH, EMBED_DIM)
        assert reg.ndim == 0  # scalar

    def test_all_ratings(self):
        A = _make_bipartite_adj()
        adj = build_lightgcn_adj(A)
        adj_t = scipy_to_torch_sparse(adj)

        model = build_model(
            "lightgcn",
            n_users=N_USERS, n_items=N_ITEMS,
            embed_dim=EMBED_DIM, n_layers=2,
            adj_matrix=adj_t,
        )

        users = torch.arange(3)
        scores = model.get_all_ratings(users)
        assert scores.shape == (3, N_ITEMS)


class TestNGCF:
    def test_forward_shapes(self):
        A = _make_bipartite_adj()
        adj = build_lightgcn_adj(A)
        adj_t = scipy_to_torch_sparse(adj)

        model = build_model(
            "ngcf",
            n_users=N_USERS, n_items=N_ITEMS,
            embed_dim=EMBED_DIM, n_layers=2,
            adj_matrix=adj_t,
            node_dropout=0.0,
            message_dropout=0.0,
        )

        users, pos, neg = _make_sample_batch()
        user_e, pos_e, neg_e, reg = model(users, pos, neg)

        # NGCF concatenates layers: dim = (n_layers+1) * embed_dim
        expected_dim = (2 + 1) * EMBED_DIM
        assert user_e.shape == (BATCH, expected_dim)
        assert pos_e.shape == (BATCH, expected_dim)


class TestSGL:
    def test_forward_and_contrastive(self):
        A_scipy = _make_bipartite_adj()
        adj = build_lightgcn_adj(A_scipy)
        adj_t = scipy_to_torch_sparse(adj)

        model = build_model(
            "sgl",
            n_users=N_USERS, n_items=N_ITEMS,
            embed_dim=EMBED_DIM, n_layers=2,
            adj_matrix=adj_t,
            bipartite_adj_scipy=A_scipy,
            augment_ratio=0.1,
            device="cpu",
        )

        users, pos, neg = _make_sample_batch()
        user_e, pos_e, neg_e, reg = model(users, pos, neg)

        assert user_e.shape == (BATCH, EMBED_DIM)

        # Contrastive views should be available
        uz1, uz2, iz1, iz2 = model.get_contrastive_views()
        assert uz1.shape == (N_USERS, EMBED_DIM)
        assert iz1.shape == (N_ITEMS, EMBED_DIM)


class TestLightCCNFlat:
    def test_forward_shapes(self):
        R = _make_interaction_matrix()
        # Fake item-item adjacency
        S = sp.eye(N_ITEMS, dtype=np.float32) * 0  # no faces = zero S
        adj = build_lightccn_flat_adj(R, S, N_USERS, N_ITEMS, gamma=0.1)
        adj_t = scipy_to_torch_sparse(adj)

        model = build_model(
            "lightccn_flat",
            n_users=N_USERS, n_items=N_ITEMS,
            embed_dim=EMBED_DIM, n_layers=2,
            adj_matrix=adj_t,
        )

        users, pos, neg = _make_sample_batch()
        user_e, pos_e, neg_e, reg = model(users, pos, neg)

        assert user_e.shape == (BATCH, EMBED_DIM)


class TestLightCCNMulti:
    def test_forward_shapes(self):
        A_scipy = _make_bipartite_adj()
        n_nodes = N_USERS + N_ITEMS

        # Create tiny B1 and B2
        n_edges = 3
        n_faces = 1
        B1 = sp.csr_matrix(
            np.array([
                [0]*n_edges,  # user nodes
            ] * N_USERS + [
                [1, 1, 0],  # item 0 -> edges 0,1
                [1, 0, 1],  # item 1 -> edges 0,2
                [0, 1, 1],  # item 2 -> edges 1,2
            ] + [[0]*n_edges] * (N_ITEMS - 3),
            dtype=np.float32)
        )
        B2 = sp.csr_matrix(
            np.array([[1], [1], [1]], dtype=np.float32)
        )

        operators = build_multi_operators(A_scipy, B1, B2)
        operators_t = {k: scipy_to_torch_sparse(v) for k, v in operators.items()}

        model = build_model(
            "lightccn_multi",
            n_users=N_USERS, n_items=N_ITEMS,
            embed_dim=EMBED_DIM, n_layers=2,
            n_edges=n_edges, n_faces=n_faces,
            edge_embed_dim=EMBED_DIM, face_embed_dim=EMBED_DIM,
            operators=operators_t,
        )

        users, pos, neg = _make_sample_batch()
        user_e, pos_e, neg_e, reg = model(users, pos, neg)

        assert user_e.shape == (BATCH, EMBED_DIM)
        assert pos_e.shape == (BATCH, EMBED_DIM)
        assert reg.ndim == 0
