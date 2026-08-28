"""Tests for cell complex construction on synthetic data."""

import numpy as np
import scipy.sparse as sp
import pytest

from light_ccn.complex.cell_complex import CellComplexBuilder


def _make_synthetic_R(n_users=50, n_items=6):
    """Create a synthetic interaction matrix where items 0,1,2 form a face.

    At least 20 users interact with all of items 0, 1, 2.
    """
    rows, cols = [], []
    # First 25 users interact with items 0, 1, 2 (all three)
    for u in range(25):
        for i in [0, 1, 2]:
            rows.append(u)
            cols.append(i)

    # Some extra interactions to make it realistic
    for u in range(25, 50):
        for i in [0, 1]:
            rows.append(u)
            cols.append(i)
    # Add item 3,4,5 interactions
    for u in range(50):
        rows.append(u)
        cols.append(3 + u % 3)

    values = np.ones(len(rows), dtype=np.float32)
    R = sp.csr_matrix((values, (rows, cols)), shape=(n_users, n_items))
    return R


class TestCellComplexBuilder:
    def test_find_faces_basic(self, tmp_path):
        R = _make_synthetic_R()
        builder = CellComplexBuilder(
            R, tau=20,
            cache_dir=str(tmp_path / "cache"),
            dataset_name="synthetic",
        )
        faces = builder.find_faces()

        # Should find at least the (0, 1, 2) face
        assert len(faces) >= 1
        # Check that (0,1,2) is in the faces
        assert (0, 1, 2) in faces

    def test_find_faces_high_tau(self, tmp_path):
        R = _make_synthetic_R()
        builder = CellComplexBuilder(
            R, tau=30,
            cache_dir=str(tmp_path / "cache"),
            dataset_name="synthetic",
        )
        faces = builder.find_faces()
        # With tau=30, only 25 users share all 3, so face should NOT appear
        assert (0, 1, 2) not in faces

    def test_item_item_adjacency(self):
        faces = [(0, 1, 2), (1, 2, 3)]
        S = CellComplexBuilder.build_item_item_adjacency(faces, n_items=5)

        assert S.shape == (5, 5)
        # Edge (1,2) appears in both faces, so S[1,2] = 2
        assert S[1, 2] == 2
        assert S[2, 1] == 2
        # Edge (0,1) appears in 1 face
        assert S[0, 1] == 1
        assert S[1, 0] == 1
        # No edge between 0 and 3
        assert S[0, 3] == 0

    def test_incidence_B1(self):
        edges = [(0, 1), (0, 2), (1, 2)]
        B1 = CellComplexBuilder.build_incidence_B1(edges, n_nodes=4)

        assert B1.shape == (4, 3)
        # Edge 0 connects nodes 0 and 1
        assert B1[0, 0] == 1
        assert B1[1, 0] == 1
        assert B1[2, 0] == 0
        # Each edge has exactly 2 boundary nodes
        for e in range(3):
            assert B1[:, e].sum() == 2

    def test_incidence_B2(self):
        edges = [(0, 1), (0, 2), (1, 2)]
        faces = [(0, 1, 2)]
        B2 = CellComplexBuilder.build_incidence_B2(edges, faces)

        assert B2.shape == (3, 1)
        # All 3 edges are boundary of the single face
        for e in range(3):
            assert B2[e, 0] == 1

    def test_build_and_cache(self, tmp_path):
        R = _make_synthetic_R()
        builder = CellComplexBuilder(
            R, tau=20,
            cache_dir=str(tmp_path / "cache"),
            dataset_name="synthetic",
        )

        # First build
        result = builder.build_and_cache()
        assert "faces" in result
        assert "edges" in result
        assert "S" in result
        assert "B1" in result
        assert "B2" in result

        n_faces = len(result["faces"])
        n_edges = len(result["edges"])
        assert n_faces >= 1
        assert n_edges >= 3  # At least 3 edges from 1 face

        # B2 shape: (n_edges, n_faces)
        assert result["B2"].shape == (n_edges, n_faces)

        # Second load from cache
        result2 = builder.build_and_cache()
        assert len(result2["faces"]) == n_faces
