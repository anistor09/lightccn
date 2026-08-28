"""Tests for dataset loading and adjacency construction."""

import tempfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import pytest

from light_ccn.data.dataset import CFDataset


def _create_toy_data(tmpdir: Path) -> Path:
    """Create a tiny dataset in LightGCN format."""
    dataset_dir = tmpdir / "toy"
    dataset_dir.mkdir(parents=True)

    # 4 users, 5 items
    # train.txt: user_id item1 item2 ...
    (dataset_dir / "train.txt").write_text(
        "0 0 1 2\n"
        "1 1 2 3\n"
        "2 0 3 4\n"
        "3 2 3 4\n"
    )
    (dataset_dir / "test.txt").write_text(
        "0 3 4\n"
        "1 0 4\n"
        "2 1 2\n"
        "3 0 1\n"
    )
    return dataset_dir


class TestCFDataset:
    def test_load_shapes(self, tmp_path):
        dataset_dir = _create_toy_data(tmp_path)
        ds = CFDataset.__new__(CFDataset)
        ds.name = "toy"
        ds.data_dir = str(tmp_path)
        ds.dataset_dir = dataset_dir
        ds.train_dict = {}
        ds.test_dict = {}
        ds.n_users = 0
        ds.n_items = 0
        ds.n_train = 0
        ds.n_test = 0
        ds._load()

        assert ds.n_users == 4
        assert ds.n_items == 5
        assert ds.n_train == 12  # 3+3+3+3
        assert ds.n_test == 8   # 2+2+2+2

    def test_interaction_matrix(self, tmp_path):
        dataset_dir = _create_toy_data(tmp_path)
        ds = CFDataset.__new__(CFDataset)
        ds.name = "toy"
        ds.data_dir = str(tmp_path)
        ds.dataset_dir = dataset_dir
        ds.train_dict = {}
        ds.test_dict = {}
        ds.n_users = 0
        ds.n_items = 0
        ds.n_train = 0
        ds.n_test = 0
        ds._load()

        R = ds.get_interaction_matrix()
        assert R.shape == (4, 5)
        assert R.nnz == 12
        # User 0 interacts with items 0, 1, 2
        assert R[0, 0] == 1.0
        assert R[0, 1] == 1.0
        assert R[0, 2] == 1.0
        assert R[0, 3] == 0.0

    def test_bipartite_adjacency(self, tmp_path):
        dataset_dir = _create_toy_data(tmp_path)
        ds = CFDataset.__new__(CFDataset)
        ds.name = "toy"
        ds.data_dir = str(tmp_path)
        ds.dataset_dir = dataset_dir
        ds.train_dict = {}
        ds.test_dict = {}
        ds.n_users = 0
        ds.n_items = 0
        ds.n_train = 0
        ds.n_test = 0
        ds._load()

        A = ds.get_bipartite_adjacency()
        n = ds.n_users + ds.n_items
        assert A.shape == (n, n)
        # Should be symmetric
        diff = (A - A.T).nnz
        assert diff == 0
        # Top-left and bottom-right blocks should be zero
        assert A[:ds.n_users, :ds.n_users].nnz == 0
        assert A[ds.n_users:, ds.n_users:].nnz == 0

    def test_train_users(self, tmp_path):
        dataset_dir = _create_toy_data(tmp_path)
        ds = CFDataset.__new__(CFDataset)
        ds.name = "toy"
        ds.data_dir = str(tmp_path)
        ds.dataset_dir = dataset_dir
        ds.train_dict = {}
        ds.test_dict = {}
        ds.n_users = 0
        ds.n_items = 0
        ds.n_train = 0
        ds.n_test = 0
        ds._load()

        users = ds.get_train_users()
        assert len(users) == 4
        assert list(users) == [0, 1, 2, 3]
