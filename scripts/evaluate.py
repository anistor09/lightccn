"""Evaluate a saved model checkpoint.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/lightgcn_gowalla_best.pt
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch

from light_ccn.config import ExperimentConfig, _dict_to_config
from light_ccn.data.dataset import CFDataset
from light_ccn.complex.adjacency import (
    build_lightgcn_adj,
    build_lightccn_flat_adj,
    build_multi_operators,
)
from light_ccn.evaluation.metrics import Evaluator
from light_ccn.utils.helpers import set_seed, scipy_to_torch_sparse


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = _dict_to_config(ckpt["config"])
    set_seed(config.train.seed)

    device = torch.device(config.train.device if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = CFDataset(config.data.name, config.data.data_dir)

    # Import models to trigger registration
    import light_ccn.models.lightgcn
    import light_ccn.models.ngcf
    import light_ccn.models.sgl
    import light_ccn.models.lightccn_flat
    import light_ccn.models.lightccn_multi

    from light_ccn.models import build_model

    model_name = config.model.name

    # Build model and adjacency (same logic as train.py)
    if model_name in ("lightgcn", "ngcf"):
        adj = build_lightgcn_adj(dataset.get_bipartite_adjacency())
        adj_tensor = scipy_to_torch_sparse(adj).to(device)
        kwargs = dict(
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            embed_dim=config.model.embed_dim,
            n_layers=config.model.n_layers,
            adj_matrix=adj_tensor,
        )
        if model_name == "ngcf":
            kwargs["node_dropout"] = config.model.node_dropout
            kwargs["message_dropout"] = config.model.message_dropout
        model = build_model(model_name, **kwargs)

    elif model_name == "sgl":
        adj = build_lightgcn_adj(dataset.get_bipartite_adjacency())
        adj_tensor = scipy_to_torch_sparse(adj).to(device)
        model = build_model(
            model_name,
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            embed_dim=config.model.embed_dim,
            n_layers=config.model.n_layers,
            adj_matrix=adj_tensor,
            bipartite_adj_scipy=dataset.get_bipartite_adjacency(),
            augment_ratio=config.model.augment_ratio,
            device=device,
        )

    elif model_name == "lightccn_flat":
        from light_ccn.complex.cell_complex import CellComplexBuilder
        R = dataset.get_interaction_matrix()
        builder = CellComplexBuilder(
            R, tau=config.complex.tau,
            cache_dir=config.complex.cache_dir,
            dataset_name=config.data.name,
        )
        complex_data = builder.build_and_cache()
        S = complex_data["S"]
        adj = build_lightccn_flat_adj(
            R, S, dataset.n_users, dataset.n_items, gamma=config.complex.gamma
        )
        adj_tensor = scipy_to_torch_sparse(adj).to(device)
        model = build_model(
            model_name,
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            embed_dim=config.model.embed_dim,
            n_layers=config.model.n_layers,
            adj_matrix=adj_tensor,
        )

    elif model_name == "lightccn_multi":
        from light_ccn.complex.cell_complex import CellComplexBuilder
        R = dataset.get_interaction_matrix()
        builder = CellComplexBuilder(
            R, tau=config.complex.tau,
            cache_dir=config.complex.cache_dir,
            dataset_name=config.data.name,
        )
        complex_data = builder.build_and_cache()
        B1 = complex_data["B1"]
        B2 = complex_data["B2"]
        n_edges = B1.shape[1]
        n_faces = B2.shape[1]

        bipartite_adj = dataset.get_bipartite_adjacency()
        operators = build_multi_operators(bipartite_adj, B1, B2)
        operators_torch = {
            k: scipy_to_torch_sparse(v).to(device) for k, v in operators.items()
        }
        model = build_model(
            model_name,
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            embed_dim=config.model.embed_dim,
            n_layers=config.model.n_layers,
            n_edges=n_edges,
            n_faces=n_faces,
            edge_embed_dim=config.model.edge_embed_dim,
            face_embed_dim=config.model.face_embed_dim,
            operators=operators_torch,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Load weights
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    # Evaluate
    evaluator = Evaluator(topk=config.eval.topk)
    metrics = evaluator.evaluate(
        model, dataset,
        eval_batch_size=config.eval.eval_batch_size,
        device=device,
    )

    print(f"\nEvaluation Results: {config.model.name} on {config.data.name}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', '?')}")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
