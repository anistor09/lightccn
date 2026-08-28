"""Main training entry point.

Usage:
    python scripts/train.py --config configs/lightgcn_gowalla.yaml
    python scripts/train.py --config configs/lightccn_flat_gowalla.yaml

    # Dry run — smoke test with 500 users, 3 epochs (~1-2 min)
    python scripts/train.py --config configs/lightccn_flat_gowalla.yaml --dry-run

    # Quick ablation — 100 epochs max, early stop patience 20 (~10-15 min per run)
    python scripts/train.py --config configs/ablation/lightccn_flat_gowalla_g0.3_t20.yaml --quick
"""

import argparse
import json
import math
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import torch

from light_ccn.config import load_config
from light_ccn.data.dataset import CFDataset
from light_ccn.complex.adjacency import (
    build_lightgcn_adj,
    build_lightccn_flat_adj,
    build_lightccn_flat_split_adj,
    build_hodge_operators,
    build_multi_operators,
)
from light_ccn.complex.cell_complex import filter_faces
from light_ccn.utils.helpers import set_seed, scipy_to_torch_sparse


def main():
    parser = argparse.ArgumentParser(description="Train a CF model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--dry-run", action="store_true",
                        help="Smoke test: 500 users, 3 epochs, no checkpoint.")
    parser.add_argument("--quick", action="store_true",
                        help="Cheap ablation: 100 epochs max, patience=20, eval every 5.")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip checkpoint and result file saving.")
    parser.add_argument("--freeze-w2", type=float, default=None,
                        help="Freeze w2 (edge->node weight) at this value.")
    parser.add_argument("--n-layers", type=int, default=None,
                        help="Override model.n_layers from config.")
    parser.add_argument("--signed-weights", action="store_true",
                        help="(Deprecated) Use --weight-mode signed instead.")
    parser.add_argument("--weight-mode", type=str, default="softmax",
                        choices=["softmax", "signed", "tanh", "softplus"],
                        help="Weight parameterization mode for multi model.")
    parser.add_argument("--weight-reg", type=float, default=0.0,
                        help="L2 reg coefficient for attention weights.")
    parser.add_argument("--nodes-only", action="store_true",
                        help="Deduce edge/face from nodes (no trainable higher-rank embeddings).")
    parser.add_argument("--no-faces", action="store_true",
                        help="Skip faces (2-cells); edges-only topology.")
    parser.add_argument("--tail-topology", type=str, default=None,
                        choices=["all_tail", "at_least_one"],
                        help="Filter faces for tail-item topology experiment.")
    parser.add_argument("--tail-pct", type=float, default=20.0,
                        help="Bottom percentile for tail items (default 20).")
    parser.add_argument("--user-topology", action="store_true",
                        help="Add user-side cell complex (user-user adjacency).")
    parser.add_argument("--user-tau", type=int, default=None,
                        help="Tau for user-side complex (default: same as item tau).")
    parser.add_argument("--user-tail-filter", type=str, default=None,
                        choices=["all_tail", "at_least_one"],
                        help="Filter user faces for tail-user topology experiment.")
    parser.add_argument("--gamma-user", type=float, default=None,
                        help="Weight for user-user edges (default: same as gamma).")
    parser.add_argument("--eval-tail", action="store_true",
                        help="Also evaluate on tail items only (no topology change).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed (default: from config, 2020).")
    args = parser.parse_args()

    config = load_config(args.config)

    # Override training params for dry-run / quick mode
    if args.dry_run:
        config.train.epochs = 3
        config.train.eval_every = 1
        config.train.early_stop_patience = 10
        config.results_dir = "results/dry_run"
        config.checkpoint_dir = "checkpoints/dry_run"
        print("[DRY RUN] 500 users, 3 epochs — smoke test only")
    elif args.quick:
        config.train.epochs = 50
        config.train.eval_every = 5
        config.train.early_stop_patience = 15
        config.results_dir = "results/ablation"
        print("[QUICK] 50 epochs max, patience=15, eval every 5")

    if args.n_layers is not None:
        config.model.n_layers = args.n_layers
        print(f"[OVERRIDE] n_layers = {args.n_layers}")

    if args.signed_weights:
        print("[SIGNED] Using unconstrained signed weights (no softmax)")
        if args.weight_reg > 0:
            print(f"[WEIGHT-REG] L2 reg on attention weights: {args.weight_reg}")

    seed = args.seed if args.seed is not None else config.train.seed
    set_seed(seed)
    print(f"[SEED] {seed}")
    requested = config.train.device
    if requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif requested == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load dataset
    dataset = CFDataset(config.data.name, config.data.data_dir)

    if args.dry_run:
        dataset.subsample(n_users=500)

    # Import models to trigger registration
    import light_ccn.models.lightgcn
    import light_ccn.models.ngcf
    import light_ccn.models.sgl
    import light_ccn.models.lightccn_flat
    import light_ccn.models.lightccn_multi
    import light_ccn.models.lightccn_hodge

    from light_ccn.models import build_model

    model_name = config.model.name

    if model_name == "lightgcn":
        adj = build_lightgcn_adj(dataset.get_bipartite_adjacency())
        adj_tensor = scipy_to_torch_sparse(adj).to(device)
        model = build_model(
            model_name,
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            embed_dim=config.model.embed_dim,
            n_layers=config.model.n_layers,
            adj_matrix=adj_tensor,
        )

    elif model_name == "ngcf":
        adj = build_lightgcn_adj(dataset.get_bipartite_adjacency())
        adj_tensor = scipy_to_torch_sparse(adj).to(device)
        model = build_model(
            model_name,
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            embed_dim=config.model.embed_dim,
            n_layers=config.model.n_layers,
            adj_matrix=adj_tensor,
            node_dropout=config.model.node_dropout,
            message_dropout=config.model.message_dropout,
        )

    elif model_name == "sgl":
        adj = build_lightgcn_adj(dataset.get_bipartite_adjacency())
        adj_tensor = scipy_to_torch_sparse(adj).to(device)
        bipartite_adj = dataset.get_bipartite_adjacency()
        model = build_model(
            model_name,
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            embed_dim=config.model.embed_dim,
            n_layers=config.model.n_layers,
            adj_matrix=adj_tensor,
            bipartite_adj_scipy=bipartite_adj,
            augment_ratio=config.model.augment_ratio,
            device=device,
        )

    elif model_name == "lightccn_flat":
        from light_ccn.complex.cell_complex import CellComplexBuilder, UserCellComplexBuilder

        R = dataset.get_interaction_matrix()
        builder = CellComplexBuilder(
            R, tau=config.complex.tau,
            cache_dir=config.complex.cache_dir,
            dataset_name=config.data.name,
        )
        complex_data = builder.build_and_cache()

        # Item face filtering for tail-item topology experiments
        if args.tail_topology:
            faces = complex_data["faces"]
            tail_items_set = dataset.get_tail_items(percentile=args.tail_pct)
            filtered = filter_faces(faces, tail_items_set, mode=args.tail_topology)
            print(f"[TAIL-ITEM] {len(faces)} -> {len(filtered)} faces "
                  f"(filter='{args.tail_topology}', {len(tail_items_set)} tail items)")
            S = CellComplexBuilder.build_item_item_adjacency(filtered, dataset.n_items)
        else:
            S = complex_data["S"]

        # User-side cell complex
        S_user = None
        if args.user_topology:
            user_tau = args.user_tau or config.complex.tau
            user_builder = UserCellComplexBuilder(
                R, tau=user_tau,
                cache_dir=config.complex.cache_dir,
                dataset_name=config.data.name,
            )
            user_complex = user_builder.build_and_cache()

            if args.user_tail_filter:
                user_faces = user_complex["faces"]
                tail_users_set = dataset.get_tail_users(percentile=args.tail_pct)
                filtered_uf = filter_faces(user_faces, tail_users_set, mode=args.user_tail_filter)
                print(f"[TAIL-USER] {len(user_faces)} -> {len(filtered_uf)} user faces "
                      f"(filter='{args.user_tail_filter}', {len(tail_users_set)} tail users)")
                S_user = UserCellComplexBuilder.build_user_user_adjacency(
                    filtered_uf, dataset.n_users,
                )
            else:
                S_user = user_complex["S_user"]

            nnz = S_user.nnz
            print(f"[USER-TOPO] S_user: {nnz} nonzeros, "
                  f"{S_user.shape[0]}x{S_user.shape[1]}")

        if args.signed_weights:
            # Split adjacency for learnable weights
            A_gcn, A_ii = build_lightccn_flat_split_adj(
                R, S, dataset.n_users, dataset.n_items
            )
            adj_tensor = scipy_to_torch_sparse(A_gcn).to(device)
            adj_ii_tensor = scipy_to_torch_sparse(A_ii).to(device)
            model = build_model(
                model_name,
                n_users=dataset.n_users,
                n_items=dataset.n_items,
                embed_dim=config.model.embed_dim,
                n_layers=config.model.n_layers,
                adj_matrix=adj_tensor,
                adj_item_item=adj_ii_tensor,
                signed_weights=True,
            )
        else:
            adj = build_lightccn_flat_adj(
                R, S, dataset.n_users, dataset.n_items,
                gamma=config.complex.gamma,
                S_user=S_user,
                gamma_user=args.gamma_user,
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

    elif model_name == "lightccn_hodge":
        from light_ccn.complex.cell_complex import (
            CellComplexBuilder, UserCellComplexBuilder,
        )
        import scipy.sparse as sp_sci

        R = dataset.get_interaction_matrix()
        builder = CellComplexBuilder(
            R, tau=config.complex.tau,
            cache_dir=config.complex.cache_dir,
            dataset_name=config.data.name,
        )
        complex_data = builder.build_and_cache()
        faces = complex_data["faces"]
        edges = complex_data["edges"]

        # Tail-item face filtering: rebuild B1 from filtered faces
        if args.tail_topology:
            tail_items_set = dataset.get_tail_items(percentile=args.tail_pct)
            filtered = filter_faces(faces, tail_items_set, mode=args.tail_topology)
            print(f"[TAIL-ITEM] {len(faces)} -> {len(filtered)} faces "
                  f"(filter='{args.tail_topology}', {len(tail_items_set)} tail items)")
            faces = filtered
            edge_set: set[tuple[int, int]] = set()
            for i, j, k in faces:
                edge_set.add((min(i, j), max(i, j)))
                edge_set.add((min(i, k), max(i, k)))
                edge_set.add((min(j, k), max(j, k)))
            edges = sorted(edge_set)

        n_nodes = dataset.n_users + dataset.n_items
        edges_bipartite = [(dataset.n_users + a, dataset.n_users + b) for a, b in edges]
        B1 = CellComplexBuilder.build_incidence_B1(edges_bipartite, n_nodes)

        # Build bipartite adjacency (optionally with user-user topology)
        bipartite_adj = dataset.get_bipartite_adjacency()

        if args.user_topology:
            user_tau = args.user_tau or config.complex.tau
            user_builder = UserCellComplexBuilder(
                R, tau=user_tau,
                cache_dir=config.complex.cache_dir,
                dataset_name=config.data.name,
            )
            user_complex = user_builder.build_and_cache()

            if args.user_tail_filter:
                user_faces = user_complex["faces"]
                tail_users_set = dataset.get_tail_users(percentile=args.tail_pct)
                filtered_uf = filter_faces(user_faces, tail_users_set, mode=args.user_tail_filter)
                print(f"[TAIL-USER] {len(user_faces)} -> {len(filtered_uf)} user faces")
                S_user = UserCellComplexBuilder.build_user_user_adjacency(
                    filtered_uf, dataset.n_users,
                )
            else:
                S_user = user_complex["S_user"]

            gamma_user = args.gamma_user if args.gamma_user is not None else config.complex.gamma
            S_user = S_user.tocsr().astype(np.float32)
            print(f"[USER-TOPO] S_user nnz={S_user.nnz}, gamma_user={gamma_user}")

            zero_ii = sp_sci.csr_matrix((dataset.n_items, dataset.n_items), dtype=np.float32)
            bipartite_adj = sp_sci.bmat([
                [gamma_user * S_user, R.tocsr().astype(np.float32)],
                [R.T.tocsr().astype(np.float32), zero_ii],
            ], format="csr")

        operators = build_hodge_operators(bipartite_adj, B1)
        operators_torch = {
            k: scipy_to_torch_sparse(v).to(device) for k, v in operators.items()
        }

        weight_mode = "signed" if args.signed_weights else args.weight_mode
        model = build_model(
            model_name,
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            embed_dim=config.model.embed_dim,
            n_layers=config.model.n_layers,
            operators=operators_torch,
            weight_mode=weight_mode,
        )
        print(f"[WEIGHTS] mode={weight_mode}, edges={B1.shape[1]}")

    elif model_name == "lightccn_multi":
        from light_ccn.complex.cell_complex import (
            CellComplexBuilder, UserCellComplexBuilder,
        )
        import scipy.sparse as sp_sci

        R = dataset.get_interaction_matrix()
        builder = CellComplexBuilder(
            R, tau=config.complex.tau,
            cache_dir=config.complex.cache_dir,
            dataset_name=config.data.name,
        )
        complex_data = builder.build_and_cache()
        faces = complex_data["faces"]
        edges = complex_data["edges"]

        # Tail-item face filtering: rebuild B1/B2 from filtered faces
        if args.tail_topology:
            tail_items_set = dataset.get_tail_items(percentile=args.tail_pct)
            filtered = filter_faces(faces, tail_items_set, mode=args.tail_topology)
            print(f"[TAIL-ITEM] {len(faces)} -> {len(filtered)} faces "
                  f"(filter='{args.tail_topology}', {len(tail_items_set)} tail items)")
            faces = filtered
            # Rebuild edges from filtered faces
            edge_set: set[tuple[int, int]] = set()
            for i, j, k in faces:
                edge_set.add((min(i, j), max(i, j)))
                edge_set.add((min(i, k), max(i, k)))
                edge_set.add((min(j, k), max(j, k)))
            edges = sorted(edge_set)

        n_nodes = dataset.n_users + dataset.n_items
        edges_bipartite = [(dataset.n_users + a, dataset.n_users + b) for a, b in edges]
        B1 = CellComplexBuilder.build_incidence_B1(edges_bipartite, n_nodes)
        B2 = CellComplexBuilder.build_incidence_B2(edges, faces)
        n_edges = B1.shape[1]

        if args.no_faces:
            n_faces = 0
            B2 = sp_sci.csr_matrix((n_edges, 0), dtype=np.float32)
            print(f"[NO-FACES] Skipping faces, edges-only topology ({n_edges} edges)")
        else:
            n_faces = B2.shape[1]

        # Build bipartite adjacency (optionally with user-user topology)
        bipartite_adj = dataset.get_bipartite_adjacency()

        if args.user_topology:
            user_tau = args.user_tau or config.complex.tau
            user_builder = UserCellComplexBuilder(
                R, tau=user_tau,
                cache_dir=config.complex.cache_dir,
                dataset_name=config.data.name,
            )
            user_complex = user_builder.build_and_cache()

            if args.user_tail_filter:
                user_faces = user_complex["faces"]
                tail_users_set = dataset.get_tail_users(percentile=args.tail_pct)
                filtered_uf = filter_faces(user_faces, tail_users_set, mode=args.user_tail_filter)
                print(f"[TAIL-USER] {len(user_faces)} -> {len(filtered_uf)} user faces")
                S_user = UserCellComplexBuilder.build_user_user_adjacency(
                    filtered_uf, dataset.n_users,
                )
            else:
                S_user = user_complex["S_user"]

            gamma_user = args.gamma_user if args.gamma_user is not None else config.complex.gamma
            S_user = S_user.tocsr().astype(np.float32)
            print(f"[USER-TOPO] S_user nnz={S_user.nnz}, gamma_user={gamma_user}")

            # Inject S_user into bipartite_adj: [[gamma*S_user, R], [R^T, 0]]
            zero_ii = sp_sci.csr_matrix((dataset.n_items, dataset.n_items), dtype=np.float32)
            bipartite_adj = sp_sci.bmat([
                [gamma_user * S_user, R.tocsr().astype(np.float32)],
                [R.T.tocsr().astype(np.float32), zero_ii],
            ], format="csr")

        operators = build_multi_operators(bipartite_adj, B1, B2)
        operators_torch = {
            k: scipy_to_torch_sparse(v).to(device) for k, v in operators.items()
        }

        # Resolve weight mode
        weight_mode = "signed" if args.signed_weights else args.weight_mode

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
            weight_mode=weight_mode,
            nodes_only=args.nodes_only,
        )
        if args.nodes_only:
            print(f"[NODES-ONLY] Edge/face deduced from nodes each layer")
        print(f"[WEIGHTS] mode={weight_mode}, edges={n_edges}, faces={n_faces}")

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Freeze w2 (edge->node weight) if requested
    if args.freeze_w2 is not None:
        w2_val = args.freeze_w2
        mode = getattr(model, "weight_mode", None)
        if hasattr(model, "node_weights"):
            # Signed mode — set directly
            model.node_weights.data = torch.tensor([1.0 - w2_val, w2_val])
            model.node_weights.requires_grad = False
            print(f"[FREEZE] w2 frozen at {model.node_weights[1].item():.4f} (signed)")
        elif hasattr(model, "node_logits"):
            # Invert the parameterization so the *effective* w2 == w2_val.
            EPS = 1e-8
            if mode == "softplus":
                # softplus(alpha) = log(1 + exp(alpha)). Inverse: log(exp(w2_val) - 1).
                # For w2_val == 0, use alpha = -50 so softplus(alpha) ~= 1.9e-22.
                if w2_val <= 0:
                    a1 = -50.0
                else:
                    a1 = math.log(math.expm1(w2_val))
                # Keep w1 at its default initial value (zeros -> softplus = ln(2) ~ 0.693).
                # Use the same target for w1 unless caller wants otherwise; here we keep
                # w1 learnable, so we initialize its logit at 0 (softplus(0)=ln 2) by default.
                w1_init = math.log(math.expm1(max(math.log(2.0), 1e-6)))  # ~= 0
                model.node_logits.data = torch.tensor([w1_init, a1], dtype=model.node_logits.dtype)
                # Freeze only w2 (index 1) — w1 (index 0) should stay learnable.
                # Cleanest way: make a parameter mask via a backward hook.
                w2_idx = 1
                def _zero_w2_grad(grad, idx=w2_idx):
                    g = grad.clone()
                    g[idx] = 0.0
                    return g
                model.node_logits.register_hook(_zero_w2_grad)
                # Re-affirm desired weight after any future overwrite by hook user:
                import torch.nn.functional as _F
                effective = _F.softplus(model.node_logits).detach().cpu().tolist()
                print(f"[FREEZE] w2 frozen at {effective[1]:.6f} (softplus, w1 still learnable)")
            elif mode == "softmax":
                # softmax([alpha_0, alpha_1]) -> we want w2 (index 1) == w2_val.
                # For w2_val == 0, set alpha_1 = -50, alpha_0 = 0.
                if w2_val <= 0:
                    a0, a1 = 0.0, -50.0
                elif w2_val >= 1:
                    a0, a1 = -50.0, 0.0
                else:
                    a0 = math.log(1.0 - w2_val)
                    a1 = math.log(w2_val)
                model.node_logits.data = torch.tensor([a0, a1], dtype=model.node_logits.dtype)
                model.node_logits.requires_grad = False
                import torch.nn.functional as _F
                w_check = _F.softmax(model.node_logits, dim=0)
                print(f"[FREEZE] w2 frozen at {w_check[1].item():.4f} (softmax, both weights frozen)")
            elif mode == "tanh":
                # tanh(alpha) = w2_val -> alpha = atanh(w2_val).
                a1 = math.atanh(max(min(w2_val, 1 - EPS), -1 + EPS)) if abs(w2_val) < 1 else (50.0 if w2_val > 0 else -50.0)
                # Keep w1 at atanh(0.5) ~= 0.549 (the default init for tanh mode).
                a0 = math.atanh(0.5)
                model.node_logits.data = torch.tensor([a0, a1], dtype=model.node_logits.dtype)
                w2_idx = 1
                def _zero_w2_grad(grad, idx=w2_idx):
                    g = grad.clone()
                    g[idx] = 0.0
                    return g
                model.node_logits.register_hook(_zero_w2_grad)
                effective = torch.tanh(model.node_logits).detach().cpu().tolist()
                print(f"[FREEZE] w2 frozen at {effective[1]:.4f} (tanh, w1 still learnable)")
            else:
                # Fallback: treat as softmax with safe clamping
                w_safe = max(min(w2_val, 1.0 - EPS), EPS)
                a0 = math.log(1.0 - w_safe)
                a1 = math.log(w_safe)
                model.node_logits.data = torch.tensor([a0, a1], dtype=model.node_logits.dtype)
                model.node_logits.requires_grad = False
                print(f"[FREEZE] w2 frozen at {w_safe:.4f} (default/softmax fallback)")

    # Set L2 weight regularization (signed mode)
    if args.weight_reg > 0:
        model.weight_reg = args.weight_reg

    # Compute tail items for dual evaluation
    if args.tail_topology or args.user_topology or args.eval_tail:
        tail_items = dataset.get_tail_items(percentile=args.tail_pct)
        print(f"[TAIL-EVAL] {len(tail_items)} tail items (bottom {args.tail_pct}%)")
    else:
        tail_items = None

    # Select trainer
    save_enabled = not args.no_save
    if model_name == "sgl":
        from light_ccn.training.trainer import SGLTrainer
        trainer = SGLTrainer(model, dataset, config, save_enabled=save_enabled,
                             tail_items=tail_items)
    else:
        from light_ccn.training.trainer import Trainer
        trainer = Trainer(model, dataset, config, save_enabled=save_enabled,
                          tail_items=tail_items)

    results = trainer.train()

    # Always save to /tmp for programmatic access (e.g., from notebooks)
    tmp_path = Path("/tmp/lightccn_last_result.json")
    with open(tmp_path, "w") as f:
        json.dump(results, f)
    print(f"\nTemp results: {tmp_path}")

    # Print final metrics
    print("\n" + "=" * 60)
    print(f"Final Results: {config.model.name} on {config.data.name}")
    print("=" * 60)
    if results["final_metrics"]:
        print("  All items:")
        for k, v in results["final_metrics"].items():
            if k != "epoch":
                print(f"    {k}: {v:.4f}")
    if results.get("tail_final_metrics"):
        print("  Tail items:")
        for k, v in results["tail_final_metrics"].items():
            if k != "epoch":
                print(f"    {k}: {v:.4f}")
    print(f"  Best epoch: {results['best_epoch']}")

    # Print learned attention weights
    if "attention_weights" in results and results["attention_weights"]:
        aw = results["attention_weights"]
        print(f"\n  Attention weights (learned):")
        if "face" in aw:
            # Multi model: 7 weights
            n = aw["node"]
            print(f"    Node: w1={n['w1_same_level']:.4f} (same-level)  "
                  f"w2={n['w2_from_edges']:.4f} (from edges)")
            e = aw["edge"]
            print(f"    Edge: w3={e['w3_from_nodes']:.4f} (from nodes)  "
                  f"w4={e['w4_same_level']:.4f} (same-level)  "
                  f"w5={e['w5_from_faces']:.4f} (from faces)")
            f_ = aw["face"]
            print(f"    Face: w6={f_['w6_from_edges']:.4f} (from edges)  "
                  f"w7={f_['w7_same_level']:.4f} (same-level)")
        elif "w1_cf" in aw.get("node", {}):
            # Hodge model: 2 weights (CF + Hodge)
            n = aw["node"]
            print(f"    w1={n['w1_cf']:.4f} (CF)  "
                  f"w2={n['w2_hodge']:.4f} (Hodge)")
        else:
            # Flat model: 2 weights
            n = aw.get("node", {})
            e = aw.get("edge", {})
            print(f"    w_node={n.get('w_node', 0):.4f} (user-item)  "
                  f"w_edge={e.get('w_edge', 0):.4f} (item-item)")


if __name__ == "__main__":
    main()
