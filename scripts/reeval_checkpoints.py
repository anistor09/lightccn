"""Re-evaluate saved checkpoints — recompute ANY metric without retraining.

Each checkpoint (written by run_multi_experiment when checkpoint_dir is set)
is self-describing: it carries the best-epoch ``state_dict`` plus a ``rebuild``
dict with every constructor argument. This script reconstructs each model,
re-supplies the cell-complex operators, and re-scores it — so you can compute
corrected Recall@K (denominator = n_rel), Precision@K, MRR, etc., at a few
seconds per model instead of a 6-hour retrain.

Usage:
    python scripts/reeval_checkpoints.py --ckpt-dir /path/to/Drive/LightCCN_Checkpoints/<run> \
        [--device cpu] [--out reeval.json]

The default metric set re-computes BOTH the capped recall (matching the sweep)
and the standard recall (hits / n_rel), side by side, so you can see the
difference the denominator makes. See `evaluate_standard_recall` below.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from light_ccn.data.dataset import CFDataset
from light_ccn.complex.cell_complex import CellComplexBuilder
from light_ccn.complex.adjacency import build_multi_operators, symmetric_norm
from light_ccn.models.lightccn_multi import LightCCNMulti
from light_ccn.models.lightgcn import LightGCN
from light_ccn.utils.helpers import scipy_to_torch_sparse


def _rebuild_model(rb, ops, device):
    if rb["model_name"] == "lightgcn":
        m = LightGCN(n_users=rb["n_users"], n_items=rb["n_items"],
                     embed_dim=rb["embed_dim"], n_layers=rb["n_layers"])
        return m
    m = LightCCNMulti(
        n_users=rb["n_users"], n_items=rb["n_items"], embed_dim=rb["embed_dim"],
        n_layers=rb["n_layers"], n_edges=rb["n_edges"], n_faces=rb["n_faces"],
        edge_embed_dim=rb["edge_embed_dim"], face_embed_dim=rb["face_embed_dim"],
        operators=ops, weight_mode=rb["weight_mode"],
        propagation_mode=rb["propagation_mode"],
        weight_granularity=rb["weight_granularity"],
        edge_face_self_loop=rb["edge_face_self_loop"], init_mode=rb["init_mode"],
    )
    return m


@torch.no_grad()
def evaluate_both_recalls(model, dataset, topk=(5, 10, 20, 50), device="cpu"):
    """Return BOTH capped (hits/min(n_rel,K)) and standard (hits/n_rel) recall,
    plus NDCG, at each K. Mirrors the trainer's scoring path exactly so the
    capped numbers reproduce the sweep, and adds the standard variant."""
    model.eval()
    user_all, item_all = model.propagate()
    test_dict = dataset.test_dict
    train_dict = dataset.train_dict
    max_k = max(topk)
    discount = 1.0 / np.log2(np.arange(2, max_k + 2))
    cum_discount = np.cumsum(discount)

    capped = {k: [] for k in topk}
    standard = {k: [] for k in topk}
    ndcg = {k: [] for k in topk}

    users = [u for u in test_dict if test_dict[u]]
    B = 1024
    for s in range(0, len(users), B):
        bu = users[s:s + B]
        ue = user_all[torch.tensor(bu, device=user_all.device)]
        scores = ue @ item_all.T
        for i, u in enumerate(bu):
            ti = train_dict.get(u, [])
            if ti:
                scores[i, ti] = -float("inf")
        _, topi = torch.topk(scores, max_k, dim=1)
        topi = topi.cpu().numpy()
        for i, u in enumerate(bu):
            rel = set(test_dict[u]); n_rel = len(rel)
            hit = np.array([1.0 if it in rel else 0.0 for it in topi[i]])
            for k in topk:
                hk = hit[:k].sum()
                capped[k].append(hk / min(n_rel, k))
                standard[k].append(hk / n_rel)
                dcg = (hit[:k] * discount[:k]).sum()
                idcg = cum_discount[min(n_rel, k) - 1] if n_rel > 0 else 0.0
                ndcg[k].append(dcg / idcg if idcg > 0 else 0.0)
    out = {}
    for k in topk:
        out[f"recall_capped@{k}"] = float(np.mean(capped[k]))
        out[f"recall_standard@{k}"] = float(np.mean(standard[k]))
        out[f"ndcg@{k}"] = float(np.mean(ndcg[k]))
    return out


def _operators_for(dataset, rb, device):
    if rb["model_name"] == "lightgcn":
        A = scipy_to_torch_sparse(symmetric_norm(dataset.get_bipartite_adjacency())).to(device)
        return {"_lightgcn_adj": A}
    cx = CellComplexBuilder(dataset.get_interaction_matrix(), tau=rb["tau"],
                            cache_dir="data/complex_cache",
                            dataset_name=dataset.name).build_and_cache()
    ops_sp = build_multi_operators(dataset.get_bipartite_adjacency(), cx["B1"], cx["B2"],
                                   add_self_loops=rb.get("operator_self_loop", False))
    return {k: scipy_to_torch_sparse(v).to(device) for k, v in ops_sp.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="reeval_results.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    ckpts = sorted(Path(args.ckpt_dir).glob("*.pt"))
    print(f"Found {len(ckpts)} checkpoints in {args.ckpt_dir}")
    # cache dataset + operators per (dataset, model kind) to avoid rebuilding
    ds_cache, op_cache = {}, {}
    out = []
    for p in ckpts:
        blob = torch.load(p, map_location=device, weights_only=False)
        rb = blob["rebuild"]; dsname = blob["dataset"]
        if dsname not in ds_cache:
            ds_cache[dsname] = CFDataset(name=dsname)
        ds = ds_cache[dsname]
        opkey = (dsname, rb["model_name"])
        if opkey not in op_cache:
            op_cache[opkey] = _operators_for(ds, rb, device)
        ops = op_cache[opkey]
        model = _rebuild_model(rb, ops if rb["model_name"] != "lightgcn" else None, device).to(device)
        if rb["model_name"] == "lightgcn":
            model.set_adj_matrix(ops["_lightgcn_adj"])
        else:
            model.set_operators(ops)
        model.load_state_dict(blob["state_dict"], strict=True)
        metrics = evaluate_both_recalls(model, ds, device=device)
        row = {"label": blob["label"], "dataset": dsname,
               "best_epoch": blob["best_epoch"], **metrics}
        out.append(row)
        print(f"  {blob['label']:32s} {dsname:14s} "
              f"Rcap@20={metrics['recall_capped@20']:.4f} "
              f"Rstd@20={metrics['recall_standard@20']:.4f}")
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out} ({len(out)} models)")


if __name__ == "__main__":
    main()
