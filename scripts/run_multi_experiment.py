"""Driver for the Multi-fix experiments notebook.

Single entry point that runs one LightCCN-Multi configuration end-to-end.
The notebook just calls this with different option dicts.

Each run:
- Builds the cell complex (current C-TAU3 by default; or NMF E1/E2 if requested)
- Constructs LightCCN-Multi with the fixes (F1 weight mode, F2 bootstrap init,
  F3 observability, F4 topology aux loss)
- Trains for the specified number of epochs
- Saves per-run JSON to results_dir
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

# Make sure the package is importable when running from notebook
import sys
SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from light_ccn.config import ExperimentConfig
from light_ccn.data.dataset import CFDataset
from light_ccn.complex.cell_complex import CellComplexBuilder
from light_ccn.complex.adjacency import (
    build_lightccn_flat_adj,
    build_multi_operators,
    symmetric_norm,
)
from light_ccn.complex.nmf_faces import (
    find_faces_e1,
    find_faces_e2,
    build_complex_from_faces,
)
from light_ccn.models.lightccn_multi import LightCCNMulti
from light_ccn.models.lightgcn import LightGCN
from light_ccn.models.mf import MF
from light_ccn.models.ngcf import NGCF
from light_ccn.models.simgcl import SimGCL
from light_ccn.models.hccf import HCCF
from light_ccn.models.sgl import SGL
from light_ccn.models.multvae import MultVAE, MultVAETrainer
from light_ccn.models.ease import fit_ease, EASEScorer
from light_ccn.training.trainer import Trainer
from light_ccn.utils.helpers import scipy_to_torch_sparse, set_seed


# Per-dataset default tau for the C-TAU3 construction. Tuned so each
# dataset produces a non-empty cell complex at the default value.
DEFAULT_TAU = {
    "gowalla": 20,
    "yelp2018": 20,
    "amazon-book": 40,
    "beidian": 3,
    "beibei": 5,
    "yelp": 5,
    "douban": 10,
    "amazon-music": 3,  # Beidian-twin density (~0.27%) → same tau; full dataset has 24k faces, 13k edges
    "amazon-beauty": 3,  # same Amazon family knob as amazon-music; tau=3 → ~25k edges
    "amazon-toys-and-games": 3,  # same family; tau=3 → ~6.9k edges
    "epinions": 8,  # 30k users / 119k items, item-tail; tau=3 explodes (118k edges), tau=8 → 8.3k edges
    "ciaodvd": 2,  # social/trust DVD, sparse (int/item 3.8); tau=2 → 74k edges/137k faces, 28.8% item coverage (beidian-parity; tau>=5 collapses to lastfm-style 4%)
    "amazon-office-products": 2,  # tiny; tau=2 → 68% item coverage
    "foursquare-tky": 10,  # 2.3k users, heavy co-engagement; tau=3 → 195k edges, tau=10 → 19.9k edges
    "foursquare-nyc": 3,  # very sparse, small user count; tau=3 keeps non-degenerate complex
    "foursquare-tky": 3,
    "lastfm-2k": 8,  # HetRec-2011; real C-TAU3 on the train split: tau=8 -> ~56.6k faces /
                     # 10.5k edges / 522 items (Beidian-scale). (tau=15 only gives 16.6k/280.)
}


def _select_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _save_best_checkpoint(trainer, checkpoint_dir, label, dataset_name, rebuild):
    """Persist the best-epoch weights for cheap later re-evaluation.

    Saves ONLY when ``checkpoint_dir`` is a real, existing directory — the
    sweep notebooks point this at a mounted-Drive path, so headless GitHub-
    backend runs (no Drive) pass ``None`` and skip silently. fp32, weights
    only (no optimizer), plus a ``rebuild`` dict with every constructor arg
    needed to reconstruct the model and re-score it. One file per config.
    """
    if not checkpoint_dir:
        return None
    cdir = Path(checkpoint_dir)
    if not cdir.exists():
        print(f"  [checkpoint] dir {cdir} missing (Drive not mounted?) — skipped")
        return None
    state = getattr(trainer, "best_state", None)
    if state is None:  # never improved past 0 / keep_best_state off → fall back
        state = {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()}
    safe = label.replace("/", "-").replace(" ", "_")
    path = cdir / f"{safe}_{dataset_name}.pt"
    torch.save({
        "state_dict": state,
        "best_epoch": getattr(trainer, "best_epoch", None),
        "best_metric": getattr(trainer, "best_metric", None),
        "label": label,
        "dataset": dataset_name,
        "rebuild": rebuild,
    }, path)
    print(f"  [checkpoint] saved best-epoch weights -> {path}")
    return str(path)


def _run_simple_baseline(
    *, model_name, label, dataset, R, device, device_pref, embed_dim, n_layers,
    epochs, eval_every, early_stop_patience, lr, reg_weight, seed,
    results_dir, eval_tail, tail_pct, observability, checkpoint_dir=None,
    node_dropout=0.1, message_dropout=0.1,
    cl_eps=0.1, cl_temp=0.2, cl_weight=0.5,
    hyper_num=128, hccf_leaky=0.5, hccf_keep_rate=0.5, hccf_mult=1.0,
    hccf_xavier_init=False,
    sgl_augment_ratio=0.1, ssl_weight=0.1, ssl_temp=0.2,
    mv_batch_size=512, mv_anneal_cap=0.2, mv_anneal_steps=200000,
    mv_dropout=0.5, batch_size=None, early_stop_monitor="auto",
    val_view=None, test_view=None, val_source=None,
    ngcf_faithful=False, mv_anneal_faithful=False,
) -> dict:
    """Plain CF baselines that need only the bipartite user-item graph:
    ``mf`` (BPR-MF, no propagation), ``ngcf`` (Wang et al. 2019),
    ``lightgcn`` (He et al. 2020), and ``simgcl`` (Yu et al. 2022). Same
    training loop / eval / JSON-save format as the LightCCN-Multi runs, so
    every baseline is directly comparable to our model and to each other.
    No cell complex is built.

    - mf:       MF(n_users, n_items, embed_dim) — score = <e_u, e_i>.
    - ngcf:     NGCF on the symmetric-normalized bipartite adjacency, with the
                paper's W1/W2 + bi-interaction + LeakyReLU + dropout + concat.
    - lightgcn: LightGCN on the same adjacency.
    - simgcl:   LightGCN backbone + noise-perturbed contrastive views (InfoNCE).
    """
    print(f"  Model: {model_name} (CF baseline, no cell complex)")

    # MF needs no graph; the others use the symmetric-normalized bipartite adj
    # (identical to LightCCN's A_hat_0 — same input, fair comparison).
    A_hat_0 = None
    if model_name in ("ngcf", "lightgcn", "simgcl", "hccf", "sgl"):
        A_hat_0_sp = symmetric_norm(dataset.get_bipartite_adjacency())
        A_hat_0 = scipy_to_torch_sparse(A_hat_0_sp).to(device)
    if model_name == "ngcf" and ngcf_faithful:
        # Official NGCF code's operator ('norm' adj_type): row-normalized
        # D^-1 (A + I) with self-loops — deviates from the paper's symmetric
        # formula, but it is what produced the published numbers.
        A_raw = dataset.get_bipartite_adjacency().tocsr()
        A_loop = (A_raw + sp.eye(A_raw.shape[0], format="csr")).tocsr()
        rowsum = np.asarray(A_loop.sum(axis=1)).flatten()
        A_hat_0_sp = sp.diags(1.0 / rowsum) @ A_loop
        A_hat_0 = scipy_to_torch_sparse(A_hat_0_sp.tocoo()).to(device)

    if model_name == "mf":
        model = MF(n_users=dataset.n_users, n_items=dataset.n_items,
                   embed_dim=embed_dim).to(device)
    elif model_name == "ngcf":
        model = NGCF(n_users=dataset.n_users, n_items=dataset.n_items,
                     embed_dim=embed_dim, n_layers=n_layers, adj_matrix=A_hat_0,
                     node_dropout=node_dropout, message_dropout=message_dropout,
                     faithful=ngcf_faithful).to(device)
        model.set_adj_matrix(A_hat_0)
    elif model_name == "lightgcn":
        model = LightGCN(n_users=dataset.n_users, n_items=dataset.n_items,
                         embed_dim=embed_dim, n_layers=n_layers, adj_matrix=A_hat_0).to(device)
        model.set_adj_matrix(A_hat_0)
    elif model_name == "simgcl":
        model = SimGCL(n_users=dataset.n_users, n_items=dataset.n_items,
                       embed_dim=embed_dim, n_layers=n_layers, adj_matrix=A_hat_0,
                       eps=cl_eps, cl_temp=cl_temp, cl_weight=cl_weight).to(device)
        model.set_adj_matrix(A_hat_0)
    elif model_name == "hccf":
        model = HCCF(n_users=dataset.n_users, n_items=dataset.n_items,
                     embed_dim=embed_dim, n_layers=n_layers, adj_matrix=A_hat_0,
                     hyper_num=hyper_num, cl_temp=cl_temp, cl_weight=cl_weight,
                     leaky=hccf_leaky, keep_rate=hccf_keep_rate, mult=hccf_mult).to(device)
        if hccf_xavier_init:
            model.init_table_xavier()
        model.set_adj_matrix(A_hat_0)
    elif model_name == "sgl":
        model = SGL(n_users=dataset.n_users, n_items=dataset.n_items,
                    embed_dim=embed_dim, n_layers=n_layers, adj_matrix=A_hat_0,
                    bipartite_adj_scipy=dataset.get_bipartite_adjacency(),
                    augment_ratio=sgl_augment_ratio, device=device).to(device)
        model.set_adj_matrix(A_hat_0)
    elif model_name == "multvae":
        model = MultVAE(n_users=dataset.n_users, n_items=dataset.n_items,
                        anneal_cap=mv_anneal_cap, dropout=mv_dropout,
                        total_anneal_steps=mv_anneal_steps).to(device)
        model.set_interactions(R)
    else:
        raise ValueError(f"_run_simple_baseline: unsupported model_name {model_name!r}")

    config = ExperimentConfig()
    config.data.name = dataset.name
    config.model.name = model_name
    config.model.embed_dim = embed_dim
    config.model.n_layers = n_layers
    config.train.epochs = epochs
    config.train.lr = lr
    config.train.reg_weight = reg_weight
    config.train.early_stop_patience = early_stop_patience
    config.train.eval_every = eval_every
    if batch_size is not None:
        config.data.batch_size = batch_size
    config.train.early_stop_monitor = early_stop_monitor
    config.train.seed = seed
    config.train.device = device_pref
    config.results_dir = results_dir
    if model_name == "sgl":
        config.model.ssl_weight = ssl_weight
        config.model.ssl_temp = ssl_temp

    tail_items = dataset.get_tail_items(percentile=tail_pct) if eval_tail else None
    if tail_items is not None:
        print(f"  Tail eval enabled: {len(tail_items)} tail items (bottom {tail_pct:.0f}%)")
    from light_ccn.training.trainer import SGLTrainer
    trainer_cls = Trainer
    trainer_kwargs = {}
    if model_name == "sgl":
        trainer_cls = SGLTrainer
    elif model_name == "multvae":
        trainer_cls = MultVAETrainer
        trainer_kwargs["mv_batch_size"] = mv_batch_size
        trainer_kwargs["mv_anneal_faithful"] = mv_anneal_faithful
        observability = False  # no factor embeddings -> no cosine telemetry
    trainer = trainer_cls(
        model=model,
        dataset=dataset,
        config=config,
        save_enabled=False,
        tail_items=tail_items,
        topology_aux_weight=0.0,  # no faces, no aux loss
        faces=None,
        observability=observability,
        keep_best_state=bool(checkpoint_dir),
        val_dataset=val_view,
        eval_dataset=test_view,
        **trainer_kwargs,
    )

    t0 = time.time()
    results = trainer.train()
    elapsed = time.time() - t0
    results["wall_time_sec"] = elapsed
    results["use_validation"] = val_view is not None
    results["val_source"] = val_source
    results["label"] = label
    results["model_name"] = model_name
    results["construction"] = "none"
    results["propagation_mode"] = model_name
    results["weight_mode"] = "n/a"
    results["edge_face_self_loop"] = False
    results["operator_self_loop"] = False
    results["n_edges"] = 0
    results["n_faces"] = 0
    if model_name == "ngcf":
        results["node_dropout"] = node_dropout
        results["message_dropout"] = message_dropout
        results["ngcf_faithful"] = ngcf_faithful
    if model_name == "simgcl":
        results["cl_eps"] = cl_eps
        results["cl_temp"] = cl_temp
        results["cl_weight"] = cl_weight
    if model_name == "sgl":
        results["sgl_augment_ratio"] = sgl_augment_ratio
        results["ssl_weight"] = ssl_weight
        results["ssl_temp"] = ssl_temp
    if model_name == "multvae":
        results["mv_anneal_cap"] = mv_anneal_cap
        results["mv_anneal_steps"] = mv_anneal_steps
        results["mv_anneal_faithful"] = mv_anneal_faithful
    if model_name == "hccf":
        results["hyper_num"] = hyper_num
        results["cl_temp"] = cl_temp
        results["cl_weight"] = cl_weight
        results["hccf_leaky"] = hccf_leaky
        results["hccf_keep_rate"] = hccf_keep_rate
        results["hccf_mult"] = hccf_mult
        results["hccf_xavier_init"] = bool(hccf_xavier_init)

    rebuild = dict(
        model_name=model_name,
        n_users=dataset.n_users, n_items=dataset.n_items,
        embed_dim=embed_dim, n_layers=n_layers,
    )
    if model_name == "ngcf":
        rebuild.update(node_dropout=node_dropout, message_dropout=message_dropout,
                       faithful=ngcf_faithful)
    if model_name == "simgcl":
        rebuild.update(cl_eps=cl_eps, cl_temp=cl_temp, cl_weight=cl_weight)
    if model_name == "hccf":
        rebuild.update(hyper_num=hyper_num, cl_temp=cl_temp, cl_weight=cl_weight,
                       leaky=hccf_leaky, keep_rate=hccf_keep_rate, mult=hccf_mult)
    ckpt_path = _save_best_checkpoint(
        trainer, checkpoint_dir, label, dataset.name, rebuild=rebuild,
    )
    if ckpt_path:
        results["_checkpoint"] = ckpt_path

    # Save with the same descriptive filename pattern.
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_label = label.replace("/", "-").replace(" ", "_")
    save_path = Path(results_dir) / f"{safe_label}_{dataset.name}_{timestamp}.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [{label}] Saved: {save_path} (wall {elapsed/60:.1f} min)")
    results["_saved_to"] = str(save_path)
    ranks_path = trainer.save_ranks_npz(str(save_path)[:-5] + ".ranks.npz")
    if ranks_path:
        results["_ranks_saved_to"] = ranks_path
    return results


def _run_ease(
    *, label, dataset, R, device, device_pref, results_dir, seed,
    ease_lambdas, eval_tail, tail_pct,
    val_view=None, test_view=None, val_source=None,
) -> dict:
    """EASE^R closed-form baseline. The lambda grid plays the epoch axis:
    one eval row (and one rank-dump row) per lambda, so both frames' selection
    conventions work unchanged downstream."""
    from light_ccn.evaluation.metrics import Evaluator
    print(f"  Model: ease (closed-form, lambda grid = {ease_lambdas})")
    lambdas = [float(x) for x in str(ease_lambdas).split(",") if x.strip()]
    evaluator = Evaluator()
    tail_items = dataset.get_tail_items(percentile=tail_pct) if eval_tail else None
    eval_ds = test_view if test_view is not None else dataset
    t0 = time.time()
    eval_results, val_rows, tail_rows = [], [], []
    rank_rows, rank_meta, rank_epochs = [], None, []
    for i, lam in enumerate(lambdas):
        B = fit_ease(R, lam, device)
        scorer = EASEScorer(R, B, device)
        row, rank_data = evaluator.evaluate(scorer, eval_ds, device=device,
                                            return_ranks=True)
        row["epoch"] = i
        row["ease_lambda"] = lam
        eval_results.append(row)
        if rank_data is not None:
            rank_rows.append(rank_data["ranks"])
            rank_epochs.append(i)
            rank_meta = {"user_ids": rank_data["user_ids"],
                         "n_rel": rank_data["n_rel"]}
        if tail_items is not None:
            trow = evaluator.evaluate(scorer, eval_ds, device=device,
                                      item_filter=set(tail_items))
            trow["epoch"] = i
            trow["ease_lambda"] = lam
            tail_rows.append(trow)
        if val_view is not None:
            vrow = evaluator.evaluate(scorer, val_view, device=device)
            vrow["epoch"] = i
            vrow["ease_lambda"] = lam
            val_rows.append(vrow)
        print(f"    lambda={lam:g}: test R@20={row['recall@20']:.4f}"
              + (f"  val R@20={val_rows[-1]['recall@20']:.4f}" if val_rows else ""))
        del B, scorer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if val_rows:
        best_i = max(range(len(val_rows)), key=lambda j: val_rows[j]["recall@20"])
    else:
        best_i = max(range(len(eval_results)),
                     key=lambda j: eval_results[j]["recall@20"])
    results = {
        "label": label, "model_name": "ease", "construction": "none",
        "propagation_mode": "ease", "weight_mode": "n/a",
        "edge_face_self_loop": False, "operator_self_loop": False,
        "n_edges": 0, "n_faces": 0,
        "eval_results": eval_results,
        "tail_eval_results": tail_rows or None,
        "val_eval_results": val_rows or None,
        "best_epoch": int(eval_results[best_i]["epoch"]),
        "best_metric": float((val_rows or eval_results)[best_i]["recall@20"]),
        "ease_lambdas": lambdas,
        "ease_lambda_selected": lambdas[best_i],
        "use_validation": val_view is not None,
        "val_source": val_source,
        "final_metrics": eval_results[best_i],
        "wall_time_sec": time.time() - t0,
        "config": {"data": {"name": dataset.name},
                   "model": {"name": "ease", "embed_dim": 0, "n_layers": 0},
                   "train": {"seed": seed}},
    }
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_label = label.replace("/", "-").replace(" ", "_")
    save_path = Path(results_dir) / f"{safe_label}_{dataset.name}_{timestamp}.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [{label}] Saved: {save_path}")
    results["_saved_to"] = str(save_path)
    if rank_rows and rank_meta is not None:
        ranks = np.stack(rank_rows).astype(np.uint32)
        npz_path = str(save_path)[:-5] + ".ranks.npz"
        np.savez_compressed(
            npz_path,
            epochs=np.array(rank_epochs, dtype=np.int32),
            ranks=ranks,
            user_ids=rank_meta["user_ids"].astype(np.int64),
            n_rel=rank_meta["n_rel"].astype(np.int32),
        )
        results["_ranks_saved_to"] = npz_path
        print(f"  [ranks] saved {ranks.shape[0]} lambdas x {ranks.shape[1]} -> {npz_path}")
    return results


def _run_lightgcn_cc(
    *, label, dataset, R, complex_data, construction, tau,
    device, device_pref, embed_dim, n_layers,
    epochs, eval_every, early_stop_patience, lr, reg_weight, seed,
    results_dir, eval_tail, tail_pct, observability, checkpoint_dir=None,
    val_view=None, test_view=None, val_source=None,
    cc_variant: str = "binary",
) -> dict:
    """Structure-vs-propagation ablation: PLAIN LightGCN propagation over the
    bipartite graph augmented with the cell complex's item-item 1-cells as
    ordinary binary graph edges:

        A_cc = sym_norm([[0, R], [R^T, S_bin]]),  S_bin[i,j] = 1 iff {i,j} is
        a 1-cell (face-boundary co-engagement pair at the campaign tau).

    Identical LightGCN module, loss, trainer, eval, and complex construction
    as every other run — the only change vs the ``lightgcn`` baseline is the
    extra item-item edges. Any delta vs ``lightgcn`` therefore isolates what
    the complex's STRUCTURE is worth when consumed by vanilla propagation;
    the remaining gap to LightCCN(+B) isolates the higher-order
    propagation/readout machinery consuming the same cells.
    """
    edges = complex_data["edges"]
    n_edges = len(edges)
    n_faces = len(complex_data["faces"])
    print(f"  Model: lightgcn_cc (LightGCN over bipartite + {n_edges:,} "
          f"complex item-item edges, variant={cc_variant})")

    if cc_variant == "weighted":
        # Face-count multiplicities: S[i,j] = number of shared faces.
        S_bin = complex_data["S"].tocsr().astype(np.float32)
    elif cc_variant == "random":
        # Control: SAME number of item-item edges, endpoints drawn uniformly
        # (seeded) from items with at least one training interaction. Separates
        # "these specific cells" from "any extra item-item connectivity".
        rng = np.random.default_rng(seed)
        active = np.flatnonzero(np.asarray(R.sum(axis=0)).ravel() > 0)
        chosen: set[tuple[int, int]] = set()
        while len(chosen) < n_edges and len(active) > 1:
            a, b = rng.choice(active, size=2, replace=False)
            chosen.add((min(int(a), int(b)), max(int(a), int(b))))
        e = np.asarray(sorted(chosen), dtype=np.int64)
        rows = np.concatenate([e[:, 0], e[:, 1]])
        cols = np.concatenate([e[:, 1], e[:, 0]])
        S_bin = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(dataset.n_items, dataset.n_items),
        )
        print(f"  [cc-random] {len(chosen):,} random item-item edges "
              f"among {len(active):,} active items (seed={seed})")
    elif n_edges > 0:
        e = np.asarray(edges, dtype=np.int64)
        rows = np.concatenate([e[:, 0], e[:, 1]])
        cols = np.concatenate([e[:, 1], e[:, 0]])
        S_bin = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(dataset.n_items, dataset.n_items),
        )
    else:
        S_bin = sp.csr_matrix((dataset.n_items, dataset.n_items), dtype=np.float32)
    A_cc_sp = build_lightccn_flat_adj(
        R, S_bin, dataset.n_users, dataset.n_items, gamma=1.0,
    )
    A_cc = scipy_to_torch_sparse(A_cc_sp).to(device)

    model = LightGCN(n_users=dataset.n_users, n_items=dataset.n_items,
                     embed_dim=embed_dim, n_layers=n_layers, adj_matrix=A_cc).to(device)
    model.set_adj_matrix(A_cc)

    config = ExperimentConfig()
    config.data.name = dataset.name
    config.model.name = "lightgcn_cc"
    config.model.embed_dim = embed_dim
    config.model.n_layers = n_layers
    config.train.epochs = epochs
    config.train.lr = lr
    config.train.reg_weight = reg_weight
    config.train.early_stop_patience = early_stop_patience
    config.train.eval_every = eval_every
    config.train.seed = seed
    config.train.device = device_pref
    config.results_dir = results_dir

    tail_items = dataset.get_tail_items(percentile=tail_pct) if eval_tail else None
    if tail_items is not None:
        print(f"  Tail eval enabled: {len(tail_items)} tail items (bottom {tail_pct:.0f}%)")
    trainer = Trainer(
        model=model,
        dataset=dataset,
        config=config,
        save_enabled=False,
        tail_items=tail_items,
        topology_aux_weight=0.0,
        faces=None,
        observability=observability,
        keep_best_state=bool(checkpoint_dir),
        val_dataset=val_view,
        eval_dataset=test_view,
    )

    t0 = time.time()
    results = trainer.train()
    elapsed = time.time() - t0
    results["wall_time_sec"] = elapsed
    results["use_validation"] = val_view is not None
    results["val_source"] = val_source
    results["label"] = label
    results["model_name"] = "lightgcn_cc"
    results["construction"] = construction
    results["tau"] = tau
    results["propagation_mode"] = "lightgcn_cc"
    results["weight_mode"] = "n/a"
    results["cc_edge_weighting"] = cc_variant
    results["edge_face_self_loop"] = False
    results["operator_self_loop"] = False
    results["n_edges"] = n_edges
    results["n_faces"] = n_faces

    ckpt_path = _save_best_checkpoint(
        trainer, checkpoint_dir, label, dataset.name,
        rebuild=dict(model_name="lightgcn_cc",
                     n_users=dataset.n_users, n_items=dataset.n_items,
                     embed_dim=embed_dim, n_layers=n_layers,
                     construction=construction, tau=tau),
    )
    if ckpt_path:
        results["_checkpoint"] = ckpt_path

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_label = label.replace("/", "-").replace(" ", "_")
    save_path = Path(results_dir) / f"{safe_label}_{dataset.name}_{timestamp}.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [{label}] Saved: {save_path} (wall {elapsed/60:.1f} min)")
    results["_saved_to"] = str(save_path)
    ranks_path = trainer.save_ranks_npz(str(save_path)[:-5] + ".ranks.npz")
    if ranks_path:
        results["_ranks_saved_to"] = ranks_path
    return results


def _build_complex(
    construction: str,
    dataset: CFDataset,
    R: sp.spmatrix,
    tau: int,
    n_factors: int,
    cache_dir: str,
) -> dict:
    """Dispatch on construction string. Returns the standard complex dict."""
    if construction == "tau3":
        builder = CellComplexBuilder(
            R, tau=tau, cache_dir=cache_dir, dataset_name=dataset.name,
        )
        return builder.build_and_cache()
    elif construction == "nmf_e1":
        faces = find_faces_e1(
            R, n_factors=n_factors,
            cache_dir=os.path.join(cache_dir, "..", "nmf_cache"),
            dataset_name=dataset.name,
        )
        return build_complex_from_faces(faces, dataset.n_users, dataset.n_items)
    elif construction == "nmf_e2":
        faces = find_faces_e2(
            R, n_factors=n_factors,
            cache_dir=os.path.join(cache_dir, "..", "nmf_cache"),
            dataset_name=dataset.name,
        )
        return build_complex_from_faces(faces, dataset.n_users, dataset.n_items)
    elif construction == "nmf_e3":
        from light_ccn.complex.nmf_faces import find_faces_e3, build_complex_from_polygonal_faces
        faces = find_faces_e3(
            R, n_factors=n_factors,
            cache_dir=os.path.join(cache_dir, "..", "nmf_cache"),
            dataset_name=dataset.name,
        )
        return build_complex_from_polygonal_faces(
            faces, dataset.n_users, dataset.n_items, bipartite_indexed=False,
        )
    elif construction == "nmf_e4":
        from light_ccn.complex.nmf_faces import find_faces_e4, build_complex_from_polygonal_faces
        faces = find_faces_e4(
            R, n_factors=n_factors, n_users=dataset.n_users,
            cache_dir=os.path.join(cache_dir, "..", "nmf_cache"),
            dataset_name=dataset.name,
        )
        return build_complex_from_polygonal_faces(
            faces, dataset.n_users, dataset.n_items, bipartite_indexed=True,
        )
    raise ValueError(f"Unknown construction: {construction!r}")


def run(
    *,
    label: str,
    dataset_name: str = "gowalla",
    construction: str = "tau3",
    tau: int | None = None,
    n_factors: int = 1000,
    include_ui_edges: bool = False,
    nodes_only: bool = False,
    propagation_mode: str | None = None,
    edge_face_self_loop: bool = False,
    operator_self_loop: bool = False,
    model_name: str = "lightccn_multi",
    weight_mode: str = "softmax",
    init_mode: str = "table",
    fc_normalize_R: bool = False,
    fc_freeze_after_epoch: int | None = None,
    fc_snapshot_to_table_at_epoch: int | None = None,
    freeze_topology_channel: bool = False,
    weight_granularity: str = "global",
    readout_mode: str = "none",
    readout_source: str = "combined",
    bootstrap_init: bool = True,
    topology_aux_weight: float = 0.0,
    observability: bool = True,
    epochs: int = 50,
    eval_every: int = 10,
    early_stop_patience: int = 30,
    lr: float = 1e-3,
    reg_weight: float = 1e-4,
    embed_dim: int = 64,
    n_layers: int = 2,
    seed: int = 2020,
    device_pref: str = "cuda",
    results_dir: str = "results/multi_fix_experiments",
    dry_run: bool = False,
    dry_run_users: int = 500,
    dry_run_epochs: int = 3,
    eval_tail: bool = True,
    tail_pct: float = 20.0,
    use_validation: bool = False,
    cc_variant: str = "binary",   # lightgcn_cc only: binary | weighted | random
    checkpoint_dir: str | None = None,
    node_dropout: float = 0.1,      # NGCF only
    message_dropout: float = 0.1,   # NGCF only
    cl_eps: float = 0.1,            # SimGCL only: noise magnitude
    cl_temp: float = 0.2,          # SimGCL/HCCF: InfoNCE temperature
    cl_weight: float = 0.5,        # SimGCL/HCCF: contrastive weight λ
    hyper_num: int = 128,          # HCCF only: number of hyperedges K
    hccf_leaky: float = 0.5,       # HCCF only: LeakyReLU slope (paper: 0.5)
    hccf_keep_rate: float = 0.5,   # HCCF only: edge/incidence keep prob
    hccf_mult: float = 1.0,        # HCCF only: incidence scaling
    hccf_xavier_init: bool = False,  # HCCF only: SSLRec-style xavier table init (parity runs)
    sgl_augment_ratio: float = 0.1,  # SGL only: edge-dropout ratio rho
    ssl_weight: float = 0.1,         # SGL only: InfoNCE weight lambda1
    ssl_temp: float = 0.2,           # SGL only: InfoNCE temperature
    mv_batch_size: int = 512,        # Mult-VAE only
    mv_anneal_cap: float = 0.2,      # Mult-VAE only: max beta
    mv_anneal_steps: int = 200000,   # Mult-VAE only: linear anneal steps
    mv_dropout: float = 0.5,         # Mult-VAE only: input dropout
    batch_size: int | None = None,   # BPR batch override (None = config default)
    early_stop_monitor: str = "auto",  # 'auto' | 'test' (papers' rule)
    ngcf_faithful: bool = False,  # NGCF only: official-code semantics (D^-1(A+I), split LeakyReLU, per-layer L2 norm)
    mv_anneal_faithful: bool = False,  # MultVAE only: vae_cf beta ramp min(cap, t/T) instead of min(cap, cap*t/T)
    ease_lambdas: str = "10,50,200,500,1000",  # EASE only: lambda grid (epoch axis)
) -> dict:
    """Run one Multi experiment end-to-end. Returns the results dict."""
    print(f"\n{'='*70}\n[{label}] Starting run\n{'='*70}")
    set_seed(seed)
    device = _select_device(device_pref)
    print(f"  Device: {device}")

    # Resolve τ. In dry-run mode we use a small τ across the board because
    # the subsampled dataset will rarely have enough co-engagement to
    # support the per-dataset production default.
    if tau is None:
        if dry_run:
            tau = 3
            print(f"  [dry-run] Using tau={tau} for {dataset_name}")
        else:
            tau = DEFAULT_TAU.get(dataset_name, 20)
            print(f"  Using default tau={tau} for {dataset_name}")

    # ─ Dataset
    dataset = CFDataset(name=dataset_name)
    if dry_run:
        # Don't subsample if the dataset is already small (subsampling would
        # leave too few interactions to support the cell-complex threshold).
        if dataset.n_users > dry_run_users * 2:
            dataset.subsample(n_users=dry_run_users, seed=seed)
        else:
            print(f"  [dry-run] dataset already small ({dataset.n_users} users); skipping subsample")
        epochs = dry_run_epochs
        # In dry-run, evaluate every epoch so we get metrics within the short run
        eval_every = 1
        early_stop_patience = max(early_stop_patience, dry_run_epochs)
        # Cap NMF factors for dry-run — fitting NMF with the production
        # alpha=0.5 (e.g. 14,929 factors on Gowalla) would take ~hours on CPU.
        if construction.startswith("nmf") and n_factors > 100:
            print(f"  [dry-run] Capping n_factors {n_factors} -> 100 for fast NMF fit")
            n_factors = 100

    # ─ Validation protocol (unbiased early stopping). Carve a leave-one-out
    # val split from TRAIN before any matrix/complex/sampler exists, so the
    # graph the model propagates over excludes the held-out interactions.
    # Views (duck-typed for the Evaluator):
    #   val_view : mask = reduced train, targets = val  -> early stopping
    #   test_view: mask = train ∪ val,   targets = test -> reported metrics
    val_view = test_view = None
    val_source = None
    if use_validation:
        from types import SimpleNamespace
        # Prefer the dataset's OFFICIAL validation split (train stays intact,
        # matching the source benchmark's protocol); carve from train only
        # when the dataset ships no val file.
        val_dict = dataset.load_official_validation()
        val_source = "official"
        if val_dict is None:
            val_dict = dataset.carve_validation(seed=seed)
            val_source = "carved"
        mask_tv = {
            u: dataset.train_dict.get(u, []) + val_dict.get(u, [])
            for u in set(dataset.train_dict) | set(val_dict)
        }
        val_view = SimpleNamespace(train_dict=dataset.train_dict, test_dict=val_dict)
        test_view = SimpleNamespace(train_dict=mask_tv, test_dict=dataset.test_dict)

    R = dataset.get_interaction_matrix()

    # ─ LightGCN baseline shortcut: no cell complex needed, just the bipartite
    # adjacency. We dispatch here so the heavy NMF / tau3 face construction is
    # skipped entirely when the user requested the LightGCN baseline.
    if model_name == "ease":
        return _run_ease(
            label=label, dataset=dataset, R=R, device=device, device_pref=device_pref,
            results_dir=results_dir, seed=seed,
            ease_lambdas=ease_lambdas, eval_tail=eval_tail, tail_pct=tail_pct,
            val_view=val_view, test_view=test_view, val_source=val_source,
        )
    if model_name in ("lightgcn", "mf", "ngcf", "simgcl", "hccf", "sgl", "multvae"):
        return _run_simple_baseline(
            model_name=model_name,
            label=label, dataset=dataset, R=R, device=device, device_pref=device_pref,
            embed_dim=embed_dim, n_layers=n_layers,
            epochs=epochs, eval_every=eval_every,
            early_stop_patience=early_stop_patience,
            lr=lr, reg_weight=reg_weight, seed=seed,
            results_dir=results_dir,
            eval_tail=eval_tail, tail_pct=tail_pct,
            observability=observability,
            checkpoint_dir=checkpoint_dir,
            node_dropout=node_dropout, message_dropout=message_dropout,
            cl_eps=cl_eps, cl_temp=cl_temp, cl_weight=cl_weight,
            hyper_num=hyper_num, hccf_leaky=hccf_leaky,
            hccf_keep_rate=hccf_keep_rate, hccf_mult=hccf_mult,
            hccf_xavier_init=hccf_xavier_init,
            sgl_augment_ratio=sgl_augment_ratio, ssl_weight=ssl_weight,
            ssl_temp=ssl_temp, mv_batch_size=mv_batch_size,
            mv_anneal_cap=mv_anneal_cap, mv_anneal_steps=mv_anneal_steps,
            mv_dropout=mv_dropout, batch_size=batch_size,
            early_stop_monitor=early_stop_monitor,
            val_view=val_view, test_view=test_view, val_source=val_source,
            ngcf_faithful=ngcf_faithful, mv_anneal_faithful=mv_anneal_faithful,
        )

    # ─ Cell complex. Only a CARVED split changes the co-engagement graph
    # (val edges removed from train) — namespace the cache in that case. An
    # official split leaves train intact, so the v1 complex cache is reused.
    # The carve depends on the seed, so non-default seeds get their own
    # namespace (seed 2020 keeps the original name and its existing caches).
    if use_validation and val_source == "carved":
        dataset.name = f"{dataset.name}__val" + (f"_s{seed}" if seed != 2020 else "")
    complex_data = _build_complex(
        construction=construction,
        dataset=dataset,
        R=R,
        tau=tau,
        n_factors=n_factors,
        cache_dir="data/complex_cache",
    )

    # ─ Structure-vs-propagation ablation: plain LightGCN over the bipartite
    # graph augmented with the complex's item-item 1-cells as ordinary edges.
    # Dispatched AFTER the complex build so it uses the exact same cached
    # complex (incl. the __val namespace under a carved split) as the
    # LightCCN-Multi runs it is compared against.
    if model_name == "lightgcn_cc":
        return _run_lightgcn_cc(
            label=label, dataset=dataset, R=R, complex_data=complex_data,
            construction=construction, tau=tau,
            device=device, device_pref=device_pref,
            embed_dim=embed_dim, n_layers=n_layers,
            epochs=epochs, eval_every=eval_every,
            early_stop_patience=early_stop_patience,
            lr=lr, reg_weight=reg_weight, seed=seed,
            results_dir=results_dir,
            eval_tail=eval_tail, tail_pct=tail_pct,
            observability=observability,
            checkpoint_dir=checkpoint_dir,
            val_view=val_view, test_view=test_view, val_source=val_source,
            cc_variant=cc_variant,
        )

    # Feature flag: augment the rank-1 edge set with user-item interaction edges.
    if include_ui_edges:
        from light_ccn.complex.cell_complex import augment_with_ui_edges
        complex_data = augment_with_ui_edges(
            complex_data, R, dataset.n_users, dataset.n_items,
        )
        print(f"  [UI-EDGES] flag ON -> added {complex_data.get('n_ui_edges', 0):,} "
              f"user-item edges on top of {complex_data.get('n_item_edges', 0):,} item-item edges")

    faces = complex_data["faces"]
    B1 = complex_data["B1"]
    B2 = complex_data["B2"]
    n_edges = B2.shape[0]
    n_faces = B2.shape[1]

    # ─ Operators
    bipartite_adj = dataset.get_bipartite_adjacency()
    operators_sp = build_multi_operators(bipartite_adj, B1, B2, add_self_loops=operator_self_loop)
    if operator_self_loop:
        print(f"  Operator self-loop (Option A): A_hat_1/A_hat_2 built with setdiag(1)")
    operators_pt = {k: scipy_to_torch_sparse(v).to(device) for k, v in operators_sp.items()}

    # ─ Model
    model = LightCCNMulti(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_edges=n_edges,
        n_faces=n_faces,
        operators=operators_pt,
        weight_mode=weight_mode,
        nodes_only=nodes_only,
        propagation_mode=propagation_mode,
        edge_face_self_loop=edge_face_self_loop,
        init_mode=init_mode,
        fc_normalize_R=fc_normalize_R,
        freeze_topology_channel=freeze_topology_channel,
        weight_granularity=weight_granularity,
        readout_mode=readout_mode,
        readout_source=readout_source,
    )
    model = model.to(device)
    resolved_prop_mode = model.propagation_mode
    print(f"  Propagation mode: {resolved_prop_mode}")
    print(f"  Init mode:        {init_mode}")
    print(f"  Weight granularity: {weight_granularity}")
    print(f"  Readout mode:     {readout_mode}"
          + (f" (faces: {model.readout_faces}, source: {readout_source})"
             if readout_mode != 'none' else ""))

    # When weight_granularity != 'global', populate per-cell bucket indices
    # from raw degree information. Nodes get bipartite-adj degree; edges/faces
    # get neighbour-count degree via B2 (B2 @ B2^T for edges, B2^T @ B2 for
    # faces) — same matrix used to build A_1 / A_2 in the operator stack.
    if weight_granularity != "global":
        node_deg_np = np.asarray(bipartite_adj.sum(axis=1)).reshape(-1)
        edge_deg_np = np.asarray((B2 @ B2.T).sum(axis=1)).reshape(-1) if n_edges > 0 else np.zeros(0)
        face_deg_np = np.asarray((B2.T @ B2).sum(axis=1)).reshape(-1) if n_faces > 0 else np.zeros(0)
        model.set_cell_buckets(
            node_degrees=torch.from_numpy(node_deg_np).float(),
            edge_degrees=torch.from_numpy(edge_deg_np).float(),
            face_degrees=torch.from_numpy(face_deg_np).float() if n_faces > 0 else None,
        )

    # FC init needs the interaction matrix R registered on-device after .to(device).
    if init_mode == "fc":
        model.set_interaction_matrix(R)

    # F2 boundary-bootstrap only applies to full_multi (the only mode with
    # learnable cochain tables). All derived/stateful modes seed themselves.
    if bootstrap_init and resolved_prop_mode == "full_multi":
        print("  Applying F2 boundary-bootstrap init")
        model.bootstrap_higher_rank_init()

    # ─ Config (minimal — Trainer reads what it needs)
    config = ExperimentConfig()
    config.data.name = dataset_name
    config.model.name = "lightccn_multi"
    config.model.embed_dim = embed_dim
    config.model.n_layers = n_layers
    config.train.epochs = epochs
    config.train.lr = lr
    config.train.reg_weight = reg_weight
    config.train.early_stop_patience = early_stop_patience
    config.train.eval_every = eval_every
    config.train.seed = seed
    config.train.device = device_pref
    config.complex.tau = tau
    config.results_dir = results_dir

    # ─ Trainer (with tail-item set for tail R@K and N@K)
    tail_items = dataset.get_tail_items(percentile=tail_pct) if eval_tail else None
    if tail_items is not None:
        print(f"  Tail eval enabled: {len(tail_items)} tail items (bottom {tail_pct:.0f}%)")
    trainer = Trainer(
        model=model,
        dataset=dataset,
        config=config,
        save_enabled=False,  # we save manually below with the descriptive label
        tail_items=tail_items,
        topology_aux_weight=topology_aux_weight,
        faces=faces if topology_aux_weight > 0 else None,
        observability=observability,
        fc_freeze_after_epoch=fc_freeze_after_epoch,
        fc_snapshot_to_table_at_epoch=fc_snapshot_to_table_at_epoch,
        keep_best_state=bool(checkpoint_dir),
        val_dataset=val_view,
        eval_dataset=test_view,
    )

    t0 = time.time()
    results = trainer.train()
    elapsed = time.time() - t0
    results["wall_time_sec"] = elapsed
    results["use_validation"] = use_validation
    results["val_source"] = val_source
    results["label"] = label
    results["construction"] = construction
    results["nodes_only"] = nodes_only
    results["model_name"] = model_name
    results["propagation_mode"] = resolved_prop_mode
    results["edge_face_self_loop"] = edge_face_self_loop
    results["operator_self_loop"] = operator_self_loop
    results["weight_mode"] = weight_mode
    results["init_mode"] = init_mode
    results["fc_normalize_R"] = fc_normalize_R
    results["fc_freeze_after_epoch"] = fc_freeze_after_epoch
    results["fc_snapshot_to_table_at_epoch"] = fc_snapshot_to_table_at_epoch
    results["freeze_topology_channel"] = freeze_topology_channel
    results["weight_granularity"] = weight_granularity
    results["readout_mode"] = readout_mode
    results["readout_source"] = readout_source
    results["readout_faces"] = bool(getattr(model, "readout_faces", False))
    if readout_mode != "none":
        results["readout_gates"] = model.get_readout_gates()
    results["bootstrap_init"] = bootstrap_init
    results["topology_aux_weight"] = topology_aux_weight
    results["n_edges"] = n_edges
    results["n_faces"] = n_faces
    results["include_ui_edges"] = include_ui_edges
    results["n_ui_edges"] = int(complex_data.get("n_ui_edges", 0)) if include_ui_edges else 0

    # ─ Best-epoch checkpoint (only when checkpoint_dir exists, e.g. Drive
    # mounted). Stores everything needed to rebuild this exact model and
    # re-score it later (any metric, any K ≤ ranked) without retraining.
    ckpt_path = _save_best_checkpoint(
        trainer, checkpoint_dir, label, dataset_name,
        rebuild=dict(
            model_name="lightccn_multi",
            n_users=dataset.n_users, n_items=dataset.n_items,
            # Pull the ACTUAL dims off the constructed model — edge/face dims
            # default to 64 in the constructor and are NOT necessarily equal to
            # embed_dim, which also controls whether projection layers exist.
            embed_dim=getattr(model, "embed_dim", embed_dim),
            n_layers=n_layers,
            n_edges=n_edges, n_faces=n_faces,
            edge_embed_dim=getattr(model, "edge_embed_dim", embed_dim),
            face_embed_dim=getattr(model, "face_embed_dim", embed_dim),
            propagation_mode=resolved_prop_mode, weight_mode=weight_mode,
            weight_granularity=weight_granularity,
            readout_mode=readout_mode, readout_source=readout_source,
            edge_face_self_loop=edge_face_self_loop,
            operator_self_loop=operator_self_loop,
            init_mode=init_mode, construction=construction, tau=tau,
        ),
    )
    if ckpt_path:
        results["_checkpoint"] = ckpt_path

    # ─ Save with descriptive filename
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_label = label.replace("/", "-").replace(" ", "_")
    save_path = Path(results_dir) / f"{safe_label}_{dataset_name}_{timestamp}.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [{label}] Saved: {save_path} (wall {elapsed/60:.1f} min)")
    results["_saved_to"] = str(save_path)
    ranks_path = trainer.save_ranks_npz(str(save_path)[:-5] + ".ranks.npz")
    if ranks_path:
        results["_ranks_saved_to"] = ranks_path
    return results
