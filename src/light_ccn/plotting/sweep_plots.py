"""Plot helpers for the 4 weight_modes × 4 granularities × 5 propagation
models sweep.

All helpers accept a "runs" list — each run is the JSON dict that
``scripts.run_multi_experiment.run`` returns / saves. They internally
project the runs into a pandas DataFrame and emit a figure. Keep the
helpers parameterized by metric so the notebook can loop over
{R@5, R@20, R@50, N@5, N@20, N@50} × {full, tail}.

Design goals:
  - Notebook authors call ``make_sweep_plots(runs, baseline=lightgcn_run, out_dir=...)``
    and get the full figure suite in one shot.
  - Every figure is also exposed as a standalone helper so notebooks can
    pick and choose / customize.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---- canonical orderings (used for axis labels everywhere) ----
WEIGHT_MODES = ("softmax", "softplus", "tanh", "signed")
GRANULARITIES = ("global", "type", "freq", "freq_type")
PROP_MODES = ("derived_e", "derived_ef", "stateful_e", "stateful_ef", "full_multi")

# Metrics actually written by the trainer. Pairs of (full_key, tail_key).
DEFAULT_METRIC_KEYS = [
    ("recall@3",  "tail_recall@3"),
    ("recall@5",  "tail_recall@5"),
    ("recall@10", "tail_recall@10"),
    ("recall@20", "tail_recall@20"),
    ("recall@50", "tail_recall@50"),
    ("ndcg@3",    "tail_ndcg@3"),
    ("ndcg@5",    "tail_ndcg@5"),
    ("ndcg@10",   "tail_ndcg@10"),
    ("ndcg@20",   "tail_ndcg@20"),
    ("ndcg@50",   "tail_ndcg@50"),
]


def _safe_metric(run: dict, key: str) -> float | None:
    """Pull a final metric out of one results dict.

    The trainer stores eval results under either ``final_metrics`` or the
    last entry in ``eval_history``. Try both, return None if absent.
    """
    fm = run.get("final_metrics")
    if fm and key in fm and fm[key] is not None:
        return float(fm[key])
    hist = run.get("eval_history") or []
    if hist:
        last = hist[-1]
        if key in last and last[key] is not None:
            return float(last[key])
    return run.get(key)  # last resort


def runs_to_df(runs: Iterable[dict]) -> pd.DataFrame:
    """Project a list of results dicts into a DataFrame with the columns the
    plot helpers need."""
    rows = []
    for r in runs:
        if r is None:
            continue
        row = {
            "label":              r.get("label", ""),
            "weight_mode":        r.get("weight_mode", "n/a"),
            "weight_granularity": r.get("weight_granularity", "global"),
            "propagation_mode":   r.get("propagation_mode", "n/a"),
            "model_name":         r.get("model_name", "n/a"),
            "wall_time_sec":      float(r.get("wall_time_sec", 0.0)),
            "n_edges":            int(r.get("n_edges", 0)),
            "n_faces":            int(r.get("n_faces", 0)),
        }
        # All metric keys we might ever plot.
        for full_k, tail_k in DEFAULT_METRIC_KEYS:
            row[full_k] = _safe_metric(r, full_k)
            row[tail_k] = _safe_metric(r, tail_k)
        rows.append(row)
    return pd.DataFrame(rows)


# ---- heatmap (per propagation mode) ----
def _ax_heatmap(ax, mat: np.ndarray, row_labels, col_labels, title, vmin=None,
                vmax=None, fmt: str = "{:.4f}", cmap: str = "viridis",
                divergent: bool = False):
    if divergent:
        amax = float(np.nanmax(np.abs(mat))) if np.isfinite(np.nanmax(np.abs(mat))) else 1.0
        amax = max(amax, 1e-9)
        vmin, vmax, cmap = -amax, amax, "RdBu_r"
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
    ax.set_title(title, fontsize=10)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isfinite(v):
                txt = "—"
            else:
                txt = fmt.format(v)
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if (np.isfinite(v) and abs(v - (vmin or 0)) > (vmax or 1) * 0.5) else "black",
                    fontsize=7)
    return im


def heatmap_per_prop_mode(
    df: pd.DataFrame,
    metric_key: str,
    save_path: str | Path,
    title_suffix: str = "",
    baseline_value: float | None = None,
    divergent: bool = False,
    fmt: str = "{:.4f}",
) -> None:
    """One heatmap per propagation mode in a row.

    Rows = weight_mode, cols = weight_granularity. If ``baseline_value`` is
    given, plots ``value - baseline_value`` with a divergent colormap.
    """
    prop_modes = [pm for pm in PROP_MODES if (df["propagation_mode"] == pm).any()]
    n = len(prop_modes)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n + 1.2, 3.4), squeeze=False)
    axes = axes.flatten()
    for ax, pm in zip(axes, prop_modes):
        sub = df[df["propagation_mode"] == pm]
        mat = np.full((len(WEIGHT_MODES), len(GRANULARITIES)), np.nan)
        for i, wm in enumerate(WEIGHT_MODES):
            for j, g in enumerate(GRANULARITIES):
                vals = sub[(sub["weight_mode"] == wm) & (sub["weight_granularity"] == g)][metric_key]
                if len(vals) > 0 and vals.notna().any():
                    mat[i, j] = float(vals.dropna().iloc[-1])
        if baseline_value is not None:
            mat = mat - baseline_value
        _ax_heatmap(ax, mat, WEIGHT_MODES, GRANULARITIES,
                    title=f"{pm}{title_suffix}", divergent=divergent or baseline_value is not None,
                    fmt=fmt)
    suptitle = f"{metric_key}" + (f" (Δ from LightGCN={baseline_value:.4f})" if baseline_value is not None else "")
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---- Pareto scatter ----
def pareto_scatter(
    df: pd.DataFrame,
    metric_key: str,
    save_path: str | Path,
    x_key: str = "wall_time_sec",
    baseline: dict | None = None,
    title: str | None = None,
) -> None:
    """Cost vs. metric. Colors by weight_mode, marker by granularity, faceted
    by propagation_mode. ``baseline`` dict {x, y, label} drops a reference."""
    prop_modes = [pm for pm in PROP_MODES if (df["propagation_mode"] == pm).any()]
    n = len(prop_modes)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n + 1.0, 3.6), sharey=True, squeeze=False)
    axes = axes.flatten()
    color_map = {wm: c for wm, c in zip(WEIGHT_MODES, plt.cm.tab10.colors[:4])}
    marker_map = {"global": "o", "type": "s", "freq": "^", "freq_type": "D"}
    for ax, pm in zip(axes, prop_modes):
        sub = df[df["propagation_mode"] == pm]
        for _, row in sub.iterrows():
            x = row.get(x_key); y = row.get(metric_key)
            if x is None or y is None or not np.isfinite(y):
                continue
            ax.scatter(x, y, c=[color_map.get(row["weight_mode"], "gray")],
                       marker=marker_map.get(row["weight_granularity"], "x"),
                       s=60, edgecolors="black", linewidths=0.5, alpha=0.8)
        if baseline is not None:
            ax.axhline(baseline["y"], color="red", linestyle="--", alpha=0.6,
                       label=f"{baseline.get('label', 'baseline')}={baseline['y']:.4f}")
            ax.scatter(baseline["x"], baseline["y"], c="red", marker="*", s=140,
                       edgecolors="black", linewidths=0.6, zorder=10)
        ax.set_title(pm, fontsize=10)
        ax.set_xlabel(x_key)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(metric_key)
    # Legend (single)
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], color=color_map[wm], marker="o", linestyle="",
                      markersize=8, label=wm) for wm in WEIGHT_MODES]
    handles += [Line2D([0],[0], color="gray", marker=marker_map[g], linestyle="",
                       markersize=8, label=g) for g in GRANULARITIES]
    if baseline is not None:
        handles.append(Line2D([0],[0], color="red", marker="*", linestyle="--",
                              markersize=12, label=baseline.get("label", "baseline")))
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 6),
               bbox_to_anchor=(0.5, 1.05), frameon=False, fontsize=9)
    fig.suptitle(title or f"Pareto: {x_key} vs {metric_key}", y=1.10, fontsize=12)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---- per-bucket weight bars (one run) ----
_CHANNEL_NAMES = {
    "node": ["w1_same", "w2_from_edges"],
    "edge": ["w3_from_nodes", "w4_same", "w5_from_faces", "w8_self"],
    "face": ["w6_from_edges", "w7_same", "w9_self"],
}


def per_bucket_weight_bars(
    weights_per_group: dict,
    save_path: str | Path,
    title: str = "",
) -> None:
    """One figure: 3 rows (node/edge/face) × n_groups columns.

    ``weights_per_group`` is the dict returned by
    ``LightCCNMulti.get_weights_per_group()``.
    """
    granularity = weights_per_group.get("granularity", "global")
    fig, axes = plt.subplots(3, 1, figsize=(8, 7))
    for ax, level in zip(axes, ("node", "edge", "face")):
        W = np.asarray(weights_per_group[level], dtype=float)  # (n_channels, n_groups)
        n_ch, n_groups = W.shape
        names = _CHANNEL_NAMES[level][:n_ch]
        x = np.arange(n_ch)
        bar_w = 0.8 / max(n_groups, 1)
        for g in range(n_groups):
            ax.bar(x + (g - (n_groups - 1) / 2) * bar_w, W[:, g], bar_w,
                   label=f"g{g}")
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
        ax.set_title(f"{level} weights ({granularity})", fontsize=10)
        ax.axhline(0, color="black", linewidth=0.5)
        if n_groups > 1:
            ax.legend(fontsize=7, ncol=min(n_groups, 6))
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---- metric trajectory across runs ----
def metric_trajectory(
    runs: Iterable[dict],
    metric_key: str,
    save_path: str | Path,
    title: str = "",
    label_fn=None,
    baseline: dict | None = None,
    max_lines: int = 25,
) -> None:
    """One line per run, ``eval_history[metric_key]`` over epochs."""
    runs = [r for r in runs if r and (r.get("eval_history") or [])]
    if not runs:
        return
    runs = runs[:max_lines]
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in runs:
        hist = r["eval_history"]
        epochs = [h.get("epoch", i) for i, h in enumerate(hist)]
        vals = [h.get(metric_key) for h in hist]
        if all(v is None for v in vals):
            continue
        label = (label_fn(r) if label_fn else r.get("label", ""))
        ax.plot(epochs, vals, marker="o", markersize=2, alpha=0.7, label=label)
    if baseline is not None:
        ax.axhline(baseline["y"], color="red", linestyle="--", alpha=0.6,
                   label=baseline.get("label", "baseline"))
    ax.set_xlabel("Epoch"); ax.set_ylabel(metric_key)
    ax.set_title(title or metric_key)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---- w2 trajectory & per-cell heatmap ----
def w2_trajectory(
    runs: Iterable[dict],
    save_path: str | Path,
    label_fn=None,
    title: str = "w_2 (node <- edges) over epochs",
) -> None:
    """Plot the scalar w_2 (group-mean if granular) per epoch for each run."""
    runs = [r for r in runs if r and r.get("weights_history")]
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in runs:
        wh = r["weights_history"]
        epochs, w2s = [], []
        for entry in wh:
            ep = entry.get("epoch")
            w2 = entry.get("node", {}).get("w2_from_edges")
            if w2 is None:
                continue
            epochs.append(ep); w2s.append(w2)
        if not epochs:
            continue
        label = (label_fn(r) if label_fn else r.get("label", ""))
        ax.plot(epochs, w2s, marker="o", markersize=2, alpha=0.7, label=label)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("w_2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---- master entry point ----
def make_sweep_plots(
    runs: list[dict],
    out_dir: str | Path,
    baseline_run: dict | None = None,
    metric_keys: list[tuple[str, str]] | None = None,
) -> list[str]:
    """Generate the full figure suite for one dataset notebook.

    Args:
        runs: list of results dicts (sweep + optionally LightGCN baseline).
        out_dir: where to write PNGs.
        baseline_run: the LightGCN run dict (used for delta heatmaps + Pareto
            reference). If None, deltas are skipped.
        metric_keys: list of (full_key, tail_key) pairs to plot. Defaults to
            R@{5,20,50} + N@{5,20,50}.

    Returns: list of generated file paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_keys = metric_keys or DEFAULT_METRIC_KEYS
    df = runs_to_df(runs)
    # Exclude LightGCN (no weight_mode/granularity meaning) from the grid.
    df_grid = df[df["model_name"] == "lightccn_multi"].copy()

    paths: list[str] = []

    # 1. Heatmaps (absolute) — 12 figs (6 metrics × {full, tail})
    for full_k, tail_k in metric_keys:
        for k, suffix in [(full_k, ""), (tail_k, " (tail)")]:
            p = out_dir / f"heatmap_abs__{k.replace('@','_').replace(' ','_').replace('(','').replace(')','')}.png"
            heatmap_per_prop_mode(df_grid, k, p, title_suffix=suffix)
            paths.append(str(p))

    # 2. Heatmaps (delta vs LightGCN) — only when baseline available
    if baseline_run is not None:
        for full_k, tail_k in metric_keys:
            for k, suffix in [(full_k, ""), (tail_k, " (tail)")]:
                base_v = _safe_metric(baseline_run, k)
                if base_v is None:
                    continue
                p = out_dir / f"heatmap_delta__{k.replace('@','_').replace(' ','_').replace('(','').replace(')','')}.png"
                heatmap_per_prop_mode(df_grid, k, p, title_suffix=suffix,
                                      baseline_value=base_v, divergent=True)
                paths.append(str(p))

    # 3. Pareto scatter — wall_time vs primary metrics (incl. small-K)
    for k in ("recall@20", "recall@5", "recall@3", "tail_recall@20"):
        base = None
        if baseline_run is not None:
            base_v = _safe_metric(baseline_run, k)
            if base_v is not None:
                base = {"x": float(baseline_run.get("wall_time_sec", 0.0)),
                        "y": base_v, "label": "LightGCN"}
        p = out_dir / f"pareto__wall_time__{k.replace('@','_')}.png"
        pareto_scatter(df_grid, k, p, baseline=base)
        paths.append(str(p))

    # 4. Metric trajectories — one figure per primary metric, all runs overlaid
    def _lbl(r): return f"{r.get('propagation_mode','?')[:5]}_{r.get('weight_mode','?')[:3]}_{r.get('weight_granularity','?')[:5]}"
    for k in ("recall@20", "recall@5", "recall@3", "ndcg@3", "tail_recall@20", "ndcg@20"):
        base = None
        if baseline_run is not None:
            base_v = _safe_metric(baseline_run, k)
            if base_v is not None:
                base = {"y": base_v, "label": "LightGCN"}
        p = out_dir / f"trajectory__{k.replace('@','_')}.png"
        metric_trajectory(runs, k, p, title=f"{k} over epochs",
                          label_fn=_lbl, baseline=base, max_lines=25)
        paths.append(str(p))

    # 5. w_2 trajectory (sweep runs only)
    p = out_dir / "w2_trajectory.png"
    w2_trajectory([r for r in runs if r and r.get("model_name") == "lightccn_multi"],
                  p, label_fn=_lbl)
    paths.append(str(p))

    return paths
