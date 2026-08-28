"""Plotting utilities for training loss curves and metric comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training_loss(
    losses_dict: dict[str, list[float]],
    save_path: str | Path,
    title: str = "Training Loss",
) -> None:
    """Plot training loss curves for multiple models.

    Args:
        losses_dict: {model_name: [loss_per_epoch, ...]}.
        save_path: Where to save the figure.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, losses in losses_dict.items():
        ax.plot(range(1, len(losses) + 1), losses, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss plot saved: {save_path}")


def plot_metric_comparison(
    results_dict: dict[str, dict[str, float]],
    save_path: str | Path,
    title: str = "Model Comparison",
) -> None:
    """Plot grouped bar chart comparing metrics across models.

    Args:
        results_dict: {model_name: {metric_name: value, ...}}.
        save_path: Where to save the figure.
        title: Plot title.
    """
    models = list(results_dict.keys())
    if not models:
        return

    metrics = [k for k in results_dict[models[0]].keys() if k != "epoch"]
    n_metrics = len(metrics)
    n_models = len(models)

    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(12, n_metrics * 2), 6))
    for i, model in enumerate(models):
        values = [results_dict[model].get(m, 0) for m in metrics]
        bars = ax.bar(x + i * width - 0.4 + width / 2, values, width, label=model)
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom", fontsize=7, rotation=45,
            )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Metric comparison plot saved: {save_path}")


def plot_convergence(
    eval_results: dict[str, list[dict]],
    metric_key: str,
    save_path: str | Path,
    title: str = "Convergence",
) -> None:
    """Plot a specific metric over training epochs for multiple models.

    Args:
        eval_results: {model_name: [{"epoch": e, metric_key: v, ...}, ...]}.
        metric_key: Which metric to plot (e.g., "recall@20").
        save_path: Where to save the figure.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, evals in eval_results.items():
        epochs = [e["epoch"] for e in evals]
        values = [e.get(metric_key, 0) for e in evals]
        ax.plot(epochs, values, marker="o", markersize=3, label=name)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_key)
    ax.set_title(f"{title}: {metric_key}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Convergence plot saved: {save_path}")
