"""Generate comparison plots from saved result JSON files.

Usage:
    python scripts/plot_results.py --results-dir results --output-dir results/plots
    python scripts/plot_results.py --results-dir results --dataset gowalla
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from light_ccn.plotting.plots import (
    plot_training_loss,
    plot_metric_comparison,
    plot_convergence,
)


def load_results(results_dir: str, dataset: str | None = None) -> dict[str, dict]:
    """Load all result JSON files, optionally filtering by dataset.

    Returns dict keyed by model_name with latest result per model+dataset.
    """
    results_path = Path(results_dir)
    all_results = {}

    for f in sorted(results_path.glob("*.json")):
        with open(f) as fp:
            data = json.load(fp)
        ds = data.get("dataset", "")
        if dataset and ds != dataset:
            continue
        key = f"{data['model']}_{ds}"
        all_results[key] = data  # Latest file wins (sorted by name/timestamp)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Plot comparison results")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="results/plots")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Filter by dataset name (e.g., gowalla)")
    args = parser.parse_args()

    results = load_results(args.results_dir, args.dataset)
    if not results:
        print("No results found. Run training first.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{args.dataset}" if args.dataset else ""

    # 1. Training loss curves
    losses_dict = {}
    for key, data in results.items():
        if data.get("train_losses"):
            losses_dict[key] = data["train_losses"]
    if losses_dict:
        plot_training_loss(
            losses_dict,
            output_dir / f"training_loss{suffix}.png",
            title=f"Training Loss{' - ' + args.dataset if args.dataset else ''}",
        )

    # 2. Final metric comparison
    metrics_dict = {}
    for key, data in results.items():
        if data.get("final_metrics"):
            metrics_dict[key] = data["final_metrics"]
    if metrics_dict:
        plot_metric_comparison(
            metrics_dict,
            output_dir / f"metric_comparison{suffix}.png",
            title=f"Model Comparison{' - ' + args.dataset if args.dataset else ''}",
        )

    # 3. Convergence plots (recall@20 over epochs)
    eval_dict = {}
    for key, data in results.items():
        if data.get("eval_results"):
            eval_dict[key] = data["eval_results"]
    if eval_dict:
        for metric in ["recall@20", "ndcg@20"]:
            plot_convergence(
                eval_dict,
                metric,
                output_dir / f"convergence_{metric.replace('@', '_')}{suffix}.png",
                title=f"Convergence{' - ' + args.dataset if args.dataset else ''}",
            )

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
