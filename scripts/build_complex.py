"""Standalone script to precompute cell complex for a dataset.

Usage:
    python scripts/build_complex.py --dataset gowalla --tau 20
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from light_ccn.data.dataset import CFDataset
from light_ccn.complex.cell_complex import CellComplexBuilder
from light_ccn.utils.helpers import timer


def main():
    parser = argparse.ArgumentParser(description="Precompute cell complex")
    parser.add_argument("--dataset", type=str, default="gowalla")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--tau", type=int, default=20)
    parser.add_argument("--cache-dir", type=str, default="data/complex_cache")
    args = parser.parse_args()

    dataset = CFDataset(args.dataset, args.data_dir)
    R = dataset.get_interaction_matrix()

    with timer(f"Cell complex for {args.dataset} (tau={args.tau})"):
        builder = CellComplexBuilder(
            R, tau=args.tau,
            cache_dir=args.cache_dir,
            dataset_name=args.dataset,
        )
        result = builder.build_and_cache()

    n_items = R.shape[1]
    items_in_faces = set()
    for face in result['faces']:
        items_in_faces.update(face)
    pct = 100.0 * len(items_in_faces) / n_items if n_items > 0 else 0.0

    print(f"\nStatistics:")
    print(f"  Faces: {len(result['faces']):,}")
    print(f"  Edges: {len(result['edges']):,}")
    print(f"  Items covered: {len(items_in_faces):,} / {n_items:,} ({pct:.1f}%)")
    print(f"  S shape: {result['S'].shape}, nnz: {result['S'].nnz:,}")
    print(f"  B1 shape: {result['B1'].shape}, nnz: {result['B1'].nnz:,}")
    print(f"  B2 shape: {result['B2'].shape}, nnz: {result['B2'].nnz:,}")


if __name__ == "__main__":
    main()
