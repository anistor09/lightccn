# LightCCN — Lightweight Cell Complex Networks for Collaborative Filtering

LightCCN extends LightGCN with the higher-order structure of a rank-2 cell
complex. Item triples that at least *τ* users co-consume become faces; embeddings
are propagated at every rank of the complex under LightGCN's transformation-free
design, and a readout injects the pooled higher-order signal into the user
representations. The full model adds nine scalar parameters to its backbone, so
any change from the backbone is attributable to structure rather than capacity.

This repository contains the model, the training and evaluation pipeline, the
baselines used for comparison, and the raw result records behind the thesis
tables and figures.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Requires Python 3.10+ and PyTorch. Datasets are downloaded from their original
public sources on first use (see `src/light_ccn/data/download.py`); none are
redistributed here.

## Train a model

```bash
# LightCCN on Gowalla
python scripts/train.py --config configs/lightccn_multi_gowalla.yaml

# the LightGCN backbone, same pipeline
python scripts/train.py --config configs/lightgcn_gowalla.yaml
```

Configs for every model and dataset are in `configs/`. `scripts/train.py --help`
lists the ablation flags (depth, weight mode, readout gates, τ, tail evaluation).

## Models

The thesis model and the baselines it is compared against, all trained and
evaluated under the same pipeline: `lightccn_multi` (LightCCN — higher-order
propagation, with the user readout in the full model) and the baselines
`lightgcn`, `mf`, `ngcf`, `simgcl`, and `hccf`. See `NOTICE.md` for the reference
implementation each baseline was verified against.

## Results

`results/` holds the raw per-run JSON records: metrics at cutoffs 3/5/10/20/50
(full and tail), the run configuration, and the training curves. Each record's
`test_at_val_best` are the test-split metrics at the epoch with the best
validation Recall@20 — the values reported in the thesis.

`results/PROVENANCE.md` maps each table and figure to the records that produce it,
and `analysis/reproduce_tables.py` gathers every run's metrics into
`results/summary.csv` and regenerates the headline tables. Calibration runs
(LightGCN on Gowalla and Yelp2018 against the published numbers) are in
`results/calibration/`.

## Citation

Alexandru Nistor, *LightCCN: Lightweight Cell Complex Networks for Collaborative
Filtering*, MSc thesis, TU Delft, 2026.

Released under the MIT License (see `LICENSE`).
