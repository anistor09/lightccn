# Where each table's numbers come from

Every reported value is a metric on the official test split, at the epoch with the
best validation Recall@20 (each record stores its validation trajectory, its test
trajectory, and `test_at_val_best`, the test row at that epoch). `summary.csv`, built
by `../analysis/reproduce_tables.py`, lists those metrics for every run; its
`selection` column marks the few appendix and calibration runs that have no
validation carve and instead report their own `final_metrics`.

Run labels encode the model and configuration:

| name              | label                                       |
|-------------------|---------------------------------------------|
| LightCCN-full     | `…sef.sm.B…` (stateful edges+faces, softmax weights, user readout) |
| LightCCN-prop     | `…sef.sm…` (propagation only, no readout)   |
| LightGCN backbone | `…lightgcn…`                                |

`t<n>` in a label is the face-support threshold τ; `L<n>` the number of layers.

## RQ1 — accuracy against the backbone
Tables at K=3/5/10/20 and the NDCG@K figure.
- `depthro/` — the L3 records for each dataset (LightGCN and LightCCN at the
  selected τ), which the comparison is built from.
- `campaign_val_*/` — the matching runs with self-contained validation and test
  trajectories.

## RQ1 — external baselines
- `baseline_tuning/` — MF, NGCF and SimGCL tuned per dataset.
- `abl_base2/`, `abl_night1/`, `abl_night2/`, `baselines_repro/`,
  `baselines_mf_ngcf/`, `baselines_mf_ngcf_simgcl/`, `baselines_hccf/`, `ngcf_retest/`
  — the baseline runs (MF, NGCF, SimGCL, and an earlier HCCF pass).
- `hccf_fix/` — HCCF tuned in this pipeline; `hccf_arbiter/` — the authors' HCCF
  implementation run on the same splits, for cross-checking.

## RQ2 — propagation vs readout
- `depthro/` — the full model (`sef.sm.B`), the propagation-only model (`sef.sm`),
  and the nine learned scalars (channel weights and the two readout gates).

## RQ3 — propagation depth
- `abl_depth3/` — both models at layers 1, 2, 4, 5, 6.
- `depthro/`, `gowalla_sef_sm_l3/`, `gowalla_se_sm_l3/`, `foursquare_tky_grid/`,
  `gowalla_grid/`, `gowalla_finish/` — the layer-3 references. The smoothness
  measurements are logged inside these runs.

## RQ4 — amount of higher-order structure (τ)
- `tau_runs/` and `abl_depth3/` — the model trained across the τ grid.
- `tau_sweep/` — the face and support counts per dataset (`<ds>_taucurve.json`).

## Setup and calibration
- Dataset statistics and the complex at the selected τ use the dataset loaders and
  `tau_sweep/`.
- `calibration/` — LightGCN on Gowalla and Yelp2018 against the published numbers
  (see `calibration/README.md`).

## Appendix ablations
- Readout tower ablation: `readout_ablation/`, `abl_towers/`.
- LightGCN on the complex: `ablation_cc/`, `ablation_cc_val/`.
- User–item edges as rank-1 cells: `beidian_uiedge/`, `ciaodvd_uiedge/`.
