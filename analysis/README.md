# Analysis

`reproduce_tables.py` rebuilds the result tables from the run records in
`../results`.

```
python analysis/reproduce_tables.py
```

It writes three kinds of output:

- `../results/summary.csv` — one row per run, with Recall, NDCG and ARHR at
  cutoffs 3/5/10/20/50. The `selection` column records how each row was chosen.
- `tables/rq1_k{3,5,10}.csv` — the headline accuracy comparison, LightGCN
  against the two LightCCN variants, with the per-metric percentage gain.
- `tables/rq1_k{3,5,10}.tex` — the same tables as LaTeX.

## Selection rule

Every run stores a validation trace and a test trace logged at the same epochs.
Models are selected on validation and reported on test: the epoch with the best
validation Recall@20 is chosen, and the test metrics at that epoch are the
reported numbers. Runs evaluated without a validation carve (some appendix and
calibration runs) report their own `final_metrics`; the `selection` column
distinguishes the two cases.

The script ends by checking its own output against the numbers printed in the
thesis RQ1 table and reports whether they reproduce.

## Significance

The `$^{*}$` markers in the thesis come from paired per-user tests, which need
the per-user ranking files (`*.ranks.npz`). Those are large and are not
included here; they are available on request. Without them the tables above
carry point estimates.

`../results/PROVENANCE.md` maps each thesis table and figure to the records that
produce it.

Requirements: Python 3.10+ (standard library only).
