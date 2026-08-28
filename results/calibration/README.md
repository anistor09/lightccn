# Calibration against published numbers

To confirm the training/evaluation pipeline is sound, LightGCN was run on the
two datasets whose original paper reports numbers on the same split: Gowalla and
Yelp2018 (LightGCN, He et al., SIGIR 2020, Table 3). Our runs select the epoch on
a held-out **validation** split and report on the official **test** split at that
epoch; the paper early-stops on the test set (marked *). Our numbers sit a few
percent below the published ones, in the direction expected from honest
validation-based selection.

| dataset  | metric     | published* | ours (val-selected) | Δ      |
|----------|------------|-----------:|--------------------:|-------:|
| Gowalla  | Recall@20  | 0.1830     | 0.1762              | −3.7%  |
| Gowalla  | NDCG@20    | 0.1554     | 0.1510              | −2.8%  |
| Yelp2018 | Recall@20  | 0.0649     | 0.0627              | −3.4%  |
| Yelp2018 | NDCG@20    | 0.0530     | 0.0514              | −3.0%  |

\* Published figures use test-set early stopping (LightGCN, He et al., SIGIR 2020).

Raw run records: `lightgcn__{gowalla,yelp2018}__L3__{val,tst}.json`.
