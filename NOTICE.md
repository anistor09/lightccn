# Third-party code and baselines

LightCCN reimplements several baseline recommenders for a like-for-like
comparison under one training and evaluation pipeline. The implementations in
`src/light_ccn/models/` are our own, written from the papers and verified
against the authors' reference code where available:

- **LightGCN** — He et al., *LightGCN: Simplifying and Powering Graph Convolution
  Network for Recommendation*, SIGIR 2020. Reference: github.com/gusye1234/LightGCN-PyTorch (MIT).
- **NGCF** — Wang et al., *Neural Graph Collaborative Filtering*, SIGIR 2019.
  Reference: github.com/xiangwang1223/neural_graph_collaborative_filtering (MIT).
- **SimGCL** — Yu et al., *Are Graph Augmentations Necessary?*, SIGIR 2022.
- **HCCF** — Xia et al., *Hypergraph Contrastive Collaborative Filtering*, SIGIR 2022.
  Verified against the authors' framework SSLRec (github.com/HKUDS/SSLRec).
- **MF (BPR)** — Rendle et al., *BPR: Bayesian Personalized Ranking*, UAI 2009.

Cell-complex background follows Bodnar et al., *CW Networks* (NeurIPS 2021) and
Hajij et al., *Topological Deep Learning* (2023).

Datasets are downloaded from their original public sources at run time (see
`src/light_ccn/data/download.py`); none are redistributed in this repository.
