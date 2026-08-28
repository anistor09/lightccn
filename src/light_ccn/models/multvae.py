"""Mult-VAE (Liang et al., WWW 2018) — variational autoencoder with a
multinomial likelihood over each user's interaction vector.

Architecture per the LightGCN paper's baseline setup: I -> 600 -> (mu,logvar 200)
-> 600 -> I, tanh activations, input dropout 0.5, beta annealed linearly to
``anneal_cap`` over ``total_anneal_steps`` gradient steps.

Scoring for evaluation goes through ``full_scores_for_users`` (the Evaluator's
non-factorized hook): input is the user's TRAINING vector (the same matrix the
model was fit on — under the validation protocol that is the reduced train set),
output the decoder logits over all items.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from light_ccn.models import register_model
from light_ccn.training.trainer import Trainer


@register_model("multvae")
class MultVAE(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        hidden_dim: int = 600,
        latent_dim: int = 200,
        dropout: float = 0.5,
        total_anneal_steps: int = 200000,
        anneal_cap: float = 0.2,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.dropout = dropout
        self.total_anneal_steps = total_anneal_steps
        self.anneal_cap = anneal_cap

        self.enc1 = nn.Linear(n_items, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, 2 * latent_dim)
        self.dec1 = nn.Linear(latent_dim, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, n_items)
        for layer in (self.enc1, self.enc2, self.dec1, self.dec2):
            nn.init.xavier_normal_(layer.weight)
            nn.init.normal_(layer.bias, std=0.001)

        self._R: sp.csr_matrix | None = None  # training interaction matrix

    # ── data wiring ──────────────────────────────────────────────────────
    def set_interactions(self, R: sp.csr_matrix) -> None:
        self._R = R.tocsr().astype(np.float32)

    def _rows(self, users: np.ndarray, device) -> torch.Tensor:
        dense = np.asarray(self._R[users].todense(), dtype=np.float32)
        return torch.from_numpy(dense).to(device)

    # ── VAE forward ──────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        h = F.normalize(x, p=2, dim=1)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.tanh(self.enc1(h))
        h = self.enc2(h)
        mu, logvar = h[:, : h.shape[1] // 2], h[:, h.shape[1] // 2 :]
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        out = torch.tanh(self.dec1(z))
        logits = self.dec2(out)
        return logits, mu, logvar

    def loss(self, x: torch.Tensor, beta: float) -> torch.Tensor:
        logits, mu, logvar = self.forward(x)
        neg_ll = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * x, dim=1))
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return neg_ll + beta * kl

    # ── Evaluator hook (non-factorized scoring) ─────────────────────────
    @torch.no_grad()
    def full_scores_for_users(self, batch_users: np.ndarray, device) -> torch.Tensor:
        x = self._rows(np.asarray(batch_users, dtype=np.int64), device)
        logits, _, _ = self.forward(x)
        return logits


class MultVAETrainer(Trainer):
    """Reuses the standard Trainer's eval / early-stop / rank-dump machinery;
    only the per-epoch optimization is replaced (full-row multinomial VAE
    objective instead of BPR triplets)."""

    def __init__(self, *args, mv_batch_size: int = 512,
                 mv_anneal_faithful: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._mv_batch = mv_batch_size
        self._anneal_faithful = mv_anneal_faithful
        self._anneal_step = 0
        self._train_users = np.array(
            [u for u, its in self.dataset.train_dict.items() if len(its) > 0],
            dtype=np.int64,
        )

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        rng = np.random.default_rng(self.config.train.seed + epoch)
        users = rng.permutation(self._train_users)
        total, nb = 0.0, 0
        pbar = tqdm(range(0, len(users), self._mv_batch),
                    desc=f"Epoch {epoch}", leave=False)
        for s in pbar:
            batch = users[s : s + self._mv_batch]
            x = self.model._rows(batch, self.device)
            if self.model.total_anneal_steps > 0:
                if self._anneal_faithful:
                    # vae_cf (and RecBole) semantics: fixed 1/T ramp, the cap
                    # only clips it — reaches the cap at t = cap*T, not at T.
                    beta = min(self.model.anneal_cap,
                               self._anneal_step / self.model.total_anneal_steps)
                else:
                    beta = min(self.model.anneal_cap,
                               self.model.anneal_cap * self._anneal_step
                               / self.model.total_anneal_steps)
            else:
                beta = self.model.anneal_cap
            loss = self.model.loss(x, beta)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self._anneal_step += 1
            total += loss.item(); nb += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg = total / max(nb, 1)
        print(f"  Epoch {epoch}: avg_loss = {avg:.4f}")
        return avg
