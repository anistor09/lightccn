"""Loss functions for collaborative filtering models."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bpr_loss(
    user_e: torch.Tensor,
    pos_e: torch.Tensor,
    neg_e: torch.Tensor,
    reg_loss: torch.Tensor,
    reg_weight: float = 1e-4,
) -> torch.Tensor:
    """Bayesian Personalized Ranking loss with L2 regularization.

    loss = -mean(log(sigmoid(pos_score - neg_score))) + reg_weight * reg_loss

    Args:
        user_e: (batch, dim) user embeddings.
        pos_e: (batch, dim) positive item embeddings.
        neg_e: (batch, dim) negative item embeddings.
        reg_loss: L2 regularization term on ego embeddings.
        reg_weight: Regularization coefficient.

    Returns:
        Scalar loss tensor.
    """
    pos_scores = (user_e * pos_e).sum(dim=1)
    neg_scores = (user_e * neg_e).sum(dim=1)
    bpr = F.softplus(neg_scores - pos_scores).mean()
    return bpr + reg_weight * reg_loss


def topology_aux_loss(
    item_embeddings: torch.Tensor,
    face_indices: torch.Tensor,
    n_users: int,
) -> torch.Tensor:
    """F4 fix: topology auxiliary loss.

    For each face (i, j, k), pull the three item embeddings closer together:
        L_topo = sum_{f=(i,j,k)} (||e_i - e_j||^2 + ||e_j - e_k||^2 + ||e_i - e_k||^2)

    Returns mean over faces so the magnitude is independent of |faces|.

    Args:
        item_embeddings: (n_users + n_items, dim) stacked embeddings; rows
            n_users..n_users+n_items are item embeddings.
        face_indices: (n_faces, 3) tensor of item indices (0..n_items-1).
        n_users: Offset to apply to face indices to get bipartite row indices.

    Returns:
        Scalar topology aux loss.
    """
    if face_indices.shape[0] == 0:
        return torch.zeros(1, device=item_embeddings.device).squeeze()
    # Look up item embeddings (offset by n_users since complex is bipartite)
    bipartite_idx = face_indices + n_users
    e_i = item_embeddings[bipartite_idx[:, 0]]
    e_j = item_embeddings[bipartite_idx[:, 1]]
    e_k = item_embeddings[bipartite_idx[:, 2]]
    loss = ((e_i - e_j).pow(2).sum(dim=1)
            + (e_j - e_k).pow(2).sum(dim=1)
            + (e_i - e_k).pow(2).sum(dim=1)).mean()
    return loss


def infonce_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """InfoNCE contrastive loss for self-supervised learning (SGL).

    Args:
        z1: (batch, dim) embeddings from view 1.
        z2: (batch, dim) embeddings from view 2.
        temperature: Temperature scaling parameter.

    Returns:
        Scalar contrastive loss.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Positive pairs: (z1[i], z2[i])
    pos_score = (z1 * z2).sum(dim=1) / temperature

    # All pairs: z1 @ z2^T
    all_score = z1 @ z2.T / temperature

    # InfoNCE: -log(exp(pos) / sum(exp(all)))
    loss = -pos_score + torch.logsumexp(all_score, dim=1)
    return loss.mean()
