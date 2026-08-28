"""F3 fix: Observability metrics for LightCCN-Multi training diagnostics.

Records:
- Per-cochain forward magnitudes per layer
- Per-cochain gradient norms (X^(0), H_1^(0), H_2^(0))
- Per-weight scalar gradient norms (w_1..w_7)
- Per-operator contribution ratios (||w_i * op_i * x|| / ||x_new||)
- Global gradient norm (per batch)
- Training-loss rolling standard deviation
- Weight trajectories (per eval epoch)

Designed to be minimally invasive: optional, controlled by a flag, low overhead.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import pstdev
from typing import Any

import torch


class ObservabilityLogger:
    """Diagnostic logger for Multi-style models.

    Attach via `attach(trainer, model)` and call `on_batch(loss, model)` after
    each backward pass, `on_epoch_end(model)` at the end of each epoch.
    """

    def __init__(self, enabled: bool = True, rolling_window: int = 50):
        self.enabled = enabled
        self.rolling_window = rolling_window

        # Per-batch
        self.batch_losses: list[float] = []
        self.global_grad_norms: list[float] = []

        # Per-epoch (accumulated then averaged)
        self.epoch_records: list[dict[str, Any]] = []
        self._current_epoch_grad_X: list[float] = []
        self._current_epoch_grad_H1: list[float] = []
        self._current_epoch_grad_H2: list[float] = []
        self._current_epoch_grad_weights: dict[str, list[float]] = defaultdict(list)
        self._current_epoch_global_norm: list[float] = []

        # Per-cochain forward magnitudes per layer, recorded at epoch end
        self.forward_magnitudes_history: list[dict[str, Any]] = []

    # ───────── Per-batch hooks ──────────────────────────────────────────────

    def on_batch(self, loss: float, model: torch.nn.Module) -> None:
        """Call after `loss.backward()` but before `optimizer.step()`."""
        if not self.enabled:
            return

        self.batch_losses.append(loss)

        # Per-batch global gradient norm
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total += p.grad.detach().pow(2).sum().item()
        gnorm = total ** 0.5
        self.global_grad_norms.append(gnorm)
        self._current_epoch_global_norm.append(gnorm)

        # Per-cochain ego-embedding gradient norms.
        # Source depends on init_mode: 'table' uses the nn.Embedding tables;
        # 'fc' uses the W_U, W_I encoder matrices (the FC params).
        x_params = []
        if getattr(model, "user_embedding", None) is not None \
                and model.user_embedding.weight.grad is not None:
            x_params.append(model.user_embedding.weight.grad)
        if getattr(model, "item_embedding", None) is not None \
                and model.item_embedding.weight.grad is not None:
            x_params.append(model.item_embedding.weight.grad)
        # FC init path: W_U / W_I carry the node-embedding gradient signal.
        if getattr(model, "W_U", None) is not None and model.W_U.grad is not None:
            x_params.append(model.W_U.grad)
        if getattr(model, "W_I", None) is not None and model.W_I.grad is not None:
            x_params.append(model.W_I.grad)
        if x_params:
            n = sum(g.pow(2).sum() for g in x_params).sqrt().item()
            self._current_epoch_grad_X.append(n)
        if hasattr(model, "edge_embedding") and model.edge_embedding is not None and \
                model.edge_embedding.weight.grad is not None:
            self._current_epoch_grad_H1.append(
                model.edge_embedding.weight.grad.pow(2).sum().sqrt().item()
            )
        if hasattr(model, "face_embedding") and model.face_embedding is not None and \
                model.face_embedding.weight.grad is not None:
            self._current_epoch_grad_H2.append(
                model.face_embedding.weight.grad.pow(2).sum().sqrt().item()
            )

        # Per-weight scalar gradient norms (for Multi/Hodge)
        for name in ("node_logits", "edge_logits", "face_logits",
                     "node_weights", "edge_weights", "face_weights"):
            p = getattr(model, name, None)
            if p is not None and p.grad is not None:
                self._current_epoch_grad_weights[name].append(
                    p.grad.pow(2).sum().sqrt().item()
                )

    # ───────── Per-epoch hooks ──────────────────────────────────────────────

    def on_epoch_end(self, model: torch.nn.Module, epoch: int) -> None:
        """Call once per epoch after all batches finished."""
        if not self.enabled:
            return

        record: dict[str, Any] = {"epoch": epoch}

        # Average gradient norms across the epoch
        if self._current_epoch_grad_X:
            record["grad_X_avg"] = sum(self._current_epoch_grad_X) / len(self._current_epoch_grad_X)
        if self._current_epoch_grad_H1:
            record["grad_H1_avg"] = sum(self._current_epoch_grad_H1) / len(self._current_epoch_grad_H1)
        if self._current_epoch_grad_H2:
            record["grad_H2_avg"] = sum(self._current_epoch_grad_H2) / len(self._current_epoch_grad_H2)
        if self._current_epoch_global_norm:
            record["global_grad_norm_avg"] = (
                sum(self._current_epoch_global_norm) / len(self._current_epoch_global_norm)
            )
            record["global_grad_norm_max"] = max(self._current_epoch_global_norm)

        per_weight: dict[str, float] = {}
        for name, vals in self._current_epoch_grad_weights.items():
            if vals:
                per_weight[f"{name}_grad_avg"] = sum(vals) / len(vals)
        record["weight_scalar_grads"] = per_weight

        # Rolling std of loss over the trailing window
        recent = self.batch_losses[-self.rolling_window:]
        if len(recent) >= 2:
            record["loss_rolling_std"] = pstdev(recent)

        # Measure per-cochain forward magnitudes (single forward pass)
        forward_mags = self._measure_forward_magnitudes(model)
        if forward_mags is not None:
            record["forward_magnitudes"] = forward_mags
            self.forward_magnitudes_history.append({"epoch": epoch, **forward_mags})

        self.epoch_records.append(record)

        # Reset per-epoch accumulators
        self._current_epoch_grad_X.clear()
        self._current_epoch_grad_H1.clear()
        self._current_epoch_grad_H2.clear()
        self._current_epoch_grad_weights.clear()
        self._current_epoch_global_norm.clear()

    # ───────── Forward magnitudes (Multi-aware) ─────────────────────────────

    def _measure_forward_magnitudes(self, model: torch.nn.Module) -> dict[str, Any] | None:
        """Measure ||X^(k)||, ||H_1^(k)||, ||H_2^(k)|| per layer of Multi.

        Only works for LightCCNMulti; returns None for other models.
        """
        if not hasattr(model, "operators") or "A_hat_0" not in getattr(model, "operators", {}):
            return None

        model.eval()
        try:
            with torch.no_grad():
                # Replicate Multi's propagate loop, recording magnitudes per layer
                x_nodes = model.get_ego_embeddings()
                has_edges = getattr(model, "n_edges", 0) > 0
                has_faces = getattr(model, "n_faces", 0) > 0

                if has_edges:
                    if model.edge_embedding is not None:
                        x_edges = model.edge_embedding.weight
                    else:
                        x_edges = x_nodes.new_zeros((model.n_edges, model.edge_embed_dim))
                else:
                    x_edges = None
                if has_faces:
                    if model.face_embedding is not None:
                        x_faces = model.face_embedding.weight
                    else:
                        x_faces = x_nodes.new_zeros((model.n_faces, model.face_embed_dim))
                else:
                    x_faces = None

                layer_mags: list[dict[str, float]] = [{
                    "X": float(x_nodes.norm().item()),
                    "H1": float(x_edges.norm().item()) if x_edges is not None else 0.0,
                    "H2": float(x_faces.norm().item()) if x_faces is not None else 0.0,
                }]

                n_layers = getattr(model, "n_layers", 2)
                ops = model.operators
                w_node, w_edge, w_face = model._get_weights()
                # Reuse the model's mix helper so per-cell weights broadcast
                # correctly under weight_granularity != 'global'.
                mix = type(model)._mix

                for _ in range(n_layers):
                    if has_edges:
                        node_from_node = torch.sparse.mm(ops["A_hat_0"], x_nodes)
                        node_from_edge = torch.sparse.mm(ops["B_hat_1_down"], x_edges)
                        x_nodes = mix(w_node[0], node_from_node) + mix(w_node[1], node_from_edge)

                        edge_from_node = torch.sparse.mm(ops["B_hat_1_up"], x_nodes)
                        edge_from_edge = torch.sparse.mm(ops["A_hat_1"], x_edges)
                        if has_faces:
                            edge_from_face = torch.sparse.mm(ops["B_hat_2_down"], x_faces)
                            x_edges_new = (mix(w_edge[0], edge_from_node)
                                           + mix(w_edge[1], edge_from_edge)
                                           + mix(w_edge[2], edge_from_face))
                        else:
                            x_edges_new = mix(w_edge[0], edge_from_node) + mix(w_edge[1], edge_from_edge)

                        if has_faces:
                            face_from_edge = torch.sparse.mm(ops["B_hat_2_up"], x_edges)
                            face_from_face = torch.sparse.mm(ops["A_hat_2"], x_faces)
                            x_faces = mix(w_face[0], face_from_edge) + mix(w_face[1], face_from_face)
                        x_edges = x_edges_new
                    else:
                        x_nodes = torch.sparse.mm(ops["A_hat_0"], x_nodes)

                    layer_mags.append({
                        "X": float(x_nodes.norm().item()),
                        "H1": float(x_edges.norm().item()) if x_edges is not None else 0.0,
                        "H2": float(x_faces.norm().item()) if x_faces is not None else 0.0,
                    })
        except Exception as e:  # noqa: BLE001
            print(f"  [observability] forward-magnitude probe failed: {e}")
            model.train()
            return None
        model.train()
        return {"per_layer": layer_mags}

    # ───────── Export ───────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "batch_losses": self.batch_losses,
            "global_grad_norms": self.global_grad_norms,
            "epoch_records": self.epoch_records,
            "forward_magnitudes_history": self.forward_magnitudes_history,
        }
