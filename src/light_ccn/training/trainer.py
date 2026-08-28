"""Training loop with early stopping, checkpointing, and result logging."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from light_ccn.config import ExperimentConfig
from light_ccn.data.dataset import CFDataset
from light_ccn.data.sampler import BPRSampler
from light_ccn.evaluation.metrics import Evaluator
from light_ccn.models.base import BaseCFModel
from light_ccn.training.loss import bpr_loss, topology_aux_loss
from light_ccn.training.observability import ObservabilityLogger


class Trainer:
    """Training loop for collaborative filtering models."""

    def __init__(
        self,
        model: BaseCFModel,
        dataset: CFDataset,
        config: ExperimentConfig,
        save_enabled: bool = True,
        tail_items: set[int] | None = None,
        topology_aux_weight: float = 0.0,
        faces: list | None = None,
        observability: bool = False,
        fc_freeze_after_epoch: int | None = None,
        fc_snapshot_to_table_at_epoch: int | None = None,
        keep_best_state: bool = False,
        val_dataset=None,
        eval_dataset=None,
    ):
        self.model = model
        self.dataset = dataset
        # Evaluation views (duck-typed: anything with .train_dict/.test_dict).
        # eval_dataset: what test metrics are computed against (in validation
        # mode its train_dict masks train ∪ val). val_dataset: when set, the
        # early-stopping metric comes from THIS view (targets = held-out val
        # items) instead of the test evaluation — the unbiased protocol.
        self.eval_dataset = eval_dataset if eval_dataset is not None else dataset
        self.val_dataset = val_dataset
        self.config = config
        self.save_enabled = save_enabled
        self.tail_items = tail_items
        # FC freeze knob: at the START of this epoch (1-indexed), freeze the FC
        # encoder params (W_U, W_I, b_U, b_I) so they are no longer updated by
        # the optimizer. Aligns with HOUR's "initialization" framing (§4.2).
        # None = never freeze (FC co-trained throughout, current default).
        self.fc_freeze_after_epoch = fc_freeze_after_epoch
        self._fc_frozen: bool = False
        # FC->table snapshot knob: at the START of this epoch, convert the FC
        # encoder into free nn.Embedding tables (one row per user/item),
        # initialized from the current FC outputs. Each row becomes independently
        # trainable from then on. Literal "FC = initialization" reading.
        # Mutually exclusive with fc_freeze_after_epoch (the freeze and the
        # snapshot do different things at the same hook point).
        self.fc_snapshot_to_table_at_epoch = fc_snapshot_to_table_at_epoch
        self._fc_snapshot_done: bool = False

        # F4: topology auxiliary loss (flag-gated by weight > 0)
        self.topology_aux_weight = topology_aux_weight
        self._faces_tensor: torch.Tensor | None = None
        if topology_aux_weight > 0 and faces is not None and len(faces) > 0:
            self._faces_tensor = torch.tensor(faces, dtype=torch.long)

        # F3: observability logger
        self.observability_logger = ObservabilityLogger(enabled=observability)
        requested = config.train.device
        if requested == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif requested == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.train.lr
        )
        self.sampler = BPRSampler(
            dataset.train_dict,
            dataset.n_items,
            batch_size=config.data.batch_size,
            seed=config.train.seed,
        )
        self.evaluator = Evaluator(topk=config.eval.topk)

        # Note: AMP (FP16) is NOT used because torch.sparse.mm does not
        # support Half precision on CUDA. All models rely on sparse matmul
        # in propagate(), so AMP would error. TF32 (enabled in helpers.py)
        # already accelerates the dense matmuls on Ampere+ / Ada GPUs.

        # Tracking
        self.best_metric = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        # When enabled, keep a CPU copy of the model weights at the best epoch
        # so the driver can persist the *deployed* checkpoint (the one whose
        # metrics we report), not the final-epoch weights. Cheap: one clone,
        # overwritten each time the primary metric improves.
        self.keep_best_state = keep_best_state
        self.best_state: dict | None = None
        self.train_losses: list[float] = []
        self.eval_results: list[dict] = []
        self.val_eval_results: list[dict] = []
        self.tail_eval_results: list[dict] = []
        self.weight_history: list[dict] = []
        # Per-eval-epoch test-item ranks (sufficient statistic for every
        # rank-based metric + per-user significance tests). Same test set at
        # every eval -> rectangular; persisted via save_ranks_npz().
        self._rank_meta: dict | None = None     # user_ids, n_rel (constant)
        self._rank_epochs: list[int] = []
        self._rank_rows: list = []              # one uint32 array per eval
        self.train_time_sec = 0.0
        self.eval_time_sec = 0.0

    def train(self) -> dict:
        """Run the full training loop.

        Returns:
            Dict with final results including best metrics and training history.
        """
        print(f"\nTraining {self.config.model.name} on {self.config.data.name}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.train.epochs}, "
              f"Early stop patience: {self.config.train.early_stop_patience}")
        print()

        # Snapshot initial weights (epoch 0, before any training)
        if hasattr(self.model, "get_attention_weights"):
            self.weight_history.append({
                "epoch": 0,
                **self.model.get_attention_weights(),
            })

        for epoch in range(1, self.config.train.epochs + 1):
            # FC->table snapshot hook (checked first; this changes the model's
            # init_mode to 'table' so the freeze hook below will no-op).
            if (
                self.fc_snapshot_to_table_at_epoch is not None
                and not self._fc_snapshot_done
                and epoch >= self.fc_snapshot_to_table_at_epoch
                and getattr(self.model, "init_mode", None) == "fc"
            ):
                self._snapshot_fc_to_table(epoch)

            # FC freeze hook: at the chosen epoch, freeze W_U/W_I/b_U/b_I and
            # rebuild the Adam optimizer over the remaining trainable params
            # so frozen tensors' moment buffers are not maintained.
            if (
                self.fc_freeze_after_epoch is not None
                and not self._fc_frozen
                and epoch >= self.fc_freeze_after_epoch
                and getattr(self.model, "init_mode", None) == "fc"
            ):
                self._freeze_fc_encoder(epoch)

            # Train one epoch
            _t_train = time.time()
            loss = self._train_epoch(epoch)
            self.train_time_sec += time.time() - _t_train
            self.train_losses.append(loss)

            # Snapshot attention weights (if model supports them)
            if hasattr(self.model, "get_attention_weights"):
                self.weight_history.append({
                    "epoch": epoch,
                    **self.model.get_attention_weights(),
                })

            # Evaluate periodically
            if epoch % self.config.train.eval_every == 0:
                _t_eval = time.time()
                metrics, rank_data = self.evaluator.evaluate(
                    self.model,
                    self.eval_dataset,
                    eval_batch_size=self.config.eval.eval_batch_size,
                    device=self.device,
                    return_ranks=True,
                )
                self.eval_time_sec += time.time() - _t_eval
                self.eval_results.append({"epoch": epoch, **metrics})
                if rank_data is not None:
                    if self._rank_meta is None:
                        self._rank_meta = {
                            "user_ids": rank_data["user_ids"],
                            "n_rel": rank_data["n_rel"],
                        }
                    self._rank_epochs.append(epoch)
                    self._rank_rows.append(rank_data["ranks"])

                primary_key = f"{self.config.eval.primary_metric}@{self.config.eval.primary_k}"
                if self.val_dataset is not None:
                    # Validation protocol: model selection reads the held-out
                    # val split, never the test set. Test metrics above are
                    # logged for post-hoc analysis only.
                    _t_val = time.time()
                    val_metrics = self.evaluator.evaluate(
                        self.model,
                        self.val_dataset,
                        eval_batch_size=self.config.eval.eval_batch_size,
                        device=self.device,
                    )
                    self.eval_time_sec += time.time() - _t_val
                    self.val_eval_results.append({"epoch": epoch, **val_metrics})
                    if getattr(self.config.train, "early_stop_monitor", "auto") == "test":
                        # Baseline-tuning mode: stopping follows the papers'
                        # test-monitored rule; the val stream is still logged
                        # so selection stays computable offline.
                        current = metrics[primary_key]
                    else:
                        current = val_metrics[primary_key]
                else:
                    current = metrics[primary_key]

                # Check improvement
                if current > self.best_metric:
                    self.best_metric = current
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    if self.save_enabled:
                        self._save_checkpoint(epoch)
                    if self.keep_best_state:
                        # CPU fp32 clone of the deployed weights (overwrites
                        # the previous best). Operators are plain attributes,
                        # not buffers, so they are NOT in state_dict — re-eval
                        # re-supplies them from the rebuilt complex.
                        self.best_state = {
                            k: v.detach().cpu().clone()
                            for k, v in self.model.state_dict().items()
                        }
                    marker = " *"
                else:
                    self.patience_counter += self.config.train.eval_every
                    marker = ""

                # Full metric row is stored in eval_results; print a compact
                # subset (the row now carries ~28 keys incl. hr/mrr/arhr/cos).
                _show = ("recall@5", "recall@20", "ndcg@5", "ndcg@20", "mrr@10")
                metrics_str = ", ".join(
                    f"{k}: {metrics[k]:.4f}" for k in _show if k in metrics)
                if self.val_dataset is not None:
                    metrics_str += f" | val_{primary_key}: {current:.4f}"
                print(f"  Eval [{epoch}]: {metrics_str}{marker}")

                # Tail-item evaluation (if tail_items provided)
                if self.tail_items is not None:
                    tail_metrics = self.evaluator.evaluate(
                        self.model,
                        self.eval_dataset,
                        eval_batch_size=self.config.eval.eval_batch_size,
                        device=self.device,
                        item_filter=self.tail_items,
                    )
                    self.tail_eval_results.append({"epoch": epoch, **tail_metrics})
                    tail_str = ", ".join(f"{k}: {v:.4f}" for k, v in tail_metrics.items())
                    print(f"  Tail [{epoch}]: {tail_str}")

                # Early stopping
                if self.patience_counter >= self.config.train.early_stop_patience:
                    print(f"\nEarly stopping at epoch {epoch}. "
                          f"Best {primary_key}: {self.best_metric:.4f} at epoch {self.best_epoch}")
                    break

        # Final results
        results = self._build_results()
        if self.save_enabled:
            self._save_results(results)
        return results

    def _snapshot_fc_to_table(self, epoch: int) -> None:
        """Convert the FC encoder into a free embedding table at this epoch.

        Calls model.snapshot_fc_to_table() to create the new tables, then
        rebuilds the optimizer over the model's new parameters (which now
        include the embedding tables and exclude the old W_U/W_I/b_U/b_I).
        """
        # Capture pre/post param counts for the log
        pre_n = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.model.snapshot_fc_to_table()
        # Rebuild the optimizer over the new parameter set.
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable, lr=self.config.train.lr)
        self._fc_snapshot_done = True
        post_n = sum(p.numel() for p in trainable)
        print(
            f"  [FC->table snapshot] epoch {epoch}: converted FC encoder into "
            f"learnable embedding tables. Trainable params {pre_n:,} -> {post_n:,} "
            f"(model.init_mode is now 'table')."
        )

    def _freeze_fc_encoder(self, epoch: int) -> None:
        """Freeze the FC encoder parameters (W_U, W_I, b_U, b_I) and rebuild
        the optimizer so its momentum/RMS buffers don't carry stale state for
        the frozen tensors. Applies HOUR's literal §4.2 framing: FC produces
        the *initial* embeddings, not a co-trained encoder.
        """
        n_frozen = 0
        for name in ("W_U", "W_I", "b_U", "b_I"):
            p = getattr(self.model, name, None)
            if isinstance(p, torch.nn.Parameter):
                p.requires_grad_(False)
                n_frozen += 1
        # Rebuild Adam over remaining trainable params (mixing/topology weights,
        # and any learnable cochain tables in full_multi mode).
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable, lr=self.config.train.lr)
        self._fc_frozen = True
        print(
            f"  [FC freeze] epoch {epoch}: froze {n_frozen} FC encoder tensors; "
            f"continuing to train {sum(p.numel() for p in trainable):,} remaining params."
        )

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        # F4: cache faces tensor on the device
        faces_dev = (self._faces_tensor.to(self.device)
                     if self._faces_tensor is not None else None)

        pbar = tqdm(self.sampler, desc=f"Epoch {epoch}", leave=False)
        for users, pos_items, neg_items in pbar:
            users = torch.as_tensor(users, device=self.device)
            pos_items = torch.as_tensor(pos_items, device=self.device)
            neg_items = torch.as_tensor(neg_items, device=self.device)

            user_e, pos_e, neg_e, reg_loss = self.model(users, pos_items, neg_items)
            loss = bpr_loss(user_e, pos_e, neg_e, reg_loss, self.config.train.reg_weight)

            # F4: topology auxiliary loss
            if self.topology_aux_weight > 0 and faces_dev is not None:
                ego = self.model.get_ego_embeddings()
                aux = topology_aux_loss(ego, faces_dev, self.dataset.n_users)
                loss = loss + self.topology_aux_weight * aux

            # Opt-in model self-supervised loss (e.g. SimGCL / HCCF contrastive).
            # The model computes it for the current batch inside forward() and
            # already applies its own weight λ, so we add it as-is. Models that
            # don't define it (MF/NGCF/LightGCN/LightCCN) are unaffected.
            aux_loss = getattr(self.model, "auxiliary_loss", None)
            if callable(aux_loss):
                ssl = aux_loss()
                if ssl is not None:
                    loss = loss + ssl

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # F3: observability hook (before optimizer step so .grad is available)
            self.observability_logger.on_batch(loss.item(), self.model)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch}: avg_loss = {avg_loss:.4f}")
        # F3: per-epoch hook
        self.observability_logger.on_epoch_end(self.model, epoch)
        return avg_loss

    def _save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint."""
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"{self.config.model.name}_{self.config.data.name}_best.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config.to_dict(),
        }, path)
        print(f"  Checkpoint saved: {path}")

    def _build_results(self) -> dict:
        """Build final results dict."""
        results = {
            "model": self.config.model.name,
            "dataset": self.config.data.name,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "config": self.config.to_dict(),
            "train_losses": self.train_losses,
            "eval_results": self.eval_results,
            "final_metrics": self.eval_results[-1] if self.eval_results else {},
        }

        # Validation protocol: record the val trajectory and make the
        # selection rule explicit. eval_results above is ALWAYS the test
        # trajectory; best_epoch/best_metric refer to the val metric when
        # early_stop_on == 'validation'.
        results["early_stop_on"] = (
            "test" if getattr(self.config.train, "early_stop_monitor", "auto") == "test"
            else ("validation" if self.val_eval_results else "test"))
        if self.val_eval_results:
            results["val_eval_results"] = self.val_eval_results
            best_test = next(
                (r for r in self.eval_results if r["epoch"] == self.best_epoch), None)
            if best_test is not None:
                results["test_at_val_best"] = best_test

        # Tail-item metrics
        if self.tail_eval_results:
            results["tail_eval_results"] = self.tail_eval_results
            results["tail_final_metrics"] = self.tail_eval_results[-1]

        # Include learned attention weights for models that support them
        if hasattr(self.model, "get_attention_weights"):
            results["attention_weights"] = self.model.get_attention_weights()
            results["weight_history"] = self.weight_history

        # F3: observability metrics
        if self.observability_logger.enabled:
            results["observability"] = self.observability_logger.to_dict()
        # F4: record whether topology aux loss was active
        results["topology_aux_weight"] = self.topology_aux_weight

        # Cost accounting for the efficiency discussion
        results["param_count"] = int(sum(p.numel() for p in self.model.parameters()))
        results["train_time_sec"] = round(self.train_time_sec, 2)
        results["eval_time_sec"] = round(self.eval_time_sec, 2)
        results["n_eval_points"] = len(self.eval_results)

        return results

    def save_ranks_npz(self, path: str) -> str | None:
        """Persist the per-eval-epoch test-item ranks (compressed).

        Layout: ``ranks`` is (n_eval_epochs, n_test_interactions) uint32,
        1-based, user-major within a row (users sorted ascending, each user's
        test items in their test_dict order — identical every epoch);
        ``epochs`` aligns rows; ``user_ids``/``n_rel`` recover the grouping.
        From this file every rank-based metric at any cutoff and any per-user
        significance test is recomputable offline.
        """
        if not self._rank_rows or self._rank_meta is None:
            return None
        ranks = np.stack(self._rank_rows).astype(np.uint32)
        np.savez_compressed(
            path,
            epochs=np.array(self._rank_epochs, dtype=np.int32),
            ranks=ranks,
            user_ids=self._rank_meta["user_ids"].astype(np.int64),
            n_rel=self._rank_meta["n_rel"].astype(np.int32),
        )
        print(f"  [ranks] saved {ranks.shape[0]} epochs x {ranks.shape[1]} "
              f"test-item ranks -> {path}")
        return str(path)

    def _save_results(self, results: dict) -> None:
        """Save results to JSON."""
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = results_dir / f"{self.config.model.name}_{self.config.data.name}_{timestamp}.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {path}")


class SGLTrainer(Trainer):
    """Extended trainer for SGL with contrastive loss."""

    def _train_epoch(self, epoch: int) -> float:
        """Train one epoch with BPR + InfoNCE contrastive loss."""
        from light_ccn.training.loss import infonce_loss

        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.sampler, desc=f"Epoch {epoch}", leave=False)
        for users, pos_items, neg_items in pbar:
            users_t = torch.as_tensor(users, device=self.device)
            pos_t = torch.as_tensor(pos_items, device=self.device)
            neg_t = torch.as_tensor(neg_items, device=self.device)

            # BPR forward
            user_e, pos_e, neg_e, reg_loss = self.model(users_t, pos_t, neg_t)
            loss_bpr = bpr_loss(user_e, pos_e, neg_e, reg_loss, self.config.train.reg_weight)

            # Contrastive forward (SGL model provides augmented views)
            user_z1, user_z2, item_z1, item_z2 = self.model.get_contrastive_views()
            ssl_user = infonce_loss(
                user_z1[users_t], user_z2[users_t],
                temperature=self.config.model.ssl_temp,
            )
            ssl_item = infonce_loss(
                item_z1[pos_t], item_z2[pos_t],
                temperature=self.config.model.ssl_temp,
            )
            loss_ssl = ssl_user + ssl_item

            loss = loss_bpr + self.config.model.ssl_weight * loss_ssl

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch}: avg_loss = {avg_loss:.4f}")
        return avg_loss
