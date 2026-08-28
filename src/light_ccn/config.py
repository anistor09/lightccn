"""Experiment configuration via dataclasses + YAML loading."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    name: str = "gowalla"
    data_dir: str = "data"
    batch_size: int = 2048
    num_workers: int = 4


@dataclass
class ComplexConfig:
    tau: int = 20
    gamma: float = 0.1
    cache_dir: str = "data/complex_cache"


@dataclass
class ModelConfig:
    name: str = "lightgcn"
    embed_dim: int = 64
    n_layers: int = 3
    dropout: float = 0.0
    # NGCF-specific
    node_dropout: float = 0.1
    message_dropout: float = 0.1
    # SGL-specific
    ssl_weight: float = 0.1
    ssl_temp: float = 0.2
    augment_type: str = "edge_dropout"
    augment_ratio: float = 0.1
    # LightCCN-Multi specific
    edge_embed_dim: int = 64
    face_embed_dim: int = 64


@dataclass
class TrainConfig:
    epochs: int = 1000
    lr: float = 0.001
    reg_weight: float = 1e-4
    early_stop_patience: int = 50
    early_stop_monitor: str = "auto"  # 'auto' (val if present) | 'test'
    eval_every: int = 10
    seed: int = 2020
    device: str = "cuda"


@dataclass
class EvalConfig:
    topk: list[int] = field(default_factory=lambda: [3, 5, 10, 20, 50])
    eval_batch_size: int = 1024
    primary_metric: str = "recall"
    primary_k: int = 20


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    complex: ComplexConfig = field(default_factory=ComplexConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"

    def to_dict(self) -> dict:
        return asdict(self)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _dict_to_config(d: dict) -> ExperimentConfig:
    """Convert a nested dict to ExperimentConfig."""
    return ExperimentConfig(
        data=DataConfig(**d.get("data", {})),
        complex=ComplexConfig(**d.get("complex", {})),
        model=ModelConfig(**d.get("model", {})),
        train=TrainConfig(**d.get("train", {})),
        eval=EvalConfig(**d.get("eval", {})),
        checkpoint_dir=d.get("checkpoint_dir", "checkpoints"),
        results_dir=d.get("results_dir", "results"),
    )


def load_config(config_path: str, default_path: str | None = None) -> ExperimentConfig:
    """Load config from YAML, optionally merging with a default config.

    Args:
        config_path: Path to the experiment-specific YAML config.
        default_path: Path to the default YAML config. If None, looks for
            configs/default.yaml relative to the config_path's parent.
    """
    config_path = Path(config_path)

    if default_path is None:
        candidate = config_path.parent / "default.yaml"
        if candidate.exists():
            default_path = str(candidate)

    base: dict[str, Any] = {}
    if default_path and os.path.exists(default_path):
        with open(default_path) as f:
            base = yaml.safe_load(f) or {}

    with open(config_path) as f:
        override = yaml.safe_load(f) or {}

    merged = _deep_merge(base, override)
    return _dict_to_config(merged)
