"""Model registry and factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from light_ccn.models.base import BaseCFModel

MODEL_REGISTRY: dict[str, type] = {}


def register_model(name: str):
    """Decorator to register a model class."""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def build_model(name: str, **kwargs) -> "BaseCFModel":
    """Build a model by name from the registry."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)
