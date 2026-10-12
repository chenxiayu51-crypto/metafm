"""Explicit model registry used by public entry points."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from .models import AEMGEncoder, BendrEncoder, CBraModEncoder, RingCrissCrossEncoder, SupervisedTCN

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "ring_criss_cross": RingCrissCrossEncoder,
    "supervised_tcn": SupervisedTCN,
    "bendr": BendrEncoder,
    "cbramod": CBraModEncoder,
    "aemg": AEMGEncoder,
}


def build_model(name: str, /, **kwargs: Any) -> nn.Module:
    """Build a registered backbone."""
    try:
        return MODEL_REGISTRY[name](**kwargs)
    except KeyError as error:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODEL_REGISTRY)}") from error


def model_from_config(config: Mapping[str, Any]) -> nn.Module:
    """Build a model from a Hydra-compatible mapping with a ``name`` field."""
    values = dict(config)
    try:
        name = str(values.pop("name"))
    except KeyError as error:
        raise ValueError("model config must include name") from error
    return build_model(name, **values)
