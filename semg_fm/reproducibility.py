"""Small, explicit run-provenance helpers used by public training entry points."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import torch


def manifest_digest(path: str | Path) -> str:
    """Hash the exact manifest consumed by a training run."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment_info() -> dict[str, Any]:
    """Snapshot interpreter, platform, and accelerator visibility."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
    }


def effective_global_batch_size(data_config: Mapping[str, Any], trainer_config: Mapping[str, Any]) -> int:
    """Calculate per-device batch × requested devices × gradient accumulation."""
    batch_size = int(data_config.get("batch_size", 8))
    devices = trainer_config.get("devices", 1)
    if devices == "auto":
        devices = max(1, torch.cuda.device_count())
    if isinstance(devices, bool) or not isinstance(devices, int) or devices < 1:
        raise ValueError("trainer.devices must be positive integer or 'auto'")
    accumulation = int(trainer_config.get("accumulate_grad_batches", 1))
    if batch_size < 1 or accumulation < 1:
        raise ValueError("batch_size and accumulate_grad_batches must be positive")
    return batch_size * devices * accumulation


def _git_commit() -> str | None:
    """Return the current commit when Git metadata is available, otherwise null."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def write_run_metadata(
    output_dir: str | Path,
    *,
    manifest_path: str | Path,
    data_config: Mapping[str, Any],
    trainer_config: Mapping[str, Any],
    command: str,
) -> Path:
    """Write an auditable JSON record of data, code, device, and batch context."""
    manifest = Path(manifest_path)
    if not manifest.is_file():
        raise ValueError(f"manifest does not exist: {manifest}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "command": str(command),
        "manifest": {"path": str(manifest.resolve()), "sha256": manifest_digest(manifest)},
        "git_commit": _git_commit(),
        "trainer": dict(trainer_config),
        "data": dict(data_config),
        "effective_global_batch_size": effective_global_batch_size(data_config, trainer_config),
        "environment": environment_info(),
    }
    path = output / "run_metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


__all__ = [
    "effective_global_batch_size",
    "environment_info",
    "manifest_digest",
    "write_run_metadata",
]
