"""Contracts for run-level reproducibility metadata."""

from __future__ import annotations

import json

from semg_fm.reproducibility import effective_global_batch_size, write_run_metadata


def test_effective_global_batch_size_accounts_for_devices_and_accumulation():
    assert effective_global_batch_size({"batch_size": 8}, {"devices": 4, "accumulate_grad_batches": 2}) == 64
    assert effective_global_batch_size({"batch_size": 8}, {"devices": 1}) == 8


def test_run_metadata_records_manifest_hash_and_execution_context(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("dataset_id,path\npose,a.h5\n", encoding="utf-8")

    path = write_run_metadata(
        tmp_path / "output",
        manifest_path=manifest,
        data_config={"batch_size": 2},
        trainer_config={"devices": 2, "precision": "bf16-mixed", "accumulate_grad_batches": 4},
        command="pretrain",
    )
    metadata = json.loads(path.read_text(encoding="utf-8"))

    assert metadata["command"] == "pretrain"
    assert metadata["manifest"]["sha256"]
    assert metadata["effective_global_batch_size"] == 16
    assert metadata["trainer"]["precision"] == "bf16-mixed"
    assert "torch" in metadata["environment"]
