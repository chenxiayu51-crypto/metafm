"""Validation helpers for training configuration."""

from collections.abc import Mapping


def validate_devices(devices: int | str) -> None:
    """Validate a Lightning device count, allowing automatic selection."""
    if devices == "auto":
        return
    if isinstance(devices, bool) or not isinstance(devices, int) or not 1 <= devices <= 8:
        raise ValueError("devices must be 'auto' or an integer from 1 through 8")


def validate_seed(seed: int) -> None:
    """Require an explicit non-negative integral random seed."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")


def build_trainer_options(trainer_config: Mapping) -> dict:
    """Turn a resolved trainer config into ``Trainer(**options)`` kwargs.

    Hydra cannot construct ``ModelCheckpoint`` objects from plain dicts, so a
    trainer config may declare a ``callbacks`` mapping of named ``_target_``
    specs that this helper instantiates before the ``Trainer`` is built.
    """
    from hydra.utils import instantiate

    options = {key: value for key, value in trainer_config.items() if key != "callbacks"}
    callback_configs = trainer_config.get("callbacks")
    if callback_configs:
        if not isinstance(callback_configs, Mapping):
            raise ValueError("trainer callbacks must be a mapping of named _target_ specs")
        options["callbacks"] = [instantiate(spec) for spec in callback_configs.values()]
    return options
