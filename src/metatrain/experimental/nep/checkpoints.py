"""Checkpoint migration functions for the NEP architecture.

Functions named ``model_update_v{n}_v{n + 1}`` and
``trainer_update_v{n}_v{n + 1}`` are looked up by
``NEP.upgrade_checkpoint`` and ``Trainer.upgrade_checkpoint``.
"""

from typing import Any, Dict


def model_update_v1_v2(checkpoint: Dict[str, Any]) -> None:
    """Add the ``charge_mode`` model hyperparameter (v2, qNEP support).

    :param checkpoint: The checkpoint to update in place.
    """
    checkpoint["model_data"]["model_hypers"].setdefault("charge_mode", 0)
