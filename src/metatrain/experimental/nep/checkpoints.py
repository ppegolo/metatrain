"""Checkpoint migration functions for the NEP architecture.

Functions named ``model_update_v{n}_v{n + 1}`` and
``trainer_update_v{n}_v{n + 1}`` are looked up by
``NEP.upgrade_checkpoint`` and ``Trainer.upgrade_checkpoint``.
"""


def trainer_update_v1_v2(checkpoint: dict) -> None:
    """Add the ``finetune`` training hypers introduced with checkpoint-from-
    checkpoint fine-tuning."""
    checkpoint["train_hypers"].setdefault("finetune", {"read_from": None})
