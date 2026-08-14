"""Checkpoint migration functions for the NEP architecture.

Functions named ``model_update_v{n}_v{n + 1}`` and
``trainer_update_v{n}_v{n + 1}`` are looked up by
``NEP.upgrade_checkpoint`` and ``Trainer.upgrade_checkpoint``.
"""


def model_update_v1_v2(checkpoint: dict) -> None:
    """Add the ``mn_radial`` and ``mn_angular`` model hypers, which used to be
    hard-coded to GPUMD's ``nep.in`` defaults of 100 and 20."""
    hypers = checkpoint["model_data"]["model_hypers"]
    hypers.setdefault("mn_radial", 200)
    hypers.setdefault("mn_angular", 100)


def model_update_v2_v3(checkpoint: dict) -> None:
    """Add the ``loaded_nep`` model data flag, which records whether the
    potential was loaded from a ``nep.txt`` file (and therefore keeps its
    composition baselines at zero and its target scale at one).

    Checkpoints written before this flag existed lost that information when
    they were saved, so they are restored as regular models.
    """
    checkpoint["model_data"].setdefault("loaded_nep", False)


def model_update_v3_v4(checkpoint: dict) -> None:
    """Add the ``model_type`` model hyper introduced with the tensorial
    (dipole and polarizability) models.  Older checkpoints are all regular
    energy models."""
    checkpoint["model_data"]["model_hypers"].setdefault("model_type", 0)


def trainer_update_v1_v2(checkpoint: dict) -> None:
    """Add the ``finetune`` training hypers introduced with checkpoint-from-
    checkpoint fine-tuning."""
    checkpoint["train_hypers"].setdefault("finetune", {"read_from": None})
