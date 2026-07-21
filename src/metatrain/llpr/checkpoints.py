import torch


def model_update_v1_v2(checkpoint: dict) -> None:
    """
    Update a v1 checkpoint to v2.

    :param checkpoint: The checkpoint to update.
    """
    ensemble_sizes = {}
    for key in checkpoint["state_dict"].keys():
        if key.endswith("_ensemble_weights"):
            ensemble_sizes[key.replace("_ensemble_weights", "")] = checkpoint[
                "state_dict"
            ][key].shape[1]
    checkpoint["model_data"] = {}
    checkpoint["model_data"]["hypers"] = {
        "ensembles": {
            "num_members": ensemble_sizes,
            # correct means not needed to load the state_dict correctly
            "means": {name: [] for name in ensemble_sizes.keys()},
        },
    }
    checkpoint["model_data"]["dataset_info"] = checkpoint["wrapped_model_checkpoint"][
        "model_data"
    ]["dataset_info"]


def model_update_v2_v3(checkpoint: dict) -> None:
    """
    Update a v2 checkpoint to v3.

    :param checkpoint: The checkpoint to update.
    """
    # state_dict renamed to model_state_dict, best_model_state_dict added with the
    # new training procedure for the ensemble by backpropagation
    checkpoint["model_state_dict"] = checkpoint.pop("state_dict")
    checkpoint["best_model_state_dict"] = None
    # changed format for ensemble weights (only do energy for simplicity)
    t = checkpoint["model_state_dict"].pop("energy_ensemble_weights").T
    checkpoint["model_state_dict"]["llpr_ensemble_layers.energy.weight"] = t
    # added num_ensemble_members to hypers (only do energy for simplicity)
    num_members = t.shape[0]
    checkpoint["model_data"]["hypers"]["num_ensemble_members"] = {"energy": num_members}
    # trainer is v1
    checkpoint["trainer_ckpt_version"] = 1
    # we set the following to None and the user will probably get errors if they're
    # accessed from a restart exercise (which would be useless anyway as there was
    # no ensemble training by backpropagation before this version)
    checkpoint["epoch"] = None
    checkpoint["optimizer_state_dict"] = None
    checkpoint["scheduler_state_dict"] = None
    checkpoint["best_epoch"] = None
    checkpoint["best_metric"] = None
    checkpoint["best_optimizer_state_dict"] = None


def model_update_v3_v4(checkpoint: dict) -> None:
    """
    Update a v3 checkpoint to v4.

    :param checkpoint: The checkpoint to update.
    """
    # need to change all inv_covariance to cholesky buffers
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("inv_covariance_"):
            cholesky_key = key.replace("inv_covariance_", "cholesky_")
            covariance_key = key.replace("inv_covariance_", "covariance_")
            covariance = state_dict[covariance_key]
            # Try with an increasingly high regularization parameter until
            # the matrix is invertible
            is_not_pd = True
            regularizer = 1e-20
            while is_not_pd and regularizer < 1e16:
                try:
                    cholesky = torch.linalg.cholesky(
                        0.5 * (covariance + covariance.T)
                        + regularizer
                        * torch.eye(
                            covariance.shape[0],
                            device=covariance.device,
                            dtype=torch.float64,
                        )
                    ).to(covariance.dtype)
                    is_not_pd = False
                except RuntimeError:
                    regularizer *= 10.0
            if is_not_pd:
                raise RuntimeError(
                    "Could not compute Cholesky decomposition. Something went "
                    "wrong. Please contact the metatrain developers"
                )
            new_state_dict[cholesky_key] = cholesky
        else:
            new_state_dict[key] = value
    checkpoint["model_state_dict"] = new_state_dict


def model_update_v4_v5(checkpoint: dict) -> None:
    """
    Update a v4 checkpoint to v5.

    :param checkpoint: The checkpoint to update.
    """
    # The LLPR buffers and ensemble layers became per-block: the covariance and its
    # Cholesky factor are keyed by last-layer feature block, the multiplier and the
    # ensemble layers by target block. A v4 model could only wrap single-block
    # targets, so every old buffer maps onto that single block.
    from .model import _SINGLE_BLOCK, _block_key, _get_uncertainty_name

    # The keys are taken from the *wrapped model's* targets, which is the source
    # `set_wrapped_model` derives them from. The LLPR's own dataset info is not
    # enough: it may cover fewer targets than the model exposes outputs for (PET-MAD
    # is calibrated on `energy` alone, but the wrapped PET also predicts
    # `non_conservative_forces` and `non_conservative_stress`, and buffers were
    # registered for all three).
    dataset_info = checkpoint["wrapped_model_checkpoint"]["model_data"]["dataset_info"]

    block_keys = {
        target_name: _block_key(target_name, target_info.layout.keys.entry(0))
        for target_name, target_info in dataset_info.targets.items()
    }
    multiplier_block_keys = {
        _get_uncertainty_name(target_name): block_key
        for target_name, block_key in block_keys.items()
    }

    def rename(state_dict: dict) -> dict:
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("covariance_") or key.startswith("cholesky_"):
                # the covariance is a property of the last-layer features, which are
                # a single invariant block for any model a v4 LLPR could wrap
                new_state_dict[f"{key}_{_SINGLE_BLOCK}"] = value
            elif key.startswith("multiplier_"):
                uncertainty_name = key[len("multiplier_") :]
                if uncertainty_name not in multiplier_block_keys:
                    raise RuntimeError(
                        f"Unable to upgrade the checkpoint: no target in the "
                        f"wrapped model's dataset info corresponds to the buffer "
                        f"'{key}'."
                    )
                new_key = f"{key}_{multiplier_block_keys[uncertainty_name]}"
                new_state_dict[new_key] = value
            elif key.startswith("llpr_ensemble_layers."):
                target_name, parameter = key[len("llpr_ensemble_layers.") :].rsplit(
                    ".", 1
                )
                if target_name not in block_keys:
                    raise RuntimeError(
                        f"Unable to upgrade the checkpoint: no target in the "
                        f"wrapped model's dataset info corresponds to the ensemble "
                        f"layer '{key}'."
                    )
                new_key = (
                    f"llpr_ensemble_layers.{target_name}::"
                    f"{block_keys[target_name]}.{parameter}"
                )
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        return new_state_dict

    for state_dict_name in ("model_state_dict", "best_model_state_dict"):
        state_dict = checkpoint.get(state_dict_name)
        if state_dict is not None:
            checkpoint[state_dict_name] = rename(state_dict)


def trainer_update_v1_v2(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 1 to version 2.

    :param checkpoint: The checkpoint to update.
    """
    # Added distributed training hyperparameters
    if "train_hypers" in checkpoint:
        checkpoint["train_hypers"]["distributed"] = False
        checkpoint["train_hypers"]["distributed_port"] = 39591


def trainer_update_v2_v3(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 2 to version 3.

    :param checkpoint: The checkpoint to update.
    """
    if "train_hypers" in checkpoint:
        checkpoint["train_hypers"]["batch_atom_bounds"] = [None, None]


def trainer_update_v3_v4(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 3 to version 4.

    :param checkpoint: The checkpoint to update.
    """
    if "train_hypers" in checkpoint:
        checkpoint["train_hypers"]["calibrate_with_absolute_residuals"] = False


def trainer_update_v4_v5(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 4 to version 5.

    :param checkpoint: The checkpoint to update.
    """
    # added calibration method to pick the alpha prefactor
    if "train_hypers" in checkpoint:
        if checkpoint["train_hypers"].get("calibrate_with_absolute_residuals", False):
            checkpoint["train_hypers"]["calibration_method"] = "absolute_residuals"
        else:
            checkpoint["train_hypers"]["calibration_method"] = "squared_residuals"
