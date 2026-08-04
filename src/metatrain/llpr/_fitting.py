"""Covariance accumulation, Cholesky decomposition and calibration of an LLPR
model. Called from Python (the LLPR trainer and tests), never from the
TorchScript forward path."""

import logging
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch
from metatomic.torch import ModelOutput
from torch.utils.data import DataLoader

from metatrain.utils.data import (
    CollateFn,
    CombinedDataLoader,
    Dataset,
    unpack_batch,
)
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists_transform,
)

from .calibration import (
    GaussianCRPSCalibrator,
    RatioCalibrator,
)
from .model import LLPRUncertaintyModel, get_uncertainty_name


def _get_dataloader(
    model: LLPRUncertaintyModel,
    datasets: List[Union[Dataset, torch.utils.data.Subset]],
    batch_size: int,
    is_distributed: bool,
) -> DataLoader:
    """
    Create a DataLoader for the provided datasets. As the dataloader is only used to
    accumulate the quantities needed for LLPR calibration, there is no need to
    shuffle or drop the last non-full batch. Distributed sampling can be used or
    not, based on the `is_distributed` argument, and training with double
    precision is enforced.

    :param model: the LLPR model the dataloader is built for.
    :param datasets: List of datasets to create the dataloader from.
    :param batch_size: Batch size to use for the dataloader.
    :param is_distributed: Whether to use distributed sampling or not.
    :return: The created DataLoader.
    """
    # Create the collate function
    targets_keys = list(model.dataset_info.targets.keys())
    requested_neighbor_lists = get_requested_neighbor_lists(model)
    collate_fn = CollateFn(
        target_keys=targets_keys,
        callables=[get_system_with_neighbor_lists_transform(requested_neighbor_lists)],
    )

    # Validate dtype from datasets
    if len(datasets) == 0:
        raise ValueError(
            "Cannot create dataloader from empty datasets list. "
            "Please provide non-empty datasets for LLPR calibration."
        )
    if len(datasets[0]) == 0:
        raise ValueError(
            "Cannot create dataloader from empty dataset. "
            "Please provide non-empty datasets for LLPR calibration."
        )

    # Build the dataloaders
    samplers: List[torch.utils.data.Sampler | None]
    if is_distributed:
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        samplers = [
            NoPadDistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                seed=0,
            )
            for dataset in datasets
        ]
    else:
        samplers = [None] * len(datasets)

    dataloaders = []
    for dataset, sampler in zip(datasets, samplers, strict=True):
        if len(dataset) < batch_size:
            raise ValueError(
                f"A dataset has fewer samples "
                f"({len(dataset)}) than the batch size "
                f"({batch_size}). "
                "Please reduce the batch size."
            )
        dataloaders.append(
            DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                sampler=sampler,
                drop_last=False,
                collate_fn=collate_fn,
            )
        )

    # important to keep shuffle=False for consistent calibration results
    # in distributed training
    return CombinedDataLoader(dataloaders, shuffle=False)


def compute_covariance(
    model: LLPRUncertaintyModel,
    datasets: List[Union[Dataset, torch.utils.data.Subset]],
    batch_size: int,
    is_distributed: bool,
) -> None:
    """A function to compute the covariance matrix for a training set.

    The covariance is stored as a buffer in the model.

    :param model: the LLPR model whose covariance buffers are accumulated.
    :param datasets: List of datasets to use for covariance calculation.
    :param batch_size: Batch size to use for the dataloader.
    :param is_distributed: Whether to use distributed sampling or not.
    """
    # Create dataloader for the training datasets
    train_loader = _get_dataloader(
        model, datasets, batch_size, is_distributed=is_distributed
    )

    device = next(iter(model.buffers())).device
    dtype = next(iter(model.buffers())).dtype
    with torch.no_grad():
        for batch in train_loader:
            systems, targets, _ = unpack_batch(batch)
            n_atoms = torch.tensor(
                [len(system.positions) for system in systems], device=device
            )
            systems = [system.to(device=device, dtype=dtype) for system in systems]
            outputs_for_targets = {
                name: ModelOutput(
                    sample_kind="atom"
                    if "atom" in target.block(0).samples.names
                    else "system"
                )
                for name, target in targets.items()
            }
            outputs_for_features = {
                f"mtt::aux::{n.replace('mtt::', '')}_last_layer_features": o
                for n, o in outputs_for_targets.items()
            }
            output = model.forward(
                systems, {**outputs_for_targets, **outputs_for_features}
            )
            for name in targets.keys():
                if name not in model.feature_keys:
                    # no last-layer feature declaration for this target
                    continue
                ll_feat_tmap = output[
                    f"mtt::aux::{name.replace('mtt::', '')}_last_layer_features"
                ]
                uncertainty_name = get_uncertainty_name(name)
                feature_keys = model.feature_keys[name]

                # one covariance per feature block, accumulated from that
                # block of the feature output
                for feat_index in range(len(feature_keys)):
                    block_values = ll_feat_tmap.block(feat_index).values.detach()
                    # TODO: interface ll_feat calculation with the loss function,
                    # paying attention to normalization w.r.t. n_atoms
                    if outputs_for_targets[name].sample_kind == "system":
                        norm_shape = [n_atoms.shape[0]] + [1] * (
                            len(block_values.shape) - 1
                        )
                        block_values = block_values / n_atoms.reshape(norm_shape)

                    # Flatten components into samples: component-free features
                    # have no component axis (no-op), while equivariant features
                    # share one weight vector across components, whose
                    # Gauss-Newton Hessian is `sum_samples sum_components f f^T`
                    # -- exactly what flattening accumulates.
                    ll_feats = block_values.reshape(-1, block_values.shape[-1])

                    covariance = model._get_covariance(
                        uncertainty_name, feature_keys[feat_index]
                    )
                    covariance += ll_feats.T @ ll_feats

    if is_distributed:
        torch.distributed.barrier()
        # All-reduce the covariance matrices across all processes
        for name in model.outputs_list:
            uncertainty_name = get_uncertainty_name(name)
            for block_key in model.feature_keys[name]:
                covariance = model._get_covariance(uncertainty_name, block_key)
                torch.distributed.all_reduce(covariance)


def compute_cholesky_decomposition(
    model: LLPRUncertaintyModel, regularizer: Optional[float] = None
) -> None:
    """A function to compute the Cholesky decomposition of the covariance matrix.

    The Cholesky decomposition is stored as a buffer in the model.

    :param model: the LLPR model whose Cholesky buffers are filled.
    :param regularizer: A regularization parameter to ensure the matrix is
        positive-definite. If not provided, the function will try to compute the
        Cholesky decomposition without regularization and increase the
        regularization parameter until the matrix is positive-definite.
    """

    for name in model.outputs_list:
        uncertainty_name = get_uncertainty_name(name)
        for block_key in model.feature_keys[name]:
            covariance = model._get_covariance(uncertainty_name, block_key).to(
                dtype=torch.float64
            )
            cholesky = model._get_cholesky(uncertainty_name, block_key)
            if regularizer is not None:
                cholesky[:] = torch.linalg.cholesky(
                    0.5 * (covariance + covariance.T)
                    + regularizer
                    * torch.eye(
                        covariance.shape[0],
                        device=covariance.device,
                        dtype=torch.float64,
                    )
                ).to(cholesky.dtype)
            else:
                # Try with an increasingly high regularization parameter until
                # the matrix is invertible
                is_not_pd = True
                r = 1e-20
                while is_not_pd and r < 1e16:
                    try:
                        cholesky[:] = torch.linalg.cholesky(
                            0.5 * (covariance + covariance.T)
                            + r
                            * torch.eye(
                                covariance.shape[0],
                                device=covariance.device,
                                dtype=torch.float64,
                            )
                        ).to(cholesky.dtype)
                        is_not_pd = False
                    except RuntimeError:
                        r *= 10.0
                if is_not_pd:
                    raise RuntimeError(
                        "Could not compute Cholesky decomposition. Something "
                        "went wrong. Please contact the metatrain developers"
                    )
                else:
                    logging.info(
                        f"Used regularization parameter of {r:.1e} to "
                        f"compute the Cholesky decomposition for `{name}`"
                    )


def calibrate(
    model: LLPRUncertaintyModel,
    datasets: List[Union[Dataset, torch.utils.data.Subset]],
    batch_size: int,
    is_distributed: bool,
    calibration_method: str,
) -> None:
    """
    Calibrate the LLPR model.

    This function computes the calibration constants (one for each output)
    that are used to scale the uncertainties in the LLPR model. The
    calibration is performed in a simple way by computing either the calibration
    constant as the mean of the squared residuals divided by the mean of
    the non-calibrated uncertainties (i.e., by minimizing the NLL as a function of
    the calibration constant), or by minimizing the CRPS as a function of the
    calibration constant.

    :param model: the LLPR model whose multipliers are calibrated.
    :param datasets: List of datasets to use for calibration.
    :param batch_size: Batch size to use for the dataloader.
    :param is_distributed: Whether to use distributed sampling or not.
    :param calibration_method: The method to use for calibration. Supported methods
        are "squared_residuals", "absolute_residuals", and "crps". All methods
        assume Gaussian errors. The "squared_residuals" method minimize the negative
        log-likelihood (NLL) as a function of the calibration constant.
        The "absolute_residuals" method estimates the calibration constant based on
        the mean absolute residuals, which can help reduce the effect of large
        outliers.
        The "crps" method minimizes the Continuous Ranked Probability Score (CRPS)
        as a function of the calibration constant.
    """
    valid_loader = _get_dataloader(
        model, datasets, batch_size, is_distributed=is_distributed
    )

    device = next(iter(model.buffers())).device
    dtype = next(iter(model.buffers())).dtype

    calibrator: Union[RatioCalibrator, GaussianCRPSCalibrator]

    if calibration_method in ["squared_residuals", "absolute_residuals"]:
        calibrator = RatioCalibrator(method=calibration_method)  # type: ignore[arg-type]
    elif calibration_method == "crps":
        calibrator = GaussianCRPSCalibrator()
    else:
        raise ValueError(
            f"Unknown calibration method '{calibration_method}'! "
            "Supported methods are 'squared_residuals', 'absolute_residuals', and"
            " 'crps'."
        )

    # calibrator key -> its (uncertainty, block) pair; both parts may contain
    # "::", so they cannot be recovered by splitting the key
    calibrated_blocks: Dict[str, Tuple[str, str]] = {}

    with torch.no_grad():
        for batch in valid_loader:
            systems, targets, _ = unpack_batch(batch)
            systems = [system.to(device=device, dtype=dtype) for system in systems]
            targets = {
                name: target.to(device=device, dtype=dtype)
                for name, target in targets.items()
            }

            requested_outputs = {}
            for name in targets:
                if name not in model.feature_keys:
                    # no last-layer feature declaration for this target
                    continue
                sample_kind = (
                    "atom"
                    if "atom" in targets[name].block(0).samples.names
                    else "system"
                )
                requested_outputs[name] = ModelOutput(sample_kind=sample_kind)
                uncertainty_name = get_uncertainty_name(name)
                requested_outputs[uncertainty_name] = ModelOutput(
                    sample_kind=sample_kind
                )

            outputs = model.forward(systems, requested_outputs)

            for name, target in targets.items():
                if name not in model.feature_keys:
                    continue
                uncertainty_name = get_uncertainty_name(name)
                block_keys = model.target_block_keys[name]
                prediction = outputs[name]
                uncertainty = outputs[uncertainty_name]

                # one multiplier per target block, fitted against that
                # block's residuals
                for block_index in range(len(prediction.keys)):
                    key = prediction.keys.entry(block_index)
                    pred = prediction.block(block_index).values.detach()
                    targ = target.block(key).values
                    unc = uncertainty.block(block_index).values.detach()

                    residuals = pred - targ

                    block_key = block_keys[block_index]
                    calibrator_key = f"{uncertainty_name}/{block_key}"
                    calibrated_blocks[calibrator_key] = (
                        uncertainty_name,
                        block_key,
                    )
                    calibrator.update(
                        uncertainty_name=calibrator_key,
                        residuals=residuals.reshape(-1, residuals.shape[-1]),
                        uncertainties=unc.reshape(-1, unc.shape[-1]),
                    )

    multipliers = calibrator.finalize()

    for calibrator_key, alpha in multipliers.items():
        uncertainty_name, block_key = calibrated_blocks[calibrator_key]
        multiplier = model._get_multiplier(uncertainty_name, block_key)
        multiplier[:] = alpha.to(device=device, dtype=multiplier.dtype)


class NoPadDistributedSampler(torch.utils.data.Sampler[int]):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_replicas: int,
        rank: int,
        shuffle: bool = False,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        n = len(self.dataset)
        indices = torch.arange(n, dtype=torch.long)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = indices[torch.randperm(n, generator=g)]
        # no padding, no dropping
        return iter(indices[self.rank :: self.num_replicas].tolist())

    def __len__(self) -> int:
        n = len(self.dataset)
        return (n - self.rank + self.num_replicas - 1) // self.num_replicas
