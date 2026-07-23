import copy
import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union, cast

import torch
from metatensor.torch import TensorMap
from metatomic.torch import System
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DistributedSampler

from metatrain.composition import train_or_load_composition_model
from metatrain.utils.abc import ModelInterface, TrainerInterface
from metatrain.utils.additive import get_remove_additive_transform
from metatrain.utils.atomic_basis import get_trainer_hooks
from metatrain.utils.atomic_basis.helpers import (
    get_prepare_atomic_basis_targets_transform,
)
from metatrain.utils.augmentation import O3Augmenter
from metatrain.utils.data import (
    CollateFn,
    CombinedDataLoader,
    Dataset,
    build_train_dataloaders,
    build_val_dataloaders,
    get_num_workers,
    unpack_batch,
    validate_num_workers,
)
from metatrain.utils.data.dataset import RawExtraPayload
from metatrain.utils.distributed.distributed_data_parallel import (
    DistributedDataParallel,
)
from metatrain.utils.distributed.slurm import initialize_slurm_nccl_process_group
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.io import check_file_extension
from metatrain.utils.logging import ROOT_LOGGER, MetricLogger
from metatrain.utils.loss import LossAggregator, LossSpecification
from metatrain.utils.metrics import MAEAccumulator, RMSEAccumulator, get_selected_metric
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists_transform,
)
from metatrain.utils.per_atom import average_by_num_atoms
from metatrain.utils.scaler import get_remove_scale_transform
from metatrain.utils.system_data import get_system_data_transform
from metatrain.utils.transfer import batch_to

from . import checkpoints
from .documentation import TrainerHypers
from .model import PET
from .modules.finetuning import apply_finetuning_strategy


def _unpack_batch_to(
    batch: Any, dtype: torch.dtype, device: torch.device
) -> tuple[List[System], Dict[str, TensorMap], Dict[str, Any]]:
    """Unpack a batch and move it to ``dtype``/``device``.

    Raw payloads (:class:`~metatrain.utils.data.dataset.RawExtraPayload`, e.g.
    ragged metric matrices) are popped out before :func:`batch_to`, which is
    TorchScript-typed ``Dict[str, TensorMap]`` and cannot accept them, then
    moved and reattached. No-op for batches without such entries.

    :param batch: Collated batch as produced by the dataloader.
    :param dtype: Target dtype.
    :param device: Target device.
    :return: ``(systems, targets, extra_data)`` on the requested dtype/device.
    """
    systems, targets, extra_data = unpack_batch(batch)
    ragged = {
        key: extra_data.pop(key)
        for key in list(extra_data.keys())
        if isinstance(extra_data[key], RawExtraPayload)
    }
    systems, targets, extra_data = batch_to(
        systems, targets, extra_data, dtype=dtype, device=device
    )
    for key, value in ragged.items():
        extra_data[key] = value.to(dtype=dtype, device=device)
    return systems, targets, extra_data


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    train_hypers: TrainerHypers,
    steps_per_epoch: int,
) -> LambdaLR:
    """
    Get a CosineAnnealing learning-rate scheduler with warmup

    :param optimizer: The optimizer for which to create the scheduler.
    :param train_hypers: The training hyperparameters.
    :param steps_per_epoch: The number of steps per epoch.
    :return: The learning rate scheduler.
    """
    total_steps = train_hypers["num_epochs"] * steps_per_epoch
    warmup_steps = int(train_hypers["warmup_fraction"] * total_steps)
    min_lr_ratio = 0.0  # hardcoded for now, could be made configurable in the future

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            progress = (current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler


def _clone_state_dict_to_cpu(state: Any) -> Any:
    """Recursively clone a (possibly nested) state dict, moving tensors to CPU.

    Keeps the "best" checkpoint off the training device instead of holding a second
    copy of every model/optimizer tensor on GPU (as ``copy.deepcopy`` would).

    :param state: State dict (or nested container/tensor) to clone.
    :return: Clone of ``state`` with all tensors on CPU.
    """
    if isinstance(state, torch.Tensor):
        return state.detach().cpu().clone()
    if isinstance(state, dict):
        return {k: _clone_state_dict_to_cpu(v) for k, v in state.items()}
    if isinstance(state, list):
        return [_clone_state_dict_to_cpu(v) for v in state]
    return copy.deepcopy(state)


class Trainer(TrainerInterface[TrainerHypers]):
    __checkpoint_version__ = 14

    def __init__(self, hypers: TrainerHypers) -> None:
        super().__init__(hypers)

        self.optimizer_state_dict: Optional[Dict[str, Any]] = None
        self.scheduler_state_dict: Optional[Dict[str, Any]] = None
        self.epoch: Optional[int] = None
        self.best_epoch: Optional[int] = None
        self.best_metric: Optional[float] = None
        self.best_model_state_dict: Optional[Dict[str, Any]] = None
        self.best_optimizer_state_dict: Optional[Dict[str, Any]] = None

    def train(
        self,
        model: PET,
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ) -> None:
        assert dtype in PET.__supported_dtypes__

        is_distributed = self.hypers["distributed"]
        is_finetune = self.hypers["finetune"]["read_from"] is not None

        if is_distributed:
            if len(devices) > 1:
                raise ValueError(
                    "Requested distributed training with the `multi-gpu` device. "
                    " If you want to run distributed training with PET, please "
                    "set `device` to cuda."
                )
            # the calculation of the device number works both when GPUs on different
            # processes are not visible to each other and when they are
            device, world_size, rank = initialize_slurm_nccl_process_group(
                self.hypers["distributed_port"]
            )
        else:
            rank = 0
            world_size = 1
            device = devices[0]
            # only one device, as we don't support non-distributed multi-gpu for now

        if is_distributed:
            logging.info(f"Training on {world_size} devices with dtype {dtype}")
        else:
            logging.info(f"Training on device {device} with dtype {dtype}")

        if self.hypers.get("compile", False):
            # Opt-in torch.compile of the backbone's feature calculation: the
            # collate/loss plumbing stays eager, but the launch-bound GNN math
            # gets fused. dynamic=True avoids a recompile per batch shape.
            # runs before the DDP wrap: ``model`` is still the raw PET here
            model.backend.calculate_features = torch.compile(  # type: ignore[method-assign]
                model.backend.calculate_features, dynamic=True
            )
            logging.info("torch.compile enabled for the PET backbone")

        # Apply fine-tuning strategy if provided
        if is_finetune:
            assert self.hypers["finetune"]["read_from"] is not None  # for mypy
            # ``inherit_heads`` is a one-time weight-copy initialization step that
            # must only run when finetuning first starts (a fresh ``Trainer``), not
            # again when a restart resumes an already-started finetuning run (a
            # restarted ``Trainer`` has its ``optimizer_state_dict`` restored by
            # ``load_checkpoint``); otherwise it would clobber the head weights
            # trained so far with a fresh copy from the (possibly since-changed, or
            # already stale-pruned) source target.
            is_fresh_finetune_start = self.optimizer_state_dict is None
            model = apply_finetuning_strategy(
                model,
                self.hypers["finetune"],
                apply_inherit_heads=is_fresh_finetune_start,
            )
            method = self.hypers["finetune"]["method"]
            num_params = sum(p.numel() for p in model.parameters())
            num_trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

            logging.info(f"Applied finetuning strategy: {method}")
            logging.info(
                f"Number of trainable parameters: {num_trainable_params} "
                f"[{num_trainable_params / num_params:.2%} %]"
            )
            inherit_heads = self.hypers["finetune"]["inherit_heads"]
            if inherit_heads and is_fresh_finetune_start:
                logging.info(
                    "Inheriting initial weights for heads and last layers for targets: "
                    f"from {list(inherit_heads.values())} to "
                    f"{list(inherit_heads.keys())}"
                )

        # Move the model to the device and dtype:
        model.to(device=device, dtype=dtype)
        # The additive models of PET are always in float64 (to avoid numerical errors in
        # the composition weights, which can be very large).
        for additive_model in model.additive_models:
            additive_model.to(dtype=torch.float64)
        model.scaler.to(dtype=torch.float64)

        # Set up transformations
        dataset_info = model.dataset_info
        train_targets = dataset_info.targets
        extra_data_info = dataset_info.extra_data
        rotational_augmenter = O3Augmenter(
            target_info_dict=train_targets, extra_data_info_dict=extra_data_info
        )
        requested_neighbor_lists = get_requested_neighbor_lists(model)
        max_atoms = self.hypers["max_atoms_per_batch"]
        atomic_basis_transform, atomic_basis_reverse_transform = (
            get_prepare_atomic_basis_targets_transform(train_targets, extra_data_info)
        )
        loss_hypers = cast(Dict[str, LossSpecification], self.hypers["loss"])

        # Determined here (rather than down by the main data loaders) because the
        # composition/scaler fitting below also reads through the training data and
        # would otherwise default to synchronous, single-process loading, which is
        # very slow on large (e.g. disk-based) datasets.
        if self.hypers["num_workers"] is None:
            num_workers = get_num_workers()
            logging.info(
                "Number of workers for data-loading not provided and chosen "
                f"automatically. Using {num_workers} workers."
            )
        else:
            num_workers = self.hypers["num_workers"]
            validate_num_workers(num_workers)

        # On CUDA (especially GH200 unified memory), forking after CUDA init causes
        # workers to inherit GPU memory mappings, inflating per-worker RSS and
        # OOM. Use 'spawn' to start workers as fresh processes instead.
        mp_context = "spawn" if num_workers > 0 and device.type == "cuda" else None

        if mp_context == "spawn":
            # Some container setups (e.g. CSCS daint ml4es) restrict /dev/shm so
            # that ftruncate() fails with EINVAL.  PyTorch's 'file_descriptor' sharing
            # strategy creates POSIX shared-memory files in /dev/shm; switching to
            # 'file_system' uses regular files in /tmp instead, which always works.
            import torch.multiprocessing as _torch_mp

            _torch_mp.set_sharing_strategy("file_system")

        # The composition/scaler fitting dataloaders are short-lived (created once,
        # used for a single pass, then torn down) rather than persistent like the main
        # training loop's. On at least one observed container setup, spawning 8 workers
        # for such a short-lived pool made a worker abort during shutdown and left the
        # process's multiprocessing state unable to start a subsequent worker pool
        # (i.e. the following composition/scaler/main-loop pool would hang forever).
        # Capping at 4 avoided this in testing; the persistent main-loop pool below is
        # unaffected and keeps using the full `num_workers`.
        fitting_num_workers = min(num_workers, 4)
        fitting_mp_context = mp_context if fitting_num_workers > 0 else None

        train_or_load_composition_model(
            composition_model=model.additive_models[0],
            atomic_baseline=self.hypers["atomic_baseline"],
            train_datasets=train_datasets,
            other_additive_models=list(model.additive_models[1:]),
            batch_size=self.hypers["batch_size"],
            is_distributed=is_distributed,
            checkpoint_dir=checkpoint_dir,
            num_workers=fitting_num_workers,
            multiprocessing_context=fitting_mp_context,
        )

        if self.hypers["scale_targets"]:
            logging.info("Calculating scaling weights")
            model.scaler.train_model(
                train_datasets,
                model.additive_models,
                self.hypers["batch_size"],
                is_distributed,
                self.hypers["fixed_scaling_weights"],
                initial_transforms=[atomic_basis_transform],
                per_structure_targets=self.hypers["per_structure_targets"],
                num_workers=fitting_num_workers,
                multiprocessing_context=fitting_mp_context,
            )

        logging.info("Setting up data loaders")

        if is_distributed:
            train_samplers = [
                DistributedSampler(
                    train_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    drop_last=True,
                )
                for train_dataset in train_datasets
            ]
            val_samplers = [
                DistributedSampler(
                    val_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                    drop_last=False,
                )
                for val_dataset in val_datasets
            ]
        else:
            train_samplers = [None] * len(train_datasets)
            val_samplers = [None] * len(val_datasets)

        # Extract additive models and scaler and move them to CPU/float64 so they
        # can be used in the collate function
        model.additive_models[0].weights_to(device="cpu", dtype=torch.float64)
        additive_models = copy.deepcopy(
            model.additive_models.to(dtype=torch.float64, device="cpu")
        )
        model.additive_models.to(device)
        model.additive_models[0].weights_to(device=device, dtype=torch.float64)
        model.scaler.scales_to(device="cpu", dtype=torch.float64)
        scaler = copy.deepcopy(model.scaler.to(dtype=torch.float64, device="cpu"))
        model.scaler.to(device)
        model.scaler.scales_to(device=device, dtype=torch.float64)

        # Target-type-specific orchestration (currently: atomic-basis RI
        # density training) is delegated to hooks; the no-op NullTrainerHooks
        # is returned when nothing of the sort is configured.
        target_hooks = get_trainer_hooks(
            loss_hypers,
            train_targets,
            self.hypers["ri_aux_basis"],
            bool(self.hypers.get("log_density_loss", False)),
        )
        ri_train_transforms, ri_val_transforms = target_hooks.collate_transforms(dtype)

        # Create collate functions

        conditioning_keys = list(model.requested_inputs().keys())
        if conditioning_keys:
            splits = [("training", train_datasets), ("validation", val_datasets)]
            for split, datasets in splits:
                if len(datasets[0]) == 0:
                    continue

                fields = datasets[0][0]._asdict()
                missing_keys = [key for key in conditioning_keys if key not in fields]
                if missing_keys:
                    logging.warning(
                        f"System conditioning is enabled but {missing_keys} are not in "
                        f"the {split} data and will fall back to defaults."
                    )

        conditioning_callables = (
            [get_system_data_transform(conditioning_keys)] if conditioning_keys else []
        )

        target_keys = list(train_targets.keys())
        # Shared callables that run after `atomic_basis_transform` (and after
        # rotational augmentation in training).
        base_callables: List[Callable[..., Any]] = [
            get_system_with_neighbor_lists_transform(requested_neighbor_lists),
            *conditioning_callables,
            get_remove_additive_transform(additive_models, train_targets),
            get_remove_scale_transform(scaler),
        ]
        collate_fn_train = CollateFn(
            target_keys=target_keys,
            callables=[
                atomic_basis_transform,
                # RI transforms run on the *unrotated* systems/targets: metric
                # matrices are cached per system across epochs (the density
                # losses un-rotate their residuals via the BatchRotations the
                # augmenter stashes), and the density-fit constant is
                # rotation-invariant.
                *ri_train_transforms,
                rotational_augmenter.apply_random_augmentations,
                *base_callables,
            ],
        )
        collate_fn_val = CollateFn(
            target_keys=target_keys,
            callables=[  # no augmentation for validation
                atomic_basis_transform,
                *ri_val_transforms,
                *base_callables,
            ],
        )

        # Create dataloader for the training datasets:
        # (num_workers/mp_context were already computed above, before the
        # composition/scaler fitting steps)
        train_dataloaders, epoch_samplers = build_train_dataloaders(
            train_datasets=train_datasets,
            train_distributed_samplers=train_samplers,
            collate_fn_train=collate_fn_train,
            batch_size=self.hypers["batch_size"],
            max_atoms_per_batch=max_atoms,
            min_atoms_per_batch=self.hypers["min_atoms_per_batch"],
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=(num_workers > 0),
        )
        train_dataloader = CombinedDataLoader(train_dataloaders, shuffle=True)

        # Create dataloader for the validation datasets:
        val_dataloaders = build_val_dataloaders(
            val_datasets=val_datasets,
            val_distributed_samplers=val_samplers,
            collate_fn_val=collate_fn_val,
            batch_size=self.hypers["batch_size"],
            max_atoms_per_batch=max_atoms,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=(num_workers > 0),
        )
        val_dataloader = CombinedDataLoader(val_dataloaders, shuffle=False)

        if is_distributed:
            model = DistributedDataParallel(model, device_ids=[device])

        outputs_list = []
        for target_name, target_info in train_targets.items():
            outputs_list.append(target_name)
            for gradient_name in target_info.gradients:
                outputs_list.append(f"{target_name}_{gradient_name}_gradients")

        # Create a loss function:
        loss_fn = LossAggregator(targets=train_targets, config=loss_hypers)
        logging.info("Using the following loss functions:")
        for name, info in loss_fn.metadata.items():
            logging.info(f"{name}:")
            main = {k: v for k, v in info.items() if k != "gradients"}
            logging.info(main)
            if "gradients" not in info or len(info["gradients"]) == 0:
                continue
            logging.info("With gradients:")
            for grad, ginfo in info["gradients"].items():
                logging.info(f"\t{name}::{grad}: {ginfo}")

        if self.hypers["weight_decay"] is not None:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.hypers["learning_rate"],
                weight_decay=self.hypers["weight_decay"],
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.hypers["learning_rate"]
            )

        if self.optimizer_state_dict is not None:
            # try to load the optimizer state dict, but this is only possible
            # if there are no new targets in the model (new parameters)
            if not (model.module if is_distributed else model).has_new_targets:
                optimizer.load_state_dict(self.optimizer_state_dict)

        # Create a learning rate scheduler
        lr_scheduler = get_scheduler(optimizer, self.hypers, len(train_dataloader))

        if self.scheduler_state_dict is not None:
            # same as the optimizer, try to load the scheduler state dict
            if not (model.module if is_distributed else model).has_new_targets:
                lr_scheduler.load_state_dict(self.scheduler_state_dict)

        per_structure_targets = self.hypers["per_structure_targets"]

        # Log the initial learning rate:
        logging.info(f"Base learning rate: {self.hypers['learning_rate']}")

        start_epoch = 0 if self.epoch is None else self.epoch + 1

        # Train the model:
        if self.best_metric is None:
            self.best_metric = float("inf")
        logging.info("Starting training")
        epoch = start_epoch

        for epoch in range(start_epoch, self.hypers["num_epochs"]):
            for sampler in epoch_samplers:
                sampler.set_epoch(epoch)
            train_rmse_calculator = RMSEAccumulator(self.hypers["log_separate_blocks"])
            val_rmse_calculator = RMSEAccumulator(self.hypers["log_separate_blocks"])
            if self.hypers["log_mae"]:
                train_mae_calculator = MAEAccumulator(
                    self.hypers["log_separate_blocks"]
                )
                val_mae_calculator = MAEAccumulator(self.hypers["log_separate_blocks"])

            is_log_epoch = (
                epoch == start_epoch or epoch % self.hypers["log_interval"] == 0
            )
            # Accumulate the loss as a device tensor: a per-batch .item() (and,
            # when distributed, a per-batch all_reduce) drains the CUDA queue
            # and serializes the Python-side batch preparation with the GPU.
            train_loss_acc: Optional[torch.Tensor] = None
            for batch in train_dataloader:
                optimizer.zero_grad()

                systems, targets, extra_data = _unpack_batch_to(batch, dtype, device)
                predictions = evaluate_model(
                    model,
                    systems,
                    {key: train_targets[key] for key in targets.keys()},
                    is_training=True,
                )

                # average by the number of atoms
                predictions = average_by_num_atoms(
                    predictions, systems, per_structure_targets
                )
                targets = average_by_num_atoms(targets, systems, per_structure_targets)

                # Apply per-property scales to the predictions before loss computation.
                # The targets from the dataloader have only been scaled per-target, and
                # not per-property. This transformation only applies to targets with
                # per-property scales (i.e. multiple blocks or multiple properties), and
                # leaves the others unchanged.
                predictions = (model.module if is_distributed else model).scaler(
                    systems,
                    predictions,
                    remove=False,
                    use_per_target_scales=False,  # never before loss
                    use_per_property_scales=True,
                )

                # On log epochs, compute the per-target scaled maps once: they
                # are needed for the metrics below and reused by the RI loss
                # preparation.
                scaled_predictions: Optional[Dict[str, Any]] = None
                scaled_targets: Optional[Dict[str, Any]] = None
                if is_log_epoch:
                    scaled_predictions = (
                        model.module if is_distributed else model
                    ).scaler(
                        systems,
                        predictions,
                        remove=False,
                        use_per_target_scales=True,
                        use_per_property_scales=False,
                    )
                    scaled_targets = (model.module if is_distributed else model).scaler(
                        systems,
                        targets,
                        remove=False,
                        use_per_target_scales=True,
                        use_per_property_scales=False,
                    )

                # Rescale predictions/targets to the physical units expected by
                # the active RI loss type (density_mse_via_c / density_mse_via_w).
                # For direct-c this is a no-op.
                loss_predictions, loss_targets = target_hooks.prepare_for_loss(
                    predictions,
                    targets,
                    systems,
                    model.module if is_distributed else model,
                    scaled_predictions=scaled_predictions,
                    scaled_targets=scaled_targets,
                )
                train_loss_batch = loss_fn(loss_predictions, loss_targets, extra_data)

                if is_distributed:
                    # make sure all parameters contribute to the gradient calculation
                    # to make torch DDP happy (e.g. when a target's head is kept in
                    # the model but not part of the current run's targets)
                    train_loss_batch += 0.0 * sum(
                        p.sum() for p in model.parameters() if p.requires_grad
                    )

                train_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.hypers["grad_clip_norm"]
                )
                optimizer.step()
                lr_scheduler.step()

                detached_loss = train_loss_batch.detach()
                train_loss_acc = (
                    detached_loss
                    if train_loss_acc is None
                    else train_loss_acc + detached_loss
                )

                # Accumulate quantities for computing train metrics, but only if
                # this is an epoch to log (scaled maps were computed above)
                if is_log_epoch:
                    assert scaled_predictions is not None
                    assert scaled_targets is not None
                    if self.hypers["log_separate_blocks"]:
                        # if any atomic basis outputs are present and metrics are to be
                        # reported per-block, reverse the transform (i.e. sparsify)
                        # before calculating metrics
                        systems, scaled_targets, extra_data = (
                            atomic_basis_reverse_transform(
                                systems, scaled_targets, extra_data
                            )
                        )
                        systems, scaled_predictions, _ = atomic_basis_reverse_transform(
                            systems, scaled_predictions, {}
                        )

                    train_rmse_calculator.update(
                        scaled_predictions, scaled_targets, extra_data
                    )
                    if self.hypers["log_mae"]:
                        train_mae_calculator.update(
                            scaled_predictions, scaled_targets, extra_data
                        )

            # One sync (and one distributed reduction) per epoch for the
            # accumulated loss, instead of one per batch.
            if train_loss_acc is None:
                train_loss = 0.0
            else:
                if is_distributed:
                    torch.distributed.all_reduce(train_loss_acc)
                train_loss = train_loss_acc.item()

            # Compute train metrics if they are to be logged this epoch:
            if is_log_epoch:
                finalized_train_info = train_rmse_calculator.finalize(
                    not_per_atom=["positions_gradients"] + per_structure_targets,
                    is_distributed=is_distributed,
                    device=device,
                )
                if self.hypers["log_mae"]:
                    finalized_train_info.update(
                        train_mae_calculator.finalize(
                            not_per_atom=["positions_gradients"]
                            + per_structure_targets,
                            is_distributed=is_distributed,
                            device=device,
                        )
                    )

            with torch.set_grad_enabled(
                any(target_info.gradients for target_info in train_targets.values())
            ):  # keep gradients on if any of the targets require them
                val_loss_acc: Optional[torch.Tensor] = None
                for batch in val_dataloader:
                    systems, targets, extra_data = _unpack_batch_to(
                        batch, dtype, device
                    )
                    predictions = evaluate_model(
                        model,
                        systems,
                        {key: train_targets[key] for key in targets.keys()},
                        is_training=False,
                    )

                    # average by the number of atoms
                    predictions = average_by_num_atoms(
                        predictions, systems, per_structure_targets
                    )
                    targets = average_by_num_atoms(
                        targets, systems, per_structure_targets
                    )

                    # Apply per-property scales to the predictions before loss
                    # computation. The targets from the dataloader have only been scaled
                    # per-target, and not per-property. This transformation only applies
                    # to targets with per-property scales (i.e. multiple blocks or
                    # multiple properties), and leaves the others unchanged.
                    predictions = (model.module if is_distributed else model).scaler(
                        systems,
                        predictions,
                        remove=False,
                        use_per_target_scales=False,
                        use_per_property_scales=True,
                    )

                    # Reapply per-target scales once; used for the val metrics
                    # below and reused by the RI loss preparation, which needs
                    # the same rescaling for its targets.
                    # scaled_predictions = (c_ML - CM),  scaled_targets = (c_RI - CM).
                    scaled_predictions = (
                        model.module if is_distributed else model
                    ).scaler(
                        systems,
                        predictions,
                        remove=False,
                        use_per_target_scales=True,
                        use_per_property_scales=False,
                    )
                    scaled_targets = (model.module if is_distributed else model).scaler(
                        systems,
                        targets,
                        remove=False,
                        use_per_target_scales=True,
                        use_per_property_scales=False,
                    )

                    loss_predictions, loss_targets = target_hooks.prepare_for_loss(
                        predictions,
                        targets,
                        systems,
                        model.module if is_distributed else model,
                        scaled_predictions=scaled_predictions,
                        scaled_targets=scaled_targets,
                    )
                    val_loss_batch = loss_fn(loss_predictions, loss_targets, extra_data)

                    detached_loss = val_loss_batch.detach()
                    val_loss_acc = (
                        detached_loss
                        if val_loss_acc is None
                        else val_loss_acc + detached_loss
                    )

                    # Extra target-type metrics (e.g. the density L2 metric)
                    # accumulate on the dense maps, i.e. before the
                    # log_separate_blocks sparsification below.
                    target_hooks.update_validation_metrics(
                        scaled_predictions, scaled_targets, extra_data
                    )

                    if self.hypers["log_separate_blocks"]:
                        # if any atomic basis outputs are present and metrics are to be
                        # reported per-block, reverse the transform (i.e. sparsify)
                        # before calculating metrics
                        systems, scaled_targets, extra_data = (
                            atomic_basis_reverse_transform(
                                systems, scaled_targets, extra_data
                            )
                        )
                        systems, scaled_predictions, _ = atomic_basis_reverse_transform(
                            systems, scaled_predictions, {}
                        )

                    val_rmse_calculator.update(
                        scaled_predictions, scaled_targets, extra_data
                    )
                    if self.hypers["log_mae"]:
                        val_mae_calculator.update(
                            scaled_predictions, scaled_targets, extra_data
                        )

            # One sync (and one distributed reduction) per epoch for the
            # accumulated validation scalars.
            val_loss_total = (
                val_loss_acc
                if val_loss_acc is not None
                else torch.zeros((), device=device)
            )
            if is_distributed:
                torch.distributed.all_reduce(val_loss_total)
            val_loss = float(val_loss_total.item())

            # Compute val metrics:
            finalized_val_info = val_rmse_calculator.finalize(
                not_per_atom=["positions_gradients"] + per_structure_targets,
                is_distributed=is_distributed,
                device=device,
            )
            if self.hypers["log_mae"]:
                finalized_val_info.update(
                    val_mae_calculator.finalize(
                        not_per_atom=["positions_gradients"] + per_structure_targets,
                        is_distributed=is_distributed,
                        device=device,
                    )
                )

            # Now we log the information:
            if is_log_epoch:
                finalized_train_info = {
                    "loss": train_loss,
                    **finalized_train_info,
                }
            finalized_val_info = {
                "loss": val_loss,
                **finalized_val_info,
            }
            finalized_val_info.update(
                target_hooks.finalize_validation_metrics(is_distributed, device)
            )

            if epoch == start_epoch:
                metric_logger = MetricLogger(
                    log_obj=ROOT_LOGGER,
                    dataset_info=(
                        model.module if is_distributed else model
                    ).dataset_info,
                    initial_metrics=[finalized_train_info, finalized_val_info],
                    names=["training", "validation"],
                )
            if epoch % self.hypers["log_interval"] == 0:
                metric_logger.log(
                    metrics=[finalized_train_info, finalized_val_info],
                    epoch=epoch,
                    rank=rank,
                    learning_rate=optimizer.param_groups[0]["lr"],
                )

            val_metric = get_selected_metric(
                finalized_val_info, self.hypers["best_model_metric"]
            )
            if val_metric < self.best_metric:
                self.best_metric = val_metric
                self.best_model_state_dict = _clone_state_dict_to_cpu(
                    (model.module if is_distributed else model).state_dict()
                )
                self.best_epoch = epoch
                self.best_optimizer_state_dict = _clone_state_dict_to_cpu(
                    optimizer.state_dict()
                )

            if epoch % self.hypers["checkpoint_interval"] == 0:
                if is_distributed:
                    torch.distributed.barrier()
                self.optimizer_state_dict = optimizer.state_dict()
                self.scheduler_state_dict = lr_scheduler.state_dict()
                self.epoch = epoch
                if rank == 0:
                    self.save_checkpoint(
                        (model.module if is_distributed else model),
                        Path(checkpoint_dir) / f"model_{epoch}.ckpt",
                    )

        # prepare for the checkpoint that will be saved outside the function
        self.epoch = epoch
        self.optimizer_state_dict = optimizer.state_dict()
        self.scheduler_state_dict = lr_scheduler.state_dict()

        if is_distributed:
            torch.distributed.destroy_process_group()

    def save_checkpoint(self, model: ModelInterface, path: Union[str, Path]) -> None:
        checkpoint = model.get_checkpoint()
        if self.best_model_state_dict is not None:
            self.best_model_state_dict["finetune_config"] = model.finetune_config
        checkpoint.update(
            {
                "trainer_ckpt_version": self.__checkpoint_version__,
                "train_hypers": self.hypers,
                "epoch": self.epoch,
                "optimizer_state_dict": self.optimizer_state_dict,
                "scheduler_state_dict": self.scheduler_state_dict,
                "best_epoch": self.best_epoch,
                "best_metric": self.best_metric,
                "best_model_state_dict": self.best_model_state_dict,
                "best_optimizer_state_dict": self.best_optimizer_state_dict,
            }
        )
        torch.save(
            checkpoint,
            check_file_extension(path, ".ckpt"),
        )

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        hypers: TrainerHypers,
        context: Literal["restart", "finetune"],
    ) -> "Trainer":
        trainer = cls(hypers)
        trainer.optimizer_state_dict = checkpoint["optimizer_state_dict"]
        trainer.scheduler_state_dict = checkpoint["scheduler_state_dict"]
        if context == "restart":
            trainer.epoch = checkpoint["epoch"]
        else:
            assert context == "finetune"
            trainer.epoch = None  # interpreted as zero in the training loop
        trainer.best_epoch = checkpoint["best_epoch"]
        trainer.best_metric = checkpoint["best_metric"]
        trainer.best_model_state_dict = checkpoint["best_model_state_dict"]
        trainer.best_optimizer_state_dict = checkpoint["best_optimizer_state_dict"]

        return trainer

    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict) -> Dict:
        for v in range(1, cls.__checkpoint_version__):
            if checkpoint["trainer_ckpt_version"] == v:
                update = getattr(checkpoints, f"trainer_update_v{v}_v{v + 1}")
                update(checkpoint)
                checkpoint["trainer_ckpt_version"] = v + 1

        if checkpoint["trainer_ckpt_version"] != cls.__checkpoint_version__:
            raise RuntimeError(
                f"Unable to upgrade the checkpoint: the checkpoint is using "
                f"trainer version {checkpoint['trainer_ckpt_version']}, while the "
                f"current trainer version is {cls.__checkpoint_version__}."
            )
        return checkpoint
