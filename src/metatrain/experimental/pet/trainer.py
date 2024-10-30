import datetime
import logging
import os
import pickle
import time
import warnings
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from pet.analysis import adapt_hypers
from pet.data_preparation import (
    get_all_species,
    get_corrected_energies,
    get_forces,
    get_pyg_graphs,
    get_self_contributions,
    update_pyg_graphs,
)
from pet.hypers import Hypers, save_hypers
from pet.pet import (
    PET,
    FlagsWrapper,
    PETMLIPWrapper,
    PETUtilityWrapper,
    SelfContributionsWrapper,
)
from pet.utilities import (
    FullLogger,
    ModelKeeper,
    dtype2string,
    get_calc_names,
    get_data_loaders,
    get_loss,
    get_optimizer,
    get_rmse,
    get_scheduler,
    load_checkpoint,
    log_epoch_stats,
    set_reproducibility,
    string2dtype,
)
from torch_geometric.nn import DataParallel

from ...utils.data import Dataset, check_datasets
from . import PET as WrappedPET
from .utils import dataset_to_ase, update_hypers

from ...utils.distributed.slurm import DistributedEnvironment
from ...utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)
from ...utils.distributed.distributed_data_parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from ...utils.data import CombinedDataLoader, Dataset, TargetInfoDict, collate_fn
from ...utils.data.extract_targets import get_targets_dict
from ...utils.external_naming import to_external_name

from ...utils.loss import TensorMapDictLoss

from ...utils.metrics import MAEAccumulator, RMSEAccumulator
from ...utils.transfer import (
    systems_and_targets_to_device,
    systems_and_targets_to_dtype,
)
from ...utils.additive import remove_additive
from ...utils.evaluate_model import evaluate_model
from ...utils.per_atom import average_by_num_atoms

from ...utils.logging import MetricLogger

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, train_hypers):
        self.hypers = {"FITTING_SCHEME": train_hypers}
        self.pet_dir = None
        self.pet_checkpoint = None

    def train(
        self,
        model: WrappedPET,
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ):
        assert dtype in WrappedPET.__supported_dtypes__

        is_distributed = self.hypers["DISTRIBUTED"]

        if is_distributed:
            distr_env = DistributedEnvironment(self.hypers["DISTRIBUTED_PORT"])
            torch.distributed.init_process_group(backend="nccl")
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        if is_distributed:
            if len(devices) > 1:
                raise ValueError(
                    "Requested distributed training with the `multi-gpu` device. "
                    " If you want to run distributed training with SOAP-BPNN, please "
                    "set `device` to cuda."
                )
            # the calculation of the device number works both when GPUs on different
            # processes are not visible to each other and when they are
            device_number = distr_env.local_rank % torch.cuda.device_count()
            device = torch.device("cuda", device_number)
        else:
            device = devices[
                0
            ]  # only one device, as we don't support multi-gpu for now

        if is_distributed:
            logger.info(f"Training on {world_size} devices with dtype {dtype}")
        else:
            logger.info(f"Training on device {device} with dtype {dtype}")

        # Calculate the neighbor lists in advance (in particular, this
        # needs to happen before the additive models are trained, as they
        # might need them):
        logger.info("Calculating neighbor lists for the datasets")
        requested_neighbor_lists = get_requested_neighbor_lists(model)
        for dataset in train_datasets + val_datasets:
            for i in range(len(dataset)):
                system = dataset[i]["system"]
                # The following line attaches the neighbors lists to the system,
                # and doesn't require to reassign the system to the dataset:
                _ = get_system_with_neighbor_lists(system, requested_neighbor_lists)

        # Move the model to the device and dtype:
        model.to(device=device, dtype=dtype)
        # The additive models of the SOAP-BPNN are always in float64 (to avoid
        # numerical errors in the composition weights, which can be very large).
        for additive_model in model.additive_models:
            additive_model.to(dtype=torch.float64)

        logger.info("Calculating composition weights")
        if self.hypers["SELF_CONTRIBUTIONS_PATH"] is not None:
            composition_weights = torch.tensor(
                np.load(self.hypers["SELF_CONTRIBUTIONS_PATH"]),
                device=device,
                dtype=dtype,
            )
        else:
            composition_weights = None

        model.additive_models[0].train_model(  # this is the composition model
            train_datasets, composition_weights
        )

        if is_distributed:
            model = DistributedDataParallel(model, device_ids=[device])

        logger.info("Setting up data loaders")

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

        # Create dataloader for the training datasets:
        train_dataloaders = []
        for dataset, sampler in zip(train_datasets, train_samplers):
            train_dataloaders.append(
                DataLoader(
                    dataset=dataset,
                    batch_size=self.hypers["batch_size"],
                    sampler=sampler,
                    shuffle=(
                        sampler is None
                    ),  # the sampler takes care of this (if present)
                    drop_last=(
                        sampler is None
                    ),  # the sampler takes care of this (if present)
                    collate_fn=collate_fn,
                )
            )
        train_dataloader = CombinedDataLoader(train_dataloaders, shuffle=True)

        # Create dataloader for the validation datasets:
        val_dataloaders = []
        for dataset, sampler in zip(val_datasets, val_samplers):
            val_dataloaders.append(
                DataLoader(
                    dataset=dataset,
                    batch_size=self.hypers["batch_size"],
                    sampler=sampler,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=collate_fn,
                )
            )
        val_dataloader = CombinedDataLoader(val_dataloaders, shuffle=False)

        # Extract all the possible outputs and their gradients:
        train_targets = get_targets_dict(
            train_datasets, (model.module if is_distributed else model).dataset_info
        )
        outputs_list = []
        for target_name, target_info in train_targets.items():
            outputs_list.append(target_name)
            for gradient_name in target_info.gradients:
                outputs_list.append(f"{target_name}_{gradient_name}_gradients")
        # Create a loss weight dict:
        loss_weights_dict = {}
        for output_name in outputs_list:
            loss_weights_dict[output_name] = (
                self.hypers["loss_weights"][
                    to_external_name(output_name, train_targets)
                ]
                if to_external_name(output_name, train_targets)
                in self.hypers["loss_weights"]
                else 1.0
            )
        loss_weights_dict_external = {
            to_external_name(key, train_targets): value
            for key, value in loss_weights_dict.items()
        }
        logging.info(f"Training with loss weights: {loss_weights_dict_external}")

        # Create a loss function:
        loss_fn = TensorMapDictLoss(loss_weights_dict)

        # Create an optimizer:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.hypers["learning_rate"]
        )
        if self.optimizer_state_dict is not None:
            optimizer.load_state_dict(self.optimizer_state_dict)

        # Create a scheduler:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.hypers["scheduler_factor"],
            patience=self.hypers["scheduler_patience"],
        )
        if self.scheduler_state_dict is not None:
            lr_scheduler.load_state_dict(self.scheduler_state_dict)

        # counters for early stopping:
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        # per-atom targets:
        per_structure_targets = self.hypers["per_structure_targets"]

        # Log the initial learning rate:
        old_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Initial learning rate: {old_lr}")

        start_epoch = 0 if self.epoch is None else self.epoch + 1

        # Train the model:
        logger.info("Starting training")
        for epoch in range(start_epoch, start_epoch + self.hypers["num_epochs"]):
            if is_distributed:
                sampler.set_epoch(epoch)

            train_rmse_calculator = RMSEAccumulator()
            val_rmse_calculator = RMSEAccumulator()
            if self.hypers["log_mae"]:
                train_mae_calculator = MAEAccumulator()
                val_mae_calculator = MAEAccumulator()

            train_loss = 0.0
            for batch in train_dataloader:
                optimizer.zero_grad()

                systems, targets = batch
                systems, targets = systems_and_targets_to_device(
                    systems, targets, device
                )
                for additive_model in (
                    model.module if is_distributed else model
                ).additive_models:
                    targets = remove_additive(
                        systems, targets, additive_model, train_targets
                    )
                systems, targets = systems_and_targets_to_dtype(systems, targets, dtype)
                predictions = evaluate_model(
                    model,
                    systems,
                    TargetInfoDict(
                        **{key: train_targets[key] for key in targets.keys()}
                    ),
                    is_training=True,
                )

                # average by the number of atoms
                predictions = average_by_num_atoms(
                    predictions, systems, per_structure_targets
                )
                targets = average_by_num_atoms(targets, systems, per_structure_targets)

                train_loss_batch = loss_fn(predictions, targets)

                train_loss_batch.backward()
                optimizer.step()

                if is_distributed:
                    # sum the loss over all processes
                    torch.distributed.all_reduce(train_loss_batch)
                train_loss += train_loss_batch.item()
                train_rmse_calculator.update(predictions, targets)
                if self.hypers["log_mae"]:
                    train_mae_calculator.update(predictions, targets)

            finalized_train_info = train_rmse_calculator.finalize(
                not_per_atom=["positions_gradients"] + per_structure_targets,
                is_distributed=is_distributed,
                device=device,
            )
            if self.hypers["log_mae"]:
                finalized_train_info.update(
                    train_mae_calculator.finalize(
                        not_per_atom=["positions_gradients"] + per_structure_targets,
                        is_distributed=is_distributed,
                        device=device,
                    )
                )

            val_loss = 0.0
            for batch in val_dataloader:
                systems, targets = batch
                systems, targets = systems_and_targets_to_device(
                    systems, targets, device
                )
                for additive_model in (
                    model.module if is_distributed else model
                ).additive_models:
                    targets = remove_additive(
                        systems, targets, additive_model, train_targets
                    )
                systems, targets = systems_and_targets_to_dtype(systems, targets, dtype)
                predictions = evaluate_model(
                    model,
                    systems,
                    TargetInfoDict(
                        **{key: train_targets[key] for key in targets.keys()}
                    ),
                    is_training=False,
                )

                # average by the number of atoms
                predictions = average_by_num_atoms(
                    predictions, systems, per_structure_targets
                )
                targets = average_by_num_atoms(targets, systems, per_structure_targets)

                val_loss_batch = loss_fn(predictions, targets)

                if is_distributed:
                    # sum the loss over all processes
                    torch.distributed.all_reduce(val_loss_batch)
                val_loss += val_loss_batch.item()
                val_rmse_calculator.update(predictions, targets)
                if self.hypers["log_mae"]:
                    val_mae_calculator.update(predictions, targets)

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
            finalized_train_info = {"loss": train_loss, **finalized_train_info}
            finalized_val_info = {"loss": val_loss, **finalized_val_info}

            if epoch == start_epoch:
                metric_logger = MetricLogger(
                    log_obj=logger,
                    dataset_info=model.dataset_info,
                    initial_metrics=[finalized_train_info, finalized_val_info],
                    names=["training", "validation"],
                )
            if epoch % self.hypers["log_interval"] == 0:
                metric_logger.log(
                    metrics=[finalized_train_info, finalized_val_info],
                    epoch=epoch,
                    rank=rank,
                )

            lr_scheduler.step(val_loss)
            new_lr = lr_scheduler.get_last_lr()[0]
            if new_lr != old_lr:
                logger.info(f"Changing learning rate from {old_lr} to {new_lr}")
                old_lr = new_lr

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

            # early stopping criterion:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.hypers["early_stopping_patience"]:
                    logger.info(
                        "Early stopping criterion reached after "
                        f"{self.hypers['early_stopping_patience']} epochs "
                        "without improvement."
                    )
                    break

        name_of_calculation = "pet"
        self.pet_dir = Path(checkpoint_dir) / name_of_calculation

        if len(train_datasets) != 1:
            raise ValueError("PET only supports a single training dataset")
        if len(val_datasets) != 1:
            raise ValueError("PET only supports a single validation dataset")
        if torch.device("cpu") in devices:
            warnings.warn(
                "Training PET on a CPU is very slow! For better performance, use a "
                "CUDA GPU.",
                stacklevel=1,
            )

        logger.info("Checking datasets for consistency")
        check_datasets(train_datasets, val_datasets)

        train_dataset = train_datasets[0]
        val_dataset = val_datasets[0]

        # are we fitting on only energies or energies and forces?
        target_name = model.target_name
        do_forces = (
            next(iter(train_dataset))[target_name].block().has_gradient("positions")
        )

        ase_train_dataset = dataset_to_ase(
            train_dataset, model, do_forces=do_forces, target_name=target_name
        )
        ase_val_dataset = dataset_to_ase(
            val_dataset, model, do_forces=do_forces, target_name=target_name
        )

        self.hypers = update_hypers(self.hypers, model.hypers, do_forces)

        device = devices[0]  # only one device, as we don't support multi-gpu for now

        if self.pet_checkpoint is not None:
            # save the checkpoint to a temporary file, so that fit_pet can load it
            checkpoint_path = Path(checkpoint_dir) / "checkpoint.temp"
            torch.save(
                self.pet_checkpoint,
                checkpoint_path,
            )
        else:
            checkpoint_path = None

        ########################################
        # STARTNG THE PURE PET TRAINING SCRIPT #
        ########################################

        logging.info("Initializing PET training...")

        TIME_SCRIPT_STARTED = time.time()
        value = datetime.datetime.fromtimestamp(TIME_SCRIPT_STARTED)
        logging.info(f"Starting training at: {value.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("Training configuration:")

        print(f"Output directory: {checkpoint_dir}")
        print(f"Training using device: {device}")

        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        hypers = Hypers(self.hypers)
        dtype = string2dtype(hypers.ARCHITECTURAL_HYPERS.DTYPE)
        torch.set_default_dtype(dtype)

        FITTING_SCHEME = hypers.FITTING_SCHEME
        MLIP_SETTINGS = hypers.MLIP_SETTINGS
        ARCHITECTURAL_HYPERS = hypers.ARCHITECTURAL_HYPERS

        if FITTING_SCHEME.USE_SHIFT_AGNOSTIC_LOSS:
            raise ValueError(
                "shift agnostic loss is intended only for general target training"
            )

        ARCHITECTURAL_HYPERS.D_OUTPUT = 1  # energy is a single scalar
        ARCHITECTURAL_HYPERS.TARGET_TYPE = "structural"  # energy is structural property
        ARCHITECTURAL_HYPERS.TARGET_AGGREGATION = (
            "sum"  # energy is a sum of atomic energies
        )
        print(f"Output dimensionality: {ARCHITECTURAL_HYPERS.D_OUTPUT}")
        print(f"Target type: {ARCHITECTURAL_HYPERS.TARGET_TYPE}")
        print(f"Target aggregation: {ARCHITECTURAL_HYPERS.TARGET_AGGREGATION}")

        set_reproducibility(
            FITTING_SCHEME.RANDOM_SEED, FITTING_SCHEME.CUDA_DETERMINISTIC
        )

        print(f"Random seed: {FITTING_SCHEME.RANDOM_SEED}")
        print(f"CUDA is deterministic: {FITTING_SCHEME.CUDA_DETERMINISTIC}")

        adapt_hypers(FITTING_SCHEME, ase_train_dataset)
        dataset = ase_train_dataset + ase_val_dataset
        all_species = get_all_species(dataset)

        name_to_load, NAME_OF_CALCULATION = get_calc_names(
            os.listdir(checkpoint_dir), name_of_calculation
        )

        os.mkdir(f"{checkpoint_dir}/{NAME_OF_CALCULATION}")
        np.save(f"{checkpoint_dir}/{NAME_OF_CALCULATION}/all_species.npy", all_species)
        hypers.UTILITY_FLAGS.CALCULATION_TYPE = "mlip"
        save_hypers(hypers, f"{checkpoint_dir}/{NAME_OF_CALCULATION}/hypers_used.yaml")

        logging.info("Convering structures to PyG graphs...")

        train_graphs = get_pyg_graphs(
            ase_train_dataset,
            all_species,
            ARCHITECTURAL_HYPERS.R_CUT,
            ARCHITECTURAL_HYPERS.USE_ADDITIONAL_SCALAR_ATTRIBUTES,
            ARCHITECTURAL_HYPERS.USE_LONG_RANGE,
            ARCHITECTURAL_HYPERS.K_CUT,
            ARCHITECTURAL_HYPERS.N_TARGETS > 1,
            ARCHITECTURAL_HYPERS.TARGET_INDEX_KEY,
        )
        val_graphs = get_pyg_graphs(
            ase_val_dataset,
            all_species,
            ARCHITECTURAL_HYPERS.R_CUT,
            ARCHITECTURAL_HYPERS.USE_ADDITIONAL_SCALAR_ATTRIBUTES,
            ARCHITECTURAL_HYPERS.USE_LONG_RANGE,
            ARCHITECTURAL_HYPERS.K_CUT,
            ARCHITECTURAL_HYPERS.N_TARGETS > 1,
            ARCHITECTURAL_HYPERS.TARGET_INDEX_KEY,
        )

        logging.info("Pre-processing training data...")
        if MLIP_SETTINGS.USE_ENERGIES:
            self_contributions = get_self_contributions(
                MLIP_SETTINGS.ENERGY_KEY, ase_train_dataset, all_species
            )
            np.save(
                f"{checkpoint_dir}/{NAME_OF_CALCULATION}/self_contributions.npy",
                self_contributions,
            )

            train_energies = get_corrected_energies(
                MLIP_SETTINGS.ENERGY_KEY,
                ase_train_dataset,
                all_species,
                self_contributions,
            )
            val_energies = get_corrected_energies(
                MLIP_SETTINGS.ENERGY_KEY,
                ase_val_dataset,
                all_species,
                self_contributions,
            )

            update_pyg_graphs(train_graphs, "y", train_energies)
            update_pyg_graphs(val_graphs, "y", val_energies)

        if MLIP_SETTINGS.USE_FORCES:
            train_forces = get_forces(ase_train_dataset, MLIP_SETTINGS.FORCES_KEY)
            val_forces = get_forces(ase_val_dataset, MLIP_SETTINGS.FORCES_KEY)

            update_pyg_graphs(train_graphs, "forces", train_forces)
            update_pyg_graphs(val_graphs, "forces", val_forces)

        train_loader, val_loader = get_data_loaders(
            train_graphs, val_graphs, FITTING_SCHEME
        )

        logging.info("Initializing the model...")
        pet_model = PET(ARCHITECTURAL_HYPERS, 0.0, len(all_species)).to(device)
        pet_model = PETUtilityWrapper(pet_model, FITTING_SCHEME.GLOBAL_AUG)

        pet_model = PETMLIPWrapper(
            pet_model, MLIP_SETTINGS.USE_ENERGIES, MLIP_SETTINGS.USE_FORCES
        )
        if FITTING_SCHEME.MULTI_GPU and torch.cuda.is_available():
            logging.info(
                f"Using multi-GPU training on {torch.cuda.device_count()} GPUs"
            )
            pet_model = DataParallel(FlagsWrapper(pet_model))
            pet_model = pet_model.to(torch.device("cuda:0"))

        if FITTING_SCHEME.MODEL_TO_START_WITH is not None:
            logging.info(f"Loading model from: {FITTING_SCHEME.MODEL_TO_START_WITH}")
            pet_model.load_state_dict(torch.load(FITTING_SCHEME.MODEL_TO_START_WITH))
            pet_model = pet_model.to(dtype=dtype)

        optim = get_optimizer(pet_model, FITTING_SCHEME)
        scheduler = get_scheduler(optim, FITTING_SCHEME)

        if checkpoint_path is not None:
            logging.info(f"Loading model and checkpoint from: {checkpoint_path}\n")
            load_checkpoint(pet_model, optim, scheduler, checkpoint_path)
        elif name_to_load is not None:
            path = f"{checkpoint_dir}/{name_to_load}/checkpoint"
            logging.info(f"Loading model and checkpoint from: {path}\n")
            load_checkpoint(
                pet_model,
                optim,
                scheduler,
                f"{checkpoint_dir}/{name_to_load}/checkpoint",
            )

        history = []
        if MLIP_SETTINGS.USE_ENERGIES:
            energies_logger = FullLogger(
                FITTING_SCHEME.SUPPORT_MISSING_VALUES,
                FITTING_SCHEME.USE_SHIFT_AGNOSTIC_LOSS,
                device,
            )

        if MLIP_SETTINGS.USE_FORCES:
            forces_logger = FullLogger(
                FITTING_SCHEME.SUPPORT_MISSING_VALUES,
                FITTING_SCHEME.USE_SHIFT_AGNOSTIC_LOSS,
                device,
            )

        if MLIP_SETTINGS.USE_FORCES:
            val_forces = torch.cat(val_forces, dim=0)

            sliding_forces_rmse = get_rmse(
                val_forces.data.cpu().to(dtype=torch.float32).numpy(), 0.0
            )

            forces_rmse_model_keeper = ModelKeeper()
            forces_mae_model_keeper = ModelKeeper()

        if MLIP_SETTINGS.USE_ENERGIES:
            if FITTING_SCHEME.ENERGIES_LOSS == "per_structure":
                sliding_energies_rmse = get_rmse(val_energies, np.mean(val_energies))
            else:
                val_n_atoms = np.array(
                    [len(struc.positions) for struc in ase_val_dataset]
                )
                val_energies_per_atom = val_energies / val_n_atoms
                sliding_energies_rmse = get_rmse(
                    val_energies_per_atom, np.mean(val_energies_per_atom)
                )

            energies_rmse_model_keeper = ModelKeeper()
            energies_mae_model_keeper = ModelKeeper()

        if MLIP_SETTINGS.USE_ENERGIES and MLIP_SETTINGS.USE_FORCES:
            multiplication_rmse_model_keeper = ModelKeeper()
            multiplication_mae_model_keeper = ModelKeeper()

        logging.info(f"Starting training for {FITTING_SCHEME.EPOCH_NUM} epochs")
        if FITTING_SCHEME.EPOCHS_WARMUP > 0:
            remaining_lr_scheduler_steps = (
                FITTING_SCHEME.EPOCHS_WARMUP - scheduler.last_epoch
            )
            logging.info(
                f"Performing {remaining_lr_scheduler_steps} epochs of LR warmup"
            )
        TIME_TRAINING_STARTED = time.time()
        last_elapsed_time = 0
        print("=" * 50)
        for epoch in range(1, FITTING_SCHEME.EPOCH_NUM + 1):
            pet_model.train(True)
            for batch in train_loader:
                if not FITTING_SCHEME.MULTI_GPU:
                    batch.to(device)

                if FITTING_SCHEME.MULTI_GPU:
                    pet_model.module.augmentation = True
                    pet_model.module.create_graph = True
                    predictions_energies, predictions_forces = pet_model(batch)
                else:
                    predictions_energies, predictions_forces = pet_model(
                        batch, augmentation=True, create_graph=True
                    )

                if FITTING_SCHEME.MULTI_GPU:
                    y_list = [el.y for el in batch]
                    batch_y = torch.tensor(
                        y_list, dtype=torch.get_default_dtype(), device=device
                    )

                    n_atoms_list = [el.n_atoms for el in batch]
                    batch_n_atoms = torch.tensor(
                        n_atoms_list, dtype=torch.get_default_dtype(), device=device
                    )
                    # print('batch_y: ', batch_y.shape)
                    # print('batch_n_atoms: ', batch_n_atoms.shape)

                else:
                    batch_y = batch.y
                    batch_n_atoms = batch.n_atoms

                if FITTING_SCHEME.ENERGIES_LOSS == "per_atom":
                    predictions_energies = predictions_energies / batch_n_atoms
                    ground_truth_energies = batch_y / batch_n_atoms
                else:
                    ground_truth_energies = batch_y

                if MLIP_SETTINGS.USE_ENERGIES:
                    energies_logger.train_logger.update(
                        predictions_energies, ground_truth_energies
                    )
                    loss_energies = get_loss(
                        predictions_energies,
                        ground_truth_energies,
                        FITTING_SCHEME.SUPPORT_MISSING_VALUES,
                        FITTING_SCHEME.USE_SHIFT_AGNOSTIC_LOSS,
                    )
                if MLIP_SETTINGS.USE_FORCES:

                    if FITTING_SCHEME.MULTI_GPU:
                        forces_list = [el.forces for el in batch]
                        batch_forces = torch.cat(forces_list, dim=0).to(device)
                    else:
                        batch_forces = batch.forces

                    forces_logger.train_logger.update(predictions_forces, batch_forces)
                    loss_forces = get_loss(
                        predictions_forces,
                        batch_forces,
                        FITTING_SCHEME.SUPPORT_MISSING_VALUES,
                        FITTING_SCHEME.USE_SHIFT_AGNOSTIC_LOSS,
                    )

                if MLIP_SETTINGS.USE_ENERGIES and MLIP_SETTINGS.USE_FORCES:
                    loss = FITTING_SCHEME.ENERGY_WEIGHT * loss_energies / (
                        sliding_energies_rmse**2
                    ) + loss_forces / (sliding_forces_rmse**2)
                    loss.backward()

                if MLIP_SETTINGS.USE_ENERGIES and (not MLIP_SETTINGS.USE_FORCES):
                    loss_energies.backward()
                if MLIP_SETTINGS.USE_FORCES and (not MLIP_SETTINGS.USE_ENERGIES):
                    loss_forces.backward()

                if FITTING_SCHEME.DO_GRADIENT_CLIPPING:
                    torch.nn.utils.clip_grad_norm_(
                        pet_model.parameters(),
                        max_norm=FITTING_SCHEME.GRADIENT_CLIPPING_MAX_NORM,
                    )
                optim.step()
                optim.zero_grad()

            pet_model.train(False)
            for batch in val_loader:
                if not FITTING_SCHEME.MULTI_GPU:
                    batch.to(device)

                if FITTING_SCHEME.MULTI_GPU:
                    pet_model.module.augmentation = False
                    pet_model.module.create_graph = False
                    predictions_energies, predictions_forces = pet_model(batch)
                else:
                    predictions_energies, predictions_forces = pet_model(
                        batch, augmentation=False, create_graph=False
                    )

                if FITTING_SCHEME.MULTI_GPU:
                    y_list = [el.y for el in batch]
                    batch_y = torch.tensor(
                        y_list, dtype=torch.get_default_dtype(), device=device
                    )

                    n_atoms_list = [el.n_atoms for el in batch]
                    batch_n_atoms = torch.tensor(
                        n_atoms_list, dtype=torch.get_default_dtype(), device=device
                    )

                    # print('batch_y: ', batch_y.shape)
                    # print('batch_n_atoms: ', batch_n_atoms.shape)
                else:
                    batch_y = batch.y
                    batch_n_atoms = batch.n_atoms

                if FITTING_SCHEME.ENERGIES_LOSS == "per_atom":
                    predictions_energies = predictions_energies / batch_n_atoms
                    ground_truth_energies = batch_y / batch_n_atoms
                else:
                    ground_truth_energies = batch_y

                if MLIP_SETTINGS.USE_ENERGIES:
                    energies_logger.val_logger.update(
                        predictions_energies, ground_truth_energies
                    )
                if MLIP_SETTINGS.USE_FORCES:
                    if FITTING_SCHEME.MULTI_GPU:
                        forces_list = [el.forces for el in batch]
                        batch_forces = torch.cat(forces_list, dim=0).to(device)
                    else:
                        batch_forces = batch.forces
                    forces_logger.val_logger.update(predictions_forces, batch_forces)

            now = {}
            if FITTING_SCHEME.ENERGIES_LOSS == "per_structure":
                energies_key = "energies per structure"
            else:
                energies_key = "energies per atom"

            if MLIP_SETTINGS.USE_ENERGIES:
                now[energies_key] = energies_logger.flush()

            if MLIP_SETTINGS.USE_FORCES:
                now["forces"] = forces_logger.flush()
            now["lr"] = scheduler.get_last_lr()
            now["epoch"] = epoch

            now["elapsed_time"] = time.time() - TIME_TRAINING_STARTED
            now["epoch_time"] = now["elapsed_time"] - last_elapsed_time
            now["estimated_remaining_time"] = (now["elapsed_time"] / epoch) * (
                FITTING_SCHEME.EPOCH_NUM - epoch
            )
            last_elapsed_time = now["elapsed_time"]

            if MLIP_SETTINGS.USE_ENERGIES:
                sliding_energies_rmse = (
                    FITTING_SCHEME.SLIDING_FACTOR * sliding_energies_rmse
                    + (1.0 - FITTING_SCHEME.SLIDING_FACTOR)
                    * now[energies_key]["val"]["rmse"]
                )

                energies_mae_model_keeper.update(
                    pet_model, now[energies_key]["val"]["mae"], epoch
                )
                energies_rmse_model_keeper.update(
                    pet_model, now[energies_key]["val"]["rmse"], epoch
                )

            if MLIP_SETTINGS.USE_FORCES:
                sliding_forces_rmse = (
                    FITTING_SCHEME.SLIDING_FACTOR * sliding_forces_rmse
                    + (1.0 - FITTING_SCHEME.SLIDING_FACTOR)
                    * now["forces"]["val"]["rmse"]
                )
                forces_mae_model_keeper.update(
                    pet_model, now["forces"]["val"]["mae"], epoch
                )
                forces_rmse_model_keeper.update(
                    pet_model, now["forces"]["val"]["rmse"], epoch
                )

            if MLIP_SETTINGS.USE_ENERGIES and MLIP_SETTINGS.USE_FORCES:
                multiplication_mae_model_keeper.update(
                    pet_model,
                    now["forces"]["val"]["mae"] * now[energies_key]["val"]["mae"],
                    epoch,
                    additional_info=[
                        now[energies_key]["val"]["mae"],
                        now["forces"]["val"]["mae"],
                    ],
                )
                multiplication_rmse_model_keeper.update(
                    pet_model,
                    now["forces"]["val"]["rmse"] * now[energies_key]["val"]["rmse"],
                    epoch,
                    additional_info=[
                        now[energies_key]["val"]["rmse"],
                        now["forces"]["val"]["rmse"],
                    ],
                )
            last_lr = scheduler.get_last_lr()[0]
            log_epoch_stats(epoch, FITTING_SCHEME.EPOCH_NUM, now, last_lr, energies_key)

            history.append(now)
            scheduler.step()
            elapsed = time.time() - TIME_SCRIPT_STARTED
            if epoch > 0 and epoch % FITTING_SCHEME.CHECKPOINT_INTERVAL == 0:
                checkpoint_dict = {
                    "model_state_dict": pet_model.state_dict(),
                    "optim_state_dict": optim.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "dtype_used": dtype2string(dtype),
                }
                torch.save(
                    checkpoint_dict,
                    f"{checkpoint_dir}/{NAME_OF_CALCULATION}/checkpoint_{epoch}",
                )
                torch.save(
                    {
                        "checkpoint": checkpoint_dict,
                        "hypers": self.hypers,
                        "dataset_info": model.dataset_info,
                        "self_contributions": np.load(
                            self.pet_dir / "self_contributions.npy"  # type: ignore
                        ),
                    },
                    f"{checkpoint_dir}/model.ckpt_{epoch}",
                )

            if FITTING_SCHEME.MAX_TIME is not None:
                if elapsed > FITTING_SCHEME.MAX_TIME:
                    logging.info("Reached maximum time\n")
                    break
        logging.info("Training is finished\n")
        logging.info("Saving the model and history...")
        torch.save(
            {
                "model_state_dict": pet_model.state_dict(),
                "optim_state_dict": optim.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "dtype_used": dtype2string(dtype),
            },
            f"{checkpoint_dir}/{NAME_OF_CALCULATION}/checkpoint",
        )
        with open(f"{checkpoint_dir}/{NAME_OF_CALCULATION}/history.pickle", "wb") as f:
            pickle.dump(history, f)

        def save_model(model_name, model_keeper):
            torch.save(
                model_keeper.best_model.state_dict(),
                f"{checkpoint_dir}/{NAME_OF_CALCULATION}/{model_name}_state_dict",
            )

        summary = ""
        if MLIP_SETTINGS.USE_ENERGIES:
            if FITTING_SCHEME.ENERGIES_LOSS == "per_structure":
                postfix = "per structure"
            if FITTING_SCHEME.ENERGIES_LOSS == "per_atom":
                postfix = "per atom"
            save_model("best_val_mae_energies_model", energies_mae_model_keeper)
            summary += f"best val mae in energies {postfix}: "
            summary += f"{energies_mae_model_keeper.best_error} "
            summary += f"at epoch {energies_mae_model_keeper.best_epoch}\n"

            save_model("best_val_rmse_energies_model", energies_rmse_model_keeper)
            summary += f"best val rmse in energies {postfix}: "
            summary += f"{energies_rmse_model_keeper.best_error} "
            summary += f"at epoch {energies_rmse_model_keeper.best_epoch}\n"

        if MLIP_SETTINGS.USE_FORCES:
            save_model("best_val_mae_forces_model", forces_mae_model_keeper)
            summary += f"best val mae in forces: {forces_mae_model_keeper.best_error} "
            summary += f"at epoch {forces_mae_model_keeper.best_epoch}\n"

            save_model("best_val_rmse_forces_model", forces_rmse_model_keeper)
            summary += (
                f"best val rmse in forces: {forces_rmse_model_keeper.best_error} "
            )
            summary += f"at epoch {forces_rmse_model_keeper.best_epoch}\n"

        if MLIP_SETTINGS.USE_ENERGIES and MLIP_SETTINGS.USE_FORCES:
            save_model("best_val_mae_both_model", multiplication_mae_model_keeper)
            summary += f"best both (multiplication) mae in energies {postfix}: "
            summary += (
                f"{multiplication_mae_model_keeper.additional_info[0]} in forces: "
            )
            summary += f"{multiplication_mae_model_keeper.additional_info[1]} "
            summary += f"at epoch {multiplication_mae_model_keeper.best_epoch}\n"

            save_model("best_val_rmse_both_model", multiplication_rmse_model_keeper)
            summary += f"best both (multiplication) rmse in energies {postfix}: "
            summary += (
                f"{multiplication_rmse_model_keeper.additional_info[0]} in forces: "
            )
            summary += (
                f"{multiplication_rmse_model_keeper.additional_info[1]} at epoch "
            )
            summary += f"{multiplication_rmse_model_keeper.best_epoch}\n"

        with open(f"{checkpoint_dir}/{NAME_OF_CALCULATION}/summary.txt", "wb") as f:
            f.write(summary.encode())
        logging.info(f"Total elapsed time: {time.time() - TIME_SCRIPT_STARTED}")

        ##########################################
        # FINISHING THE PURE PET TRAINING SCRIPT #
        ##########################################

        if self.pet_checkpoint is not None:
            # remove the temporary file
            os.remove(Path(checkpoint_dir) / "checkpoint.temp")

        if do_forces:
            load_path = self.pet_dir / "best_val_rmse_forces_model_state_dict"
        else:
            load_path = self.pet_dir / "best_val_rmse_energies_model_state_dict"

        state_dict = torch.load(load_path, weights_only=False)

        ARCHITECTURAL_HYPERS = Hypers(model.hypers)
        raw_pet = PET(ARCHITECTURAL_HYPERS, 0.0, len(model.atomic_types))

        new_state_dict = {}
        for name, value in state_dict.items():
            name = name.replace("model.pet_model.", "")
            new_state_dict[name] = value

        raw_pet.load_state_dict(new_state_dict)

        self_contributions_path = self.pet_dir / "self_contributions.npy"
        self_contributions = np.load(self_contributions_path)
        wrapper = SelfContributionsWrapper(raw_pet, self_contributions)

        model.set_trained_model(wrapper)

    def save_checkpoint(self, model, path: Union[str, Path]):
        # This function takes a checkpoint from the PET folder and saves it
        # together with the hypers inside a file that will act as a metatrain
        # checkpoint
        checkpoint_path = self.pet_dir / "checkpoint"  # type: ignore
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        torch.save(
            {
                "checkpoint": checkpoint,
                "hypers": self.hypers,
                "dataset_info": model.dataset_info,
                "self_contributions": np.load(
                    self.pet_dir / "self_contributions.npy"  # type: ignore
                ),
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path], train_hypers) -> "Trainer":
        # This function loads a metatrain PET checkpoint and returns a Trainer
        # instance with the hypers, while also saving the checkpoint in the
        # class
        checkpoint = torch.load(path, weights_only=False)
        trainer = cls(train_hypers)
        trainer.pet_checkpoint = checkpoint["checkpoint"]
        return trainer
