"""Standalone tests for the ``explicit_gradients`` support added to the
``energy_ensemble`` output of :class:`LLPRUncertaintyModel` (see
``_add_energy_ensemble_gradients``). These do not share fixtures with
``test_llpr.py``/``test_basic.py``: they train their own minimal PET+LLPR model and
call ``LLPRUncertaintyModel.forward`` directly, to stay fast and focused on this one
feature.
"""

import copy
import random
import tempfile

import numpy as np
import pytest
import torch
from metatomic.torch import (
    ModelEvaluationOptions,
    ModelOutput,
    System,
    load_atomistic_model,
)
from omegaconf import OmegaConf

from metatrain.llpr import LLPRUncertaintyModel
from metatrain.llpr import Trainer as LLPRTrainer
from metatrain.pet import PET
from metatrain.pet import Trainer as PETTrainer
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo, get_atomic_types, get_dataset
from metatrain.utils.data.readers import read_systems
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

from . import DATASET_WITH_FORCES_PATH


# Kept small: `_add_energy_ensemble_gradients` loops once per member, so runtime
# scales linearly with this. Only needs to be large enough to check that members
# actually differ from one another (see test_gradients_vary_across_members).
NUM_ENSEMBLE_MEMBERS = 8

DTYPE = torch.float64


def _build_llpr_model() -> LLPRUncertaintyModel:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    dataset_targets = {
        "energy": {
            "quantity": "energy",
            "read_from": DATASET_WITH_FORCES_PATH,
            "reader": "ase",
            "key": "energy",
            "unit": "eV",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    dataset, targets_info, _ = get_dataset(
        {
            "systems": {"read_from": DATASET_WITH_FORCES_PATH, "reader": "ase"},
            "targets": dataset_targets,
        }
    )
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=get_atomic_types(dataset),
        targets=targets_info,
    )

    pet_hypers = copy.deepcopy(get_default_hypers("pet"))
    pet_model_hypers = pet_hypers["model"]
    pet_model_hypers.update(
        d_pet=1,
        d_head=1,
        d_node=1,
        d_feedforward=1,
        num_heads=1,
        num_attention_layers=1,
        num_gnn_layers=1,
    )
    pet_model = PET(pet_model_hypers, dataset_info)

    pet_hypers["training"]["num_epochs"] = 1
    loss_hypers = OmegaConf.create(
        {k: init_with_defaults(LossSpecification) for k in dataset_targets}
    )
    pet_hypers["training"]["loss"] = OmegaConf.to_container(loss_hypers, resolve=True)

    pet_trainer = PETTrainer(pet_hypers["training"])
    pet_trainer.train(
        pet_model,
        dtype=DTYPE,
        devices=[torch.device("cpu")],
        train_datasets=[dataset],
        val_datasets=[dataset],
        checkpoint_dir="",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = f"{tmpdir}/pet_checkpoint.ckpt"
        pet_trainer.save_checkpoint(pet_model, checkpoint_path)

        llpr_hypers = copy.deepcopy(get_default_hypers("llpr"))
        llpr_hypers["model"]["num_ensemble_members"] = {"energy": NUM_ENSEMBLE_MEMBERS}
        llpr_model = LLPRUncertaintyModel(llpr_hypers["model"], dataset_info)

        llpr_hypers["training"]["model_checkpoint"] = checkpoint_path
        llpr_hypers["training"]["batch_size"] = 4
        llpr_trainer = LLPRTrainer(llpr_hypers["training"])
        llpr_trainer.train(
            llpr_model,
            dtype=DTYPE,
            devices=[torch.device("cpu")],
            train_datasets=[dataset],
            val_datasets=[dataset],
            checkpoint_dir="",
        )

    return llpr_model


@pytest.fixture(scope="module")
def llpr_model() -> LLPRUncertaintyModel:
    return _build_llpr_model()


def _grad_systems(model: LLPRUncertaintyModel, n: int):
    """Fresh `System`s (periodic, from the carbon dataset) with positions/cell as
    leaf tensors requiring grad, and neighbor lists attached -- exactly what a
    caller like `metatomic_ase.MetatomicCalculator.run_model` sets up before
    invoking the model with a non-empty `explicit_gradients` request.
    """
    systems = [s.to(DTYPE) for s in read_systems(DATASET_WITH_FORCES_PATH)[:n]]
    requested_neighbor_lists = get_requested_neighbor_lists(model)
    grad_systems = []
    for system in systems:
        positions = system.positions.clone().detach().requires_grad_(True)
        cell = system.cell.clone().detach().requires_grad_(True)
        new_system = System(system.types, positions, cell, system.pbc)
        get_system_with_neighbor_lists(new_system, requested_neighbor_lists)
        grad_systems.append(new_system)
    return grad_systems


def _plain_systems(model: LLPRUncertaintyModel, n: int):
    """Fresh `System`s as an engine hands them over when it is NOT doing its own
    autograd: plain positions/cell (no grad), neighbor lists attached but not
    autograd-registered. Explicit gradients then rely entirely on the in-forward
    rebuild (`_systems_with_grad`).
    """
    systems = [s.to(DTYPE) for s in read_systems(DATASET_WITH_FORCES_PATH)[:n]]
    requested_neighbor_lists = get_requested_neighbor_lists(model)
    for system in systems:
        assert not system.positions.requires_grad
        get_system_with_neighbor_lists(system, requested_neighbor_lists)
    return systems


def _reference_energy_gradients(model, systems):
    """dE/dr and the virial-style dE/dstrain (positions^T @ dE/dr + cell^T @ dE/dcell,
    evaluated at strain=identity) for the plain, deterministic "energy" output --
    computed independently of `_add_energy_ensemble_gradients`, via a direct
    autograd.grad call on this output alone.
    """
    result = model(systems, {"energy": ModelOutput(sample_kind="system")})
    energy = result["energy"].block().values.sum()

    inputs = [s.positions for s in systems] + [s.cell for s in systems]
    grads = torch.autograd.grad(energy, inputs)
    n = len(systems)
    pos_grads = grads[:n]
    cell_grads = grads[n:]

    positions_grad = torch.cat(pos_grads, dim=0)
    strain_grad = torch.stack(
        [
            systems[i].positions.detach().t() @ pos_grads[i]
            + systems[i].cell.detach().t() @ cell_grads[i]
            for i in range(n)
        ]
    )
    return positions_grad, strain_grad


def test_energy_ensemble_capabilities(llpr_model):
    """Only the core "energy_ensemble" output should declare explicit_gradients;
    it's the only ensemble quantity `_add_energy_ensemble_gradients` supports."""
    outputs = llpr_model.capabilities.outputs
    assert outputs["energy_ensemble"].explicit_gradients == ["positions", "strain"]
    for name, output in outputs.items():
        if name != "energy_ensemble":
            assert output.explicit_gradients == []


def test_energy_ensemble_properties(llpr_model):
    systems = _grad_systems(llpr_model, n=1)
    result = llpr_model(systems, {"energy_ensemble": ModelOutput(sample_kind="system")})
    block = result["energy_ensemble"].block()
    assert block.properties.names == ["energy"]
    assert block.values.shape == (1, NUM_ENSEMBLE_MEMBERS)


def test_energy_ensemble_gradients_match_energy_mean(llpr_model):
    positions_ref, strain_ref = _reference_energy_gradients(
        llpr_model, _grad_systems(llpr_model, n=2)
    )

    systems = _grad_systems(llpr_model, n=2)
    result = llpr_model(
        systems,
        {
            "energy_ensemble": ModelOutput(
                sample_kind="system", explicit_gradients=["positions", "strain"]
            )
        },
    )
    block = result["energy_ensemble"].block()

    positions_grad = block.gradient("positions").values.mean(dim=-1)
    assert torch.allclose(positions_grad, positions_ref, atol=1e-5, rtol=1e-5)

    strain_grad = block.gradient("strain").values.mean(dim=-1)
    assert torch.allclose(strain_grad, strain_ref, atol=1e-5, rtol=1e-5)


def test_energy_ensemble_gradients_vary_across_members(llpr_model):
    systems = _grad_systems(llpr_model, n=2)
    result = llpr_model(
        systems,
        {
            "energy_ensemble": ModelOutput(
                sample_kind="system", explicit_gradients=["positions", "strain"]
            )
        },
    )
    block = result["energy_ensemble"].block()

    positions_grad = block.gradient("positions").values
    assert positions_grad.std(dim=-1).abs().max() > 1e-8

    strain_grad = block.gradient("strain").values
    assert strain_grad.std(dim=-1).abs().max() > 1e-8


@pytest.mark.parametrize(
    "explicit_gradients", [[], ["positions"], ["strain"], ["positions", "strain"]]
)
def test_energy_ensemble_gradients_partial_request(llpr_model, explicit_gradients):
    systems = _grad_systems(llpr_model, n=1)
    result = llpr_model(
        systems,
        {
            "energy_ensemble": ModelOutput(
                sample_kind="system", explicit_gradients=explicit_gradients
            )
        },
    )
    gradients_list = result["energy_ensemble"].block().gradients_list()
    assert ("positions" in gradients_list) == ("positions" in explicit_gradients)
    assert ("strain" in gradients_list) == ("strain" in explicit_gradients)


def test_energy_ensemble_gradients_batch(llpr_model):
    """A batch of independent systems must give the same per-system gradients as
    running each system through the model on its own -- guards against cross-system
    mixing in `_add_energy_ensemble_gradients`'s atom-offset bookkeeping."""
    batched_systems = _grad_systems(llpr_model, n=2)
    batched_result = llpr_model(
        batched_systems,
        {
            "energy_ensemble": ModelOutput(
                sample_kind="system", explicit_gradients=["positions", "strain"]
            )
        },
    )
    batched_block = batched_result["energy_ensemble"].block()
    n_atoms_0 = batched_systems[0].positions.shape[0]

    single_systems = _grad_systems(llpr_model, n=1)
    single_result = llpr_model(
        single_systems,
        {
            "energy_ensemble": ModelOutput(
                sample_kind="system", explicit_gradients=["positions", "strain"]
            )
        },
    )
    single_block = single_result["energy_ensemble"].block()

    assert torch.allclose(
        batched_block.gradient("positions").values[:n_atoms_0],
        single_block.gradient("positions").values,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        batched_block.gradient("strain").values[0],
        single_block.gradient("strain").values[0],
        atol=1e-6,
        rtol=1e-6,
    )


def test_energy_ensemble_gradients_plain_systems(llpr_model):
    """Explicit gradients must not depend on the caller having enabled grad:
    engine-style plain systems go through the in-forward rebuild
    (`_systems_with_grad`) and must give the same gradients as caller-prepared
    grad-enabled systems. This is the contract that lets any engine request
    ensemble gradients without engine-side autograd setup."""
    outputs = {
        "energy_ensemble": ModelOutput(
            sample_kind="system", explicit_gradients=["positions", "strain"]
        )
    }
    ref_block = llpr_model(_grad_systems(llpr_model, n=2), outputs)[
        "energy_ensemble"
    ].block()
    block = llpr_model(_plain_systems(llpr_model, n=2), outputs)[
        "energy_ensemble"
    ].block()

    assert torch.allclose(block.values, ref_block.values, atol=1e-10, rtol=1e-10)
    for gradient_name in ("positions", "strain"):
        assert torch.allclose(
            block.gradient(gradient_name).values,
            ref_block.gradient(gradient_name).values,
            atol=1e-10,
            rtol=1e-10,
        )


def test_energy_ensemble_gradients_torchscript(llpr_model, tmp_path):
    """The whole explicit-gradient path -- including the in-forward system rebuild
    -- must survive `AtomisticModel.save()`'s `torch.jit.script` and produce the
    same gradients from a re-loaded scripted model fed engine-style plain systems.
    This is the deployment path (export -> load in an engine) the feature exists
    for; the other tests only exercise the eager Python model."""
    outputs = {
        "energy_ensemble": ModelOutput(
            sample_kind="system", explicit_gradients=["positions", "strain"]
        )
    }
    ref_block = llpr_model(_grad_systems(llpr_model, n=2), outputs)[
        "energy_ensemble"
    ].block()

    path = str(tmp_path / "llpr_ensemble.pt")
    llpr_model.export().save(path)
    loaded = load_atomistic_model(path)

    options = ModelEvaluationOptions(
        length_unit="angstrom", outputs=outputs, selected_atoms=None
    )
    result = loaded(_plain_systems(llpr_model, n=2), options, check_consistency=True)
    block = result["energy_ensemble"].block()

    assert torch.allclose(block.values, ref_block.values, atol=1e-10, rtol=1e-10)
    for gradient_name in ("positions", "strain"):
        assert torch.allclose(
            block.gradient(gradient_name).values,
            ref_block.gradient(gradient_name).values,
            atol=1e-10,
            rtol=1e-10,
        )
