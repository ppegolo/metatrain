"""Tests for the tensorial NEP models (dipole and polarizability)."""

import copy
import math
import re

import ase
import ase.io
import pytest
import torch
from metatomic.torch import System
from omegaconf import OmegaConf

from metatrain.experimental.nep.model import NEP
from metatrain.experimental.nep.trainer import Trainer
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import get_dataset
from metatrain.utils.data.dataset import DatasetInfo
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

from . import MODEL_HYPERS


TARGET = "mtt::tensor"
ATOMIC_TYPES = [1, 8]


def _target_info(rank, sample_kind="system", num_subtargets=1):
    return get_generic_target_info(
        TARGET,
        OmegaConf.create(
            {
                "quantity": "",
                "unit": "",
                "type": {"cartesian": {"rank": rank}},
                "sample_kind": sample_kind,
                "num_subtargets": num_subtargets,
            }
        ),
    )


def _dataset_info(rank, sample_kind="system", num_subtargets=1):
    return DatasetInfo(
        length_unit="angstrom",
        atomic_types=ATOMIC_TYPES,
        targets={TARGET: _target_info(rank, sample_kind, num_subtargets)},
    )


def _model(model_type, sample_kind="system", **hyper_overrides):
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["model_type"] = model_type
    hypers["cutoff_radial"] = 5.0
    hypers["cutoff_angular"] = 4.0
    hypers.update(hyper_overrides)
    rank = 1 if model_type == 1 else 2
    return NEP(hypers, _dataset_info(rank, sample_kind)).to(torch.float64)


def _atoms(seed=0):
    """Small periodic water-like structure."""
    rng = torch.Generator().manual_seed(seed)
    positions = (torch.rand(6, 3, generator=rng, dtype=torch.float64) * 4.0).numpy()
    return ase.Atoms(
        numbers=[8, 1, 1, 8, 1, 1], positions=positions, cell=[6.0, 6.0, 6.0], pbc=True
    )


def _system(atoms, model):
    system = System(
        types=torch.tensor(atoms.numbers, dtype=torch.int32),
        positions=torch.tensor(atoms.get_positions(), dtype=torch.float64),
        cell=torch.tensor(atoms.get_cell().array, dtype=torch.float64),
        pbc=torch.tensor([True, True, True]),
    )
    return get_system_with_neighbor_lists(system, get_requested_neighbor_lists(model))


def _predict(model, atoms):
    """Total (summed over atoms) prediction in evaluation mode."""
    model.eval()
    with torch.no_grad():
        out = model([_system(atoms, model)], {TARGET: model.outputs[TARGET]})
    return out[TARGET].block().values.sum(dim=0).squeeze(-1)


@pytest.mark.parametrize("model_type, rank", [(1, 1), (2, 2)])
@pytest.mark.parametrize("sample_kind", ["system", "atom"])
def test_output_layout(model_type, rank, sample_kind):
    """Dipoles are Cartesian vectors and polarizabilities rank-2 tensors."""
    model = _model(model_type, sample_kind)
    atoms = _atoms()

    out = model([_system(atoms, model)], {TARGET: model.outputs[TARGET]})
    block = out[TARGET].block()

    assert block.values.shape == (len(atoms),) + (3,) * rank + (1,)
    assert [component.names for component in block.components] == (
        [["xyz"]] if rank == 1 else [["xyz_1"], ["xyz_2"]]
    )


@pytest.mark.parametrize("model_type", [1, 2])
def test_per_atom_sums_to_per_structure(model_type):
    """A per-atom model predicts the same total as a per-structure one."""
    # both are built from the same `seed` hyper, so the networks are identical
    # (their scalers are not: a per-atom target gets one scale per type)
    per_structure = _model(model_type, "system")
    per_atom = _model(model_type, "atom")
    assert torch.equal(per_atom.potential.ann, per_structure.potential.ann)
    atoms = _atoms()

    assert torch.allclose(_predict(per_atom, atoms), _predict(per_structure, atoms))


@pytest.mark.parametrize("model_type", [1, 2])
def test_rotational_equivariance(model_type):
    model = _model(model_type)
    atoms = _atoms()
    predicted = _predict(model, atoms)

    angle = 0.7
    rotation = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    rotated = atoms.copy()
    rotated.set_positions(atoms.get_positions() @ rotation.numpy().T)
    rotated.set_cell(atoms.get_cell().array @ rotation.numpy().T)

    expected = (
        rotation @ predicted
        if model_type == 1
        else rotation @ predicted @ rotation.transpose(0, 1)
    )
    assert torch.allclose(_predict(model, rotated), expected, atol=1e-10)


@pytest.mark.parametrize("model_type", [1, 2])
def test_export_matches_nep_txt(model_type, tmp_path):
    """The exported nep.txt reproduces the model, target scale included."""
    calorine = pytest.importorskip("calorine")
    from calorine.calculators import CPUNEP

    model = _model(model_type)
    atoms = _atoms()
    unscaled = _predict(model, atoms)
    # note: `sync_tensor_maps` would reload the scales from the buffers and
    # undo this, making the test vacuous
    model.scaler.model.scales[TARGET].block().values[:] = 2.5

    path = tmp_path / "nep.txt"
    model.export_nep(path)

    predicted = _predict(model, atoms)
    # the scale really is part of what is being checked below
    assert torch.allclose(predicted, 2.5 * unscaled)
    atoms.calc = CPUNEP(str(path))
    reference = (
        atoms.get_dipole_moment()
        if model_type == 1
        else atoms.calc.get_polarizability(atoms)
    )
    reference = torch.tensor(reference, dtype=torch.float64).reshape(predicted.shape)

    # the file format keeps 7 digits
    assert torch.allclose(predicted, reference, rtol=1e-5, atol=1e-6)
    assert calorine  # silence the unused-import linter


def test_per_atom_dipole_export_folds_per_type_scales(tmp_path):
    """A per-atom target has one scale per type, which NEP4 can fold exactly.

    The scaler labels those scales with the *position* of the type in its
    sorted type list, not with the atomic number, so this also pins down the
    mapping used by ``_energy_scales_per_type``.
    """
    pytest.importorskip("calorine")
    from calorine.calculators import CPUNEP

    model = _model(1, "atom")
    scales = model.scaler.model.scales[TARGET].block()
    assert scales.samples.names == ["atomic_type"]
    assert scales.values.shape[0] == len(ATOMIC_TYPES)
    # note: `sync_tensor_maps` would undo this
    scales.values[0] = 1.5  # hydrogen, the first of the sorted types
    scales.values[1] = 2.5  # oxygen

    path = tmp_path / "nep.txt"
    model.export_nep(path)

    atoms = _atoms()
    predicted = _predict(model, atoms)
    atoms.calc = CPUNEP(str(path))
    reference = torch.tensor(atoms.get_dipole_moment(), dtype=torch.float64)

    assert torch.allclose(predicted, reference, rtol=1e-5, atol=1e-6)


def test_polarizability_per_type_scale_export_raises(tmp_path):
    """Per-type scales cannot be folded into the single isotropic bias."""
    model = _model(2, "atom")
    scales = model.scaler.model.scales[TARGET].block()
    if scales.values.shape[0] < 2:
        pytest.skip("this target has a single global scale")
    scales.values[0] = 1.5
    scales.values[1] = 2.5

    with pytest.raises(ValueError, match="single global bias"):
        model.export_nep(tmp_path / "nep.txt")


def test_scalar_target_with_tensorial_model_raises():
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["model_type"] = 1
    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=ATOMIC_TYPES,
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )
    message = (
        "A NEP model with `model_type: 1` predicts a dipole, so its target must "
        "be a Cartesian tensor of rank 1, but the target 'energy' is not."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        NEP(hypers, dataset_info)


def test_tensorial_target_with_energy_model_raises():
    message = (
        "A NEP model with `model_type: 0` can only predict scalars, but the "
        "target 'mtt::tensor' is not a scalar. Use `model_type: 1` for dipoles "
        "or `model_type: 2` for polarizabilities."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        NEP(copy.deepcopy(MODEL_HYPERS), _dataset_info(rank=1))


def test_wrong_rank_raises():
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["model_type"] = 2
    with pytest.raises(ValueError, match="Cartesian tensor of rank 2"):
        NEP(hypers, _dataset_info(rank=1))


def test_multiple_properties_raise():
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["model_type"] = 1
    with pytest.raises(ValueError, match="single property"):
        NEP(hypers, _dataset_info(rank=1, num_subtargets=5))


@pytest.mark.parametrize(
    "override, message",
    [
        ({"version": 5}, "require `version: 4`"),
        ({"charge_mode": 1}, "cannot use NEP-Charge"),
        ({"zbl_outer_cutoff": 2.0}, "cannot use ZBL"),
    ],
)
def test_unsupported_combinations_raise(override, message):
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["model_type"] = 1
    hypers.update(override)
    with pytest.raises(ValueError, match=message):
        NEP(hypers, _dataset_info(rank=1))


@pytest.mark.parametrize("model_type", [1, 2])
def test_checkpoint_roundtrip(model_type):
    """`model_type` survives a checkpoint, and so do the predictions."""
    model = _model(model_type)
    atoms = _atoms()

    checkpoint = model.get_checkpoint()
    assert checkpoint["model_data"]["model_hypers"]["model_type"] == model_type
    rebuilt = NEP.load_checkpoint(checkpoint, "restart").to(torch.float64)

    assert rebuilt.model_type == model_type
    assert torch.allclose(_predict(rebuilt, atoms), _predict(model, atoms))


@pytest.mark.parametrize("model_type", [1, 2])
def test_torchscript(model_type):
    model = _model(model_type)
    atoms = _atoms()
    system = _system(atoms, model)

    scripted = torch.jit.script(model)
    expected = model([system], {TARGET: model.outputs[TARGET]})[TARGET].block().values
    obtained = (
        scripted([system], {TARGET: model.outputs[TARGET]})[TARGET].block().values
    )

    assert torch.allclose(obtained, expected)


def _write_dataset(path, sample_kind, rank):
    frames = []
    for seed in range(5):
        atoms = _atoms(seed)
        values = torch.rand(
            [len(atoms)] + [3] * rank if sample_kind == "atom" else [3] * rank,
            generator=torch.Generator().manual_seed(seed),
            dtype=torch.float64,
        ).numpy()
        if sample_kind == "atom":
            atoms.arrays[TARGET.replace("mtt::", "")] = values.reshape(len(atoms), -1)
        else:
            atoms.info[TARGET.replace("mtt::", "")] = values.reshape(-1)
        frames.append(atoms)
    ase.io.write(path, frames)
    return path


@pytest.mark.parametrize("model_type, rank", [(1, 1), (2, 2)])
@pytest.mark.parametrize("sample_kind", ["system", "atom"])
def test_training(model_type, rank, sample_kind, tmp_path):
    """A tensorial model trains end to end, composition model included."""
    path = _write_dataset(tmp_path / "data.xyz", sample_kind, rank)
    dataset, targets_info, _ = get_dataset(
        {
            "systems": {"read_from": str(path), "reader": "ase"},
            "targets": {
                TARGET: {
                    "quantity": "",
                    "read_from": str(path),
                    "reader": "ase",
                    "key": TARGET.replace("mtt::", ""),
                    "unit": "",
                    "type": {"cartesian": {"rank": rank}},
                    "sample_kind": sample_kind,
                    "num_subtargets": 1,
                }
            },
        }
    )
    dataset_info = DatasetInfo(
        length_unit="angstrom", atomic_types=ATOMIC_TYPES, targets=targets_info
    )

    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["model_type"] = model_type
    hypers["cutoff_radial"] = 5.0
    hypers["cutoff_angular"] = 4.0
    model = NEP(hypers, dataset_info).to(torch.float64)

    train_hypers = copy.deepcopy(get_default_hypers("experimental.nep")["training"])
    train_hypers["num_epochs"] = 2
    train_hypers["batch_size"] = 2
    train_hypers["loss"] = {TARGET: init_with_defaults(LossSpecification)}

    before = model.potential.ann.clone()
    Trainer(train_hypers).train(
        model,
        torch.float64,
        [torch.device("cpu")],
        [dataset],
        [dataset],
        str(tmp_path),
    )

    assert not torch.allclose(model.potential.ann, before)
    # the composition model does not support Cartesian targets and is skipped
    assert TARGET not in model.additive_models[0].model.weights
