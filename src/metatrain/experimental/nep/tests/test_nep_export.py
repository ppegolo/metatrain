import copy

import pytest
import torch
from metatomic.torch import System
from torchnep import NepPotential, build_neighbor_lists, load_nep

from metatrain.experimental.nep.model import NEP
from metatrain.utils.data.dataset import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

from . import MODEL_HYPERS


def _dataset_info(atomic_types):
    return DatasetInfo(
        length_unit="angstrom",
        atomic_types=atomic_types,
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )


def _make_model(atomic_types, version, scale, composition, zbl=None, charge_mode=0):
    """Build a NEP model and manually set scaler + composition values."""
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["version"] = version
    hypers["cutoff_radial"] = 5.0
    hypers["cutoff_angular"] = 4.0
    hypers["charge_mode"] = charge_mode
    if zbl is not None:
        hypers["zbl_outer_cutoff"] = zbl
    model = NEP(hypers, _dataset_info(atomic_types)).to(torch.float64)

    model.scaler.model.scales["energy"].block().values[:] = scale
    comp_block = model.additive_models[0].model.weights["energy"].block()
    for row, z in enumerate(comp_block.samples.values[:, 0].tolist()):
        comp_block.values[row, 0] = composition[int(z)]
    return model


def _test_system(atomic_types, dtype=torch.float64):
    """Small periodic system containing all the given atomic types."""
    torch.manual_seed(0)
    n_atoms = 8
    types = torch.tensor([atomic_types[i % len(atomic_types)] for i in range(n_atoms)])
    positions = torch.rand(n_atoms, 3, dtype=dtype) * 4.0
    cell = torch.eye(3, dtype=dtype) * 4.5
    return System(
        types=types,
        positions=positions,
        cell=cell,
        pbc=torch.tensor([True, True, True]),
    )


def _nep_txt_per_atom_energies(path, system, atomic_types):
    params = load_nep(str(path))
    potential = NepPotential(params)
    type_ids = torch.tensor(
        [atomic_types.index(int(z)) for z in system.types], dtype=torch.long
    )
    radial_edges, radial_shifts, angular_edges, angular_shifts = build_neighbor_lists(
        system.positions,
        system.cell,
        type_ids,
        potential.rc_radial,
        potential.rc_angular,
        pbc=(True, True, True),
    )
    return potential(
        system.positions,
        system.cell,
        type_ids,
        radial_edges,
        angular_edges,
        filter_edges=True,
        radial_shifts=radial_shifts,
        angular_shifts=angular_shifts,
    ).detach()


def _model_per_atom_energies(model, system):
    system = get_system_with_neighbor_lists(system, get_requested_neighbor_lists(model))
    model.eval()
    out = model([system], model.outputs)
    return out["energy"].block().values.squeeze(-1).detach()


@pytest.mark.parametrize(
    "version, atomic_types, composition",
    [
        (3, [6], {6: -3.5}),
        (4, [6], {6: -3.5}),
        (4, [6, 14], {6: -2.0, 14: -2.0}),  # uniform composition: foldable
        (5, [6, 14], {6: -2.0, 14: -7.5}),  # per-type composition: NEP5 only
    ],
)
def test_export_nep_matches_model(tmp_path, version, atomic_types, composition):
    """Native evaluation of the exported nep.txt reproduces the metatrain
    predictions, including scaler and composition contributions."""
    # non-uniform composition with a global scale requires uniform
    # s*b1 - c_t for NEP3/4, so those cases use uniform composition
    model = _make_model(atomic_types, version, scale=0.6, composition=composition)
    path = tmp_path / "nep.txt"
    model.export_nep(path)

    system = _test_system(atomic_types)
    e_model = _model_per_atom_energies(model, system)
    e_native = _nep_txt_per_atom_energies(path, system, atomic_types)

    # nep.txt stores values with 7 significant digits
    assert torch.allclose(
        e_model, e_native, rtol=1e-6, atol=1e-5 * float(e_model.abs().max())
    )


def test_export_nep4_nonuniform_composition_raises(tmp_path):
    """NEP4 has a single global bias: per-type composition cannot be folded."""
    model = _make_model([6, 14], 4, scale=0.6, composition={6: -2.0, 14: -7.5})
    with pytest.raises(ValueError, match="version: 5"):
        model.export_nep(tmp_path / "nep.txt")


@pytest.mark.parametrize("scale", [1.0, 0.6])
def test_export_nep_zbl(tmp_path, scale):
    """ZBL models export at any scale: the ZBL term is an additive
    contribution excluded from the scaler, matching native NEP."""
    model = _make_model([6], 4, scale=scale, composition={6: -2.0}, zbl=2.0)
    path = tmp_path / "nep.txt"
    model.export_nep(path)

    with open(path) as fd:
        assert fd.readline().startswith("nep4_zbl")

    system = _test_system([6])
    e_model = _model_per_atom_energies(model, system)
    e_native = _nep_txt_per_atom_energies(path, system, [6])
    assert torch.allclose(
        e_model, e_native, rtol=1e-6, atol=1e-5 * float(e_model.abs().max())
    )


def test_zbl_torchscript():
    """The ZBL additive model is TorchScript-compatible (needed for export)."""
    model = _make_model([6, 14], 4, scale=1.0, composition={6: 0.0, 14: 0.0}, zbl=2.0)
    system = get_system_with_neighbor_lists(
        _test_system([6, 14]), get_requested_neighbor_lists(model)
    )
    model.eval()
    e_eager = model([system], model.outputs)["energy"].block().values
    scripted = torch.jit.script(model)
    e_scripted = scripted([system], model.outputs)["energy"].block().values
    # close pairs in the random box give large ZBL energies; the TorchScript
    # executor introduces ~1e-8 relative noise
    assert torch.allclose(e_eager, e_scripted, rtol=1e-6, atol=1e-8)
