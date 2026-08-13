import copy
import re

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
    message = (
        "NEP4 has a single global bias, but the folded per-type constants "
        "differ (spread 5.500e+00). Export this model as NEP5 with "
        "`export_nep(path, version=5)`, whose per-type bias makes the "
        "composition fold exact for multi-element models."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        model.export_nep(tmp_path / "nep.txt")


def test_export_nep_zbl(tmp_path):
    """ZBL models export at any scale: the ZBL term is an additive
    contribution excluded from the scaler, matching native NEP."""
    model = _make_model([6], 4, scale=0.6, composition={6: -2.0}, zbl=2.0)
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


@pytest.mark.parametrize("version", [3, 4])
def test_export_as_nep5_folds_per_type_composition(tmp_path, version, caplog):
    """NEP3/NEP4 models can be written as NEP5, whose per-type bias absorbs
    per-type composition baselines that NEP3/NEP4 cannot represent."""
    atomic_types = [6, 14]
    model = _make_model(
        atomic_types, version, scale=0.6, composition={6: -2.0, 14: -7.5}
    )
    path = tmp_path / "nep.txt"
    model.export_nep(path, version=5)

    assert load_nep(str(path)).version == 5
    system = _test_system(atomic_types)
    e_model = _model_per_atom_energies(model, system)
    e_native = _nep_txt_per_atom_energies(path, system, atomic_types)
    assert torch.allclose(
        e_model, e_native, rtol=1e-6, atol=1e-5 * float(e_model.abs().max())
    )


def test_export_as_nep5_is_the_same_potential(tmp_path):
    """Promoting to NEP5 does not change the potential itself."""
    model = _make_model([6, 14], 4, scale=1.0, composition={6: 0.0, 14: 0.0})
    plain = tmp_path / "nep4.txt"
    promoted = tmp_path / "nep5.txt"
    model.export_nep(plain)
    model.export_nep(promoted, version=5)

    system = _test_system([6, 14])
    e_plain = _nep_txt_per_atom_energies(plain, system, [6, 14])
    e_promoted = _nep_txt_per_atom_energies(promoted, system, [6, 14])
    assert torch.allclose(e_plain, e_promoted, rtol=1e-10, atol=1e-10)


def test_export_nep_neighbor_capacity_from_hypers(tmp_path):
    """`mn_radial`/`mn_angular` end up on the `cutoff` line of the nep.txt."""
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["mn_radial"] = 173
    hypers["mn_angular"] = 91
    model = NEP(hypers, _dataset_info([6])).to(torch.float64)

    path = tmp_path / "nep.txt"
    model.export_nep(path)

    params = load_nep(str(path))
    assert (params.mn_radial, params.mn_angular) == (173, 91)


def test_loaded_nep_neighbor_capacity_comes_from_hypers(tmp_path):
    """A loaded nep.txt does not carry its own capacity into the export."""
    source = tmp_path / "source.txt"
    _make_model([6], 4, scale=1.0, composition={6: 0.0}).export_nep(source)
    with open(source) as fd:
        lines = fd.read().splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("cutoff "))
    lines[index] = " ".join(lines[index].split()[:-2] + ["100", "20"])
    with open(source, "w") as fd:
        fd.write("\n".join(lines) + "\n")

    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["nep_model"] = str(source)
    hypers["mn_radial"] = 250
    hypers["mn_angular"] = 125
    model = NEP(hypers, _dataset_info([6])).to(torch.float64)

    exported = tmp_path / "nep.txt"
    model.export_nep(exported)
    params = load_nep(str(exported))
    assert (params.mn_radial, params.mn_angular) == (250, 125)


@pytest.mark.parametrize("key", ["mn_radial", "mn_angular"])
def test_nonpositive_neighbor_capacity_raises(key):
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers[key] = 0
    with pytest.raises(ValueError, match="must be positive"):
        NEP(hypers, _dataset_info([6]))


def test_export_unsupported_version_raises(tmp_path):
    model = _make_model([6], 4, scale=1.0, composition={6: 0.0})
    message = (
        "Cannot export a NEP4 model as NEP3: only exporting NEP3 and NEP4 "
        "models as NEP5 is supported."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        model.export_nep(tmp_path / "nep.txt", version=3)
