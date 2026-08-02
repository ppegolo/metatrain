import copy

import pytest
import torch
from metatomic.torch import System

from metatrain.experimental.nep.model import NEP
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import MODEL_HYPERS
from .test_nep_export import (
    _dataset_info,
    _make_model,
    _model_per_atom_energies,
    _nep_txt_per_atom_energies,
    _test_system,
)


def _qnep_model(atomic_types, charge_mode, scale=1.0, composition=None):
    if composition is None:
        composition = {z: 0.0 for z in atomic_types}
    return _make_model(
        atomic_types,
        version=4,
        scale=scale,
        composition=composition,
        charge_mode=charge_mode,
    )


@pytest.mark.parametrize("charge_mode", [1, 2, 3])
def test_qnep_forward(charge_mode):
    """qNEP models produce finite per-atom energies on periodic systems."""
    model = _qnep_model([6, 14], charge_mode)
    energies = _model_per_atom_energies(model, _test_system([6, 14]))
    assert torch.isfinite(energies).all()
    # charge terms change the prediction with respect to a regular NEP with
    # the same energy-head parameters being absent from a plain model
    plain = _make_model([6, 14], version=4, scale=1.0, composition={6: 0.0, 14: 0.0})
    e_plain = _model_per_atom_energies(plain, _test_system([6, 14]))
    assert not torch.allclose(energies, e_plain)


def test_qnep_nonperiodic_raises():
    model = _qnep_model([6, 14], 2)
    system = System(
        types=torch.tensor([6, 14]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]], dtype=torch.float64),
        cell=torch.zeros(3, 3, dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    model.eval()
    with pytest.raises(ValueError, match="periodic"):
        model([system], model.outputs)


def test_qnep_requires_version_4():
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["version"] = 5
    hypers["charge_mode"] = 2
    with pytest.raises(ValueError, match="version: 4"):
        NEP(hypers, _dataset_info([6]))


def test_qnep_torchscript():
    model = _qnep_model([6, 14], 2)
    system = get_system_with_neighbor_lists(
        _test_system([6, 14]), model.requested_neighbor_lists()
    )
    model.eval()
    e_eager = model([system], model.outputs)["energy"].block().values
    scripted = torch.jit.script(model)
    e_scripted = scripted([system], model.outputs)["energy"].block().values
    assert torch.allclose(e_eager, e_scripted, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("charge_mode", [1, 2, 3])
def test_qnep_export_roundtrip(tmp_path, charge_mode):
    """Exported qNEP files reproduce the metatrain predictions natively."""
    model = _qnep_model([6, 14], charge_mode, composition={6: -2.0, 14: -2.0})
    path = tmp_path / "nep.txt"
    model.export_nep(path)

    with open(path) as fd:
        assert fd.readline().startswith(f"nep4_charge{charge_mode}")

    system = _test_system([6, 14])
    e_model = _model_per_atom_energies(model, system)
    e_native = _nep_txt_per_atom_energies(path, system, [6, 14])
    assert torch.allclose(
        e_model, e_native, rtol=1e-6, atol=1e-5 * float(e_model.abs().max())
    )


def test_qnep_export_with_scale_raises(tmp_path):
    model = _qnep_model([6, 14], 2, scale=0.6)
    with pytest.raises(ValueError, match="scale_targets"):
        model.export_nep(tmp_path / "nep.txt")


def test_qnep_finetune_roundtrip(tmp_path):
    """qNEP nep.txt files load back for fine-tuning."""
    original = _qnep_model([6, 14], 2)
    path = tmp_path / "nep.txt"
    original.export_nep(path)

    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["nep_model"] = str(path)
    loaded = NEP(hypers, _dataset_info([6, 14])).to(torch.float64)
    assert loaded.charge_mode == 2
    assert loaded.hypers["charge_mode"] == 2

    e_original = _model_per_atom_energies(original, _test_system([6, 14]))
    e_loaded = _model_per_atom_energies(loaded, _test_system([6, 14]))
    assert torch.allclose(
        e_original, e_loaded, rtol=1e-6, atol=1e-5 * float(e_original.abs().max())
    )
