"""Tests for the input validation of the NEP model.

A NEP potential has a single output head and a fixed element list, so the
model rejects anything it cannot represent instead of silently ignoring it.
"""

import copy
import re

import pytest
import torch
from metatomic.torch import System

from metatrain.experimental.nep.model import NEP
from metatrain.utils.data import TargetInfo
from metatrain.utils.data.dataset import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

from . import MODEL_HYPERS
from .test_nep_export import _dataset_info, _make_model, _test_system


def _energy_info() -> TargetInfo:
    return get_energy_target_info("energy", {"quantity": "energy", "unit": "eV"})


def test_multiple_targets_raise():
    """A NEP model cannot be built for more than one target."""
    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[6, 14],
        targets={"energy": _energy_info(), "mtt::other": _energy_info()},
    )
    message = (
        "The NEP architecture can only predict a single target, but 2 were "
        "requested: ['energy', 'mtt::other']."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        NEP(copy.deepcopy(MODEL_HYPERS), dataset_info)


def test_restart_with_new_target_raises():
    """A restart cannot add a second target to an existing model."""
    model = NEP(copy.deepcopy(MODEL_HYPERS), _dataset_info([6, 14]))
    new_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[6, 14],
        targets={"mtt::other": _energy_info()},
    )
    message = (
        "New targets found in the dataset: ['mtt::other']. The NEP model can "
        "only predict the single target it was trained on ('energy')."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        model.restart(new_info)


def _system_with_neighbor_lists(model, system):
    return get_system_with_neighbor_lists(system, get_requested_neighbor_lists(model))


def test_forward_without_its_target_raises():
    """Asking the model for an output it does not compute is an error."""
    model = _make_model([6, 14], version=4, scale=1.0, composition={6: 0.0, 14: 0.0})
    system = _system_with_neighbor_lists(model, _test_system([6, 14]))
    with pytest.raises(ValueError, match="which was not requested"):
        model([system], {"mtt::other": model.outputs["energy"]})


@pytest.mark.parametrize(
    "unknown_type",
    # oxygen is inside the lookup table (which goes up to silicon) but was
    # never trained on, so its type id is -1; tin is past the end of the table
    [8, 50],
)
def test_unknown_atomic_type_raises(unknown_type):
    """Systems with elements the model does not know about are rejected."""
    model = _make_model([6, 14], version=4, scale=1.0, composition={6: 0.0, 14: 0.0})
    original = _test_system([6, 14])
    types = original.types.clone()
    types[0] = unknown_type
    broken = _system_with_neighbor_lists(
        model,
        System(
            types=types,
            positions=original.positions,
            cell=original.cell,
            pbc=original.pbc,
        ),
    )
    with pytest.raises(ValueError, match="atomic types the NEP model does not"):
        model([broken], {"energy": model.outputs["energy"]})


def test_loaded_nep_survives_checkpoint(tmp_path):
    """A model loaded from a nep.txt keeps fixing its composition baselines
    and its target scale after a checkpoint round trip."""
    original = _make_model(
        [6, 14], version=5, scale=0.6, composition={6: -2.0, 14: -7.5}
    )
    path = tmp_path / "nep.txt"
    original.export_nep(path)

    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["nep_model"] = str(path)
    loaded = NEP(hypers, _dataset_info([6, 14])).to(torch.float64)
    assert loaded.loaded_nep
    assert loaded.get_fixed_composition_weights() == {"energy": {6: 0.0, 14: 0.0}}
    assert loaded.get_fixed_scaling_weights() == {"energy": 1.0}

    rebuilt = NEP.load_checkpoint(loaded.get_checkpoint(), "restart")
    assert rebuilt.loaded_nep
    assert rebuilt.get_fixed_composition_weights() == {"energy": {6: 0.0, 14: 0.0}}
    assert rebuilt.get_fixed_scaling_weights() == {"energy": 1.0}


def test_regular_model_checkpoint_is_not_loaded_nep():
    """A model trained from scratch keeps fitting its composition and scale."""
    model = NEP(copy.deepcopy(MODEL_HYPERS), _dataset_info([6, 14]))
    assert not model.loaded_nep

    rebuilt = NEP.load_checkpoint(model.get_checkpoint(), "restart")
    assert not rebuilt.loaded_nep
    assert rebuilt.get_fixed_composition_weights() == {}
    assert rebuilt.get_fixed_scaling_weights() == {}
