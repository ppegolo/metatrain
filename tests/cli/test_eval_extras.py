"""Tests for the `dump_dataset` and `density_error` eval options.

Uses PET (not soap_bpnn like test_eval_model.py) so the tests do not depend on
the optional `spex` package.
"""

import shutil

import pytest
import torch
from omegaconf import OmegaConf

from metatrain.cli.eval import eval_model
from metatrain.pet import PET
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo, DiskDataset
from metatrain.utils.data.readers.ase import read_systems
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.pydantic import MetatrainValidationError, validate_eval_options

from ..conftest import RESOURCES_PATH


@pytest.fixture
def exported_pet_model():
    hypers = get_default_hypers("pet")["model"]
    hypers.update(
        {
            "d_pet": 16,
            "d_head": 16,
            "d_node": 16,
            "d_feedforward": 16,
            "num_heads": 1,
            "num_attention_layers": 1,
            "num_gnn_layers": 1,
        }
    )
    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 6, 7, 8, 9],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )
    return PET(hypers, dataset_info).export()


def test_eval_dump_dataset(monkeypatch, tmp_path, exported_pet_model):
    """`dump_dataset` must write the evaluated subset — the indices-selected
    systems and their reference targets, in evaluation order — as a DiskDataset
    that can be read back and reused as an input."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    indices = [5, 2, 9]
    options = OmegaConf.create(
        {
            "systems": "qm9_reduced_100.xyz",
            "targets": {"energy": {"key": "U0", "unit": "eV"}},
            "indices": indices,
            "dump_dataset": "dumped.zip",
        }
    )

    eval_model(
        model=exported_pet_model,
        options=options,
        output="predictions.xyz",
        batch_size=2,
        warm_up=False,
    )

    original_systems = read_systems("qm9_reduced_100.xyz")
    dumped = DiskDataset("dumped.zip", fields=["energy"])
    assert len(dumped) == len(indices)
    for position, index in enumerate(indices):
        sample = dumped[position]
        assert len(sample.system) == len(original_systems[index])
        torch.testing.assert_close(
            sample.system.positions.to(torch.float64),
            original_systems[index].positions.to(torch.float64),
        )
        # reference targets (not predictions) must be dumped
        assert sample.energy.block().values.shape == (1, 1)


def test_eval_dump_dataset_requires_targets(monkeypatch, tmp_path, exported_pet_model):
    monkeypatch.chdir(tmp_path)
    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    options = OmegaConf.create(
        {"systems": "qm9_reduced_100.xyz", "dump_dataset": "dumped.zip"}
    )
    with pytest.raises(ValueError, match="`dump_dataset` requires a `targets`"):
        eval_model(
            model=exported_pet_model,
            options=options,
            output="predictions.xyz",
            warm_up=False,
        )


def test_eval_options_schema_accepts_new_keys():
    validate_eval_options(
        {
            "systems": "data.xyz",
            "targets": {"energy": {"key": "U0"}},
            "indices": [0, 1],
            "dump_dataset": "out.zip",
            "density_error": {"aux_basis": "def2-universal-jfit"},
        }
    )
    validate_eval_options(
        {
            "systems": "data.xyz",
            "targets": {"energy": {"key": "U0"}},
            "density_error": {"aux_basis": {"mtt::ri": "def2-universal-jfit"}},
        }
    )


def test_eval_options_schema_rejects_unknown_density_error_keys():
    with pytest.raises(MetatrainValidationError):
        validate_eval_options(
            {
                "systems": "data.xyz",
                "density_error": {"aux_basis": "x", "typo_key": 1},
            }
        )
