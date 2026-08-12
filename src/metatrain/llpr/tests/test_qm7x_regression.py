import copy
import gzip
import shutil
from pathlib import Path

import pytest
import torch
from metatomic.torch import ModelOutput
from omegaconf import OmegaConf

from metatrain.llpr import LLPRUncertaintyModel
from metatrain.llpr import Trainer as LLPRTrainer
from metatrain.llpr.model import get_uncertainty_name
from metatrain.utils.architectures import import_architecture
from metatrain.utils.data import DatasetInfo, get_atomic_types, get_dataset
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)
from metatrain.utils.omegaconf import expand_dataset_config

from . import DEFAULT_HYPERS_LLPR


HERE = Path(__file__).parent
DATASET_PATH = HERE.parents[3] / "tests" / "resources" / "qm7x_spherical_100.zip"

BACKBONES = {
    "pet-qm7x": "pet",
    "space-qm7x": "experimental.space",
}

QM7X_TARGET_CONFIG = {
    "energy": {
        "quantity": "energy",
        "unit": "eV",
    },
    "non_conservative_force": {
        "quantity": "",
        "unit": "eV/angstrom",
        "sample_kind": "atom",
        "type": {"cartesian": {"rank": 1}},
        "num_subtargets": 1,
    },
    "mtt::dipole": {
        "quantity": "",
        "unit": "",
        "sample_kind": "system",
        "type": {"spherical": {"irreps": [{"o3_lambda": 1, "o3_sigma": 1}]}},
        "num_subtargets": 1,
    },
    "mtt::polarizability": {
        "quantity": "",
        "unit": "",
        "sample_kind": "system",
        "type": {
            "spherical": {
                "irreps": [
                    {"o3_lambda": 0, "o3_sigma": 1},
                    {"o3_lambda": 2, "o3_sigma": 1},
                ]
            }
        },
        "num_subtargets": 1,
    },
}


def _get_qm7x_datasets():
    """Load the QM7X dataset with the same 90:10 split the shipped checkpoints
    were trained with: the last 10 structures are the held-out calibration set."""
    conf = expand_dataset_config(
        OmegaConf.create(
            {
                "systems": {"read_from": str(DATASET_PATH), "length_unit": "angstrom"},
                "targets": copy.deepcopy(QM7X_TARGET_CONFIG),
            }
        )
    )[0]
    dataset, target_infos, _ = get_dataset(conf)
    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=get_atomic_types(dataset),
        targets=target_infos,
    )
    train_dataset = torch.utils.data.Subset(dataset, list(range(90)))
    holdout_dataset = torch.utils.data.Subset(dataset, list(range(90, 100)))
    return train_dataset, holdout_dataset, dataset_info


# mean calibrated uncertainty over the 10 held-out structures, per target
# block, computed once from the shipped checkpoints; recompute them whenever
# the checkpoints are regenerated (see qm7x_checkpoints/readme.txt). The
# tolerance absorbs float32 cross-platform drift (BLAS, instruction sets)
EXPECTED_MEAN_UNCERTAINTIES = {
    "pet-qm7x": {
        ("energy", (0,)): 0.7568486928939819,
        ("non_conservative_force", (0,)): 1.3553270101547241,
        ("mtt::dipole", (1, 1)): 0.18728281557559967,
        ("mtt::polarizability", (0, 1)): 8.49271297454834,
        ("mtt::polarizability", (2, 1)): 8.952874183654785,
    },
    "space-qm7x": {
        ("energy", (0,)): 0.6724218130111694,
        ("non_conservative_force", (0,)): 1.2135006189346313,
        ("mtt::dipole", (1, 1)): 0.1230238750576973,
        ("mtt::polarizability", (0, 1)): 7.877081394195557,
        ("mtt::polarizability", (2, 1)): 3.312208652496338,
    },
}


@pytest.fixture(scope="module", params=list(BACKBONES))
def calibrated_llpr(request, tmp_path_factory):
    """An LLPR model wrapped around a shipped checkpoint, with covariance from
    the 90 train structures and calibration from the 10 held-out ones."""
    tmp_path = tmp_path_factory.mktemp(request.param)
    checkpoint_path = tmp_path / f"{request.param}.ckpt"
    with gzip.open(HERE / "qm7x_checkpoints" / f"{request.param}.ckpt.gz", "rb") as f:
        with open(checkpoint_path, "wb") as out:
            shutil.copyfileobj(f, out)

    architecture = import_architecture(BACKBONES[request.param])
    dtype = architecture.__model__.__supported_dtypes__[0]

    train_dataset, holdout_dataset, dataset_info = _get_qm7x_datasets()

    model = LLPRUncertaintyModel({"num_ensemble_members": {}}, dataset_info)
    hypers = copy.deepcopy(DEFAULT_HYPERS_LLPR["training"])
    hypers["model_checkpoint"] = str(checkpoint_path)
    trainer = LLPRTrainer(hypers)
    trainer.train(
        model,
        dtype=dtype,
        devices=[torch.device("cpu")],
        train_datasets=[train_dataset],
        val_datasets=[holdout_dataset],
        checkpoint_dir=str(tmp_path),
    )

    systems = [
        get_system_with_neighbor_lists(
            sample["system"].to(dtype=dtype), get_requested_neighbor_lists(model)
        )
        for sample in holdout_dataset
    ]
    return request.param, model, systems, holdout_dataset, dataset_info


def test_calibrated_uncertainties_on_all_targets(calibrated_llpr):
    """Uncertainties mirror each target's layout, match the scale of the
    held-out residuals, and reproduce the stored per-block references."""
    name, model, systems, holdout_dataset, dataset_info = calibrated_llpr

    requested = {}
    for target_name, target_info in dataset_info.targets.items():
        requested[target_name] = ModelOutput(sample_kind=target_info.sample_kind)
        requested[get_uncertainty_name(target_name)] = ModelOutput(
            sample_kind=target_info.sample_kind
        )

    outputs = model(systems, requested)

    actual = {}
    for target_name in dataset_info.targets:
        prediction = outputs[target_name]
        uncertainty = outputs[get_uncertainty_name(target_name)]

        assert uncertainty.keys == prediction.keys
        for index in range(len(prediction.keys)):
            prediction_block = prediction.block(index)
            uncertainty_block = uncertainty.block(index)
            assert uncertainty_block.values.shape == prediction_block.values.shape
            assert uncertainty_block.samples == prediction_block.samples
            assert uncertainty_block.components == prediction_block.components
            assert uncertainty_block.properties == prediction_block.properties
            assert torch.all(uncertainty_block.values > 0.0)

            # deterministically-wrong calibration (e.g. against predictions
            # missing the scaler or composition) would pass the exact pins
            # below; the residual scale catches it
            references = torch.cat(
                [
                    sample[target_name]
                    .block(index)
                    .values.to(dtype=prediction_block.values.dtype)
                    for sample in holdout_dataset
                ]
            )
            mean_abs_residual = (prediction_block.values - references).abs().mean()
            ratio = (uncertainty_block.values.mean() / mean_abs_residual).item()
            key = tuple(int(v) for v in prediction.keys.values[index])
            assert 0.2 < ratio < 5.0, f"{(target_name, key)}: ratio {ratio}"

            actual[(target_name, key)] = uncertainty_block.values.mean().item()

    # per-atom force samples survive the wrapper
    forces = outputs[get_uncertainty_name("non_conservative_force")]
    assert forces.block(0).samples.names == ["system", "atom"]

    expected = EXPECTED_MEAN_UNCERTAINTIES[name]
    assert set(actual) == set(expected)
    for block_id, expected_value in expected.items():
        torch.testing.assert_close(
            actual[block_id],
            expected_value,
            rtol=1e-2,
            atol=1e-12,
            msg=f"{block_id}: expected {expected_value}, got {actual[block_id]}",
        )


def test_uncertainty_only_request(calibrated_llpr):
    """Requesting only uncertainties returns only them: the base outputs and
    last-layer features requested internally are stripped from the result."""
    _, model, systems, _, dataset_info = calibrated_llpr

    requested = {
        get_uncertainty_name(target_name): ModelOutput(
            sample_kind=target_info.sample_kind
        )
        for target_name, target_info in dataset_info.targets.items()
    }
    outputs = model(systems, requested)
    assert set(outputs) == set(requested)
