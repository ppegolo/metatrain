import logging
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch import load as metatensor_load
from metatomic.torch import NeighborListOptions, systems_to_torch
from omegaconf import OmegaConf

from metatrain.cli.eval import eval_model
from metatrain.soap_bpnn import __model__
from metatrain.utils.data import DatasetInfo, DiskDataset
from metatrain.utils.data.dataset import MemmapDataset
from metatrain.utils.data.readers.ase import read
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.data.writers import DiskDatasetWriter
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists
from metatrain.utils.pydantic import MetatrainValidationError

from ..conftest import EVAL_OPTIONS_PATH, MODEL_HYPERS, RESOURCES_PATH


@pytest.fixture
def model(MODEL_PATH):
    return torch.jit.load(MODEL_PATH)


@pytest.fixture
def options():
    return OmegaConf.load(EVAL_OPTIONS_PATH)


@pytest.mark.parametrize("warm_up", [True, False])
def test_eval_cli(monkeypatch, tmp_path, MODEL_PATH, warm_up):
    """Test succesful run of the eval script via the CLI with default arguments"""
    monkeypatch.chdir(tmp_path)
    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    command = [
        "mtt",
        "eval",
        str(MODEL_PATH),
        str(EVAL_OPTIONS_PATH),
        "-e",
        str(MODEL_PATH.parent / "extensions"),
        "--check-consistency",
    ]
    if not warm_up:
        command.append("--no-warm-up")

    result = subprocess.run(
        command,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
    )

    if result.returncode != 0:
        print(
            f"Eval logs:\n{result.stdout.decode()}",
            file=sys.stderr,
        )
        raise RuntimeError(
            "Failed to evaluate model via CLI. Logs should be printed above."
        )

    log_text = result.stdout.decode()

    assert "100%|██████████" in log_text
    assert "energy RMSE" in log_text

    # Check that the warm-up flag is correctly used
    warm_up_str = "Warming up the model with 10 batches..."
    no_warm_up_str = "Skipping warm-up of the model."
    if warm_up:
        assert warm_up_str in log_text
        assert no_warm_up_str not in log_text
    else:
        assert warm_up_str not in log_text
        assert no_warm_up_str in log_text

    assert Path("output.xyz").is_file()


@pytest.mark.parametrize("model_type", ["32-bit", "64-bit"])
def test_eval(request, monkeypatch, tmp_path, caplog, model_type, options):
    """Test that eval via python API runs without an error raise."""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO)

    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    fixture_name = {
        "32-bit": "MODEL_PATH",
        "64-bit": "MODEL_PATH_64_BIT",
    }.get(model_type)

    model_path = request.getfixturevalue(fixture_name)

    model = torch.jit.load(model_path)

    eval_model(
        model=model,
        options=options,
        output="foo.xyz",
        check_consistency=True,
    )

    # Test target predictions
    log = "".join([rec.message for rec in caplog.records])
    assert "energy RMSE (per atom)" in log
    assert "energy MAE (per atom)" in log
    assert "dataset with index" not in log
    assert "Evaluation time" in log
    assert "ms per atom" in log

    # Test file is written predictions
    frames = read("foo.xyz", ":")
    frames[0].info["energy"]


@pytest.mark.parametrize("model_type", ["32-bit", "64-bit"])
def test_eval_batch_size(request, monkeypatch, tmp_path, caplog, model_type, options):
    """Test that eval via python API runs without an error raise."""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.DEBUG)

    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    fixture_name = {
        "32-bit": "MODEL_PATH",
        "64-bit": "MODEL_PATH_64_BIT",
    }.get(model_type)

    model_path = request.getfixturevalue(fixture_name)

    model = torch.jit.load(model_path)

    eval_model(
        model=model,
        options=options,
        output="foo.xyz",
        batch_size=13,
        check_consistency=True,
    )

    # Test target predictions
    log = "".join([rec.message for rec in caplog.records])
    assert "energy RMSE (per atom)" in log
    assert "energy MAE (per atom)" in log
    assert "dataset with index" not in log
    assert "Evaluation time" in log
    assert "ms per atom" in log
    assert "inaccurate average timings" in log

    # Test file is written predictions
    frames = read("foo.xyz", ":")
    frames[0].info["energy"]


def test_eval_export(monkeypatch, tmp_path, options):
    """Test evaluation of a trained model exported but not saved to disk."""
    monkeypatch.chdir(tmp_path)
    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types={1, 6, 7, 8},
        targets={"energy": get_energy_target_info("energy", {"unit": "eV"})},
    )
    model = __model__(hypers=MODEL_HYPERS, dataset_info=dataset_info)

    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    exported_model = model.export()

    eval_model(
        model=exported_model,
        options=options,
        output="foo.xyz",
        check_consistency=True,
    )


def test_eval_multi_dataset(monkeypatch, tmp_path, caplog, model, options):
    """Test that eval runs for multiple datasets should be evaluated."""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO)

    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    eval_model(
        model=model,
        options=OmegaConf.create([options, options]),
        output="foo.xyz",
        check_consistency=True,
    )

    # Test target predictions
    log = "".join([rec.message for rec in caplog.records])
    assert "index 0" in log
    assert "index 1" in log

    # Test file is written predictions
    for i in range(2):
        frames = read(f"foo_{i}.xyz", ":")
        frames[0].info["energy"]


def test_eval_no_targets(monkeypatch, tmp_path, model, options):
    monkeypatch.chdir(tmp_path)

    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    options.pop("targets")

    eval_model(
        model=model,
        options=options,
        check_consistency=True,
    )

    assert Path("output.xyz").is_file()


@pytest.mark.parametrize("suffix", [".zip", "/"])
def test_eval_no_targets_disallowed_for_dataset_writers(
    monkeypatch, tmp_path, model, options, suffix
):
    """DiskDataset/MemmapDataset outputs require explicit targets in the input."""
    monkeypatch.chdir(tmp_path)

    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    options.pop("targets")

    with pytest.raises(ValueError, match="not allowed without explicitly"):
        eval_model(
            model=model,
            options=options,
            output=f"output{suffix}",
            check_consistency=True,
        )


@pytest.mark.parametrize("suffix", [".zip", ".mts", "/"])
def test_eval_disk_dataset(monkeypatch, tmp_path, caplog, suffix, MODEL_PATH):
    """Test that eval via python API runs without an error raise."""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO)

    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    model = torch.jit.load(MODEL_PATH)

    options = OmegaConf.create(
        {
            "systems": {"read_from": "qm9_reduced_100.zip"},
            "targets": {"energy": {"read_from": "qm9_reduced_100.zip"}},
        }
    )

    # Write a disk dataset
    disk_dataset_writer = DiskDatasetWriter("qm9_reduced_100.zip")
    for i in range(100):
        frame = read("qm9_reduced_100.xyz", index=i)
        system = systems_to_torch(frame, dtype=torch.float64)
        system = get_system_with_neighbor_lists(
            system,
            [NeighborListOptions(cutoff=5.0, full_list=True, strict=True)],
        )
        energy = TensorMap(
            keys=Labels.single(),
            blocks=[
                TensorBlock(
                    values=torch.tensor([[frame.info["U0"]]], dtype=torch.float64),
                    samples=Labels(
                        names=["system"],
                        values=torch.tensor([[0]]),
                    ),
                    components=[],
                    properties=Labels("energy", torch.tensor([[0]])),
                )
            ],
        )
        disk_dataset_writer.write([system], {"energy": energy})
    disk_dataset_writer.finish()

    eval_model(
        model=model,
        options=options,
        output=f"foo{suffix}",
        check_consistency=True,
    )

    # Test target predictions
    log = "".join([rec.message for rec in caplog.records])
    assert "energy RMSE (per atom)" in log
    assert "energy MAE (per atom)" in log
    assert "dataset with index" not in log
    assert "Evaluation time" in log
    assert "ms per atom" in log

    # Test file is written predictions
    if suffix == ".mts":
        pred = metatensor_load("foo_energy.mts")
        assert pred.keys == Labels(["_"], torch.tensor([[0]]))
    elif suffix == "/":
        target_options = {
            "energy": {
                "key": "energy",
                "quantity": "energy",
                "sample_kind": "system",
                "type": "scalar",
                "num_subtargets": 1,
                "forces": False,
                "stress": False,
                "virial": False,
            }
        }
        pred = MemmapDataset("foo", target_options)
        assert len(pred) == 100
        assert pred[0].energy.keys == Labels(["_"], torch.tensor([[0]]))
    else:
        pred = DiskDataset("foo.zip")
        assert pred[0]["energy"].keys == Labels(["_"], torch.tensor([[0]]))


@pytest.mark.parametrize("indices_mode", ["list", "file"])
def test_eval_indices(monkeypatch, tmp_path, caplog, options, MODEL_PATH, indices_mode):
    """Checks that the indices option is correctly used to evaluate only a subset"""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO)

    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    options = options.copy()
    if indices_mode == "file":
        with open("indices.txt", "w") as f:
            f.write("0\n")
        options["indices"] = "indices.txt"
    else:
        options["indices"] = [0]

    model = torch.jit.load(MODEL_PATH)

    eval_model(
        model=model,
        options=options,
        output="foo.xyz",
        check_consistency=True,
    )

    # Test target predictions
    log = "".join([rec.message for rec in caplog.records])
    assert "energy RMSE (per atom)" in log
    assert "energy MAE (per atom)" in log
    assert "dataset with index" not in log
    assert "Evaluation time" in log
    assert "ms per atom" in log

    # Test file is written predictions
    frames = read("foo.xyz", ":")
    assert len(frames) == 1


@pytest.mark.parametrize("equivariance", [True, {"rotation_batch_size": 8}])
def test_eval_equivariance(monkeypatch, tmp_path, caplog, model, options, equivariance):
    """Test that the equivariance error is computed and logged during eval."""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO)

    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    options["equivariance"] = equivariance
    options["indices"] = [0, 1, 2]

    eval_model(
        model=model,
        options=options,
        output="foo.xyz",
        warm_up=False,
    )

    log = "".join([rec.message for rec in caplog.records])
    assert "Equivariance error evaluation enabled" in log
    assert "energy equivariance RMSE (per atom)" in log
    # the fixture model is float32, so the noise floor must be reported
    assert "equivariance errors below the round-off" in log
    assert "nan" not in log


def test_eval_equivariance_unknown_option(monkeypatch, tmp_path, model, options):
    """Unknown equivariance sub-options must be rejected by validation."""
    monkeypatch.chdir(tmp_path)

    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    options["equivariance"] = {"bad_option": 1}

    with pytest.raises(MetatrainValidationError, match="bad_option"):
        eval_model(model=model, options=options)


def test_eval_equivariance_requires_targets(monkeypatch, tmp_path, model, options):
    """The equivariance error needs target information, so a `targets` section
    must be present."""
    monkeypatch.chdir(tmp_path)

    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    options.pop("targets")
    options["equivariance"] = True

    with pytest.raises(ValueError, match="requires a `targets` section"):
        eval_model(model=model, options=options)


@pytest.mark.parametrize("gradients", [True, False])
def test_eval_equivariance_gradients(monkeypatch, tmp_path, caplog, model, gradients):
    """The equivariance error of conservative forces is only computed when
    explicitly requested with the `gradients` option (it is expensive)."""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO)

    shutil.copy(RESOURCES_PATH / "ethanol_reduced_100.xyz", "ethanol_reduced_100.xyz")

    options = OmegaConf.create(
        {
            "systems": "ethanol_reduced_100.xyz",
            "targets": {"energy": {"key": "energy", "unit": "eV", "forces": True}},
            "indices": [0, 1, 2],
            "equivariance": {"gradients": gradients},
        }
    )

    eval_model(
        model=model,
        options=options,
        output="foo.xyz",
        warm_up=False,
    )

    log = "".join([rec.message for rec in caplog.records])
    assert "energy equivariance RMSE (per atom)" in log
    gradient_metric = "energy_positions_gradients equivariance RMSE"
    if gradients:
        assert gradient_metric in log
    else:
        assert gradient_metric not in log
