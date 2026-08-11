import copy
import re

import pytest
import torch
from torchnep import write_nep

from metatrain.experimental.nep.model import NEP, _permute_type_params
from metatrain.experimental.nep.trainer import Trainer
from metatrain.utils.architectures import (
    check_architecture_options,
    get_default_hypers,
)
from metatrain.utils.data import get_atomic_types, get_dataset
from metatrain.utils.data.dataset import DatasetInfo
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification

from . import DATASET_PATH, MODEL_HYPERS
from .test_nep_export import (
    _dataset_info,
    _make_model,
    _model_per_atom_energies,
    _test_system,
)


def _finetune_model(path, atomic_types):
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["nep_model"] = str(path)
    return NEP(hypers, _dataset_info(atomic_types)).to(torch.float64)


def test_load_nep_roundtrip(tmp_path):
    """A model loaded from an exported nep.txt reproduces the original
    predictions (with untrained scaler and composition)."""
    original = _make_model(
        [6, 14], version=5, scale=0.6, composition={6: -2.0, 14: -7.5}
    )
    path = tmp_path / "nep.txt"
    original.export_nep(path)

    loaded = _finetune_model(path, [6, 14])
    assert loaded.loaded_nep
    # architecture hypers synced from the file
    assert loaded.hypers["version"] == 5
    assert loaded.hypers["cutoff_radial"] == 5.0

    system = _test_system([6, 14])
    e_original = _model_per_atom_energies(original, system)
    e_loaded = _model_per_atom_energies(loaded, _test_system([6, 14]))
    assert torch.allclose(
        e_original, e_loaded, rtol=1e-6, atol=1e-5 * float(e_original.abs().max())
    )


def test_load_nep_permuted_order(tmp_path):
    """nep.txt files whose element order differs from the dataset order are
    reordered on load."""
    original = _make_model([6, 14], version=5, scale=1.0, composition={6: 0.0, 14: 0.0})
    params = original.potential.to_nep_parameters()
    permuted = _permute_type_params(params, [1, 0])  # file order: Si, C
    path = tmp_path / "nep.txt"
    write_nep(permuted, path)

    loaded = _finetune_model(path, [6, 14])
    system = _test_system([6, 14])
    e_original = _model_per_atom_energies(original, system)
    e_loaded = _model_per_atom_energies(loaded, _test_system([6, 14]))
    assert torch.allclose(
        e_original, e_loaded, rtol=1e-6, atol=1e-5 * float(e_original.abs().max())
    )


def test_load_nep_element_mismatch_raises(tmp_path):
    model = _make_model([6, 14], version=4, scale=1.0, composition={6: 0.0, 14: 0.0})
    path = tmp_path / "nep.txt"
    model.export_nep(path)
    message = (
        "The dataset elements [6] do not match the elements of the NEP file "
        "[6, 14]. Fine-tuning requires the dataset to contain exactly the "
        "elements of the potential."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        _finetune_model(path, [6])


def test_checkpoint_roundtrip_without_file(tmp_path):
    """Checkpoints of fine-tuned models can be loaded after the nep.txt file
    is gone: the synced hypers and the state dict are self-contained."""
    original = _make_model(
        [6, 14], version=5, scale=0.6, composition={6: -2.0, 14: -7.5}
    )
    path = tmp_path / "nep.txt"
    original.export_nep(path)

    loaded = _finetune_model(path, [6, 14])
    checkpoint = loaded.get_checkpoint()
    assert "nep_model" not in checkpoint["model_data"]["model_hypers"]

    path.unlink()
    rebuilt = NEP.load_checkpoint(checkpoint, "export").to(torch.float64)

    e_loaded = _model_per_atom_energies(loaded, _test_system([6, 14]))
    e_rebuilt = _model_per_atom_energies(rebuilt, _test_system([6, 14]))
    assert torch.allclose(e_loaded, e_rebuilt, rtol=1e-12, atol=1e-12)


def test_finetune_training(tmp_path):
    """A loaded nep.txt can be fine-tuned; the composition stays zero and the
    target scale stays one."""
    targets = {
        "energy": {
            "quantity": "energy",
            "read_from": DATASET_PATH,
            "reader": "ase",
            "key": "U0",
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
        {"systems": {"read_from": DATASET_PATH, "reader": "ase"}, "targets": targets}
    )
    atomic_types = get_atomic_types(dataset)
    dataset_info = DatasetInfo(
        length_unit="angstrom", atomic_types=atomic_types, targets=targets_info
    )

    # export a pretrained file covering the dataset elements
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["cutoff_radial"] = 5.0
    hypers["cutoff_angular"] = 4.0
    pretrained = NEP(hypers, dataset_info).to(torch.float64)
    path = tmp_path / "nep.txt"
    pretrained.export_nep(path)

    finetune_hypers = copy.deepcopy(MODEL_HYPERS)
    finetune_hypers["nep_model"] = str(path)
    model = NEP(finetune_hypers, dataset_info).to(torch.float64)

    train_hypers = copy.deepcopy(get_default_hypers("experimental.nep")["training"])
    train_hypers["num_epochs"] = 1
    train_hypers["batch_size"] = 10
    train_hypers["loss"] = {"energy": init_with_defaults(LossSpecification)}
    trainer = Trainer(train_hypers)
    subset = torch.utils.data.Subset(dataset, list(range(10)))
    trainer.train(
        model, torch.float64, [torch.device("cpu")], [subset], [subset], str(tmp_path)
    )

    comp_block = model.additive_models[0].model.weights["energy"].block()
    assert torch.all(comp_block.values == 0.0)
    scales_block = model.scaler.model.scales["energy"].block()
    assert torch.all(scales_block.values == 1.0)


def _qm9_dataset_and_info():
    targets = {
        "energy": {
            "quantity": "energy",
            "read_from": DATASET_PATH,
            "reader": "ase",
            "key": "U0",
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
        {"systems": {"read_from": DATASET_PATH, "reader": "ase"}, "targets": targets}
    )
    atomic_types = get_atomic_types(dataset)
    dataset_info = DatasetInfo(
        length_unit="angstrom", atomic_types=atomic_types, targets=targets_info
    )
    return dataset, dataset_info


def _train_hypers(**overrides):
    hypers = copy.deepcopy(get_default_hypers("experimental.nep")["training"])
    hypers["num_epochs"] = 1
    hypers["batch_size"] = 10
    hypers["loss"] = {"energy": init_with_defaults(LossSpecification)}
    hypers.update(overrides)
    return hypers


def test_finetune_hypers_are_valid_options():
    """`training.finetune.read_from` passes architecture option validation."""
    options = copy.deepcopy(get_default_hypers("experimental.nep"))
    options["training"]["finetune"] = {"read_from": "model.ckpt"}
    check_architecture_options("experimental.nep", options)


def test_finetune_from_checkpoint(tmp_path):
    """Fine-tuning from a metatrain checkpoint keeps the pretrained composition
    weights, target scales and descriptor normalisation."""
    dataset, dataset_info = _qm9_dataset_and_info()

    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["cutoff_radial"] = 5.0
    hypers["cutoff_angular"] = 4.0
    pretrained = NEP(hypers, dataset_info).to(torch.float64)
    (tmp_path / "pretrain").mkdir()
    (tmp_path / "finetune").mkdir()

    # pretrain on the first half of the dataset, so that the composition
    # weights, the scales and the descriptor normalisation are all fitted
    pretrain_subset = torch.utils.data.Subset(dataset, list(range(10)))
    Trainer(_train_hypers()).train(
        pretrained,
        torch.float64,
        [torch.device("cpu")],
        [pretrain_subset],
        [pretrain_subset],
        str(tmp_path / "pretrain"),
    )
    checkpoint = pretrained.get_checkpoint()

    q_scaler = pretrained.potential.q_scaler.clone()
    composition = (
        pretrained.additive_models[0].model.weights["energy"].block().values.clone()
    )
    scales = pretrained.scaler.model.scales["energy"].block().values.clone()
    assert not torch.all(composition == 0.0)
    assert not torch.all(q_scaler == 1.0)

    # fine-tune on the second half
    model = NEP.load_checkpoint(checkpoint, "finetune").to(torch.float64)
    model = model.restart(dataset_info)
    ann_before = model.potential.ann.clone()

    finetune_subset = torch.utils.data.Subset(dataset, list(range(10, 20)))
    trainer = Trainer(_train_hypers(finetune={"read_from": "unused.ckpt"}))
    trainer.train(
        model,
        torch.float64,
        [torch.device("cpu")],
        [finetune_subset],
        [finetune_subset],
        str(tmp_path / "finetune"),
    )

    assert torch.all(model.potential.q_scaler == q_scaler)
    comp_block = model.additive_models[0].model.weights["energy"].block()
    assert torch.all(comp_block.values == composition)
    scales_block = model.scaler.model.scales["energy"].block()
    assert torch.all(scales_block.values == scales)
    # the network itself was trained further
    assert not torch.all(model.potential.ann == ann_before)
