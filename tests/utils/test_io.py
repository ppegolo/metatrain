from pathlib import Path

import pytest
from metatensor.torch.atomistic import MetatensorAtomisticModel

from metatrain.experimental.soap_bpnn.model import SoapBpnn
from metatrain.utils.io import check_file_extension, is_exported_file, load_model

from . import RESOURCES_PATH


def is_None(*args, **kwargs) -> None:
    return None


@pytest.mark.parametrize("filename", ["example.txt", Path("example.txt")])
def test_check_suffix(filename):
    result = check_file_extension(filename, ".txt")

    assert str(result) == "example.txt"
    assert isinstance(result, type(filename))


@pytest.mark.parametrize("filename", ["example", Path("example")])
def test_warning_on_missing_suffix(filename):
    match = r"The file name should have a '\.txt' file extension."
    with pytest.warns(UserWarning, match=match):
        result = check_file_extension(filename, ".txt")

    assert str(result) == "example.txt"
    assert isinstance(result, type(filename))


def test_is_exported_file():
    assert is_exported_file(RESOURCES_PATH / "model-32-bit.pt")
    assert not is_exported_file(RESOURCES_PATH / "model-32-bit.ckpt")


@pytest.mark.parametrize(
    "path",
    [
        RESOURCES_PATH / "model-32-bit.ckpt",
        str(RESOURCES_PATH / "model-32-bit.ckpt"),
        f"file:{str(RESOURCES_PATH / 'model-32-bit.ckpt')}",
    ],
)
def test_load_model_checkpoint(path):
    model = load_model(path, architecture_name="experimental.soap_bpnn")
    assert type(model) is SoapBpnn


@pytest.mark.parametrize(
    "path",
    [
        RESOURCES_PATH / "model-32-bit.pt",
        str(RESOURCES_PATH / "model-32-bit.pt"),
        f"file:{str(RESOURCES_PATH / 'model-32-bit.pt')}",
    ],
)
def test_load_model_exported(path):
    model = load_model(path, architecture_name="experimental.soap_bpnn")
    assert type(model) is MetatensorAtomisticModel


@pytest.mark.parametrize("suffix", [".yml", ".yaml"])
def test_load_model_yaml(suffix):
    match = f"path 'foo{suffix}' seems to be a YAML option file and no model"
    with pytest.raises(ValueError, match=match):
        load_model(
            f"foo{suffix}",
            architecture_name="experimental.soap_bpnn",
        )


def test_load_model_unknown_model():
    architecture_name = "experimental.pet"
    path = RESOURCES_PATH / "model-32-bit.ckpt"

    match = (
        f"path '{path}' is not a valid model file for the {architecture_name} "
        "architecture"
    )
    with pytest.raises(ValueError, match=match):
        load_model(path, architecture_name=architecture_name)


def test_extensions_directory_and_architecture_name():
    match = (
        r"Both ``extensions_directory`` \('.'\) and ``architecture_name`` \('foo'\) "
        r"are given which are mutually exclusive. An ``extensions_directory`` is only "
        r"required for \*exported\* models while an ``architecture_name`` is only "
        r"needed for model \*checkpoints\*."
    )
    with pytest.raises(ValueError, match=match):
        load_model("model.pt", extensions_directory=".", architecture_name="foo")
