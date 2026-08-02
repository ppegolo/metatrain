import copy

import pytest

from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.testing import (
    ArchitectureTests,
    AutogradTests,
    CheckpointTests,
    ExportedTests,
    InputTests,
    OutputTests,
    TorchscriptTests,
    TrainingTests,
)


def _minimal_hypers(arch: str) -> dict:
    hypers = copy.deepcopy(get_default_hypers(arch)["model"])
    hypers["n_max_radial"] = 1
    hypers["n_max_angular"] = 1
    hypers["basis_size_radial"] = 1
    hypers["basis_size_angular"] = 1
    hypers["l_max_3body"] = 1
    hypers["l_max_4body"] = 0
    hypers["neurons"] = 2
    return hypers


class NEPTests(ArchitectureTests):
    architecture = "experimental.nep"

    @pytest.fixture
    def minimal_model_hypers(self) -> dict:
        """Minimal hyperparameters for a NEP model for the smallest
        checkpoint possible.

        :return: Hyperparameters for the model.
        """
        return _minimal_hypers(self.architecture)


class TestInput(InputTests, NEPTests): ...


class TestOutput(OutputTests, NEPTests):
    supports_multiscalar_outputs = False
    supports_spherical_outputs = False
    supports_vector_outputs = False
    supports_features = False
    supports_last_layer_features = False


class TestAutograd(AutogradTests, NEPTests):
    @pytest.fixture
    def model_hypers(self) -> dict:
        return _minimal_hypers(self.architecture)


class TestTorchscript(TorchscriptTests, NEPTests):
    float_hypers = ["cutoff_radial", "cutoff_angular"]
    supports_spherical_outputs = False


class TestExported(ExportedTests, NEPTests): ...


class TestTraining(TrainingTests, NEPTests): ...


class TestCheckpoints(CheckpointTests, NEPTests):
    incompatible_trainer_checkpoints = []
