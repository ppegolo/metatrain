import copy

import pytest

from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.testing import (
    ArchitectureTests,
    ExportedTests,
    InputTests,
    TorchscriptTests,
)


class GLETests(ArchitectureTests):
    architecture = "gle"

    @pytest.fixture
    def minimal_model_hypers(self):
        hypers = copy.deepcopy(get_default_hypers(self.architecture)["model"])
        hypers["d_pet"] = 1
        hypers["d_head"] = 1
        hypers["d_node"] = 1
        hypers["d_feedforward"] = 1
        hypers["num_heads"] = 1
        hypers["num_attention_layers"] = 1
        hypers["num_gnn_layers"] = 1
        # a 4x4 drift matrix instead of the default 16x16, to keep the tests light
        hypers["num_auxiliary_variables"] = 1
        return hypers


class TestInput(InputTests, GLETests): ...


class TestTorchscript(TorchscriptTests, GLETests):
    float_hypers = ["cutoff", "cutoff_width"]


class TestExported(ExportedTests, GLETests): ...
