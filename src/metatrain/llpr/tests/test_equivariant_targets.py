"""LLPR on targets made of several irreps.

The uncertainty must mirror the target's block layout, and every block must get its
own calibration factor: PET and SOAP-BPNN expose a single, invariant last-layer
feature block, so ``sqrt(f^T C^-1 f)`` is identical for every block and component of
a given sample and the multiplier is the only thing that can tell one block's
uncertainty from another's.

Ensembles are more demanding, as they are sampled in the weight space of the last
layer: they need every block to be read directly off the last-layer features, which
holds for PET but not for SOAP-BPNN's equivariant blocks.
"""

import copy

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System

from metatrain.llpr.model import LLPRUncertaintyModel
from metatrain.pet import PET
from metatrain.soap_bpnn import SoapBpnn
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.target_info import get_generic_target_info
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)


TARGET = "mtt::spherical"
UNCERTAINTY = "mtt::aux::spherical_uncertainty"
ENSEMBLE = "mtt::aux::spherical_ensemble"
NUM_ENSEMBLE_MEMBERS = 128


def _dataset_info() -> DatasetInfo:
    """A target with a lambda=0 and a lambda=2 block."""
    target_info = get_generic_target_info(
        TARGET,
        {
            "quantity": "",
            "unit": "",
            "type": {
                "spherical": {
                    "irreps": [
                        {"o3_lambda": 0, "o3_sigma": 1},
                        {"o3_lambda": 2, "o3_sigma": 1},
                    ],
                }
            },
            "num_subtargets": 1,
            "sample_kind": "system",
        },
    )
    return DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6], targets={TARGET: target_info}
    )


def _soap_bpnn(dataset_info: DatasetInfo) -> SoapBpnn:
    hypers = copy.deepcopy(get_default_hypers("soap_bpnn")["model"])
    hypers["soap"]["max_angular"] = 2
    hypers["soap"]["max_radial"] = 2
    hypers["bpnn"]["num_neurons_per_layer"] = 4
    hypers["bpnn"]["num_hidden_layers"] = 1
    return SoapBpnn(hypers, dataset_info)


def _pet(dataset_info: DatasetInfo) -> PET:
    hypers = copy.deepcopy(get_default_hypers("pet")["model"])
    for key in ("d_pet", "d_head", "d_node", "d_feedforward"):
        hypers[key] = 4
    for key in ("num_heads", "num_attention_layers", "num_gnn_layers"):
        hypers[key] = 1
    return PET(hypers, dataset_info)


BACKBONES = {"soap_bpnn": _soap_bpnn, "pet": _pet}


def _wrapped_model(backbone_name: str, dataset_info: DatasetInfo, ensembles=False):
    backbone = BACKBONES[backbone_name](dataset_info).to(torch.float64)
    num_ensemble_members = {TARGET: NUM_ENSEMBLE_MEMBERS} if ensembles else {}
    model = LLPRUncertaintyModel(
        {"num_ensemble_members": num_ensemble_members}, dataset_info
    )
    model.set_wrapped_model(backbone)
    return model.to(torch.float64)


def _systems(model, n_systems: int):
    torch.manual_seed(0)
    systems = []
    for _ in range(n_systems):
        system = System(
            types=torch.tensor([6, 1, 1, 1]),
            positions=torch.tensor(
                [[0.0, 0, 0], [1.1, 0, 0], [0, 1.1, 0], [0, 0, 1.1]],
                dtype=torch.float64,
            )
            + 0.1 * torch.randn(4, 3, dtype=torch.float64),
            cell=torch.zeros((3, 3), dtype=torch.float64),
            pbc=torch.tensor([False, False, False]),
        )
        systems.append(
            get_system_with_neighbor_lists(system, get_requested_neighbor_lists(model))
        )
    return systems


def _with_identity_covariance(model):
    """A covariance the Cholesky decomposition can be taken of. Its exact value does
    not matter to these tests, only that the uncertainty derives from it."""
    covariance = model._get_covariance(UNCERTAINTY)
    covariance[:] = torch.eye(covariance.shape[0], dtype=covariance.dtype)
    model.compute_cholesky_decomposition(regularizer=1e-8)
    return model


@pytest.mark.parametrize("backbone", list(BACKBONES))
def test_uncertainty_mirrors_the_target_layout(backbone):
    """The uncertainty must have the target's own keys, components and properties.

    A single-block output (the previous behaviour) silently drops the lambda=2
    block, and a caller selecting on ``o3_lambda`` does not find it.
    """
    dataset_info = _dataset_info()
    model = _with_identity_covariance(_wrapped_model(backbone, dataset_info))
    system = _systems(model, 1)[0]

    outputs = model(
        [system],
        {
            TARGET: ModelOutput(sample_kind="system"),
            UNCERTAINTY: ModelOutput(sample_kind="system"),
        },
    )
    prediction = outputs[TARGET]
    uncertainty = outputs[UNCERTAINTY]

    assert uncertainty.keys == prediction.keys
    assert prediction.keys.names == ["o3_lambda", "o3_sigma"]
    assert prediction.keys.values.tolist() == [[0, 1], [2, 1]]

    for index in range(len(prediction.keys)):
        prediction_block = prediction.block(index)
        uncertainty_block = uncertainty.block(index)

        assert uncertainty_block.values.shape == prediction_block.values.shape
        assert uncertainty_block.components == prediction_block.components
        assert uncertainty_block.properties == prediction_block.properties
        assert torch.all(uncertainty_block.values > 0.0)


def test_ensemble_mirrors_the_target_layout():
    """The ensemble must have the target's own keys, and its mean must reproduce the
    prediction block by block.

    PET only: its last layer holds one independent weight row per component of every
    block, which is what makes sampling in weight space equivalent to sampling the
    block.
    """
    dataset_info = _dataset_info()
    model = _with_identity_covariance(
        _wrapped_model("pet", dataset_info, ensembles=True)
    )
    model.generate_ensemble()
    system = _systems(model, 1)[0]

    outputs = model(
        [system],
        {
            TARGET: ModelOutput(sample_kind="system"),
            ENSEMBLE: ModelOutput(sample_kind="system"),
        },
    )
    prediction = outputs[TARGET]
    ensemble = outputs[ENSEMBLE]

    assert ensemble.keys == prediction.keys

    for index in range(len(prediction.keys)):
        prediction_block = prediction.block(index)
        ensemble_block = ensemble.block(index)

        # the ensemble stacks its members into the property dimension
        num_properties = prediction_block.values.shape[-1]
        assert ensemble_block.components == prediction_block.components
        assert ensemble_block.values.shape[-1] == (
            NUM_ENSEMBLE_MEMBERS * num_properties
        )

        # the ensemble is re-centered on the prediction, so its mean is exact
        members = ensemble_block.values.reshape(
            list(ensemble_block.values.shape[:-1])
            + [NUM_ENSEMBLE_MEMBERS, num_properties]
        )
        torch.testing.assert_close(
            members.mean(dim=-2), prediction_block.values, rtol=1e-10, atol=1e-10
        )


def test_ensemble_refused_when_a_block_is_not_a_direct_readout():
    """SOAP-BPNN builds its lambda>0 blocks by contracting invariant coefficients
    with a geometry-dependent tensor basis, so the last-layer weights alone do not
    determine the block and an ensemble sampled from them is meaningless.

    This must fail loudly: for lambda=1 the coefficients are as many as the
    components, so nothing downstream can tell the two cases apart from the shapes.
    """
    dataset_info = _dataset_info()
    with pytest.raises(ValueError, match="does not expose a last layer reading"):
        _wrapped_model("soap_bpnn", dataset_info, ensembles=True)


@pytest.mark.parametrize("backbone", list(BACKBONES))
def test_calibration_is_per_block(backbone):
    """Each block must be calibrated against its own residuals.

    The two blocks here are given residuals that differ by two orders of magnitude,
    while the uncalibrated ``sqrt(f^T C^-1 f)`` is identical for both (the last-layer
    features are a single invariant block). A single shared multiplier would then
    leave one block badly over-confident and the other badly under-confident; with a
    multiplier per block, both come out calibrated.
    """
    dataset_info = _dataset_info()
    model = _wrapped_model(backbone, dataset_info)
    systems = _systems(model, 8)

    # `squared_residuals` minimizes the Gaussian NLL over the multiplier, whose
    # optimum is alpha = sqrt(mean((prediction - target)^2 / sigma^2)). References
    # far larger than anything the (untrained) model predicts make the residuals of
    # the lambda=0 block ~100x those of the lambda=2 block.
    layout = dataset_info.targets[TARGET].layout
    scales = {0: 100.0, 2: 1.0}
    references = []
    for system_index in range(len(systems)):
        blocks = []
        for key, layout_block in layout.items():
            shape = (1, len(layout_block.components[0]), 1)
            blocks.append(
                TensorBlock(
                    values=torch.full(
                        shape, scales[int(key["o3_lambda"])], dtype=torch.float64
                    ),
                    samples=Labels(
                        names=["system"], values=torch.tensor([[system_index]])
                    ),
                    components=layout_block.components,
                    properties=layout_block.properties,
                )
            )
        references.append(TensorMap(keys=layout.keys, blocks=blocks))

    datasets = [Dataset.from_dict({"system": systems, TARGET: references})]

    model.compute_covariance(datasets, batch_size=2, is_distributed=False)
    model.compute_cholesky_decomposition()
    model.calibrate(
        datasets,
        batch_size=2,
        is_distributed=False,
        calibration_method="squared_residuals",
    )

    block_keys = model.target_block_keys[TARGET]
    multipliers = [
        model._get_multiplier(UNCERTAINTY, block_key).item() for block_key in block_keys
    ]
    assert multipliers[0] > 10.0 * multipliers[1], (
        "the two blocks were calibrated with the same multiplier, so a block's "
        f"residual scale is not reaching its own multiplier: {multipliers}"
    )

    # the point of a per-block multiplier: every block ends up calibrated, i.e. its
    # residuals are of the size its own uncertainty claims
    outputs = model(
        systems,
        {
            TARGET: ModelOutput(sample_kind="system"),
            UNCERTAINTY: ModelOutput(sample_kind="system"),
        },
    )
    for index in range(len(layout.keys)):
        residuals = (
            outputs[TARGET].block(index).values.detach()
            - references[0].block(index).values
        )
        uncertainties = outputs[UNCERTAINTY].block(index).values.detach()
        assert torch.allclose(
            (residuals**2 / uncertainties**2).mean(),
            torch.tensor(1.0, dtype=torch.float64),
            rtol=1e-6,
        )
