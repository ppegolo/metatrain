import copy
import re

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
from metatrain.utils.last_layer import SHARED_FEATURE_KEY
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)


TARGET = "mtt::spherical"
UNCERTAINTY = "mtt::aux::spherical_uncertainty"
ENSEMBLE = "mtt::aux::spherical_ensemble"
NUM_ENSEMBLE_MEMBERS = 8


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


def _wrapped_model(backbone_factory, dataset_info: DatasetInfo, ensembles=False):
    backbone = backbone_factory(dataset_info).to(torch.float64)
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
    covariance = model._get_covariance(UNCERTAINTY, SHARED_FEATURE_KEY)
    covariance[:] = torch.eye(covariance.shape[0], dtype=covariance.dtype)
    model.compute_cholesky_decomposition(regularizer=1e-8)
    return model


@pytest.mark.parametrize("backbone", [_soap_bpnn, _pet], ids=["soap_bpnn", "pet"])
def test_uncertainty_mirrors_the_target_layout(backbone):
    """Check the uncertainty has the target's own keys, components and
    properties."""
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
    """Check the ensemble has the target's keys and its mean reproduces the
    prediction block by block (PET)."""
    dataset_info = _dataset_info()
    model = _with_identity_covariance(
        _wrapped_model(_pet, dataset_info, ensembles=True)
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
    """Check requesting ensembles for SOAP-BPNN's lambda=2 block fails loudly."""
    dataset_info = _dataset_info()
    message = (
        f"Cannot generate LLPR ensembles for '{TARGET}': the wrapped model "
        "declares no last-layer readout weights for the block(s) "
        "['mtt::spherical_o3_lambda_2_o3_sigma_1']. Uncertainties are still "
        "available for this target; remove it from the `num_ensemble_members` "
        "section."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        _wrapped_model(_soap_bpnn, dataset_info, ensembles=True)


def test_calibration_is_per_block():
    """Check each block is calibrated against its own residuals: with residuals
    differing 100x between blocks, both must come out calibrated."""
    dataset_info = _dataset_info()
    model = _wrapped_model(_pet, dataset_info)
    systems = _systems(model, 8)

    # references far larger than the untrained model's predictions make the
    # lambda=0 residuals ~100x the lambda=2 ones
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

    # every block ends up calibrated: its residuals are of the size its own
    # uncertainty claims
    outputs = model(
        systems,
        {
            TARGET: ModelOutput(sample_kind="system"),
            UNCERTAINTY: ModelOutput(sample_kind="system"),
        },
    )
    for index in range(len(layout.keys)):
        # every reference is the same constant, so `references[0]` serves them all
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


NCF = "non_conservative_force"
NCF_UNCERTAINTY = f"mtt::aux::{NCF}_uncertainty"
NCF_ENSEMBLE = f"mtt::aux::{NCF}_ensemble"


def _small_pet_llpr(n_ensemble_members):
    """A deliberately tiny PET wrapped in LLPR, with a vector target."""
    target_info = get_generic_target_info(
        NCF,
        {
            "quantity": "force",
            "unit": "eV/Angstrom",
            "type": {"cartesian": {"rank": 1}},
            "num_subtargets": 1,
            "sample_kind": "atom",
        },
    )
    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6], targets={NCF: target_info}
    )

    pet_hypers = copy.deepcopy(get_default_hypers("pet")["model"])
    for key in ("d_pet", "d_head", "d_node", "d_feedforward"):
        pet_hypers[key] = 1
    for key in ("num_heads", "num_attention_layers", "num_gnn_layers"):
        pet_hypers[key] = 1

    backbone = PET(pet_hypers, dataset_info).to(torch.float64)
    model = LLPRUncertaintyModel(
        {"num_ensemble_members": {NCF: n_ensemble_members}}, dataset_info
    )
    model.set_wrapped_model(backbone)
    return model.to(torch.float64), dataset_info


def test_ensemble_variance_matches_analytic_uncertainty():
    """Check the ensemble variance against the analytic ``alpha^2 f^T C^-1 f``
    uncertainty for a vector target, with a non-unit calibration factor."""
    torch.manual_seed(0)
    n_ens = 20000
    model, _ = _small_pet_llpr(n_ens)

    # inject a known covariance and multiplier through the private buffers:
    # fitting/calibrating on data would leave no analytic reference to check
    covariance = model._get_covariance(NCF_UNCERTAINTY, SHARED_FEATURE_KEY)
    features = torch.randn(200, covariance.shape[0], dtype=torch.float64)
    covariance[:] = features.T @ features
    model.compute_cholesky_decomposition(regularizer=1e-8)
    # a multiplier != 1 catches mishandling of the calibration factor
    (block_key,) = model.target_block_keys[NCF]
    model._get_multiplier(NCF_UNCERTAINTY, block_key)[:] = 2.5
    model.generate_ensemble()

    system = System(
        types=torch.tensor([6, 1, 1, 1]),
        positions=torch.tensor(
            [[0.0, 0, 0], [1.1, 0, 0], [0, 1.1, 0], [0, 0, 1.1]], dtype=torch.float64
        ),
        cell=torch.zeros((3, 3), dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, get_requested_neighbor_lists(model))

    outputs = model(
        [system],
        {
            NCF: ModelOutput(sample_kind="atom"),
            NCF_UNCERTAINTY: ModelOutput(sample_kind="atom"),
            NCF_ENSEMBLE: ModelOutput(sample_kind="atom"),
        },
    )
    uncertainty = outputs[NCF_UNCERTAINTY].block().values.detach()
    ensemble = outputs[NCF_ENSEMBLE].block().values.detach()

    n_prop = outputs[NCF].block().values.shape[-1]
    ensemble = ensemble.reshape(ensemble.shape[0], 3, n_ens, n_prop)
    ensemble_var = ensemble.var(dim=-2, unbiased=True)

    # Monte Carlo error on a variance from n_ens samples is ~sqrt(2 / n_ens)
    torch.testing.assert_close(
        ensemble_var, uncertainty**2, rtol=5.0 * (2.0 / n_ens) ** 0.5, atol=0.0
    )
