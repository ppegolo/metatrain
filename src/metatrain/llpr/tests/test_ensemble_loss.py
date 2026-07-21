"""Tests for LLPR ensemble losses on targets that carry components.

The LLPR ensemble stacks its members into the property dimension of the
``_ensemble`` output. For a target with components (e.g. a vector target such as
``non_conservative_force``) those components sit between the samples and the
properties, so the number of ensemble members has to be recovered from the last
dimension. Recovering it from dimension 1 instead yields ``n_ens == 1`` for a
vector target, which silently makes the ensemble variance NaN and destroys any
training run that uses these losses.
"""

import copy

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System

from metatrain.llpr.model import LLPRUncertaintyModel
from metatrain.pet import PET
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_generic_target_info
from metatrain.utils.loss import TensorMapGaussianNLLLoss
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)


NCF = "non_conservative_force"
NCF_ENSEMBLE = f"mtt::aux::{NCF}_ensemble"
NCF_UNCERTAINTY = f"mtt::aux::{NCF}_uncertainty"


def _single_key(device=None):
    return Labels(names=["_"], values=torch.tensor([[0]], device=device))


def _tensor_map(values, components, n_prop, properties_name):
    """Build a single-block TensorMap with the given values and components."""
    n_samples = values.shape[0]
    samples = Labels(
        names=["system", "atom"],
        values=torch.stack(
            [torch.zeros(n_samples, dtype=torch.int32), torch.arange(n_samples)], dim=1
        ),
    )
    return TensorMap(
        keys=_single_key(),
        blocks=[
            TensorBlock(
                values=values,
                samples=samples,
                components=components,
                properties=Labels.range(properties_name, n_prop),
            )
        ],
    )


def _xyz_components():
    return [Labels(names=["xyz"], values=torch.tensor([[0], [1], [2]]))]


def test_ensemble_loss_recovers_members_for_vector_target():
    """The ensemble axis must be read from the last dimension, not dimension 1.

    A vector target has 3 components and (here) 1 property. Reading the ensemble
    count from dimension 1 gives ``3 // 3 == 1``, collapsing every member into a
    single one and making the variance undefined.
    """
    n_samples, n_ens, n_prop = 4, 5, 1
    torch.manual_seed(0)

    # values chosen so that per-(sample, component) mean and variance across the
    # ensemble are known independently of the implementation
    members = torch.randn(n_samples, 3, n_ens, n_prop, dtype=torch.float64)
    expected_mean = members.mean(dim=-2)
    expected_var = members.var(dim=-2, unbiased=True)

    ens_values = members.reshape(n_samples, 3, n_ens * n_prop)

    predictions = {
        NCF: _tensor_map(expected_mean, _xyz_components(), n_prop, NCF),
        NCF_ENSEMBLE: _tensor_map(
            ens_values, _xyz_components(), n_ens * n_prop, "ensemble_member"
        ),
    }
    targets = {
        NCF: _tensor_map(
            torch.zeros_like(expected_mean), _xyz_components(), n_prop, NCF
        )
    }

    loss = TensorMapGaussianNLLLoss(
        name=NCF, gradient=None, weight=1.0, reduction="mean"
    )
    value = loss.compute(predictions, targets)

    assert torch.isfinite(value), "ensemble loss must not be NaN for a vector target"

    # Reproduce the Gaussian NLL from the reference mean/variance to pin down that
    # the loss really used all `n_ens` members with the right axis ordering.
    reference = torch.nn.GaussianNLLLoss(reduction="mean")(
        expected_mean.reshape(-1),
        torch.zeros_like(expected_mean).reshape(-1),
        expected_var.reshape(-1),
    )
    assert torch.allclose(value, reference)


def test_ensemble_loss_scalar_target_unchanged():
    """The scalar (energy-like) case must keep working: it is the pre-existing path."""
    n_samples, n_ens, n_prop = 4, 5, 1
    torch.manual_seed(0)

    members = torch.randn(n_samples, n_ens, n_prop, dtype=torch.float64)
    expected_mean = members.mean(dim=-2)
    expected_var = members.var(dim=-2, unbiased=True)

    predictions = {
        "energy": _tensor_map(expected_mean, [], n_prop, "energy"),
        "energy_ensemble": _tensor_map(
            members.reshape(n_samples, n_ens * n_prop), [], n_ens * n_prop, "ensemble"
        ),
    }
    targets = {"energy": _tensor_map(torch.zeros_like(expected_mean), [], n_prop, "e")}

    loss = TensorMapGaussianNLLLoss(
        name="energy", gradient=None, weight=1.0, reduction="mean"
    )
    value = loss.compute(predictions, targets)

    reference = torch.nn.GaussianNLLLoss(reduction="mean")(
        expected_mean.reshape(-1),
        torch.zeros_like(expected_mean).reshape(-1),
        expected_var.reshape(-1),
    )
    assert torch.isfinite(value)
    assert torch.allclose(value, reference)


def test_ensemble_loss_rejects_single_member():
    """A single member makes the variance undefined; fail loudly instead of NaN."""
    n_samples, n_prop = 4, 1
    values = torch.randn(n_samples, 3, n_prop, dtype=torch.float64)

    predictions = {
        NCF: _tensor_map(values, _xyz_components(), n_prop, NCF),
        NCF_ENSEMBLE: _tensor_map(values, _xyz_components(), n_prop, "ensemble_member"),
    }
    targets = {
        NCF: _tensor_map(torch.zeros_like(values), _xyz_components(), n_prop, NCF)
    }

    loss = TensorMapGaussianNLLLoss(
        name=NCF, gradient=None, weight=1.0, reduction="mean"
    )
    with pytest.raises(ValueError, match="ensemble variance is undefined"):
        loss.compute(predictions, targets)


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
    """The LLPR ensemble must reproduce the analytic uncertainty for a vector target.

    ``_uncertainty`` is ``alpha * sqrt(f^T C^-1 f)`` while ``_ensemble`` is built by
    sampling last-layer weights from the same posterior. Their spread must agree, and
    this is what makes the ensemble losses meaningful. A wrong ensemble/component axis
    ordering breaks this agreement while leaving all shapes intact.
    """
    torch.manual_seed(0)
    n_ens = 20000
    model, _ = _small_pet_llpr(n_ens)

    n_feat = model.ll_feat_size
    features = torch.randn(200, n_feat, dtype=torch.float64)
    model._get_covariance(NCF_UNCERTAINTY)[:] = features.T @ features
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

    # PET's last layer has one independent weight row per Cartesian component, so the
    # posterior is isotropic: sigma must be identical across x, y and z.
    assert torch.allclose(uncertainty, uncertainty[:, :1, :].expand_as(uncertainty))
