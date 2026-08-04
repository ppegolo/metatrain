import copy

import torch
from metatomic.torch import ModelOutput

from metatrain.experimental.space import SPACE
from metatrain.llpr.model import LLPRUncertaintyModel
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_generic_target_info
from metatrain.utils.last_layer import LastLayerSlice, assemble_block_weights
from metatrain.utils.testing import LLPRInterfaceTests
from metatrain.utils.testing.llpr import (
    NUM_ENSEMBLE_MEMBERS,
    SPHERICAL_ENSEMBLE,
    SPHERICAL_TARGET,
    SPHERICAL_UNCERTAINTY,
    fit_llpr,
    make_systems,
    rotate_system,
    spherical_dataset_info,
    wrap_backbone,
)

from . import MODEL_HYPERS


CART_TARGET = "mtt::force_like"
CART_UNCERTAINTY = "mtt::aux::force_like_uncertainty"
CART_ENSEMBLE = "mtt::aux::force_like_ensemble"


def _make_hypers() -> dict:
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["num_element_channels"] = 2
    hypers["num_gnn_layers"] = 1
    hypers["num_tensor_products"] = 2
    # max_eigenvalue=25.0 gives l_max=2, needed for the lambda=2 block
    hypers["radial_basis"]["max_eigenvalue"] = 25.0
    hypers["radial_basis"]["mlp_expansion_ratio"] = 1
    hypers["radial_basis"]["mlp_depth"] = 2
    hypers["mlp_head_expansion_ratio"] = 1
    return hypers


def _backbone(dataset_info):
    return SPACE(_make_hypers(), dataset_info).to(torch.float64)


def _cartesian_dataset_info(rank: int = 1) -> DatasetInfo:
    target = get_generic_target_info(
        CART_TARGET,
        {
            "quantity": "",
            "unit": "",
            "num_subtargets": 1,
            "sample_kind": "atom",
            "type": {"cartesian": {"rank": rank}},
        },
    )
    return DatasetInfo(
        length_unit="Angstrom", atomic_types=[6], targets={CART_TARGET: target}
    )


class TestLLPRInterface(LLPRInterfaceTests):
    def make_backbone(self, dataset_info):
        return _backbone(dataset_info)


def test_uncertainty_and_ensemble_spread_follow_scaler():
    """Check uncertainties and ensemble spreads follow the wrapped scaler's
    per-property scales."""
    dataset_info = spherical_dataset_info()
    backbone = _backbone(dataset_info)
    scales = backbone.scaler.model.scales[SPHERICAL_TARGET]
    scales.block(1).values[:] *= 3.0  # lambda=2 block

    model = LLPRUncertaintyModel(
        {"num_ensemble_members": {SPHERICAL_TARGET: NUM_ENSEMBLE_MEMBERS}},
        dataset_info,
    )
    model.set_wrapped_model(backbone)
    model = model.to(torch.float64)
    systems = make_systems(backbone, 8)
    fit_llpr(model, systems)
    torch.manual_seed(42)
    model.generate_ensemble()

    reference = wrap_backbone(
        _backbone(spherical_dataset_info()), spherical_dataset_info(), ensembles=True
    )
    reference.model.load_state_dict(
        {k: v for k, v in backbone.state_dict().items() if not k.startswith("scaler.")},
        strict=False,
    )
    fit_llpr(reference, systems)
    torch.manual_seed(42)
    reference.generate_ensemble()

    outputs = {
        SPHERICAL_UNCERTAINTY: ModelOutput(sample_kind="system"),
        SPHERICAL_ENSEMBLE: ModelOutput(sample_kind="system"),
    }
    out = model([systems[0]], outputs)
    out_ref = reference([systems[0]], outputs)

    unc = out[SPHERICAL_UNCERTAINTY].block(1).values
    unc_ref = out_ref[SPHERICAL_UNCERTAINTY].block(1).values
    assert torch.allclose(unc, 3.0 * unc_ref, rtol=1e-8)

    members = out[SPHERICAL_ENSEMBLE].block(1).values
    members_ref = out_ref[SPHERICAL_ENSEMBLE].block(1).values
    spread = members - members.mean(dim=-1, keepdim=True)
    spread_ref = members_ref - members_ref.mean(dim=-1, keepdim=True)
    assert torch.allclose(spread, 3.0 * spread_ref, rtol=1e-8)


def test_cartesian_rank1_slices_reproduce_readout():
    """Check the declared slices and the aligned last-layer features (the l=1
    readout, components reordered to x, y, z) reproduce the prediction."""
    dataset_info = _cartesian_dataset_info()
    backbone = _backbone(dataset_info)
    system = make_systems(backbone, 1)[0]
    out = backbone(
        [system],
        {
            CART_TARGET: ModelOutput(sample_kind="atom"),
            "mtt::aux::force_like_last_layer_features": ModelOutput(sample_kind="atom"),
        },
    )
    llf = out["mtt::aux::force_like_last_layer_features"]
    prediction = out[CART_TARGET]
    assert llf.keys == prediction.keys
    (slices_as_tuples,) = backbone.last_layer_parameter_slices[CART_TARGET].values()
    slices = [LastLayerSlice(*s) for s in slices_as_tuples]
    weights = assemble_block_weights(backbone.state_dict(), slices)
    computed = torch.einsum("smk,pk->smp", llf.block().values, weights)
    assert torch.allclose(computed, prediction.block().values, atol=1e-12)


def test_cartesian_rank1_uncertainty_and_ensemble(tmp_path):
    """Check component-resolved, rotation-invariant uncertainty and ensemble on
    a rank-1 Cartesian target, including TorchScript export."""
    dataset_info = _cartesian_dataset_info()
    model = wrap_backbone(
        _backbone(dataset_info), dataset_info, ensembles=True, target=CART_TARGET
    )
    systems = make_systems(model.model, 8)
    fit_llpr(model, systems, target=CART_TARGET)
    torch.manual_seed(42)
    model.generate_ensemble()

    outputs = {
        CART_UNCERTAINTY: ModelOutput(sample_kind="atom"),
        CART_ENSEMBLE: ModelOutput(sample_kind="atom"),
    }
    out = model([systems[0]], outputs)
    unc = out[CART_UNCERTAINTY].block().values
    members = out[CART_ENSEMBLE].block().values  # (atoms, 3, members)

    # component-resolved: x, y, z uncertainties must not all coincide
    assert not torch.allclose(unc, unc.mean(dim=1, keepdim=True))
    # the ensemble spread estimates the LLPR uncertainty up to Monte Carlo
    # noise: 32 members give ~13% relative error on the std, rtol=0.5 is ~4 sigma
    spread = members - members.mean(dim=-1, keepdim=True)
    std = torch.sqrt((spread**2).mean(dim=-1))
    assert torch.allclose(std, unc.squeeze(-1), rtol=0.5)

    # the squared uncertainty summed over (x, y, z) is rotation-invariant
    rotated_system = rotate_system(model.model, systems[0])
    unc_rot = model([rotated_system], outputs)[CART_UNCERTAINTY].block().values
    assert torch.allclose((unc**2).sum(dim=1), (unc_rot**2).sum(dim=1), rtol=1e-6)

    # the LLPR-wrapped model must export and save (TorchScript end to end)
    model.export().save(str(tmp_path / "model-llpr.pt"))


def test_cartesian_rank2_uncertainty_via_shared_features():
    """Check a target without per-block features (rank-2 Cartesian) can be
    covariance-fitted and produce uncertainties through its declared shared
    invariant feature block."""
    dataset_info = _cartesian_dataset_info(rank=2)
    model = wrap_backbone(
        _backbone(dataset_info), dataset_info, ensembles=False, target=CART_TARGET
    )
    systems = make_systems(model.model, 4)
    fit_llpr(model, systems, target=CART_TARGET)
    out = model([systems[0]], {CART_UNCERTAINTY: ModelOutput(sample_kind="atom")})
    unc = out[CART_UNCERTAINTY].block().values
    assert unc.shape[1:] == (3, 3, 1)
    assert torch.all(unc > 0.0)


def test_torchscript_with_no_aligned_targets():
    """Check a model with empty `llf_aligned_targets` still scripts (it can
    only be typed through the class-level annotation)."""
    dataset_info = _cartesian_dataset_info(rank=2)
    backbone = SPACE(_make_hypers(), dataset_info)
    assert backbone.llf_aligned_targets == []
    backbone.prepare_for_export()
    torch.jit.script(backbone)


def test_pseudotensor_block_is_declared():
    """Check pseudotensor blocks are declared like any other: SPACE ignores
    o3_sigma."""
    dataset_info = spherical_dataset_info(irreps=[{"o3_lambda": 1, "o3_sigma": -1}])
    model = wrap_backbone(_backbone(dataset_info), dataset_info, ensembles=True)
    assert model.ensemble_block_keys[SPHERICAL_TARGET] == [
        "mtt::spherical_o3_lambda_1_o3_sigma_-1"
    ]
