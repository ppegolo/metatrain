import copy
import re

import pytest
import torch
from metatomic.torch import ModelOutput

from metatrain.experimental.mace import MetaMACE
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_generic_target_info
from metatrain.utils.last_layer import LastLayerSlice, assemble_block_weights
from metatrain.utils.testing import LLPRInterfaceTests
from metatrain.utils.testing.llpr import (
    SPHERICAL_TARGET,
    fit_llpr,
    make_systems,
    rotate_system,
    spherical_dataset_info,
    wrap_backbone,
)

from . import MODEL_HYPERS


def _make_hypers() -> dict:
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["hidden_irreps"] = "4x0e + 4x1o + 4x2e"
    hypers["MLP_irreps"] = "4x0e"
    hypers["num_interactions"] = 2
    return hypers


def _backbone(dataset_info):
    return MetaMACE(_make_hypers(), dataset_info).to(torch.float64)


class TestLLPRInterface(LLPRInterfaceTests):
    def make_backbone(self, dataset_info):
        return _backbone(dataset_info)


def test_missing_irrep_block_refuses_wrapping_for_ensembles():
    """Check a target whose irrep is absent from MACE's hidden features (here
    1e, masked to zero) stays undeclared: ensembles are refused, and without
    ensembles the target simply has no uncertainty output."""
    dataset_info = spherical_dataset_info(irreps=[{"o3_lambda": 1, "o3_sigma": -1}])
    message = (
        f"Output '{SPHERICAL_TARGET}' in ensembles section is not supported "
        "by the model"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        wrap_backbone(_backbone(dataset_info), dataset_info, ensembles=True)
    model = wrap_backbone(_backbone(dataset_info), dataset_info, ensembles=False)
    assert SPHERICAL_TARGET not in model.outputs_list


CART_TARGET = "mtt::force_like"
CART_UNCERTAINTY = "mtt::aux::force_like_uncertainty"
CART_ENSEMBLE = "mtt::aux::force_like_ensemble"


def _cartesian_dataset_info() -> DatasetInfo:
    target = get_generic_target_info(
        CART_TARGET,
        {
            "quantity": "",
            "unit": "",
            "num_subtargets": 1,
            "sample_kind": "atom",
            "type": {"cartesian": {"rank": 1}},
        },
    )
    return DatasetInfo(
        length_unit="Angstrom", atomic_types=[6], targets={CART_TARGET: target}
    )


def test_cartesian_rank1_slices_reproduce_readout():
    """Check the declared slices and the aligned last-layer features (the 1o
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


def test_cartesian_rank1_uncertainty_and_ensemble():
    """Check component-resolved, rotation-invariant uncertainty and recentered
    ensemble on a rank-1 Cartesian target."""
    dataset_info = _cartesian_dataset_info()
    model = wrap_backbone(
        _backbone(dataset_info), dataset_info, ensembles=True, target=CART_TARGET
    )
    systems = make_systems(model.model, 8)
    fit_llpr(model, systems, target=CART_TARGET)
    torch.manual_seed(42)
    model.generate_ensemble()

    outputs = {
        CART_TARGET: ModelOutput(sample_kind="atom"),
        CART_UNCERTAINTY: ModelOutput(sample_kind="atom"),
        CART_ENSEMBLE: ModelOutput(sample_kind="atom"),
    }
    out = model([systems[0]], outputs)
    prediction = out[CART_TARGET].block().values  # (atoms, 3, 1)
    unc = out[CART_UNCERTAINTY].block().values
    members = out[CART_ENSEMBLE].block().values  # (atoms, 3, members)

    # component-resolved: x, y, z uncertainties must not all coincide
    assert not torch.allclose(unc, unc.mean(dim=1, keepdim=True))
    # re-centering: the ensemble mean is the model prediction
    assert torch.allclose(members.mean(dim=-1, keepdim=True), prediction, atol=1e-10)

    # the squared uncertainty summed over (x, y, z) is rotation-invariant
    rotated_system = rotate_system(model.model, systems[0])
    unc_rot = model([rotated_system], outputs)[CART_UNCERTAINTY].block().values
    assert torch.allclose((unc**2).sum(dim=1), (unc_rot**2).sum(dim=1), rtol=1e-6)
