"""Tests for the metric-spec machinery and the electron-count penalty.

PySCF-free on purpose: the spec functions are pure string handling, and the
charge-penalty tests drive the loss with hand-built matrices, so these keep
running in environments where the optional ``pyscf`` dependency is absent.
"""

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatrain.utils.density_hooks import _aux_bases_by_metric
from metatrain.utils.loss import DensityMSELossViaC
from metatrain.utils.pyscf_loss import (
    DEFAULT_LR_EPS,
    coulomb_matrix_name,
    make_metric_spec,
    metric_matrix_name,
    overlap_matrix_name,
    pack_metric_matrices,
    parse_metric_spec,
)


def test_spec_round_trips_and_keeps_plain_metrics_unchanged():
    # Plain metrics must keep their exact spec string and extra_data key: the
    # spec doubles as the extra_data key and the metric-matrix cache key, so
    # any change here would silently invalidate existing configs and caches.
    assert make_metric_spec("overlap") == "overlap"
    assert make_metric_spec("coulomb", omega=0.0, charge_weight=0.0) == "coulomb"
    assert metric_matrix_name("mtt::ri", "overlap") == overlap_matrix_name("mtt::ri")
    assert metric_matrix_name("mtt::ri", "coulomb") == coulomb_matrix_name("mtt::ri")

    spec = make_metric_spec("coulomb", omega=0.15, eps=0.01, charge_weight=2.0)
    assert parse_metric_spec(spec) == ("coulomb", 0.15, 0.01, 2.0, 0.0, 0.0, 0.0, 1.4)

    # multipole terms round-trip, and their absence keeps pre-existing spec
    # strings (= extra_data and cache keys) byte-identical
    full = make_metric_spec(
        "overlap", charge_weight=1.0, dipole_weight=2.0, quadrupole_weight=3.0
    )
    assert parse_metric_spec(full) == ("overlap", 0.0, 0.01, 1.0, 2.0, 3.0, 0.0, 1.4)
    assert "d=" not in spec and "Q=" not in spec
    assert (
        make_metric_spec("overlap", dipole_weight=0.5)
        == "overlap|omega=0|eps=0.01|q=0|d=0.5"
    )

    esp = make_metric_spec("overlap", esp_weight=2.0)
    assert parse_metric_spec(esp)[6:] == (2.0, 1.4)  # default shell
    shell = parse_metric_spec(
        make_metric_spec("overlap", esp_weight=2.0, esp_shell=1.8)
    )
    assert shell[7] == 1.8
    assert "esp=" not in make_metric_spec("overlap", dipole_weight=0.5)

    # omega without an explicit eps must keep a positive-definite floor: the
    # long-range metric alone is rank-deficient
    eps = parse_metric_spec(make_metric_spec("coulomb", omega=0.15))[2]
    assert eps == DEFAULT_LR_EPS > 0.0

    # different hyper-parameters must map to different keys, or a matrix built
    # for one omega would be served for another
    other = make_metric_spec("coulomb", omega=0.30, eps=0.01, charge_weight=2.0)
    assert metric_matrix_name("mtt::ri", spec) != metric_matrix_name("mtt::ri", other)


def test_spec_rejects_invalid_combinations():
    with pytest.raises(ValueError, match="unknown metric"):
        make_metric_spec("euclidean")
    # silently ignoring omega on the overlap metric would train against a
    # different objective than configured
    with pytest.raises(ValueError, match="Coulomb-metric option"):
        make_metric_spec("overlap", omega=0.15)
    with pytest.raises(ValueError, match="omega must be"):
        make_metric_spec("coulomb", omega=-1.0)
    with pytest.raises(ValueError, match="charge_weight must be"):
        make_metric_spec("coulomb", charge_weight=-1.0)
    with pytest.raises(ValueError, match="eps must be"):
        make_metric_spec("coulomb", omega=0.15, eps=-1.0)
    with pytest.raises(ValueError, match="dipole_weight must be"):
        make_metric_spec("coulomb", dipole_weight=-1.0)
    with pytest.raises(ValueError, match="quadrupole_weight must be"):
        make_metric_spec("coulomb", quadrupole_weight=-1.0)
    with pytest.raises(ValueError, match="esp_weight must be"):
        make_metric_spec("coulomb", esp_weight=-1.0)
    with pytest.raises(ValueError, match="esp_shell must be"):
        make_metric_spec("coulomb", esp_weight=1.0, esp_shell=0.0)


def _ri_tensor_map(value_l0: float, values_l1: list) -> TensorMap:
    """A one-atom RI coefficient map with one s and one p shell (4 functions)."""
    samples = Labels(names=["system", "atom"], values=torch.tensor([[0, 0]]))
    properties = Labels(names=["n"], values=torch.tensor([[0]]))
    return TensorMap(
        keys=Labels(
            names=["o3_lambda", "o3_sigma"],
            values=torch.tensor([[0, 1], [1, 1]], dtype=torch.int32),
        ),
        blocks=[
            TensorBlock(
                values=torch.tensor([value_l0], dtype=torch.float64).reshape(1, 1, 1),
                samples=samples,
                components=[Labels(names=["o3_mu"], values=torch.tensor([[0]]))],
                properties=properties,
            ),
            TensorBlock(
                values=torch.tensor(values_l1, dtype=torch.float64).reshape(1, 3, 1),
                samples=samples,
                components=[
                    Labels(names=["o3_mu"], values=torch.tensor([[-1], [0], [1]]))
                ],
                properties=properties,
            ),
        ],
    )


def _charge_loss(charge_weight: float, pred: TensorMap, targ: TensorMap) -> float:
    """Evaluate the density loss under a pure charge-penalty metric.

    The metric matrix is exactly ``charge_weight * S_vec S_vec^T`` with the
    charge on the single s function, so the loss isolates the electron-count
    term.
    """
    target_name = "mtt::ri"
    s_vector = torch.tensor([5.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    matrix = charge_weight * torch.outer(s_vector, s_vector)
    spec = make_metric_spec("coulomb", charge_weight=charge_weight)
    extra = {metric_matrix_name(target_name, spec): pack_metric_matrices([matrix])}
    loss = DensityMSELossViaC(
        target_name,
        None,
        weight=1.0,
        reduction="mean",
        metric="coulomb",
        aux_basis="some-basis",
        charge_weight=charge_weight,
    )
    return float(loss.compute({target_name: pred}, {target_name: targ}, extra))


def test_charge_penalty_is_exactly_the_electron_count_error():
    # S.dc is the predicted electron count minus the reference's, so the term
    # must contribute charge_weight * (S.dc)^2 and nothing else. Here the s
    # coefficients differ by 2 and S carries 5 on the s function: S.dc = 10.
    value = _charge_loss(
        3.0, _ri_tensor_map(3.0, [0.0] * 3), _ri_tensor_map(1.0, [0.0] * 3)
    )
    assert value == pytest.approx(3.0 * 10.0**2)


def test_charge_penalty_ignores_charge_conserving_errors():
    # An error orthogonal to S (identical s coefficients, differing p block)
    # redistributes charge at fixed total, which a count penalty -- unlike a
    # generic coefficient penalty -- must not penalise.
    value = _charge_loss(
        3.0, _ri_tensor_map(1.0, [7.0, 0.0, 0.0]), _ri_tensor_map(1.0, [0.0] * 3)
    )
    assert value == pytest.approx(0.0)


def test_loss_and_hooks_agree_on_the_metric_spec():
    # The loss and the collate-transform hooks derive the spec independently
    # from the same config; if they diverge, the transform attaches a key the
    # loss does not look up.
    config = {
        "type": "density_mse_via_c",
        "aux_basis": "some-basis",
        "metric": "coulomb",
        "omega": 0.15,
        "charge_weight": 2.0,
    }
    (hook_spec,) = _aux_bases_by_metric({"mtt::ri": config})
    loss = DensityMSELossViaC(
        "mtt::ri",
        None,
        weight=1.0,
        reduction="mean",
        metric=config["metric"],
        aux_basis=config["aux_basis"],
        omega=config["omega"],
        charge_weight=config["charge_weight"],
    )
    assert loss.metric == hook_spec
