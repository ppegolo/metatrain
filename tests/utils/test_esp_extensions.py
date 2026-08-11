"""The multi-shell ESP surface and the buried-interface ESP penalty.

The single-shell accessible surface constrains the potential where the culling
leaves points — the solvent-exposed skin. Two extensions target what it misses:

* **multi-shell surfaces** (RESP-style, e.g. ``esp_shell=(1.0, 1.4, 2.0)``)
  add outer layers, whose potential is dominated by the low multipoles of the
  density — the far-field content the pointwise metrics underweight;
* **the interface term** (``interface_esp_weight``) samples the two fragments'
  buried contact patches, the region the accessible-surface culling removes by
  construction, and the one binding electrostatics is decided on.

Loss-side tests drive hand-made factors so they run without ``pyscf``; the
factor-construction tests need a real auxiliary basis and are skipped when it
is absent.
"""

import numpy as np
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatrain.utils.loss import DensityMSELossViaC
from metatrain.utils.pyscf_loss import (
    interface_esp_factor_name,
    make_metric_spec,
    metric_matrix_name,
    pack_metric_matrices,
    parse_interface_esp_weight,
    parse_metric_spec,
    strip_esp_from_spec,
)


TARGET = "mtt::ri"
AUX_BASIS = "def2-universal-jfit"


# ── The spec ──────────────────────────────────────────────────────────────────


def test_single_shell_specs_stay_byte_identical():
    # The spec is the extra_data key and the cache key: a change would silently
    # invalidate configs and caches from before these options existed.
    assert make_metric_spec("coulomb") == "coulomb"
    assert make_metric_spec("coulomb", esp_weight=2.0, esp_shell=1.4) == (
        "coulomb|omega=0|eps=0.01|q=0|esp=2|shell=1.4"
    )
    assert make_metric_spec("coulomb", esp_weight=2.0, esp_shell=(1.4,)) == (
        make_metric_spec("coulomb", esp_weight=2.0, esp_shell=1.4)
    )


def test_multi_shell_specs_are_canonical_and_round_trip():
    spec = make_metric_spec("coulomb", esp_weight=2.0, esp_shell=[2.0, 1.0, 1.4, 1.0])
    assert spec.endswith("|esp=2|shell=1,1.4,2")  # sorted, deduplicated
    assert parse_metric_spec(spec)[6:] == (2.0, (1.0, 1.4, 2.0))
    # A single shell parses back to the historical float, not a 1-tuple.
    single = make_metric_spec("coulomb", esp_weight=2.0, esp_shell=1.6)
    assert parse_metric_spec(single)[7] == 1.6

    base, esp_weight, esp_shell = strip_esp_from_spec(spec)
    assert base == "coulomb"  # nothing else lives in the dense matrix
    assert esp_weight == 2.0
    assert esp_shell == (1.0, 1.4, 2.0)

    with pytest.raises(ValueError, match="esp_shell"):
        make_metric_spec("coulomb", esp_weight=1.0, esp_shell=[1.0, -0.5])
    with pytest.raises(ValueError, match="esp_shell"):
        make_metric_spec("coulomb", esp_weight=1.0, esp_shell=[])


def test_the_interface_weight_has_its_own_token():
    assert parse_interface_esp_weight("coulomb") == 0.0
    assert parse_interface_esp_weight(make_metric_spec("coulomb", esp_weight=1.0)) == 0

    spec = make_metric_spec("coulomb", interface_esp_weight=5.0)
    assert spec.endswith("|iesp=5")
    assert parse_interface_esp_weight(spec) == 5.0
    # The documented 8-tuple of parse_metric_spec is unchanged by the token.
    assert parse_metric_spec(spec) == ("coulomb", 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 1.4)
    # The dense part of the spec carries no factored term.
    assert strip_esp_from_spec(spec)[0] == "coulomb"

    with pytest.raises(ValueError, match="interface_esp_weight must be >= 0"):
        make_metric_spec("coulomb", interface_esp_weight=-1.0)


def test_the_hooks_ask_for_the_same_spec_the_loss_reads():
    """A divergence here would lose the factor without any error."""
    from metatrain.utils.density_hooks import _aux_bases_by_metric

    spec = {
        "type": "density_mse_via_c",
        "aux_basis": AUX_BASIS,
        "metric": "coulomb",
        "esp_weight": 2.0,
        "esp_shell": [1.0, 1.4, 2.0],
        "interface_esp_weight": 5.0,
    }
    (built,) = _aux_bases_by_metric({TARGET: spec}).keys()
    loss = DensityMSELossViaC(
        TARGET, None, 1.0, "mean", **{k: v for k, v in spec.items() if k != "type"}
    )
    assert built == loss.metric
    assert parse_interface_esp_weight(built) == 5.0


# ── The loss ──────────────────────────────────────────────────────────────────


def _target_pair(values, reference):
    """A one-system, one-block coefficient target and prediction."""
    maps = []
    for data in (values, reference):
        maps.append(
            TensorMap(
                Labels(
                    ["o3_lambda", "o3_sigma", "atom_type"],
                    torch.tensor([[0, 1, 1]], dtype=torch.int32),
                ),
                [
                    TensorBlock(
                        values=torch.tensor(data, dtype=torch.float64).reshape(
                            -1, 1, 1
                        ),
                        samples=Labels(
                            ["system", "atom"],
                            torch.tensor(
                                [[0, i] for i in range(len(data))], dtype=torch.int32
                            ),
                        ),
                        components=[
                            Labels("o3_mu", torch.zeros((1, 1), dtype=torch.int32))
                        ],
                        properties=Labels(
                            "properties", torch.zeros((1, 1), dtype=torch.int32)
                        ),
                    )
                ],
            )
        )
    return maps


def _interface_loss(weight: float) -> DensityMSELossViaC:
    return DensityMSELossViaC(
        TARGET,
        None,
        1.0,
        "sum",
        metric="coulomb",
        aux_basis=AUX_BASIS,
        interface_esp_weight=weight,
    )


def test_the_loss_applies_the_interface_factor():
    predicted, reference = _target_pair([1.5, 0.5], [1.0, 1.0])
    # Zero metric, so only the penalty contributes and the value is exact:
    # dc = (0.5, -0.5), F dc = (0.5, -1.0), weight * |F dc|^2 = 3 * 1.25.
    factor = torch.tensor([[1.0, 0.0], [1.0, 3.0]], dtype=torch.float64)
    spec = make_metric_spec("coulomb", interface_esp_weight=3.0)
    extra = {
        metric_matrix_name(TARGET, spec): pack_metric_matrices(
            [torch.zeros((2, 2), dtype=torch.float64)]
        ),
        interface_esp_factor_name(TARGET, spec): pack_metric_matrices([factor]),
    }
    loss = _interface_loss(3.0)
    assert loss.metric == spec
    value = float(loss.compute({TARGET: predicted}, {TARGET: reference}, extra))
    assert value == pytest.approx(3.0 * 1.25)

    # A structure with no usable patch ships a zero-row factor and contributes
    # exactly zero.
    empty = torch.zeros((0, 2), dtype=torch.float64)
    extra[interface_esp_factor_name(TARGET, spec)] = pack_metric_matrices([empty])
    assert float(
        loss.compute({TARGET: predicted}, {TARGET: reference}, extra)
    ) == pytest.approx(0.0)


def test_a_missing_interface_factor_is_reported_clearly():
    predicted, reference = _target_pair([1.5, 0.5], [1.0, 1.0])
    spec = make_metric_spec("coulomb", interface_esp_weight=3.0)
    extra = {
        metric_matrix_name(TARGET, spec): pack_metric_matrices(
            [torch.zeros((2, 2), dtype=torch.float64)]
        )
    }
    with pytest.raises(RuntimeError, match="interface_esp_factor"):
        _interface_loss(3.0).compute({TARGET: predicted}, {TARGET: reference}, extra)


# ── The factors ───────────────────────────────────────────────────────────────


def _dimer():
    """Two triatomic fragments in contact, apex atoms facing each other."""
    from metatomic.torch import System

    return System(
        types=torch.tensor([8, 1, 1, 8, 1, 1], dtype=torch.int32),
        positions=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.8, -0.6],
                [0.0, -0.8, -0.6],
                [0.0, 0.0, 3.0],
                [0.0, 0.8, 3.6],
                [0.0, -0.8, 3.6],
            ],
            dtype=torch.float64,
        ),
        cell=torch.zeros(3, 3, dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
    )


def test_multi_shell_surfaces_concatenate_the_single_shells():
    pytest.importorskip("pyscf")
    from metatrain.utils.pyscf_loss import compute_esp_factor, compute_surface_points

    system = _dimer()
    shells = (1.0, 1.4, 2.0)
    coords, weights = compute_surface_points(system, AUX_BASIS, shells)
    parts = [compute_surface_points(system, AUX_BASIS, s) for s in shells]
    np.testing.assert_allclose(
        coords.numpy(), np.concatenate([p[0].numpy() for p in parts])
    )
    np.testing.assert_allclose(
        weights.numpy(), np.concatenate([p[1].numpy() for p in parts])
    )

    factor = compute_esp_factor(system, AUX_BASIS, shells)
    stacked = torch.cat([compute_esp_factor(system, AUX_BASIS, s) for s in shells])
    torch.testing.assert_close(factor, stacked)


def test_the_interface_factor_is_the_weighted_esp_on_the_contact_patches():
    pytest.importorskip("pyscf")
    from pyscf import gto

    from metatrain.utils.pyscf_loss import (
        _ec_interface_patch,
        build_auxiliary_molecule,
        compute_interface_esp_factor,
    )

    system = _dimer()
    split = np.array([0, 0, 0, 1, 1, 1])
    factor = compute_interface_esp_factor(system, AUX_BASIS, split)

    auxmol = build_auxiliary_molecule(system, AUX_BASIS)
    assert factor.shape[1] == auxmol.nao
    assert factor.shape[0] > 0  # the fragments are in contact

    # |F c|^2 must equal the area-weighted squared potential of the fitted
    # density over the union of the two orientations' contact patches.
    centres, charges = auxmol.atom_coords(), auxmol.atom_charges()
    points, areas = [], []
    for orientation in (split, 1 - split):
        p, a = _ec_interface_patch(centres, charges, orientation)
        points.append(p)
        areas.append(a)
    points, areas = np.concatenate(points), np.concatenate(areas)
    assert factor.shape[0] == len(points)

    rng = np.random.default_rng(0)
    coefficients = rng.normal(size=auxmol.nao)
    esp = (
        gto.mole.intor_cross("int2c2e", auxmol, gto.fakemol_for_charges(points)).T
        @ coefficients
    )
    assert float((factor.numpy() @ coefficients) ** 2 @ np.ones(len(points))) == (
        pytest.approx(float(areas @ esp**2))
    )

    # A monomer has no interface: the factor is empty, never an error.
    monomer = compute_interface_esp_factor(system, AUX_BASIS, np.zeros(6, dtype=int))
    assert monomer.shape == (0, auxmol.nao)


def test_the_transform_attaches_the_interface_factor_per_target():
    pytest.importorskip("pyscf")
    from metatrain.utils.pyscf_loss import (
        _metric_matrices_transform,
        compute_interface_esp_factor,
        ec_fragment_name,
    )

    system = _dimer()
    split = [0, 0, 0, 1, 1, 1]
    labels = TensorMap(
        Labels.single(),
        [
            TensorBlock(
                values=torch.tensor(split, dtype=torch.float64).reshape(-1, 1),
                samples=Labels(
                    ["system", "atom"],
                    torch.tensor([[0, a] for a in range(6)], dtype=torch.int32),
                ),
                components=[],
                properties=Labels("label", torch.zeros((1, 1), dtype=torch.int32)),
            )
        ],
    )
    spec = make_metric_spec("coulomb", interface_esp_weight=5.0)
    extra = {ec_fragment_name(TARGET): labels}
    _metric_matrices_transform({TARGET: AUX_BASIS}, spec, [system], {}, extra)

    packed = extra[interface_esp_factor_name(TARGET, spec)]
    expected = compute_interface_esp_factor(system, AUX_BASIS, np.array(split))
    torch.testing.assert_close(packed.block(0).values, expected)

    # Without fragment labels the transform fails loudly, like the EC loss.
    with pytest.raises(RuntimeError, match="fragment"):
        _metric_matrices_transform({TARGET: AUX_BASIS}, spec, [system], {}, {})
