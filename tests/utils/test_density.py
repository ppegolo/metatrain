"""Tests of the density loss, against references computed with PySCF (see
``tests/resources/density/generate.py``)."""

import json
import re

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

from metatrain.utils.augmentation import O3Augmenter
from metatrain.utils.data.target_info import get_generic_target_info
from metatrain.utils.density.basis import integrals, load_element_basis
from metatrain.utils.density.loss import DensityMSELoss, get_density_transforms
from metatrain.utils.density.metric import MetricBuilder
from metatrain.utils.loss import create_loss

from ..conftest import RESOURCES_PATH


AUX_BASIS = str(RESOURCES_PATH / "density" / "basis.json")
REFERENCES = json.loads((RESOURCES_PATH / "density" / "reference.json").read_text())
TARGET = "mtt::density"


@pytest.fixture
def systems():
    return [
        System(
            types=torch.tensor(reference["types"]),
            positions=torch.tensor(reference["positions"], dtype=torch.float64),
            cell=torch.zeros(3, 3, dtype=torch.float64),
            pbc=torch.tensor([False, False, False]),
        )
        for reference in REFERENCES
    ]


def _target(systems, seed):
    """Random coefficients, densified and padded with NaN as by the trainer."""
    generator = torch.Generator().manual_seed(seed)
    types = [z for system in systems for z in system.types.tolist()]
    samples = torch.tensor(
        [[i, atom] for i, system in enumerate(systems) for atom in range(len(system))]
    )
    counts = {}
    for z in set(types):
        offsets = load_element_basis(AUX_BASIS, z).offsets
        counts[z] = {angular: len(set(o.tolist())) for angular, o in offsets.items()}
    blocks, keys = [], []
    for angular in sorted({a for c in counts.values() for a in c}):
        n_max = max(c.get(angular, 0) for c in counts.values())
        values = torch.full(
            (len(types), 2 * angular + 1, n_max), torch.nan, dtype=torch.float64
        )
        for atom, z in enumerate(types):
            n = counts[z].get(angular, 0)
            values[atom, :, :n] = torch.randn(
                2 * angular + 1, n, generator=generator, dtype=torch.float64
            )
        keys.append([angular, 1])
        blocks.append(
            TensorBlock(
                values=values,
                samples=Labels(["system", "atom"], samples),
                components=[
                    Labels(["o3_mu"], torch.arange(-angular, angular + 1)[:, None])
                ],
                properties=Labels(["n"], torch.arange(n_max)[:, None]),
            )
        )
    return TensorMap(Labels(["o3_lambda", "o3_sigma"], torch.tensor(keys)), blocks)


def _map(tensor, function):
    return TensorMap(
        tensor.keys,
        [
            TensorBlock(
                function(block.values, int(key["o3_lambda"])),
                block.samples,
                block.components,
                block.properties,
            )
            for key, block in tensor.items()
        ],
    )


def _dense(tensor):
    """The padding filled with a large number, as in the output of a model."""
    return _map(tensor, lambda values, _: torch.nan_to_num(values, nan=1e3))


def _vectors(tensor, systems):
    """Coefficients of each system in the order of PySCF, with explicit loops."""
    vectors, row = [], 0
    for system in systems:
        vector = []
        for _ in range(len(system)):
            for key in sorted(tensor.keys, key=lambda key: int(key[0])):
                values = tensor.block(key).values[row]
                order = [2, 0, 1] if int(key[0]) == 1 else range(values.shape[0])
                for n in range(values.shape[1]):
                    for m in order:
                        if not torch.isnan(values[m, n]):
                            vector.append(values[m, n])
            row += 1
        vectors.append(torch.stack(vector))
    return vectors


class _Scaler:
    """Divides the targets by ``1 + atomic type + angular momentum``."""

    def __init__(self, scaled=True):
        self.scaled = scaled

    def apply_scales(self, systems, targets, **kwargs):
        types = torch.cat([system.types for system in systems])
        return {
            name: _map(
                tensor,
                lambda values, angular: (
                    values / (1 + types[:, None, None] + angular if self.scaled else 1)
                ),
            )
            for name, tensor in targets.items()
        }


def _extra(systems, target, scaler=None):
    (transform,) = get_density_transforms(
        {TARGET: {"type": "density_mse"}}, scaler or _Scaler(scaled=False)
    )
    return transform(systems, {TARGET: target}, {})[2]


def _loss(**kwargs):
    return DensityMSELoss(TARGET, None, 1.0, "none", aux_basis=AUX_BASIS, **kwargs)


@pytest.mark.parametrize("metric", ["overlap", "coulomb"])
def test_matrices_match_pyscf(metric, systems):
    builder = MetricBuilder(AUX_BASIS, metric)
    geometries = [(system.types, system.positions) for system in systems]
    for dtype, rtol in ((torch.float64, 1e-9), (torch.float32, 1e-5)):
        matrices = builder.compute_batch(geometries, torch.device("cpu"), dtype)
        for matrix, reference in zip(matrices, REFERENCES, strict=True):
            vector = torch.sin(torch.arange(1, len(matrix) + 1, dtype=torch.float64))
            expected = torch.tensor(reference[metric], dtype=torch.float64)
            error = (matrix.double() @ vector - expected).norm() / expected.norm()
            assert error < rtol


def test_integrals_match_pyscf(systems):
    for system, reference in zip(systems, REFERENCES, strict=True):
        torch.testing.assert_close(
            integrals(AUX_BASIS, system.types),
            torch.tensor(reference["integrals"], dtype=torch.float64),
        )


@pytest.mark.parametrize("metric", ["overlap", "coulomb"])
def test_loss_matches_the_quadratic_form(metric, systems):
    target = _target(systems, 0)
    prediction = _dense(_target(systems, 1))
    loss = _loss(metric=metric).compute(
        {TARGET: prediction}, {TARGET: target}, _extra(systems, target)
    )

    # the prediction with the padding of the target
    padded = _map(
        prediction,
        lambda values, angular: (
            values + 0 * target.block({"o3_lambda": angular}).values
        ),
    )
    deltas = [
        p - t
        for p, t in zip(
            _vectors(padded, systems), _vectors(target, systems), strict=True
        )
    ]
    geometries = [(system.types, system.positions) for system in systems]
    matrices = MetricBuilder(AUX_BASIS, metric).compute_batch(
        geometries, torch.device("cpu")
    )
    for i, (delta, matrix) in enumerate(zip(deltas, matrices, strict=True)):
        torch.testing.assert_close(loss[i], delta @ matrix @ delta, rtol=1e-9, atol=0)


def test_loss_undoes_the_removed_scale(systems):
    target = _target(systems, 0)
    prediction = _dense(_target(systems, 1))
    expected = _loss().compute(
        {TARGET: prediction}, {TARGET: target}, _extra(systems, target)
    )

    scaler = _Scaler()
    scaled_target = scaler.apply_scales(systems, {TARGET: target})[TARGET]
    scaled_prediction = scaler.apply_scales(systems, {TARGET: prediction})[TARGET]
    loss = _loss().compute(
        {TARGET: scaled_prediction},
        {TARGET: scaled_target},
        _extra(systems, scaled_target, scaler),
    )
    torch.testing.assert_close(loss, expected, rtol=1e-9, atol=0)


@pytest.mark.parametrize("determinant", [1.0, -1.0])
def test_loss_is_invariant_under_o3(determinant, systems):
    target = _target(systems, 0)
    zeros = _dense(_map(target, lambda values, _: 0 * values))
    irreps = [
        {"o3_lambda": int(key[0]), "o3_sigma": 1, "num": len(block.properties)}
        for key, block in target.items()
    ]
    info = get_generic_target_info(
        TARGET,
        {
            "quantity": "",
            "unit": "",
            "type": {"spherical": {"irreps": irreps}},
            "num_subtargets": 1,
            "sample_kind": "atom",
        },
    )
    rotations = []
    for seed in range(len(systems)):
        generator = torch.Generator().manual_seed(seed)
        q, _ = torch.linalg.qr(
            torch.randn(3, 3, generator=generator, dtype=torch.float64)
        )
        rotations.append(q * determinant / torch.linalg.det(q))
    rotated_systems, rotated, _ = O3Augmenter({TARGET: info}).apply_augmentations(
        systems, {TARGET: target}, rotations
    )

    loss = _loss(metric="coulomb")
    original = loss.compute({TARGET: zeros}, {TARGET: target}, _extra(systems, target))
    augmented = loss.compute(
        {TARGET: zeros},
        {TARGET: rotated[TARGET]},
        _extra(rotated_systems, rotated[TARGET]),
    )
    torch.testing.assert_close(augmented, original, rtol=1e-9, atol=0)

    # rotating only the geometry changes the loss
    geometry_only = loss.compute(
        {TARGET: zeros}, {TARGET: target}, _extra(rotated_systems, target)
    )
    assert not torch.allclose(geometry_only, original, rtol=1e-4)


def test_mask_zero_targets(systems):
    n_first = len(systems[0])
    target = _map(
        _target(systems, 0),
        lambda values, _: torch.cat([values[:n_first], 0 * values[n_first:]]),
    )
    prediction = _dense(_target(systems, 1))
    args = ({TARGET: prediction}, {TARGET: target}, _extra(systems, target))
    plain = _loss().compute(*args)
    masked = _loss(mask_zero_targets=True).compute(*args)
    assert masked[0] == plain[0]
    assert plain[1] > 0 and masked[1] == 0


def test_sign_invariant_singlets(systems):
    n_first = len(systems[0])

    def l0(values, angular):
        if angular != 0:
            return values
        # the first system integrates to zero, the second far from it
        return torch.cat([0 * values[:n_first], values[n_first:].abs() + 1])

    target = _map(_target(systems, 0), l0)
    prediction = _dense(_map(target, lambda values, _: -values))
    args = ({TARGET: prediction}, {TARGET: target}, _extra(systems, target))
    plain = _loss().compute(*args)
    invariant = _loss(sign_invariant_singlets=True).compute(*args)
    assert plain[0] > 1 and abs(invariant[0]) < 1e-12 * plain[0]
    assert invariant[1] == plain[1]


def test_no_gradient_through_the_padding(systems):
    target = _target(systems, 0)
    prediction = _dense(_target(systems, 1))
    values = [block.values.requires_grad_() for block in prediction.blocks()]
    loss = _loss().compute(
        {TARGET: prediction}, {TARGET: target}, _extra(systems, target)
    )
    loss.sum().backward()
    for value, block in zip(values, target.blocks(), strict=True):
        assert (value.grad[torch.isnan(block.values)] == 0).all()


def test_errors(systems):
    loss = create_loss(
        "density_mse",
        name=TARGET,
        gradient=None,
        weight=1.0,
        reduction="mean",
        aux_basis=AUX_BASIS,
    )
    target = _target(systems, 0)
    prediction = _dense(target)

    message = (
        "the 'density_mse' loss on target 'mtt::density' requires "
        "'mtt::aux::density_geometry' in the extra data; see `get_density_transforms`."
    )
    with pytest.raises(RuntimeError, match=f"^{re.escape(message)}$"):
        loss.compute({TARGET: prediction}, {TARGET: target}, {})

    message = (
        f"target 'mtt::density' has 101 coefficients in this batch, and the "
        f"'{AUX_BASIS}' auxiliary basis 23. Check that 'aux_basis' is the basis the "
        "dataset was fitted in."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        loss.compute(
            {TARGET: prediction}, {TARGET: target}, _extra(systems[:1], target)
        )
