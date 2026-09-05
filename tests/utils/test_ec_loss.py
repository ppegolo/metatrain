"""
Tests for the electrostatic-complementarity loss and its precontracted machinery.

The load-bearing check is the contraction identity: the loss never sees a
surface point, only the geometry-only tensors ``M = V K V^T``, ``t_f = V K n_f``
and ``s_fg = n_f^T K n_g``, so its EC must equal the pointwise area-weighted
anticorrelation computed directly from the patch ingredients. The full
PySCF-order convention stack (l ordering, the l=1 permutation, NaN padding) is
exercised by round-tripping coefficient vectors through densified TensorMaps
built the way the collate pipeline builds them.
"""

import warnings

import numpy as np
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

from metatrain.utils.loss import ECMSELoss, LossType, _flatten_to_pyscf_order
from metatrain.utils.pyscf_loss import (
    atomic_numbers_of,
    build_auxiliary_molecule,
    compute_ec_machinery,
    ec_fragment_split_name,
    ec_pointwise_pieces,
    get_ec_machinery_transform,
    nuclear_charges_of,
)


pyscf = pytest.importorskip("pyscf")

AUX_BASIS = "def2-universal-jfit"
TARGET = "mtt::density"


def _hf_chain(n_molecules: int, spacing: float = 3.0) -> System:
    """``n_molecules`` HF molecules along x: two elements, easy to grow."""
    positions = []
    for i in range(n_molecules):
        positions.append([i * spacing, 0.0, 0.0])
        positions.append([i * spacing, 0.0, 0.92])
    return System(
        types=torch.tensor([9, 1] * n_molecules),
        positions=torch.tensor(positions, dtype=torch.float64),
        cell=torch.zeros(3, 3, dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
    )


def _hi_hf_dimer() -> System:
    """HI facing HF along z: one def2-ECP element (iodine, 28 core electrons)."""
    return System(
        types=torch.tensor([53, 1, 9, 1]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.61], [0.0, 0.0, 4.5], [0.0, 0.0, 5.42]],
            dtype=torch.float64,
        ),
        cell=torch.zeros(3, 3, dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
    )


class TestEffectiveCorePotential:
    """The nuclear potentials must use ``Z_eff`` when the reference density does.

    PySCF keeps the ECP only in the molecule's charge column, so an auxiliary
    molecule built without it reports ``Z`` and every nuclear term downstream
    is off by ``n_core / r`` around the heavy atom, 28/r for iodine.
    """

    def test_the_auxiliary_molecule_splits_element_from_nuclear_charge(self):
        plain = build_auxiliary_molecule(_hi_hf_dimer(), AUX_BASIS)
        ecp = build_auxiliary_molecule(_hi_hf_dimer(), AUX_BASIS, "def2-svp")
        assert list(atomic_numbers_of(plain)) == [53, 1, 9, 1]
        assert list(atomic_numbers_of(ecp)) == [53, 1, 9, 1]
        assert list(nuclear_charges_of(plain)) == [53, 1, 9, 1]
        assert list(nuclear_charges_of(ecp)) == [25, 1, 9, 1]
        assert ecp.nelectron == plain.nelectron - 28
        # The integrals do not see the ECP: same basis, same operator.
        assert ecp.nao == plain.nao
        assert np.allclose(ecp.intor("int2c2e"), plain.intor("int2c2e"))

    def test_the_nuclear_potential_follows_the_effective_charge(self):
        system = _hi_hf_dimer()
        with pytest.warns(RuntimeWarning, match="no 'ecp' set"):
            bare = ec_pointwise_pieces(system, AUX_BASIS, split=2)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with_ecp = ec_pointwise_pieces(system, AUX_BASIS, split=2, ecp="def2-svp")
        assert bare is not None and with_ecp is not None

        # Same patch (radii follow the element), same operator and masks.
        assert np.allclose(bare["weights"], with_ecp["weights"])
        assert np.allclose(bare["operator"], with_ecp["operator"])
        for a, b in zip(bare["masks"], with_ecp["masks"], strict=True):
            assert np.array_equal(a, b)

        # The HF partner has no ECP atom: its potential is unchanged. The HI
        # fragment's potential drops by exactly 28/r from the iodine nucleus.
        assert np.allclose(bare["nuclear"][1], with_ecp["nuclear"][1])
        auxmol = build_auxiliary_molecule(system, AUX_BASIS)
        centres = auxmol.atom_coords()
        points = _patch_points(system)
        distance = np.linalg.norm(points - centres[0], axis=1)
        assert np.allclose(
            bare["nuclear"][0] - with_ecp["nuclear"][0], 28.0 / distance, rtol=1e-10
        )

    def test_light_elements_do_not_warn_and_are_unchanged(self):
        system = _hf_chain(2)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            bare = ec_pointwise_pieces(system, AUX_BASIS, split=2)
            with_ecp = ec_pointwise_pieces(system, AUX_BASIS, split=2, ecp="def2-svp")
        for a, b in zip(bare["nuclear"], with_ecp["nuclear"], strict=True):
            assert np.allclose(a, b)

    def test_the_machinery_and_its_cache_distinguish_the_ecp(self):
        system = _hi_hf_dimer()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            bare = compute_ec_machinery(system, AUX_BASIS, split=2)
        with_ecp = compute_ec_machinery(system, AUX_BASIS, split=2, ecp="def2-svp")
        assert torch.allclose(bare[0], with_ecp[0])  # M = V K V^T: geometry only
        assert not torch.allclose(bare[1][0], with_ecp[1][0])  # t_0 sees iodine
        assert torch.allclose(bare[1][1], with_ecp[1][1])  # t_1 does not
        assert not torch.allclose(bare[2], with_ecp[2])

        # Through the transform, with a system id so the cache is used: the
        # ECP is part of the key, so the two configurations never alias.
        extra = {
            "mtt::aux::system_index": TensorMap(
                Labels.single(),
                [
                    TensorBlock(
                        values=torch.tensor([[7.0]], dtype=torch.float64),
                        samples=Labels(
                            "system", torch.zeros((1, 1), dtype=torch.int32)
                        ),
                        components=[],
                        properties=Labels(
                            "system_index", torch.zeros((1, 1), dtype=torch.int32)
                        ),
                    )
                ],
            ),
            ec_fragment_split_name(TARGET): _split_map([2]),
        }
        key = f"{TARGET}_ec_machinery_constants"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _, _, out_bare = get_ec_machinery_transform({TARGET: AUX_BASIS})(
                [system], {}, dict(extra)
            )
        _, _, out_ecp = get_ec_machinery_transform(
            {TARGET: AUX_BASIS}, 0.0, "def2-svp"
        )([system], {}, dict(extra))
        assert torch.allclose(out_ecp[key][0].values.reshape(2, 2), with_ecp[2])
        assert torch.allclose(out_bare[key][0].values.reshape(2, 2), bare[2])


def _patch_points(system: System) -> np.ndarray:
    """The EC patch points of ``system`` for ``split=2``, in Bohr."""
    from metatrain.utils.pyscf_loss import _ec_interface_patch

    auxmol = build_auxiliary_molecule(system, AUX_BASIS)
    points, _ = _ec_interface_patch(auxmol.atom_coords(), atomic_numbers_of(auxmol), 2)
    return points


def _radial_counts(mol) -> dict:
    """Map ``(atomic_number, l)`` to radial functions per atom of that element."""
    charges = mol.atom_charges()
    representative: dict = {}
    for i_atom, charge in enumerate(charges):
        representative.setdefault(int(charge), i_atom)
    counts: dict = {}
    for shell in range(mol.nbas):
        i_atom = mol.bas_atom(shell)
        charge = int(charges[i_atom])
        if representative[charge] != i_atom:
            continue
        key = (charge, int(mol.bas_angular(shell)))
        counts[key] = counts.get(key, 0) + int(mol.bas_nctr(shell))
    return counts


def _densified_batch(systems, vectors) -> TensorMap:
    """
    Build the batched, densified TensorMap whose flattening is ``vectors``.

    Mirrors the collate pipeline's layout — one block per ``o3_lambda``, samples
    ``(system, atom)`` over the whole batch with batch-local system ids, NaN on
    the property axis where an element has no such function — and scatters each
    system's PySCF-ordered vector into it by walking atoms, then ``l``, then
    ``n``, then ``m`` with the l=1 permutation, i.e. the inverse walk of
    :py:func:`~metatrain.utils.loss._flatten_to_pyscf_order`.
    """
    counts = _radial_counts(build_auxiliary_molecule(systems[0], AUX_BASIS))
    all_types = [int(t) for system in systems for t in system.types]
    samples = [
        [i_system, i_atom]
        for i_system, system in enumerate(systems)
        for i_atom in range(len(system))
    ]
    l_values = sorted({key[1] for key in counts})

    blocks = {}
    for angular in l_values:
        n_props = max(counts.get((z, angular), 0) for z in set(all_types))
        values = torch.full(
            (len(all_types), 2 * angular + 1, n_props), torch.nan, dtype=torch.float64
        )
        for row, z in enumerate(all_types):
            values[row, :, : counts.get((z, angular), 0)] = 0.0
        blocks[angular] = values

    cursors = [0] * len(vectors)
    row = 0
    for i_system, system in enumerate(systems):
        flat = vectors[i_system]
        for z in [int(t) for t in system.types]:
            for angular in l_values:
                order = [2, 0, 1] if angular == 1 else list(range(2 * angular + 1))
                for i_n in range(counts.get((z, angular), 0)):
                    for i_m in order:
                        blocks[angular][row, i_m, i_n] = flat[cursors[i_system]]
                        cursors[i_system] += 1
            row += 1
    for i_system, flat in enumerate(vectors):
        assert cursors[i_system] == len(flat)

    return TensorMap(
        Labels(
            names=["o3_lambda", "o3_sigma"],
            values=torch.tensor([[angular, 1] for angular in l_values]),
        ),
        [
            TensorBlock(
                values=blocks[angular],
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.tensor(samples, dtype=torch.int32),
                ),
                components=[
                    Labels(
                        names=["o3_mu"],
                        values=torch.arange(
                            -angular, angular + 1, dtype=torch.int32
                        ).reshape(-1, 1),
                    )
                ],
                properties=Labels(
                    names=["n"],
                    values=torch.arange(
                        blocks[angular].shape[2], dtype=torch.int32
                    ).reshape(-1, 1),
                ),
            )
            for angular in l_values
        ],
    )


def _pointwise_ec(pieces, coefficients) -> float:
    """EC evaluated directly on the patch points, the reference route."""
    operator, weights = pieces["operator"], pieces["weights"]
    potentials = [
        nuclear - (mask * coefficients) @ operator
        for nuclear, mask in zip(pieces["nuclear"], pieces["masks"], strict=True)
    ]
    centred = [v - weights @ v for v in potentials]
    covariance = weights @ (centred[0] * centred[1])
    spreads = [np.sqrt(weights @ v**2) for v in centred]
    return float(-covariance / (spreads[0] * spreads[1]))


def _split_map(splits) -> TensorMap:
    """Per-system fragment splits, as the dataset carries them."""
    return TensorMap(
        Labels.single(),
        [
            TensorBlock(
                values=torch.tensor([[float(s)] for s in splits]),
                samples=Labels(
                    "system",
                    torch.arange(len(splits), dtype=torch.int32).reshape(-1, 1),
                ),
                components=[],
                properties=Labels(
                    "fragment_split", torch.zeros((1, 1)).to(torch.int32)
                ),
            )
        ],
    )


def _extra_via_transform(systems, splits):
    """Attach the machinery the way the trainer does: through the transform."""
    extra = {ec_fragment_split_name(TARGET): _split_map(splits)}
    transform = get_ec_machinery_transform({TARGET: AUX_BASIS})
    _, _, extra = transform(systems, {}, extra)
    return extra


def _loss(reduction: str = "sum") -> ECMSELoss:
    return ECMSELoss(TARGET, None, weight=1.0, reduction=reduction, aux_basis=AUX_BASIS)


def _random_vectors(systems, seed: int):
    generator = np.random.default_rng(seed)
    return [
        generator.normal(size=build_auxiliary_molecule(system, AUX_BASIS).nao)
        for system in systems
    ]


def test_contracted_ec_matches_pointwise():
    system = _hf_chain(2)
    pieces = ec_pointwise_pieces(system, AUX_BASIS, split=2)
    machinery = compute_ec_machinery(system, AUX_BASIS, split=2)
    coefficients = _random_vectors([system], seed=0)[0]
    contracted = ECMSELoss._ec(torch.from_numpy(coefficients), *machinery)
    assert contracted is not None
    assert abs(float(contracted) - _pointwise_ec(pieces, coefficients)) < 1e-12


def test_autograd_matches_finite_differences():
    system = _hf_chain(2)
    machinery = compute_ec_machinery(system, AUX_BASIS, split=2)
    coefficients = torch.from_numpy(_random_vectors([system], seed=1)[0])
    coefficients.requires_grad_(True)
    ec = ECMSELoss._ec(coefficients, *machinery)
    ec.backward()
    step = 1e-6
    for index in [0, 7, len(coefficients) // 2, len(coefficients) - 1]:
        plus, minus = coefficients.detach().clone(), coefficients.detach().clone()
        plus[index] += step
        minus[index] -= step
        numerical = (
            float(ECMSELoss._ec(plus, *machinery))
            - float(ECMSELoss._ec(minus, *machinery))
        ) / (2.0 * step)
        assert abs(numerical - float(coefficients.grad[index])) < 1e-6


def test_loss_is_zero_for_an_exact_prediction():
    systems = [_hf_chain(2)]
    target = _densified_batch(systems, _random_vectors(systems, seed=2))
    value = _loss().compute(
        {TARGET: target}, {TARGET: target}, _extra_via_transform(systems, [2])
    )
    assert float(value) == 0.0


def test_dense_prediction_matches_a_padded_one():
    # The model's output carries no NaN padding: only the densified *target*
    # marks which coefficients an element actually has. A prediction flattened
    # on its own would therefore count every padded slot as a real coefficient
    # and disagree with the target on any batch mixing element basis sizes, so
    # the loss must take its layout from the target. HF has two element types,
    # which is what makes the padding non-trivial here.
    systems = [_hf_chain(2)]
    reference, prediction = _random_vectors(systems, seed=11), None
    prediction = [reference[0] + 0.05]
    target = _densified_batch(systems, reference)
    padded = _densified_batch(systems, prediction)

    dense = TensorMap(
        padded.keys,
        [
            TensorBlock(
                values=torch.nan_to_num(padded.block(key).values, nan=7.5),
                samples=padded.block(key).samples,
                components=padded.block(key).components,
                properties=padded.block(key).properties,
            )
            for key in padded.keys
        ],
    )
    assert torch.isnan(padded.block(padded.keys[0]).values).any()
    assert not torch.isnan(dense.block(dense.keys[0]).values).any()

    extra = _extra_via_transform(systems, [2])
    loss = _loss()
    assert float(
        loss.compute({TARGET: dense}, {TARGET: target}, extra)
    ) == pytest.approx(float(loss.compute({TARGET: padded}, {TARGET: target}, extra)))


@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
def test_batch_with_mixed_naux_matches_per_system(reduction):
    # Two systems with different naux: the loss must reduce exactly the
    # per-system values, with nothing contributed by padding or by the ragged
    # segmentation.
    systems = [_hf_chain(2), _hf_chain(3)]
    splits = [2, 2]
    references = _random_vectors(systems, seed=3)
    predictions = [
        reference + 0.01 * perturbation
        for reference, perturbation in zip(
            references, _random_vectors(systems, seed=4), strict=True
        )
    ]

    expected = []
    for system, split, prediction, reference in zip(
        systems, splits, predictions, references, strict=True
    ):
        pieces = ec_pointwise_pieces(system, AUX_BASIS, split)
        expected.append(
            (_pointwise_ec(pieces, prediction) - _pointwise_ec(pieces, reference)) ** 2
        )

    value = _loss(reduction).compute(
        {TARGET: _densified_batch(systems, predictions)},
        {TARGET: _densified_batch(systems, references)},
        _extra_via_transform(systems, splits),
    )
    if reduction == "none":
        assert np.allclose(value.numpy(), expected, atol=1e-12)
    elif reduction == "sum":
        assert abs(float(value) - sum(expected)) < 1e-12
    else:
        assert abs(float(value) - sum(expected) / len(expected)) < 1e-12


def test_structure_without_patch_contributes_exactly_zero():
    # 30 Angstrom between the fragments: no interface patch, so the second
    # system must contribute exactly zero, not a numerically small value.
    systems = [_hf_chain(2), _hf_chain(2, spacing=30.0)]
    assert compute_ec_machinery(systems[1], AUX_BASIS, split=2) is None
    references = _random_vectors(systems, seed=5)
    predictions = [
        reference + 0.01 * perturbation
        for reference, perturbation in zip(
            references, _random_vectors(systems, seed=6), strict=True
        )
    ]
    extra = _extra_via_transform(systems, [2, 2])
    both = _loss("none").compute(
        {TARGET: _densified_batch(systems, predictions)},
        {TARGET: _densified_batch(systems, references)},
        extra,
    )
    assert float(both[1]) == 0.0
    assert float(both[0]) > 0.0


def test_gradient_flows_through_the_batch():
    systems = [_hf_chain(2)]
    references = _random_vectors(systems, seed=7)
    predictions = _densified_batch(
        systems, [references[0] + 0.01 * _random_vectors(systems, seed=8)[0]]
    )
    # Densified values contain NaN padding, so the graph must be attached to a
    # NaN-free leaf: mask the padded entries the way the model's output has
    # them, then re-insert NaN for the flattening to drop.
    block_values = [predictions.block(key).values for key in predictions.keys]
    leaves = [torch.nan_to_num(values).requires_grad_(True) for values in block_values]
    with_graph = TensorMap(
        predictions.keys,
        [
            TensorBlock(
                values=torch.where(torch.isnan(values), values, leaf),
                samples=predictions.block(key).samples,
                components=predictions.block(key).components,
                properties=predictions.block(key).properties,
            )
            for key, values, leaf in zip(
                predictions.keys, block_values, leaves, strict=True
            )
        ],
    )
    value = _loss().compute(
        {TARGET: with_graph},
        {TARGET: _densified_batch(systems, references)},
        _extra_via_transform(systems, [2]),
    )
    value.backward()
    gradient = torch.cat([leaf.grad.reshape(-1) for leaf in leaves])
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().max()) > 0.0


def test_missing_machinery_raises():
    systems = [_hf_chain(2)]
    target = _densified_batch(systems, _random_vectors(systems, seed=9))
    with pytest.raises(RuntimeError, match="ec_machinery"):
        _loss().compute({TARGET: target}, {TARGET: target}, {})


def test_missing_split_field_raises():
    transform = get_ec_machinery_transform({TARGET: AUX_BASIS})
    with pytest.raises(RuntimeError, match="fragment_split"):
        transform([_hf_chain(2)], {}, {})


class _ScalerAdapter:
    """A ``BaseScaler`` behind the ``apply_scales`` interface of the model wrapper.

    ``get_remove_scale_transform`` only calls ``apply_scales``; delegating to
    :py:meth:`BaseScaler.forward` exercises the real scaling code path without
    the model wrapper's DatasetInfo machinery.
    """

    def __init__(self, base):
        self.base = base

    def apply_scales(
        self, systems, targets, remove, use_per_target_scales, use_per_property_scales
    ):
        return self.base.forward(
            systems,
            targets,
            remove=remove,
            use_per_target_scales=use_per_target_scales,
            use_per_property_scales=use_per_property_scales,
        )


def _fitted_scaler(template: TensorMap, scale_of_type: dict):
    """A real ``BaseScaler`` for the density layout, with per-type scales set."""
    from metatrain.scaler._base_scaler import BaseScaler

    layout = TensorMap(
        template.keys,
        [
            TensorBlock(
                values=block.values[:0],
                samples=Labels(block.samples.names, block.samples.values[:0]),
                components=block.components,
                properties=block.properties,
            )
            for block in template.blocks()
        ],
    )
    base = BaseScaler(sorted(scale_of_type), {TARGET: layout})
    for key in base.per_target_scales[TARGET].keys:
        block = base.per_target_scales[TARGET].block(key)
        # Scale samples index into the scaler's sorted atomic types.
        for row, atomic_type in enumerate(base.atomic_types):
            block.values[row] = scale_of_type[int(atomic_type)]
    return _ScalerAdapter(base)


def test_per_target_scaling_is_undone():
    # The trainer's scale-removal transform hands the loss coefficients divided
    # by fitted per-target scales (per block and atomic type, not one scalar),
    # and records the reciprocal it applied under ``removed_scale_name``. EC is
    # not homogeneous in the coefficients — the nuclear potentials are fixed —
    # so the loss must undo the scaling: with the record present it must return
    # the *true* (dEC)^2, exactly as for unscaled coefficients.
    from metatrain.utils.scaler import get_remove_scale_transform
    from metatrain.utils.scaler.remove import removed_scale_name

    systems = [_hf_chain(2)]
    references = _random_vectors(systems, seed=11)
    predictions = [references[0] + 0.01 * _random_vectors(systems, seed=12)[0]]
    prediction_map = _densified_batch(systems, predictions)
    reference_map = _densified_batch(systems, references)
    extra = _extra_via_transform(systems, [2])

    true_value = float(
        _loss().compute({TARGET: prediction_map}, {TARGET: reference_map}, extra)
    )

    # Non-uniform per-type scales: a single-scalar undo could not pass this.
    transform = get_remove_scale_transform(
        _fitted_scaler(reference_map, {1: 2.0, 9: 0.5})
    )
    _, scaled_targets, extra = transform(systems, {TARGET: reference_map}, extra)
    _, scaled_predictions, _ = transform(systems, {TARGET: prediction_map}, {})

    # Without the record the loss can only see the scaled density: a different,
    # silently wrong number. This is the hazard the record closes.
    without_record = dict(extra)
    without_record.pop(removed_scale_name(TARGET))
    wrong = float(
        _loss().compute(
            {TARGET: scaled_predictions[TARGET]}, scaled_targets, without_record
        )
    )
    assert abs(wrong - true_value) > 1e-2 * abs(true_value)

    recovered = float(
        _loss().compute({TARGET: scaled_predictions[TARGET]}, scaled_targets, extra)
    )
    assert abs(recovered - true_value) < 1e-14


def test_remove_scale_transform_records_the_removal():
    # The generic recording side: whatever ``remove_scale`` does to a target,
    # the recorded map holds the reciprocal factor per entry, NaN pattern
    # included, so any loss can undo it through the same flattening.
    from metatrain.utils.scaler import get_remove_scale_transform
    from metatrain.utils.scaler.remove import removed_scale_name

    systems = [_hf_chain(2)]
    target = _densified_batch(systems, _random_vectors(systems, seed=13))
    transform = get_remove_scale_transform(_fitted_scaler(target, {1: 2.0, 9: 4.0}))
    _, targets, extra = transform(systems, {TARGET: target}, {})

    recorded = extra[removed_scale_name(TARGET)]
    for key in target.keys:
        values = target.block(key).values
        scaled = targets[TARGET].block(key).values
        factor = recorded.block(key).values
        mask = ~torch.isnan(values)
        assert torch.equal(torch.isnan(factor), ~mask)
        assert torch.allclose(scaled[mask], (values * factor)[mask])


def test_loss_type_registration():
    assert LossType.from_key("ec_mse").cls is ECMSELoss


def test_flatten_size_matches_the_basis():
    # The scatter helper above must produce exactly the layout the loss
    # flattens; a size mismatch here would invalidate every other test.
    systems = [_hf_chain(2), _hf_chain(3)]
    vectors = _random_vectors(systems, seed=10)
    flat, counts = _flatten_to_pyscf_order(_densified_batch(systems, vectors))
    assert len(flat) == sum(len(v) for v in vectors)
    assert torch.allclose(flat, torch.from_numpy(np.concatenate(vectors)))
    assert int(counts.sum()) == len(flat)


# ── Hirshfeld partition ───────────────────────────────────────────────────────


def _pointwise_ec_from_operators(pieces, coefficients) -> float:
    """EC on the patch points from explicit fragment operators."""
    weights = pieces["weights"]
    potentials = [
        nuclear - coefficients @ operator
        for nuclear, operator in zip(
            pieces["nuclear"], pieces["operators"], strict=True
        )
    ]
    centred = [v - weights @ v for v in potentials]
    covariance = weights @ (centred[0] * centred[1])
    spreads = [np.sqrt(weights @ v**2) for v in centred]
    return float(-covariance / (spreads[0] * spreads[1]))


def test_hirshfeld_operators_are_additive_and_differ_from_ri():
    """
    The two Hirshfeld fragment operators sum exactly to the total ESP operator
    (the correction enters with opposite signs), and differ from the RI ones
    where the molecules share density.
    """
    system = _hf_chain(2)
    pieces = ec_pointwise_pieces(system, AUX_BASIS, split=2, partition="hirshfeld")
    total = pieces["operator"]
    np.testing.assert_allclose(
        pieces["operators"][0] + pieces["operators"][1], total, atol=1e-12
    )
    np.testing.assert_allclose(
        pieces["operators"][0] - pieces["operator"] * pieces["masks"][0][:, None],
        pieces["correction"],
    )
    assert np.abs(pieces["correction"]).max() > 1e-4


def test_hirshfeld_machinery_matches_pointwise_and_shifts_ec():
    system = _hf_chain(2)
    coefficients = _random_vectors([system], seed=0)[0]
    pieces = ec_pointwise_pieces(system, AUX_BASIS, split=2, partition="hirshfeld")
    machinery = compute_ec_machinery(system, AUX_BASIS, split=2, partition="hirshfeld")
    assert machinery is not None
    operators, nuclear, weights = machinery
    assert operators.shape == (2 * len(coefficients), len(pieces["weights"]))
    ec = ECMSELoss._ec_pointwise(
        torch.from_numpy(coefficients), operators, nuclear, weights
    )
    assert ec is not None
    assert abs(float(ec) - _pointwise_ec_from_operators(pieces, coefficients)) < 1e-12
    ri = ECMSELoss._ec(
        torch.from_numpy(coefficients),
        *compute_ec_machinery(system, AUX_BASIS, split=2),
    )
    assert abs(float(ec) - float(ri)) > 1e-6


def test_hirshfeld_loss_runs_through_the_transform():
    systems = [_hf_chain(2), _hf_chain(3)]
    splits = [2, 4]
    extra = {ec_fragment_split_name(TARGET): _split_map(splits)}
    transform = get_ec_machinery_transform({TARGET: AUX_BASIS}, 0.0, None, "hirshfeld")
    _, _, extra = transform(systems, {}, extra)
    reference = _random_vectors(systems, seed=1)
    perturbed = [
        v + 0.05 * np.random.default_rng(2).normal(size=len(v)) for v in reference
    ]
    targets = {TARGET: _densified_batch(systems, reference)}
    predictions = {TARGET: _densified_batch(systems, perturbed)}
    loss = ECMSELoss(
        TARGET,
        None,
        weight=1.0,
        reduction="none",
        aux_basis=AUX_BASIS,
        partition="hirshfeld",
    )
    values = loss.compute(predictions, targets, extra)
    assert values.shape == (2,)
    assert torch.all(torch.isfinite(values)) and torch.all(values > 0)
    exact = loss.compute(targets, targets, extra)
    assert torch.all(exact == 0)


def test_unknown_partition_is_rejected():
    with pytest.raises(ValueError, match="partition"):
        ECMSELoss(
            TARGET,
            None,
            weight=1.0,
            reduction="sum",
            aux_basis=AUX_BASIS,
            partition="becke",
        )
    with pytest.raises(ValueError, match="partition"):
        get_ec_machinery_transform({TARGET: AUX_BASIS}, 0.0, None, "becke")
