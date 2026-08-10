"""
Tests for the general fragment specification and the partner-jitter augmentation.

Two independent things are checked here.

The **fragment specification** must accept per-atom 0/1 labels, not only a count
of leading atoms, because a ligand inside a pocket has no contiguous split. The
load-bearing test permutes the atom order and asserts the machinery is unchanged
up to that permutation: if anything still assumed contiguity it would change.

The **jitter** must displace the partner and rebuild rather than reuse, must not
touch the cached unaugmented machinery, and must never be applied on validation
-- a reported EC has to be the real score, not a score at an invented placement.
"""

import numpy as np
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

from metatrain.utils.density_hooks import get_density_hooks
from metatrain.utils.loss import ECMSELoss
from metatrain.utils.pyscf_loss import (
    build_auxiliary_molecule,
    compute_ec_machinery,
    ec_fragment_indices,
    ec_fragment_name,
    ec_fragment_split_name,
    get_ec_machinery_transform,
    unpack_metric_matrices,
)


pyscf = pytest.importorskip("pyscf")

AUX_BASIS = "def2-universal-jfit"
TARGET = "mtt::density"


def _hf_chain(n_molecules: int, spacing: float = 3.0) -> System:
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


def _reorder(system: System, order) -> System:
    index = torch.as_tensor(order)
    return System(
        types=system.types[index],
        positions=system.positions[index],
        cell=system.cell,
        pbc=system.pbc,
    )


def _per_atom_labels(labels_per_system) -> TensorMap:
    values, samples = [], []
    for system_index, labels in enumerate(labels_per_system):
        for atom, label in enumerate(labels):
            values.append([float(label)])
            samples.append([system_index, atom])
    return TensorMap(
        Labels.single(),
        [
            TensorBlock(
                values=torch.tensor(values, dtype=torch.float64),
                samples=Labels(
                    ["system", "atom"], torch.tensor(samples, dtype=torch.int32)
                ),
                components=[],
                properties=Labels("fragment", torch.zeros((1, 1)).to(torch.int32)),
            )
        ],
    )


def _per_system_splits(splits) -> TensorMap:
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


def _machinery_via_transform(systems, extra, jitter=0.0, system_ids=None):
    if system_ids is not None:
        extra = dict(extra)
        extra["system_id"] = TensorMap(
            Labels.single(),
            [
                TensorBlock(
                    values=torch.tensor([[float(i)] for i in system_ids]),
                    samples=Labels(
                        "system",
                        torch.arange(len(system_ids), dtype=torch.int32).reshape(-1, 1),
                    ),
                    components=[],
                    properties=Labels("system_id", torch.zeros((1, 1)).to(torch.int32)),
                )
            ],
        )
    transform = get_ec_machinery_transform({TARGET: AUX_BASIS}, jitter)
    _, _, extra = transform(systems, {}, dict(extra))
    return extra


# ── the fragment specification ────────────────────────────────────────────────


def test_fragment_indices_accepts_both_spellings():
    own, partner = ec_fragment_indices(2, 4)
    assert own.tolist() == [0, 1] and partner.tolist() == [2, 3]

    own, partner = ec_fragment_indices([0, 1, 0, 1], 4)
    assert own.tolist() == [0, 2] and partner.tolist() == [1, 3]


def test_fragment_indices_rejects_bad_labels():
    with pytest.raises(ValueError, match="one per atom"):
        ec_fragment_indices([0, 1, 0], 4)
    with pytest.raises(ValueError, match="must be 0"):
        ec_fragment_indices([0, 2, 0, 1], 4)


def test_labels_reproduce_the_contiguous_split():
    """The general spelling must agree with the count where both apply."""
    system = _hf_chain(2)
    by_count = compute_ec_machinery(system, AUX_BASIS, 2)
    by_labels = compute_ec_machinery(system, AUX_BASIS, [0, 0, 1, 1])
    for a, b in zip(by_count, by_labels, strict=True):
        assert torch.allclose(a, b, atol=0.0, rtol=0.0)


def test_interleaved_fragments_match_the_contiguous_case():
    """A ligand whose atoms are not contiguous gives the same physics.

    The atoms are permuted and the labels permuted with them, so the two
    machineries describe the same structure in two atom orders. The moment
    matrix must therefore agree after the induced permutation of the auxiliary
    functions.
    """
    system = _hf_chain(2)
    order = [0, 2, 1, 3]  # interleave the two HF molecules
    permuted = _reorder(system, order)
    labels = np.zeros(4, dtype=int)
    labels[[order.index(2), order.index(3)]] = 1  # the partner's atoms, moved

    reference = compute_ec_machinery(system, AUX_BASIS, 2)
    general = compute_ec_machinery(permuted, AUX_BASIS, labels)
    assert general is not None

    # Auxiliary functions follow the atom order, so build the permutation.
    mol = build_auxiliary_molecule(system, AUX_BASIS)
    ao_loc = mol.ao_loc_nr()
    per_atom = {}
    for shell in range(mol.nbas):
        per_atom.setdefault(mol.bas_atom(shell), []).extend(
            range(ao_loc[shell], ao_loc[shell + 1])
        )
    mapping = np.concatenate([per_atom[atom] for atom in order])

    assert torch.allclose(
        general[0], reference[0][np.ix_(mapping, mapping)], atol=1e-10
    )
    assert torch.allclose(general[2], reference[2], atol=1e-10)


def test_transform_prefers_per_atom_labels():
    systems = [_hf_chain(2)]
    extra = {
        ec_fragment_name(TARGET): _per_atom_labels([[0, 0, 1, 1]]),
        # A split that would be wrong, to prove which field is used.
        ec_fragment_split_name(TARGET): _per_system_splits([3]),
    }
    from_labels = _machinery_via_transform(systems, extra)
    expected = compute_ec_machinery(systems[0], AUX_BASIS, [0, 0, 1, 1])
    (moments,) = unpack_metric_matrices(from_labels[f"{TARGET}_ec_machinery_moments"])
    assert torch.allclose(moments, expected[0], atol=1e-12)


def test_transform_reports_a_missing_specification():
    with pytest.raises(RuntimeError, match="requires a fragment specification"):
        _machinery_via_transform([_hf_chain(2)], {})


# ── the jitter ────────────────────────────────────────────────────────────────


def test_jitter_changes_the_machinery():
    systems = [_hf_chain(2)]
    extra = {ec_fragment_name(TARGET): _per_atom_labels([[0, 0, 1, 1]])}
    plain = _machinery_via_transform(systems, extra, jitter=0.0, system_ids=[0])
    shaken = _machinery_via_transform(systems, extra, jitter=0.3, system_ids=[0])
    key = f"{TARGET}_ec_machinery_moments"
    assert not torch.allclose(plain[key][0].values, shaken[key][0].values, atol=1e-8)


def test_jitter_does_not_poison_the_unaugmented_cache():
    """A jittered batch must leave no cache entry the true placement would read."""
    systems = [_hf_chain(2)]
    extra = {ec_fragment_name(TARGET): _per_atom_labels([[0, 0, 1, 1]])}
    key = f"{TARGET}_ec_machinery_moments"

    _machinery_via_transform(systems, extra, jitter=0.5, system_ids=[7])
    after = _machinery_via_transform(systems, extra, jitter=0.0, system_ids=[7])
    direct = compute_ec_machinery(systems[0], AUX_BASIS, [0, 0, 1, 1])
    (moments,) = unpack_metric_matrices(after[key])
    assert torch.allclose(moments, direct[0], atol=1e-12)


def test_jitter_is_reproducible_within_one_seed():
    systems = [_hf_chain(2)]
    extra = {ec_fragment_name(TARGET): _per_atom_labels([[0, 0, 1, 1]])}
    key = f"{TARGET}_ec_machinery_moments"
    torch.manual_seed(4)
    first = _machinery_via_transform(systems, extra, jitter=0.3, system_ids=[0])
    torch.manual_seed(4)
    second = _machinery_via_transform(systems, extra, jitter=0.3, system_ids=[0])
    assert torch.allclose(first[key][0].values, second[key][0].values, atol=0.0)


def test_different_systems_get_different_shifts():
    systems = [_hf_chain(2), _hf_chain(2)]
    extra = {ec_fragment_name(TARGET): _per_atom_labels([[0, 0, 1, 1]] * 2)}
    key = f"{TARGET}_ec_machinery_moments"
    values = _machinery_via_transform(systems, extra, jitter=0.3, system_ids=[0, 1])[
        key
    ][0].values
    assert not torch.allclose(values[0], values[1], atol=1e-8)


def test_validation_never_jitters():
    """The reported EC must be the true score, so validation uses no shift."""
    hooks = get_density_hooks(
        {
            TARGET: {
                "type": "ec_mse",
                "aux_basis": AUX_BASIS,
                "partner_jitter": 0.5,
                "weight": 1.0,
            }
        }
    )
    systems = [_hf_chain(2)]
    extra = {ec_fragment_name(TARGET): _per_atom_labels([[0, 0, 1, 1]])}
    key = f"{TARGET}_ec_machinery_moments"

    (validation,) = hooks.validation_collate_transforms()
    _, _, out = validation(systems, {}, dict(extra))
    direct = compute_ec_machinery(systems[0], AUX_BASIS, [0, 0, 1, 1])
    (true_moments,) = unpack_metric_matrices(out[key])
    assert torch.allclose(true_moments, direct[0], atol=1e-12)

    (training,) = hooks.training_collate_transforms()
    _, _, shaken = training(systems, {}, dict(extra))
    (shaken_moments,) = unpack_metric_matrices(shaken[key])
    assert not torch.allclose(shaken_moments, direct[0], atol=1e-8)


def test_conflicting_jitters_are_refused():
    with pytest.raises(ValueError, match="different 'partner_jitter'"):
        get_density_hooks(
            {
                "a": {"type": "ec_mse", "aux_basis": AUX_BASIS, "partner_jitter": 0.1},
                "b": {"type": "ec_mse", "aux_basis": AUX_BASIS, "partner_jitter": 0.4},
            }
        )


def test_negative_jitter_is_refused():
    with pytest.raises(ValueError, match="must not be negative"):
        ECMSELoss(
            TARGET,
            None,
            weight=1.0,
            reduction="sum",
            aux_basis=AUX_BASIS,
            partner_jitter=-0.1,
        )


def _touching_pair(gap: float) -> System:
    """Two HF molecules whose closest atoms sit ``gap`` Angstrom apart.

    A protein pocket holds its ligand at hydrogen-bond range, so the realistic
    starting point for the jitter is a contact of 1.5-1.8 A, not a loose dimer.
    """
    return System(
        types=torch.tensor([9, 1, 9, 1]),
        positions=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.92],
                [0.0, 0.0, 0.92 + gap],
                [0.0, 0.0, 0.92 + gap + 0.92],
            ],
            dtype=torch.float64,
        ),
        cell=torch.zeros(3, 3, dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
    )


def _closest_after(system: System, labels, shift) -> float:
    positions = system.positions.double().numpy()
    mask = np.asarray(labels) == 1
    moved = positions.copy()
    moved[mask] += shift
    gaps = np.linalg.norm(moved[~mask][:, None, :] - moved[mask][None, :, :], axis=2)
    return float(gaps.min())


def test_jitter_never_drives_the_fragments_together():
    """A shift that would overlap two atoms must be redrawn, not used.

    The patch cannot catch this: on OMol25 pockets an overlapping placement
    keeps 0.93-1.19 times its usual number of points, so nothing downstream
    reports the bad geometry.
    """
    from metatrain.utils.pyscf_loss import EC_JITTER_MIN_CONTACT, _ec_partner_shifts

    labels = [0, 0, 1, 1]
    system = _touching_pair(1.6)
    torch.manual_seed(0)
    worst = 1e9
    for _ in range(40):
        (shift,) = _ec_partner_shifts(0.6, [0], [system], [labels])
        worst = min(worst, _closest_after(system, labels, shift))
    assert worst >= EC_JITTER_MIN_CONTACT, f"closest approach fell to {worst:.2f} A"


def test_jitter_still_moves_an_already_close_structure():
    """A structure tighter than the floor is only asked not to get worse."""
    from metatrain.utils.pyscf_loss import _ec_partner_shifts

    labels = [0, 0, 1, 1]
    system = _touching_pair(0.8)  # already inside the floor
    torch.manual_seed(1)
    shifts = [_ec_partner_shifts(0.3, [0], [system], [labels])[0] for _ in range(20)]
    assert any(np.linalg.norm(s) > 1e-9 for s in shifts), "every shift was refused"
    for shift in shifts:
        assert _closest_after(system, labels, shift) >= 0.8 - 1e-9
