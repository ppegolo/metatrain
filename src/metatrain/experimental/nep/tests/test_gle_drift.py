"""Tests for the GLE drift-matrix head.

These define what "done" means for the head: the block structure and
positive-semidefiniteness of ``A``, exact-by-construction equivariance under
the full O(3) (improper rotations included) and under permutations,
smoothness across the cutoff, the log-spaced initialisation of the auxiliary
rates, and the ``markovian_block`` physics switch.
"""

import copy
import math

import pytest
import torch
from metatomic.torch import System

from metatrain.experimental.nep.gle_model import GLEDriftModel
from metatrain.experimental.nep.modules.gle import GLEDriftConfig
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

from . import MODEL_HYPERS


ATOMIC_TYPES = [1, 8]


def _hypers(**overrides):
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["cutoff_radial"] = 5.0
    hypers["cutoff_angular"] = 4.0
    hypers["n_max_radial"] = 2
    hypers["n_max_angular"] = 2
    hypers["basis_size_radial"] = 4
    hypers["basis_size_angular"] = 4
    hypers["neurons"] = 8
    hypers.update(overrides)
    return hypers


def _model(config=None, dtype=torch.float64, **hyper_overrides):
    # a bare `System` carries no topology, so the tests that do not exercise
    # the neighbor classes use a single class
    model = GLEDriftModel(
        hypers=_hypers(**hyper_overrides),
        atomic_types=ATOMIC_TYPES,
        config=config if config is not None else GLEDriftConfig(n_classes=1),
    )
    return model.to(dtype)


def _excite(model, scale=1.0, seed=11):
    """Make the head's output genuinely depend on the geometry.

    A freshly initialised head is dominated by its biases (that is what puts
    the auxiliary rates in place), so its predictions are nearly constant in
    the structure. Every structural test below would then hold trivially:
    equivariance, smoothness and symmetry are all statements about how the
    output *varies*. So the output weights are scaled up first.
    """
    generator = torch.Generator().manual_seed(seed)
    weights = model.head.output_weights
    with torch.no_grad():
        weights.copy_(
            scale
            * torch.randn(weights.shape, generator=generator, dtype=torch.float64).to(
                weights.dtype
            )
        )
    return model


def _positions(n_atoms=6, seed=0, scale=4.0, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(n_atoms, 3, generator=generator, dtype=dtype) * scale


def _system(positions, types=None, cell=None, dtype=torch.float64):
    n_atoms = positions.shape[0]
    if types is None:
        types = torch.tensor([8, 1, 1, 8, 1, 1][:n_atoms], dtype=torch.int32)
    if cell is None:
        cell = torch.zeros((3, 3), dtype=dtype)
        pbc = torch.tensor([False, False, False])
    else:
        pbc = torch.tensor([True, True, True])
    return System(types=types, positions=positions.to(dtype), cell=cell, pbc=pbc)


def _run(model, system):
    system = get_system_with_neighbor_lists(system, get_requested_neighbor_lists(model))
    return model([system])


def test_block_structure_and_psd():
    """Upper blocks vanish, ``A`` is symmetric and positive semidefinite."""
    model = _excite(_model())
    n = model.config.n_aux
    outputs = _run(model, _system(_positions()))

    blocks = outputs["L_blocks"].detach()
    assert blocks.shape == (6, n + 1, n + 1, 3, 3)
    upper = torch.triu_indices(n + 1, n + 1, offset=1)
    assert torch.all(blocks[:, upper[0], upper[1]] == 0.0)

    # every block is symmetric: the basis holds no antisymmetric element
    assert torch.all(blocks == blocks.transpose(-1, -2))
    # and the anisotropic part is not negligible, so the checks below are
    # about a genuinely tensorial object rather than a multiple of I
    identity = torch.eye(3, dtype=blocks.dtype)
    anisotropy = (
        blocks - blocks.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * identity
    )
    assert float(anisotropy.abs().max()) > 1e-2

    a_matrix = outputs["A"].detach()
    assert a_matrix.shape == (6, 3 * (n + 1), 3 * (n + 1))
    assert torch.allclose(a_matrix, a_matrix.transpose(-1, -2), atol=1e-12)
    eigenvalues = torch.linalg.eigvalsh(a_matrix)
    assert torch.all(eigenvalues >= -1e-10)


def _orthogonal(seed, improper):
    """A random rotation, optionally with determinant -1."""
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(3, 3, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(matrix)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if improper:
        q = q * torch.tensor([[-1.0], [1.0], [1.0]], dtype=torch.float64).T
    determinant = float(torch.linalg.det(q))
    assert math.isclose(determinant, -1.0 if improper else 1.0, abs_tol=1e-10)
    return q


@pytest.mark.parametrize("improper", [False, True], ids=["rotation", "reflection"])
def test_equivariance(improper):
    """``L`` blocks rotate as ``Q L Q^T``, including for improper rotations.

    Parity is covered by the ``improper`` case: the basis holds only proper
    (parity-even) tensors, so reflections act exactly like rotations.
    """
    model = _excite(_model())
    positions = _positions()
    rotation = _orthogonal(seed=3, improper=improper)

    reference = _run(model, _system(positions))["L_blocks"]
    rotated = _run(model, _system(positions @ rotation.T))["L_blocks"]

    # the rotation must actually do something, or this proves nothing
    assert not torch.allclose(rotated, reference, atol=1e-3)

    expected = torch.einsum("ij,nabjk,lk->nabil", rotation, reference, rotation)
    assert torch.allclose(rotated, expected, atol=1e-6)


def test_permutation_equivariance():
    model = _excite(_model())
    positions = _positions()
    types = torch.tensor([8, 1, 1, 8, 1, 1], dtype=torch.int32)
    permutation = torch.tensor([3, 0, 5, 1, 4, 2])

    reference = _run(model, _system(positions, types))["L_blocks"]
    permuted = _run(model, _system(positions[permutation], types[permutation]))[
        "L_blocks"
    ]

    assert not torch.allclose(permuted, reference, atol=1e-3)
    assert torch.allclose(permuted, reference[permutation], atol=1e-6)


def test_translation_invariance():
    model = _excite(_model())
    positions = _positions()
    shift = torch.tensor([1.3, -2.7, 0.4], dtype=torch.float64)

    reference = _run(model, _system(positions))["L_blocks"]
    translated = _run(model, _system(positions + shift))["L_blocks"]

    assert torch.allclose(translated, reference, atol=1e-6)


def test_smoothness_across_cutoff():
    """``L`` and its position gradients are continuous through ``r_cut``."""
    config = GLEDriftConfig(n_classes=1, r_cut=4.0)
    model = _excite(_model(config))
    r_cut = config.r_cut

    def value_and_gradient(distance):
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=torch.float64
        ).requires_grad_(True)
        system = _system(positions, types=torch.tensor([8, 1], dtype=torch.int32))
        blocks = _run(model, system)["L_blocks"]
        # a smooth probe: `abs` would kink wherever an entry crosses zero,
        # which is a property of the probe, not of the model
        total = (blocks**2).sum()
        (gradient,) = torch.autograd.grad(total, positions)
        return blocks.detach(), gradient[1, 0], total.detach()

    # a fine sweep across the cutoff must not jump
    distances = torch.linspace(r_cut - 0.3, r_cut + 0.3, 61, dtype=torch.float64)
    values = []
    gradients = []
    for distance in distances:
        blocks, gradient, _ = value_and_gradient(float(distance))
        values.append(blocks)
        gradients.append(gradient)
    values = torch.stack(values)
    gradients = torch.stack(gradients)

    step = float(distances[1] - distances[0])
    value_jumps = (values[1:] - values[:-1]).abs().amax(dim=(1, 2, 3, 4, 5))
    gradient_jumps = (gradients[1:] - gradients[:-1]).abs()
    # the sweep has to move `L` by much more than the continuity threshold,
    # otherwise a discontinuity could hide under it
    variation = float((values.amax(dim=0) - values.amin(dim=0)).abs().max())
    assert variation > 0.05, variation
    assert float(value_jumps.max()) < 0.1 * variation
    assert float(gradient_jumps.max()) < 20.0 * step

    # and the analytic gradient matches finite differences across the cutoff
    delta = 1e-6
    for distance in [r_cut - 0.05, r_cut, r_cut + 0.05]:
        _, gradient, _ = value_and_gradient(distance)
        _, _, plus = value_and_gradient(distance + delta)
        _, _, minus = value_and_gradient(distance - delta)
        finite_difference = float((plus - minus) / (2 * delta))
        assert abs(float(gradient) - finite_difference) < 1e-5


def test_initial_rate_spectrum():
    """The auxiliary rates start log-spaced over [gamma_min, gamma_max].

    The momentum-momentum block instead starts at the floor, so instantaneous
    friction has to be earned against the memory channels rather than
    competing with them from the first step.
    """
    config = GLEDriftConfig(n_aux=4, gamma_min=0.01, gamma_max=10.0, n_classes=1)
    model = _model(config)

    outputs = _run(model, _system(_positions()))

    markovian_rates = torch.linalg.eigvalsh(outputs["A"][:, :3, :3].detach())
    assert torch.all(markovian_rates > 0.5 * config.eps_floor)
    assert torch.all(markovian_rates < 2.0 * config.eps_floor)
    # and well separated below the slowest memory channel
    assert float(markovian_rates.max()) < 0.1 * config.gamma_min

    a_ss = outputs["A"][:, 3:, 3:]
    eigenvalues = torch.linalg.eigvalsh(a_ss)

    expected = torch.logspace(
        math.log10(config.gamma_min),
        math.log10(config.gamma_max),
        config.n_aux,
        dtype=torch.float64,
    )
    # every auxiliary channel contributes a three-fold degenerate rate
    expected = expected.repeat_interleave(3)
    for atom in range(eigenvalues.shape[0]):
        ratio = eigenvalues[atom] / expected
        assert torch.all(ratio > 0.5), ratio
        assert torch.all(ratio < 2.0), ratio


def test_markovian_block_disabled():
    """``markovian_block=False`` removes the momentum-momentum dissipation."""
    config = GLEDriftConfig(markovian_block=False, n_classes=1)
    model = _excite(_model(config))

    outputs = _run(model, _system(_positions()))

    assert torch.all(outputs["L_blocks"][:, 0, 0] == 0.0)
    assert torch.all(outputs["A"][:, :3, :3] == 0.0)
    eigenvalues = torch.linalg.eigvalsh(outputs["A"])
    assert torch.all(eigenvalues >= -1e-10)


def test_neighbor_classes_change_the_output():
    """Bonded and non-bonded neighbors get their own radial channels."""
    model = _model(GLEDriftConfig(n_classes=2))
    positions = _positions()
    system = _system(positions)
    system = get_system_with_neighbor_lists(system, get_requested_neighbor_lists(model))

    without_bonds = model([system], bonds=[torch.zeros((0, 2), dtype=torch.long)])
    with_bonds = model([system], bonds=[torch.tensor([[0, 1], [2, 3]])])

    assert not torch.allclose(
        without_bonds["L_blocks"], with_bonds["L_blocks"], atol=1e-8
    )


def test_config_from_yaml_options():
    config = GLEDriftConfig.from_options({"n_aux": 2, "markovian_block": False})
    assert config.n_aux == 2
    assert config.n_windows == 2  # untouched default
    assert not config.markovian_block
    assert GLEDriftConfig.from_options(None) == GLEDriftConfig()

    with pytest.raises(ValueError, match=r"unknown GLE drift options \['n_ax'\]"):
        GLEDriftConfig.from_options({"n_ax": 2})
    with pytest.raises(ValueError, match="gamma_min"):
        GLEDriftConfig(gamma_min=10.0, gamma_max=1.0)


def test_missing_topology_raises():
    model = _model(GLEDriftConfig(n_classes=2))
    system = _system(_positions())
    system = get_system_with_neighbor_lists(system, get_requested_neighbor_lists(model))
    with pytest.raises(ValueError, match="n_classes=2"):
        model([system])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_trains_without_nans(dtype):
    """100 optimizer steps on random structures and labels, no NaNs."""
    model = _model(GLEDriftConfig(n_classes=1), dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    systems = []
    labels = []
    generator = torch.Generator().manual_seed(7)
    for seed in range(4):
        system = _system(_positions(seed=seed, dtype=dtype), dtype=dtype)
        systems.append(
            get_system_with_neighbor_lists(system, get_requested_neighbor_lists(model))
        )
        n = model.config.n_aux + 1
        labels.append(
            torch.randn(6, n, n, 3, 3, generator=generator, dtype=dtype) * 0.1
        )

    for step in range(100):
        optimizer.zero_grad()
        loss = torch.zeros((), dtype=dtype)
        for system, label in zip(systems, labels, strict=True):
            predicted = model([system])["L_blocks"]
            loss = loss + ((predicted - label) ** 2).mean()
        loss.backward()
        assert torch.isfinite(loss), f"non-finite loss at step {step}"
        for name, parameter in model.named_parameters():
            assert parameter.grad is None or torch.all(
                torch.isfinite(parameter.grad)
            ), f"non-finite gradient in {name} at step {step}"
        optimizer.step()

    final = model([systems[0]])
    assert torch.all(torch.isfinite(final["L_blocks"]))
    assert torch.all(torch.isfinite(final["A"]))
