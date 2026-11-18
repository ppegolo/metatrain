"""Tests for the GLE drift-matrix head.

These define what "done" means for the head: the block structure of
``A = L L^T + Omega`` (``L`` block diagonal, ``Omega`` antisymmetric over the
channels), positive-semidefiniteness of ``Sym(A) = L L^T`` -- *not* of ``A``,
which is no longer symmetric -- exact-by-construction equivariance under the
full O(3) (improper rotations included) and under permutations, smoothness
across the cutoff, the log-spaced initialisation of the auxiliary rates, the
``markovian_block`` physics switch, and the oscillatory-pair channels
(``n_pairs > 0``, ``Omega_ss != 0``).
"""

import copy
import math

import pytest
import torch
from metatomic.torch import System

from metatrain.experimental.nep.gle_model import GLEDriftModel
from metatrain.experimental.nep.modules.gle import GLEDriftConfig, omega_matrix
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
    """``L`` is block diagonal, ``Sym(A) = L L^T`` is PSD, ``A - A^T = 2 Omega``."""
    model = _excite(_model())
    n = model.config.n_aux
    outputs = _run(model, _system(_positions()))

    blocks = outputs["L_blocks"].detach()
    couplings = outputs["W_blocks"].detach()
    assert blocks.shape == (6, n + 1, n + 1, 3, 3)
    assert couplings.shape == (6, n, 3, 3)
    off_diagonal = ~torch.eye(n + 1, dtype=torch.bool)
    assert torch.all(blocks[:, off_diagonal] == 0.0)

    # every Cartesian block is symmetric: the basis holds no antisymmetric
    # element, in `L` and in `W` alike
    assert torch.all(blocks == blocks.transpose(-1, -2))
    assert torch.all(couplings == couplings.transpose(-1, -2))
    # and the anisotropic part is not negligible, so the checks below are
    # about a genuinely tensorial object rather than a multiple of I
    identity = torch.eye(3, dtype=blocks.dtype)
    for tensor in (blocks, couplings):
        anisotropy = (
            tensor
            - tensor.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * identity
        )
        assert float(anisotropy.abs().max()) > 1e-2

    a_matrix = outputs["A"].detach()
    assert a_matrix.shape == (6, 3 * (n + 1), 3 * (n + 1))

    # the symmetric part is exactly L L^T, and it is positive semidefinite
    symmetric = 0.5 * (a_matrix + a_matrix.transpose(-1, -2))
    factor = blocks.permute(0, 1, 3, 2, 4).reshape(6, 3 * (n + 1), 3 * (n + 1))
    assert torch.allclose(symmetric, factor @ factor.transpose(-1, -2), atol=1e-12)
    assert torch.all(torch.linalg.eigvalsh(symmetric) >= -1e-10)
    # ... and the drift itself is stable: Re lambda(A) >= 0
    assert float(torch.linalg.eigvals(a_matrix).real.min()) > -1e-10

    # the antisymmetric part is exactly the coupling, in the (p, s) blocks
    antisymmetric = a_matrix - a_matrix.transpose(-1, -2)
    omega = omega_matrix(couplings)
    assert torch.allclose(antisymmetric, 2.0 * omega, atol=1e-12)
    assert float(omega.abs().max()) > 1e-2
    # Omega_ss is zero: only p couples to the auxiliaries
    assert torch.all(omega[:, 3:, 3:] == 0.0)


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
@pytest.mark.parametrize("key", ["L_blocks", "W_blocks"])
def test_equivariance(improper, key):
    """Every block rotates as ``Q X Q^T``, including for improper rotations.

    Parity is covered by the ``improper`` case: the basis holds only proper
    (parity-even) tensors, so reflections act exactly like rotations.  The
    coupling blocks ``W`` obey the same law as ``L``: channel-space
    antisymmetry says nothing about the Cartesian indices.
    """
    model = _excite(_model())
    positions = _positions()
    rotation = _orthogonal(seed=3, improper=improper)

    reference = _run(model, _system(positions))[key]
    rotated = _run(model, _system(positions @ rotation.T))[key]

    # the rotation must actually do something, or this proves nothing
    assert not torch.allclose(rotated, reference, atol=1e-3)

    letters = "ab" if key == "L_blocks" else "a"
    expected = torch.einsum(
        f"ij,n{letters}jk,lk->n{letters}il", rotation, reference, rotation
    )
    assert torch.allclose(rotated, expected, atol=1e-6)


@pytest.mark.parametrize("key", ["L_blocks", "W_blocks"])
def test_permutation_equivariance(key):
    model = _excite(_model())
    positions = _positions()
    types = torch.tensor([8, 1, 1, 8, 1, 1], dtype=torch.int32)
    permutation = torch.tensor([3, 0, 5, 1, 4, 2])

    reference = _run(model, _system(positions, types))[key]
    permuted = _run(model, _system(positions[permutation], types[permutation]))[key]

    assert not torch.allclose(permuted, reference, atol=1e-3)
    assert torch.allclose(permuted, reference[permutation], atol=1e-6)


@pytest.mark.parametrize("key", ["L_blocks", "W_blocks"])
def test_translation_invariance(key):
    model = _excite(_model())
    positions = _positions()
    shift = torch.tensor([1.3, -2.7, 0.4], dtype=torch.float64)

    reference = _run(model, _system(positions))[key]
    translated = _run(model, _system(positions + shift))[key]

    assert torch.allclose(translated, reference, atol=1e-6)


def test_smoothness_across_cutoff():
    """The head's blocks and their position gradients are continuous at ``r_cut``.

    ``L`` and ``W`` are probed together: they share the basis and the network,
    so a discontinuity in either is a discontinuity of the head.
    """
    config = GLEDriftConfig(n_classes=1, r_cut=4.0)
    model = _excite(_model(config))
    r_cut = config.r_cut

    def value_and_gradient(distance):
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=torch.float64
        ).requires_grad_(True)
        system = _system(positions, types=torch.tensor([8, 1], dtype=torch.int32))
        outputs = _run(model, system)
        n_atoms = positions.shape[0]
        blocks = torch.cat(
            [
                outputs["L_blocks"].reshape(n_atoms, -1),
                outputs["W_blocks"].reshape(n_atoms, -1),
            ],
            dim=1,
        )
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
    value_jumps = (values[1:] - values[:-1]).abs().amax(dim=(1, 2))
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
    for atom in range(eigenvalues.shape[0]):
        ratio = eigenvalues[atom] / expected.repeat_interleave(3)
        assert torch.all(ratio > 0.5), ratio
        assert torch.all(ratio < 2.0), ratio

    # the couplings start at W_c = sqrt(gamma_c) I, so no channel sits on the
    # W = 0 saddle and the initial kernel integrates to n_aux
    couplings = outputs["W_blocks"].detach()
    identity = torch.eye(3, dtype=couplings.dtype)
    target = expected.sqrt()[None, :, None, None] * identity
    # atol covers the small tensorial part the random output weights add
    assert torch.allclose(couplings, target, rtol=0.05, atol=5e-3)


def test_markovian_block_disabled():
    """``markovian_block=False`` is a real pure-memory bath, not a dead one.

    Zeroing ``L_pp`` removes the instantaneous friction *and* the white noise
    on ``p``, but the coupling ``W`` lives in ``Omega``, so the momentum still
    exchanges energy with the auxiliaries.  Under the old ``A = L L^T``
    parameterisation this was structurally impossible: a symmetric PSD matrix
    with a vanishing diagonal block has a vanishing block row, so the momentum
    decoupled entirely.  That impossibility is what the antisymmetric coupling
    removes, and this test is its functional counterpart.
    """
    config = GLEDriftConfig(markovian_block=False, n_classes=1)
    model = _excite(_model(config))

    outputs = _run(model, _system(_positions()))
    a_matrix = outputs["A"].detach()

    assert torch.all(outputs["L_blocks"][:, 0, 0] == 0.0)
    assert torch.all(a_matrix[:, :3, :3] == 0.0)

    # the momentum is still coupled, and the drift is still stable
    assert float(a_matrix[:, :3, 3:].abs().min(dim=0).values.max()) > 1e-2
    symmetric = 0.5 * (a_matrix + a_matrix.transpose(-1, -2))
    assert torch.all(torch.linalg.eigvalsh(symmetric) >= -1e-10)
    assert float(torch.linalg.eigvals(a_matrix).real.min()) > 1e-6

    # and the memory kernel it produces is positive semidefinite at every lag,
    # which the old structure could never deliver
    a_ps = a_matrix[:, :3, 3:]
    a_ss = a_matrix[:, 3:, 3:]
    for time in [0.0, 0.1, 1.0]:
        kernel = -a_ps @ torch.matrix_exp(-a_ss * time) @ a_matrix[:, 3:, :3]
        assert torch.allclose(kernel, kernel.transpose(-1, -2), atol=1e-10)
        assert torch.all(torch.linalg.eigvalsh(kernel) >= -1e-10)


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
            (
                torch.randn(6, n, n, 3, 3, generator=generator, dtype=dtype) * 0.1,
                torch.randn(6, n - 1, 3, 3, generator=generator, dtype=dtype) * 0.1,
            )
        )

    for step in range(100):
        optimizer.zero_grad()
        loss = torch.zeros((), dtype=dtype)
        for system, (label, coupling_label) in zip(systems, labels, strict=True):
            predicted = model([system])
            loss = loss + ((predicted["L_blocks"] - label) ** 2).mean()
            loss = loss + ((predicted["W_blocks"] - coupling_label) ** 2).mean()
        loss.backward()
        assert torch.isfinite(loss), f"non-finite loss at step {step}"
        for name, parameter in model.named_parameters():
            assert parameter.grad is None or torch.all(
                torch.isfinite(parameter.grad)
            ), f"non-finite gradient in {name} at step {step}"
        optimizer.step()

    final = model([systems[0]])
    assert torch.all(torch.isfinite(final["L_blocks"]))
    assert torch.all(torch.isfinite(final["W_blocks"]))
    assert torch.all(torch.isfinite(final["A"]))


# --------------------------------------------------------------------------
# oscillatory auxiliary pairs
#
# `glecg.tests.test_gle` holds the statements about the *kernel* the pairs
# produce; these are the statements about the head that emits them.
# --------------------------------------------------------------------------


def _pair_config(n_aux=2, n_pairs=1, **overrides):
    return GLEDriftConfig(n_aux=n_aux, n_pairs=n_pairs, n_classes=1, **overrides)


def test_pair_layout_and_structure():
    """Pairs widen the channel space and add ``Omega_ss``, and nothing else.

    ``L`` stays block diagonal over the wider channel space, ``Sym(A)`` stays
    exactly ``L L^T`` -- the pointwise FDT is untouched by an antisymmetric
    addition -- and the only new entries of ``Omega`` are the intra-pair
    rotations, which sit strictly inside the auxiliary block and pair the two
    members of each pair with nothing else.
    """
    config = _pair_config(n_aux=2, n_pairs=2)
    model = _excite(_model(config=config))
    outputs = _run(model, _system(_positions()))

    channels = config.n_channels
    assert config.n_aux_total == 6 and channels == 7
    assert outputs["L_blocks"].shape == (6, channels, channels, 3, 3)
    assert outputs["W_blocks"].shape == (6, config.n_coupling, 3, 3)
    assert outputs["pair_frequencies"].shape == (6, config.n_pairs)
    assert outputs["A"].shape == (6, 3 * channels, 3 * channels)

    blocks = outputs["L_blocks"].detach()
    off_diagonal = ~torch.eye(channels, dtype=torch.bool)
    assert torch.all(blocks[:, off_diagonal] == 0.0)

    a_matrix = outputs["A"].detach()
    symmetric = 0.5 * (a_matrix + a_matrix.transpose(-1, -2))
    factor = blocks.permute(0, 1, 3, 2, 4).reshape(6, 3 * channels, 3 * channels)
    assert torch.allclose(symmetric, factor @ factor.transpose(-1, -2), atol=1e-12)
    assert torch.all(torch.linalg.eigvalsh(symmetric) >= -1e-10)
    assert float(torch.linalg.eigvals(a_matrix).real.min()) > -1e-10

    omega = omega_matrix(
        outputs["W_blocks"].detach(), outputs["pair_frequencies"].detach()
    )
    assert torch.allclose(
        a_matrix - a_matrix.transpose(-1, -2), 2.0 * omega, atol=1e-12
    )
    assert torch.allclose(omega, -omega.transpose(-1, -2), atol=1e-14)

    # Omega_ss holds exactly the intra-pair rotations: one 3x3 multiple of the
    # identity per pair, and zero everywhere else in the auxiliary block
    aux = omega[:, 3:, 3:]
    identity = torch.eye(3, dtype=aux.dtype)
    expected = torch.zeros_like(aux)
    for index in range(config.n_pairs):
        first, second = config.pair_channels(index)
        rows = slice(3 * first, 3 * first + 3)
        columns = slice(3 * second, 3 * second + 3)
        rotation = outputs["pair_frequencies"].detach()[:, index].reshape(-1, 1, 1)
        expected[:, rows, columns] = rotation * identity
        expected[:, columns, rows] = -rotation * identity
    assert torch.allclose(aux, expected, atol=1e-14)
    assert float(aux.abs().max()) > 1e-2


def test_pair_couplings_reach_only_the_plus_member():
    """``W^- = 0``, structurally: the ``-`` member has no coupling slot at all.

    This is what makes ``K(t) = K(t)^T`` hold at every lag rather than at the
    parameter values where a penalty happens to be satisfied; see
    ``glecg/docs/oscillatory_channels.md`` §3.
    """
    config = _pair_config(n_aux=1, n_pairs=2)
    model = _excite(_model(config=config))
    outputs = _run(model, _system(_positions()))

    row = outputs["A"].detach()[:, :3, 3:].reshape(6, 3, config.n_aux_total, 3)
    for index in range(config.n_pairs):
        _, minus = config.pair_channels(index)
        assert torch.all(row[:, :, minus] == 0.0)
    # ... while the coupled members are not silently zero too
    assert float(row.abs().max()) > 1e-2


def test_pair_frequencies_are_invariant_scalars():
    """``omega_c`` multiplies the Cartesian identity, so it is an O(3) scalar."""
    model = _excite(_model(config=_pair_config()))
    positions = _positions()
    reference = _run(model, _system(positions))["pair_frequencies"].detach()
    for improper in (False, True):
        rotation = _orthogonal(seed=5, improper=improper)
        rotated = _run(model, _system(positions @ rotation.T))
        assert torch.allclose(
            rotated["pair_frequencies"].detach(), reference, atol=1e-12
        )


def test_initial_frequencies_are_nonzero_and_log_spaced():
    """``omega = 0`` is a stationary point, so no pair may be initialised there.

    ``K`` depends on ``omega`` only through ``omega^2``
    (``docs/oscillatory_channels.md`` §4), so a pair started at zero frequency
    receives no gradient in it and stays monotone for the whole fit.  The bias
    initialisation therefore places the frequencies log-spaced over
    ``[omega_min, omega_max]``, and a single pair lands at the geometric mean
    rather than at the left endpoint.
    """
    config = _pair_config(n_aux=0, n_pairs=3, omega_min=0.5, omega_max=50.0)
    model = _model(config=config)
    outputs = _run(model, _system(_positions()))
    frequencies = outputs["pair_frequencies"].detach()[0]
    # `rtol`, not exact: the output biases carry the initialisation, but the
    # small random output weights perturb it at the 1e-3 level
    assert torch.allclose(
        frequencies, torch.tensor([0.5, 5.0, 50.0], dtype=frequencies.dtype), rtol=1e-2
    )

    single = _model(
        config=_pair_config(n_aux=0, n_pairs=1, omega_min=1.0, omega_max=100.0)
    )
    value = _run(single, _system(_positions()))["pair_frequencies"].detach()[0, 0]
    assert math.isclose(float(value), 10.0, rel_tol=1e-2)


def test_pair_config_rejects_a_zero_frequency_range():
    with pytest.raises(ValueError, match="omega_min"):
        GLEDriftConfig(n_pairs=1, omega_min=0.0)
    with pytest.raises(ValueError, match="at least one auxiliary channel"):
        GLEDriftConfig(n_aux=0, n_pairs=0)


def test_pair_config_from_yaml_options():
    config = GLEDriftConfig.from_options(
        {"n_aux": 1, "n_pairs": 2, "omega_min": 2.0, "omega_max": 20.0}
    )
    assert (config.n_aux, config.n_pairs) == (1, 2)
    assert config.n_aux_total == 5 and config.n_coupling == 3
    assert config.n_terms == config.n_blocks + config.n_coupling + config.n_pairs
