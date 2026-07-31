"""Rotationally COVARIANT construction of the GLE drift matrix.

Why this exists
---------------
The Mori-Zwanzig memory kernel is a rank-2 Cartesian TENSOR field:

    dP_i/dt = F_i(Q) - sum_j int K_ij(Q(s), t-s) . P_j(s)/m_j ds + R_i(t)

and rotational invariance of the underlying Hamiltonian forces

    K(R Q, tau) = R K(Q, tau) R^T .

So the drift must be COVARIANT. Treating ``theta`` as rotation-invariant scalars is correct
only under the additional assumption that the friction is isotropic AT FIXED Q, which is
false: at fixed Q the neighbour configuration -- and for a peptide, the molecule itself --
breaks the symmetry. Isotropy is a property of the Q-AVERAGED kernel, not of ``K(Q)``.

Measured consequence of getting this wrong (`check_A_isotropy.py`, on the trained bulk-water
model): the effective friction ``Gamma = A_pp - A_ps A_ss^-1 A_sp`` came out **15%
anisotropic** with eigenvalues 0.034-0.051. That anisotropy is not noise -- it is the model
trying to represent a real effect -- but with invariant ``theta`` it is LOCKED TO THE LAB
FRAME: the same local configuration in a different orientation receives the same ``A``. It is
therefore arbitrary rather than wrong-by-a-little.

What changes relative to the scalar construction
------------------------------------------------
**Auxiliary variables must be 3-vectors, not scalars.** With scalar ``s_k``, the term
``ds_k/dt = -sum_j (A_sp)_{kj} p_j`` is a linear functional of a VECTOR with fixed
coefficients, which transforms like a vector component and not like a scalar. Covariance
would force ``A_sp = A_ps = 0``, decoupling the auxiliaries and destroying the memory
entirely. So scalar auxiliaries admit no covariant GLE with memory.

The state per bead is therefore ``(1 + n_aux)`` three-vectors,
``Y = (p, s_1, ..., s_n_aux)``, of dimension ``d = 3 (1 + n_aux)``, ordered BLOCK-MAJOR and
Cartesian-minor: ``(p_x, p_y, p_z, s1_x, s1_y, s1_z, ...)``.

**Cholesky cannot be used.** The scalar construction writes ``A = 1/2 L L^T + K`` with ``L``
lower-triangular, the triangularity being there only to make the symmetric part positive
semi-definite. A triangular constraint in the full ``d``-dimensional space is NOT preserved by
rotations, so it would break covariance. It is also unnecessary: ``M M^T`` is PSD for an
ARBITRARY ``M``. Building ``M`` from equivariant 3x3 blocks gives positive semi-definiteness
and covariance together, at the cost of a redundant (non-unique) parametrisation -- which is
harmless, since nothing here needs ``M`` to be recoverable from ``A``.

Parametrisation
---------------
Per bead the network emits, for a ``b = (1 + n_aux)`` block grid:

* ``M``: ``b^2`` blocks of 3x3  -> symmetric part ``1/2 M M^T``, PSD by construction
* ``N``: ``b^2`` blocks of 3x3  -> antisymmetric part ``K = N - N^T``, energy-conserving

i.e. ``theta`` of length ``2 * 9 * b^2``. Each 3x3 block must be predicted EQUIVARIANTLY by
the backbone (as an ``l = 0 + 1 + 2`` Cartesian tensor); this module only assembles them, and
:func:`assert_covariant` checks the assembly does not itself break the symmetry.

Cost. Reducing ``n_aux`` is expected and intended: a tensor auxiliary carries three times the
information of a scalar one. At ``n_aux = 3`` the state is ``d = 12`` against the scalar
model's 16 at ``n_aux = 13``, so the ``matrix_exp`` in the training loss gets CHEAPER, not
dearer -- which matters, because CLAUDE.md records that backward as the campaign's memory
ceiling (it scales as beads x lags x d^2).
"""

from typing import Dict, Tuple

import torch


# --- irreducible (spherical) <-> Cartesian rank-2 ---------------------------------------
#
# A 3x3 Cartesian tensor decomposes as l = 0 (+) 1 (+) 2:
#
#   l = 0 : the trace,               1 component,  T_iso = (tr T / 3) * delta_ij
#   l = 1 : the antisymmetric part,  3 components, T_anti <-> a pseudo-vector
#   l = 2 : symmetric traceless,     5 components
#
# **l = 1 IS DELIBERATELY OMITTED. The Cartesian blocks are SYMMETRIC.** Two independent
# reasons, one physical and one practical:
#
#   * Onsager reciprocity. The Mori-Zwanzig kernel obeys K_ij(tau) = K_ji(tau)^T, so the
#     diagonal block K_ii is symmetric. An antisymmetric part of the friction would be a
#     magnetic / Coriolis-like term, forbidden without a magnetic field or broken
#     time-reversal symmetry.
#   * A pseudo-vector (l = 1, sigma = -1) is parity-ODD and cannot be built from
#     parity-EVEN descriptors. MEASURED with a SOAP-BPNN backbone: l = 0 and l = 2 came out
#     equivariant to 1e-16 while the l = 1 channel was ~1e-7 -- six orders below the others
#     and not equivariant, i.e. numerical noise rather than a pseudo-vector, and it was the
#     ONLY non-equivariant part of the output.
#
# This costs no oscillatory memory. The antisymmetry that makes a GLE non-Markovian lives in
# the BLOCK indices (the p <-> s coupling), not the Cartesian ones: with symmetric 3x3 blocks
# N_ij, the antisymmetric part K = N - N^T has blocks N_ij - N_ji, still antisymmetric in the
# block index. Onsager in the Cartesian slots, oscillatory in the block structure.
#
# Emitting these three irreps SEPARATELY -- rather than 9 unstructured numbers -- is what
# lets metatrain's architectures deliver equivariance themselves: SOAP-BPNN builds a
# `TensorBasis` per (o3_lambda, o3_sigma), and SPACE is equivariant by construction. PET is
# invariant and would need rotational augmentation to learn it, which is a training choice,
# not a change to this assembly.
#
# The l = 2 basis below is the standard real symmetric-traceless set, ordered
# (xy, yz, 2z^2-x^2-y^2, xz, x^2-y^2) with the normalisation that makes the round trip
# Cartesian -> irreps -> Cartesian the identity. It is TESTED (`assert_irrep_roundtrip`)
# rather than trusted: a wrong normalisation or ordering still round-trips within a
# consistent pair of functions, so the test also checks that each l-block transforms WITHIN
# ITSELF under rotation, which a wrong basis does not.

_SQRT2 = 2.0 ** 0.5
_SQRT3 = 3.0 ** 0.5


def _l2_basis(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """The five symmetric traceless 3x3 basis tensors, shape ``[5, 3, 3]``."""
    b = torch.zeros(5, 3, 3, dtype=dtype, device=device)
    b[0, 0, 1] = b[0, 1, 0] = 1.0 / _SQRT2                       # xy
    b[1, 1, 2] = b[1, 2, 1] = 1.0 / _SQRT2                       # yz
    b[2, 0, 0] = b[2, 1, 1] = -1.0 / (_SQRT2 * _SQRT3)           # 2z^2 - x^2 - y^2
    b[2, 2, 2] = 2.0 / (_SQRT2 * _SQRT3)
    b[3, 0, 2] = b[3, 2, 0] = 1.0 / _SQRT2                       # xz
    b[4, 0, 0] = 1.0 / _SQRT2                                    # x^2 - y^2
    b[4, 1, 1] = -1.0 / _SQRT2
    return b


def irreps_to_cartesian(l0: torch.Tensor, l2: torch.Tensor) -> torch.Tensor:
    """``(l0 [...,1], l2 [...,5])`` -> SYMMETRIC Cartesian blocks ``[..., 3, 3]``.

    There is no ``l = 1`` argument: see the module docstring -- Onsager reciprocity makes
    the Cartesian friction tensor symmetric, and a pseudo-vector cannot be built from
    parity-even descriptors anyway.
    """
    dtype, device = l0.dtype, l0.device
    eye = torch.eye(3, dtype=dtype, device=device)
    trace = l0[..., 0, None, None] * eye / _SQRT3
    basis = _l2_basis(dtype, device)
    traceless = (l2[..., :, None, None] * basis).sum(dim=-3)
    return trace + traceless


def cartesian_to_irreps(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`irreps_to_cartesian`. Returns ``(l0, l2)``.

    Any antisymmetric part of ``tensor`` is DISCARDED, not an error: the construction only
    ever produces symmetric blocks, and this is used to check that.
    """
    trace = tensor.diagonal(dim1=-2, dim2=-1).sum(-1)
    l0 = (trace / _SQRT3)[..., None]
    basis = _l2_basis(tensor.dtype, tensor.device)
    sym = 0.5 * (tensor + tensor.transpose(-1, -2))
    sym = sym - (trace / 3.0)[..., None, None] * torch.eye(
        3, dtype=tensor.dtype, device=tensor.device
    )
    l2 = (sym[..., None, :, :] * basis).sum(dim=(-2, -1))
    return l0, l2


def irrep_property_counts(n_aux: int) -> Dict[int, int]:
    """Number of properties per ``o3_lambda`` block of the ``mtt::A`` spherical target.

    Two block grids (``M`` and ``N``) of ``b x b`` Cartesian tensors, so each irrep carries
    ``2 b^2`` properties; the o3_mu components are handled by the TensorMap layout.
    """
    b = 1 + n_aux
    return {0: 2 * b * b, 2: 2 * b * b}


# --- the `mtt::A` target, declared SPHERICALLY -----------------------------------------
#
# Emitting the drift blocks as l = 0 (+) 1 (+) 2 rather than as a flat vector of scalars is
# what makes equivariance metatrain's job instead of ours: SOAP-BPNN builds a `TensorBasis`
# per (o3_lambda, o3_sigma), and SPACE is equivariant by construction. PET is invariant and
# would need rotational augmentation to learn the l > 0 channels; training it without is a
# useful CONTROL -- it isolates what covariance buys -- not a blocker.
#
# Parities. `A` maps momenta to momenta, so under inversion (p -> -p) it is EVEN: a proper
# rank-2 tensor. Its decomposition therefore carries
#     l = 0  o3_sigma = +1   (trace)
#     l = 1  o3_sigma = -1   (antisymmetric part <-> a PSEUDO-vector)
#     l = 2  o3_sigma = +1   (symmetric traceless)
# Getting the l = 1 parity wrong would let the network fit an object of the wrong symmetry
# and would not be caught by any shape check.

_IRREPS: Tuple[Tuple[int, int], ...] = ((0, 1), (2, 1))


def gle_target_info_covariant(n_aux: int) -> "TargetInfo":
    """``TargetInfo`` for the covariant ``mtt::A``: a per-atom spherical tensor target.

    Properties index the ``2 b^2`` Cartesian blocks (the ``M`` and ``N`` grids, ``b = 1 +
    n_aux``); the ``o3_mu`` components carry the ``2l + 1`` parts of each irrep.
    """
    from metatensor.torch import Labels, TensorBlock, TensorMap

    from metatrain.utils.data import TargetInfo

    counts = irrep_property_counts(n_aux)
    blocks = []
    for o3_lambda, o3_sigma in _IRREPS:
        n_props = counts[o3_lambda]
        blocks.append(
            TensorBlock(
                values=torch.empty(0, 2 * o3_lambda + 1, n_props),
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.empty((0, 2), dtype=torch.long),
                ),
                components=[
                    Labels(
                        names=["o3_mu"],
                        values=torch.arange(
                            -o3_lambda, o3_lambda + 1, dtype=torch.long
                        ).unsqueeze(1),
                    )
                ],
                properties=Labels(
                    names=["A"],
                    values=torch.arange(n_props, dtype=torch.long).unsqueeze(1),
                ),
            )
        )
    layout = TensorMap(
        keys=Labels(
            names=["o3_lambda", "o3_sigma"],
            values=torch.tensor([list(irrep) for irrep in _IRREPS], dtype=torch.long),
        ),
        blocks=blocks,
    )
    return TargetInfo(quantity="", unit="", layout=layout)



def theta_size(n_aux: int) -> int:
    """Number of raw network outputs per bead for :func:`make_A_covariant`."""
    b = 1 + n_aux
    return 2 * 6 * b * b


def _blocks(theta: torch.Tensor, n_aux: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split ``theta`` into the ``M`` and ``N`` block grids, each ``[..., b, b, 3, 3]``."""
    b = 1 + n_aux
    expected = theta_size(n_aux)
    if theta.shape[-1] != expected:
        raise ValueError(
            f"expected theta.shape[-1] = {expected} for n_aux = {n_aux}, "
            f"got {theta.shape[-1]}"
        )
    batch = theta.shape[:-1]
    half = 6 * b * b
    def to_cartesian(flat):
        irreps = flat.reshape(*batch, b, b, 6)
        return irreps_to_cartesian(irreps[..., :1], irreps[..., 1:])
    return to_cartesian(theta[..., :half]), to_cartesian(theta[..., half:])


def _assemble(blocks: torch.Tensor) -> torch.Tensor:
    """``[..., b, b, 3, 3]`` block grid -> a dense ``[..., 3b, 3b]`` matrix.

    Block-major, Cartesian-minor: entry ``(3i + a, 3j + c)`` is ``blocks[i, j, a, c]``, so the
    state vector reads ``(p_x, p_y, p_z, s1_x, ...)``. The permutation matters: assembling
    Cartesian-major instead would silently interleave the auxiliaries and produce a matrix
    that is still PSD, still the right shape, and physically meaningless.
    """
    *batch, b, _, _, _ = blocks.shape
    return (
        blocks.permute(*range(len(batch)), -4, -2, -3, -1)
        .reshape(*batch, 3 * b, 3 * b)
    )


def make_A_covariant(
    theta: torch.Tensor, n_aux: int, eps: float = 1e-6
) -> torch.Tensor:
    """Map raw per-bead network output to a covariant, stable GLE drift matrix.

    :param theta: ``[..., theta_size(n_aux)]``. Each 3x3 block must have been predicted as an
        equivariant Cartesian tensor by the backbone; this function assembles, it does not
        impose equivariance.
    :param n_aux: number of auxiliary 3-VECTORS (not scalars).
    :param eps: floor added to the diagonal so the symmetric part is strictly positive
        definite rather than merely semi-definite, keeping the stationary covariance
        invertible. The scalar construction uses the same value for the same reason.
    :return: ``[..., 3(1+n_aux), 3(1+n_aux)]`` drift matrix whose symmetric part is positive
        definite.
    """
    m_blocks, n_blocks = _blocks(theta, n_aux)
    m = _assemble(m_blocks)
    n = _assemble(n_blocks)

    symmetric = 0.5 * (m @ m.transpose(-1, -2))
    antisymmetric = n - n.transpose(-1, -2)
    eye = torch.eye(symmetric.shape[-1], dtype=theta.dtype, device=theta.device)
    return symmetric + eps * eye + antisymmetric


def rotate_theta(theta: torch.Tensor, rotation: torch.Tensor, n_aux: int) -> torch.Tensor:
    """Apply ``T -> R T R^T`` to every 3x3 block of ``theta``.

    This is what an equivariant backbone must do to its prediction when the input geometry is
    rotated. Used by :func:`assert_covariant` to test the assembly in isolation from any
    particular backbone.
    """
    m, n = _blocks(theta, n_aux)
    rt = rotation.transpose(-1, -2)
    out = []
    for blocks in (rotation @ m @ rt, rotation @ n @ rt):
        l0, l2 = cartesian_to_irreps(blocks)
        out.append(torch.cat([l0, l2], dim=-1).flatten(start_dim=-3))
    return torch.cat(out, dim=-1)


def assert_covariant(n_aux: int, seed: int = 0, tol: float = 1e-9) -> None:
    """Check ``A(R.theta) == (I_b (x) R) A(theta) (I_b (x) R)^T`` and positive definiteness.

    Rotating every 3x3 block of ``theta`` must be equivalent to rotating the assembled drift
    matrix in each of its ``1 + n_aux`` Cartesian slots. If this fails, the block ordering in
    :func:`_assemble` is wrong -- a failure mode that produces a perfectly well-formed matrix
    and would not show up in training loss, only in wrong dynamics.
    """
    generator = torch.Generator().manual_seed(seed)
    theta = torch.randn(
        4, theta_size(n_aux), dtype=torch.float64, generator=generator
    )

    # a proper rotation, via QR with a sign fix so det = +1
    q, r = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64, generator=generator))
    q = q * torch.sign(torch.diagonal(r))
    if torch.det(q) < 0:
        q = q * torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)
    q = q.contiguous()  # torch.kron below rejects the strided view left by the sign fix

    a = make_A_covariant(theta, n_aux)
    a_rotated = make_A_covariant(rotate_theta(theta, q, n_aux), n_aux)

    big_r = torch.kron(torch.eye(1 + n_aux, dtype=torch.float64), q)
    expected = big_r @ a @ big_r.transpose(-1, -2)
    error = (a_rotated - expected).abs().max().item()
    if error > tol:
        raise AssertionError(
            f"covariance violated at n_aux={n_aux}: max |A(R.theta) - R A R^T| = {error:.3e}"
        )

    symmetric = 0.5 * (a + a.transpose(-1, -2))
    smallest = torch.linalg.eigvalsh(symmetric).min().item()
    if smallest <= 0.0:
        raise AssertionError(
            f"symmetric part not positive definite at n_aux={n_aux}: "
            f"min eigenvalue {smallest:.3e}"
        )


# --- fluctuation-dissipation: the noise is FIXED by A, not fitted -----------------------
#
# The extended-variable dynamics is an Ornstein-Uhlenbeck process
#
#     dY/dt = -A Y + B xi(t),      <xi(t) xi(t')^T> = delta(t - t') I ,
#
# whose stationary covariance C solves the Lyapunov equation ``A C + C A^T = B B^T``.
#
# **Convention: mass-scaled variables.** Y = (p / sqrt(m), s_1, ..., s_n_aux), so
# equipartition reads ``C = kT I_d`` exactly. The caller is responsible for the mass scaling;
# the deployment currently carries masses explicitly (`run_gle.py` builds an edge stationary
# scale ``kT (1/m_i + 1/m_j)``), so this must be applied consistently there.
#
# With ``C = kT I`` the FDT gives ``B B^T = 2 kT sym(A)``, which EXISTS precisely because
# ``sym(A)`` is positive definite by construction (``1/2 M M^T + eps I``). Nothing here is
# fitted: the noise follows from the learned drift.
#
# **Why this is written out rather than inherited.** ``C = kT I_d`` is proportional to the
# identity, hence isotropic in every Cartesian slot AND invariant under ``I_b (x) R``. So
# equipartition holds direction-by-direction even though ``A`` is anisotropic. An
# implementation that instead inferred the stationary covariance from ``A`` could land on an
# anisotropic ``C``, which would violate equipartition per Cartesian direction while leaving
# the TRACE (and hence the reported temperature) correct -- invisible in every standard
# diagnostic this project runs.


def stationary_covariance(
    n_aux: int, kT: float, dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Equilibrium covariance of the mass-scaled extended state: ``kT I_d``."""
    return kT * torch.eye(3 * (1 + n_aux), dtype=dtype, device=device)


def ou_propagator(
    A: torch.Tensor, dt: float, kT: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact one-step OU propagator for ``dY/dt = -A Y + B xi``.

    Returns ``(T, Sigma)`` with ``T = exp(-A dt)`` and the noise covariance

        Sigma = C - T C T^T = kT (I - T T^T) ,

    which is the EXACT finite-timestep result, not a small-``dt`` expansion: propagating
    ``Y -> T Y + Sigma^{1/2} z`` therefore preserves the stationary covariance for ANY ``dt``,
    so the integrator cannot heat or cool the auxiliaries however coarse the step.

    :param A: ``[..., d, d]`` drift, ``d = 3(1 + n_aux)``.
    :param dt: timestep, in the same time unit as ``A``.
    :param kT: thermal energy.
    """
    T = torch.matrix_exp(-A * dt)
    eye = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    sigma = kT * (eye - T @ T.transpose(-1, -2))
    # symmetrise: `Sigma` is symmetric analytically, but `matrix_exp` leaves an asymmetry of
    # order the arithmetic precision, and an eigendecomposition of a non-symmetric matrix can
    # return complex eigenvalues and fail far from here.
    return T, 0.5 * (sigma + sigma.transpose(-1, -2))


# --- spherical TensorMap <-> flat theta --------------------------------------------------
#
# The model emits `mtt::A` as a TensorMap with l = 0 and l = 2 blocks; `make_A_covariant`
# consumes a FLAT per-atom vector. The two orderings have the same length, so a mismatch is
# SILENT: training would converge on a drift assembled from permuted components and the only
# symptom would be wrong dynamics. `assert_theta_roundtrip` therefore checks the conversion
# on random data rather than leaving the convention to a comment.
#
# Layout, per atom: property `p` of the spherical blocks (0 .. 2 b^2 - 1, the M grid then the
# N grid) maps to the CONTIGUOUS slice `theta[6p : 6p + 6] = [l0, l2_0 ... l2_4]`, which is
# what `_blocks` reshapes.


def tensormap_to_theta(tensor_map, n_aux: int) -> torch.Tensor:
    """Flatten the spherical ``mtt::A`` prediction into ``theta`` for the assembly.

    :param tensor_map: a ``TensorMap`` with ``(o3_lambda, o3_sigma)`` keys ``(0, 1)`` and
        ``(2, 1)``, values ``[n_atoms, 2l + 1, 2 b^2]``.
    :param n_aux: number of auxiliary 3-vectors.
    :return: ``[n_atoms, theta_size(n_aux)]``.
    """
    blocks = {}
    for key, block in tensor_map.items():
        blocks[int(key["o3_lambda"])] = block.values
    if set(blocks) != {0, 2}:
        raise ValueError(
            f"expected o3_lambda blocks {{0, 2}}, got {sorted(blocks)}; the l = 1 channel "
            "is deliberately absent (Onsager reciprocity -- see the module docstring)"
        )
    l0, l2 = blocks[0], blocks[2]              # [atoms, 1, P], [atoms, 5, P]
    n_props = l0.shape[-1]
    expected = 2 * (1 + n_aux) ** 2
    if n_props != expected:
        raise ValueError(
            f"expected {expected} properties for n_aux = {n_aux}, got {n_props}"
        )
    # -> [atoms, P, 6] with the l = 0 component first, then the five l = 2 components,
    # then flatten so each block's six numbers stay contiguous.
    stacked = torch.cat([l0.transpose(1, 2), l2.transpose(1, 2)], dim=-1)
    return stacked.reshape(stacked.shape[0], -1)


def theta_to_irrep_values(
    theta: torch.Tensor, n_aux: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`tensormap_to_theta`: flat theta -> ``(l0, l2)`` block values.

    Returns the arrays in TensorMap order, ``[n_atoms, 2l + 1, 2 b^2]``.
    """
    n_props = 2 * (1 + n_aux) ** 2
    stacked = theta.reshape(theta.shape[0], n_props, 6)
    return stacked[..., :1].transpose(1, 2), stacked[..., 1:].transpose(1, 2)


def assert_theta_roundtrip(n_aux: int, seed: int = 0, tol: float = 1e-12) -> None:
    """Check the flat/spherical conversion is an exact inverse, and preserves ``A``.

    Both directions are checked, and so is the assembled drift: a permutation that happened
    to survive one direction would still change ``A``, which is the object that matters.
    """
    from metatensor.torch import Labels, TensorBlock, TensorMap

    generator = torch.Generator().manual_seed(seed)
    n_atoms = 5
    theta = torch.randn(
        n_atoms, theta_size(n_aux), dtype=torch.float64, generator=generator
    )
    l0, l2 = theta_to_irrep_values(theta, n_aux)

    samples = Labels(
        names=["system", "atom"],
        values=torch.stack(
            [torch.zeros(n_atoms, dtype=torch.long), torch.arange(n_atoms)], dim=1
        ),
    )
    properties = Labels(
        names=["A"], values=torch.arange(l0.shape[-1]).unsqueeze(1)
    )
    blocks = []
    for o3_lambda, values in ((0, l0), (2, l2)):
        blocks.append(
            TensorBlock(
                values=values,
                samples=samples,
                components=[
                    Labels(
                        names=["o3_mu"],
                        values=torch.arange(
                            -o3_lambda, o3_lambda + 1, dtype=torch.long
                        ).unsqueeze(1),
                    )
                ],
                properties=properties,
            )
        )
    tensor_map = TensorMap(
        keys=Labels(
            names=["o3_lambda", "o3_sigma"],
            values=torch.tensor([[0, 1], [2, 1]], dtype=torch.long),
        ),
        blocks=blocks,
    )

    recovered = tensormap_to_theta(tensor_map, n_aux)
    error = (recovered - theta).abs().max().item()
    if error > tol:
        raise AssertionError(
            f"theta round trip failed at n_aux={n_aux}: max |theta' - theta| = {error:.3e}"
        )
    drift_error = (
        (make_A_covariant(recovered, n_aux) - make_A_covariant(theta, n_aux))
        .abs()
        .max()
        .item()
    )
    if drift_error > tol:
        raise AssertionError(
            f"round trip changed the drift at n_aux={n_aux}: max |dA| = {drift_error:.3e}"
        )


# --- the conservative impulse, propagated ------------------------------------------------
#
# The exact conditional mean of the extended OU process driven by a known force is
#
#     E[Y(tau) | Y(0)] = e^{-A tau} Y(0) + int_0^tau e^{-A(tau-u)} F(u) du ,
#
# so the impulse enters CONVOLVED with the propagator, not bare. The transition likelihood
# has historically omitted the whole term, which makes the fitted drift absorb decorrelation
# the conservative force already produced -- the deployed dynamics then applies the force
# again and is over-damped.
#
# Subtracting the BARE impulse `int F du` is only the tau -> 0 limit of the correct term.
# MEASURED on campaign data at gamma tau / m ~ 1.7, the bare form under-corrects by ~30%
# (ratio 0.65-0.75, stable across basins and lags) and in the direction that would overshoot
# an over-damped model into UNDER-damping, because the un-propagated leftover correlates
# positively with P(0). On a synthetic process at gamma tau / m ~ 0.05 the two agree to 1%,
# which is the same statement seen from the other side.
#
# This term is A-DEPENDENT, so it belongs in the likelihood and cannot be moved into a
# preprocessing pass over the data.


def propagator_and_integral(
    A: torch.Tensor, dt: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(E, G)`` with ``E = exp(-A dt)`` and ``G = A^-1 (I - E)``.

    Both come from ONE matrix exponential by Van Loan's block trick: for

        M = [[-A, I], [0, 0]] * dt      ->      exp(M) = [[E, G], [0, I]]

    the upper-right block is exactly the integral ``int_0^dt exp(-A w) dw``. Forming ``G``
    as ``A^-1 (I - E)`` directly would need an explicit inverse, which is both more
    expensive and ill-conditioned when ``A`` has small eigenvalues -- precisely the regime
    of a weakly damped auxiliary.
    """
    d = A.shape[-1]
    batch = A.shape[:-2]
    block = torch.zeros(*batch, 2 * d, 2 * d, dtype=A.dtype, device=A.device)
    block[..., :d, :d] = -A * dt
    block[..., :d, d:] = torch.eye(d, dtype=A.dtype, device=A.device) * dt
    expanded = torch.matrix_exp(block)
    return expanded[..., :d, :d], expanded[..., :d, d:]


def propagated_impulse(
    A: torch.Tensor, forces: torch.Tensor, dt: float
) -> torch.Tensor:
    """``int_0^tau exp(-A(tau-u)) F(u) du`` for piecewise-constant ``F`` on a grid.

    :param A: ``[..., d, d]`` drift.
    :param forces: ``[..., n_steps, d]`` the driving term on each sub-interval, already
        embedded in the extended state (the conservative force acts on the momentum block
        only, so the auxiliary components are zero).
    :param dt: sub-interval width; ``tau = n_steps * dt``.

    Evaluated by the recursion ``J <- E J + G F_k``, which is ``n_steps`` matrix-VECTOR
    products after a single matrix exponential. That is the whole cost argument for this
    route over an integrator-consistent likelihood, which would need a matrix exponential
    per step rather than a matvec.
    """
    E, G = propagator_and_integral(A, dt)
    n_steps = forces.shape[-2]
    J = torch.zeros(forces.shape[:-2] + (A.shape[-1],), dtype=A.dtype, device=A.device)
    for k in range(n_steps):
        J = (E @ J.unsqueeze(-1)).squeeze(-1) + (
            G @ forces[..., k, :].unsqueeze(-1)
        ).squeeze(-1)
    return J


def assert_impulse_correct(n_aux: int = 1, seed: int = 0, tol: float = 1e-8) -> None:
    """Check the recursion against direct numerical integration of the same integral.

    A closed-form propagator that is subtly wrong still produces a smooth, plausible
    correction, so it is checked against a fine-grid quadrature of
    ``int exp(-A(tau-u)) F(u) du`` rather than against itself.
    """
    generator = torch.Generator().manual_seed(seed)
    d = 3 * (1 + n_aux)
    theta = torch.randn(theta_size(n_aux), dtype=torch.float64, generator=generator)
    A = make_A_covariant(theta.unsqueeze(0), n_aux)[0]

    n_steps, dt = 12, 0.05
    forces = torch.randn(n_steps, d, dtype=torch.float64, generator=generator)
    J = propagated_impulse(A, forces, dt)

    # reference: dense Riemann sum of exp(-A(tau-u)) F(u), F piecewise constant
    tau = n_steps * dt
    fine = 400
    reference = torch.zeros(d, dtype=torch.float64)
    for i in range(n_steps * fine):
        u = (i + 0.5) * dt / fine
        k = min(int(u / dt), n_steps - 1)
        reference = reference + (
            torch.matrix_exp(-A * (tau - u)) @ forces[k]
        ) * (dt / fine)

    error = (J - reference).abs().max().item()
    scale = reference.abs().max().item()
    if error > tol * max(scale, 1.0):
        raise AssertionError(
            f"propagated impulse disagrees with quadrature at n_aux={n_aux}: "
            f"max |dJ| = {error:.3e} against a scale of {scale:.3e}"
        )
