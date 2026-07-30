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


def irreps_to_cartesian(
    l0: torch.Tensor, l1: torch.Tensor, l2: torch.Tensor
) -> torch.Tensor:
    """``(l0 [...,1], l1 [...,3], l2 [...,5])`` -> Cartesian blocks ``[..., 3, 3]``."""
    dtype, device = l0.dtype, l0.device
    eye = torch.eye(3, dtype=dtype, device=device)
    out = l0[..., 0, None, None] * eye / _SQRT3
    # antisymmetric part from the pseudo-vector: T_ij = -eps_ijk v_k / sqrt(2)
    v = l1 / _SQRT2
    zero = torch.zeros_like(v[..., 0])
    anti = torch.stack(
        [
            torch.stack([zero, v[..., 2], -v[..., 1]], dim=-1),
            torch.stack([-v[..., 2], zero, v[..., 0]], dim=-1),
            torch.stack([v[..., 1], -v[..., 0], zero], dim=-1),
        ],
        dim=-2,
    )
    basis = _l2_basis(dtype, device)
    sym = (l2[..., :, None, None] * basis).sum(dim=-3)
    return out + anti + sym


def cartesian_to_irreps(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse of :func:`irreps_to_cartesian`. Returns ``(l0, l1, l2)``."""
    trace = tensor.diagonal(dim1=-2, dim2=-1).sum(-1)
    l0 = (trace / _SQRT3)[..., None]
    anti = 0.5 * (tensor - tensor.transpose(-1, -2))
    l1 = _SQRT2 * torch.stack(
        [anti[..., 1, 2], anti[..., 2, 0], anti[..., 0, 1]], dim=-1
    )
    basis = _l2_basis(tensor.dtype, tensor.device)
    sym = 0.5 * (tensor + tensor.transpose(-1, -2))
    sym = sym - (trace / 3.0)[..., None, None] * torch.eye(
        3, dtype=tensor.dtype, device=tensor.device
    )
    l2 = (sym[..., None, :, :] * basis).sum(dim=(-2, -1))
    return l0, l1, l2


def irrep_property_counts(n_aux: int) -> Dict[int, int]:
    """Number of properties per ``o3_lambda`` block of the ``mtt::A`` spherical target.

    Two block grids (``M`` and ``N``) of ``b x b`` Cartesian tensors, so each irrep carries
    ``2 b^2`` properties; the o3_mu components are handled by the TensorMap layout.
    """
    b = 1 + n_aux
    return {0: 2 * b * b, 1: 2 * b * b, 2: 2 * b * b}


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

_IRREPS: Tuple[Tuple[int, int], ...] = ((0, 1), (1, -1), (2, 1))


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
    return 2 * 9 * b * b


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
    half = 9 * b * b
    m = theta[..., :half].reshape(*batch, b, b, 3, 3)
    n = theta[..., half:].reshape(*batch, b, b, 3, 3)
    return m, n


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
    m = rotation @ m @ rt
    n = rotation @ n @ rt
    return torch.cat([m.flatten(start_dim=-4), n.flatten(start_dim=-4)], dim=-1)


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
