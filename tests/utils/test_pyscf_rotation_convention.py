"""Convention test for rotation-cached RI metric matrices.

The optimization caches the two-centre metric matrix M on the *unrotated*
geometry and accounts for rotational augmentation by un-rotating the
coefficient residuals with the augmenter's own real Wigner matrices. That is
only valid if the Wigner convention of metatomic's O3Transformation (which the
augmenter stashes) matches how PySCF's real spherical auxiliary functions
transform, i.e. (with D the block-diagonal per-shell Wigner matrix in PySCF AO
order):

    M(R x geometry) == D(R) @ M(geometry) @ D(R).T

for both the overlap and Coulomb metrics, including the PySCF p-shell
(x, y, z) ordering and inversions (parity (-1)^l per shell).

This test checks exactly that identity for random rotations/inversions with
an auxiliary basis reaching l = 4.
"""

import numpy as np
import pytest
import torch
from metatomic.torch import System
from metatomic.torch.o3 import O3Transformation
from scipy.spatial.transform import Rotation

from metatrain.utils.atomic_basis.pyscf import (
    build_auxiliary_molecule,
    compute_metric_matrix,
)


pyscf = pytest.importorskip("pyscf")

AUX_BASIS = "def2-svp-jkfit"  # reaches l = 4 for O

# PySCF orders real p functions as (x, y, z) = (m=+1, m=-1, m=0) relative to
# the m = (-1, 0, +1) ordering used by the Wigner matrices; this is the same
# permutation the coefficient-flattening helpers apply for l = 1.
P_SHELL_PERMUTATION = [2, 0, 1]


def _make_system(positions: np.ndarray, types: list[int]) -> System:
    return System(
        positions=torch.tensor(positions, dtype=torch.float64),
        types=torch.tensor(types, dtype=torch.int32),
        cell=torch.zeros((3, 3), dtype=torch.float64),
        pbc=torch.zeros(3, dtype=torch.bool),
    )


def _augmenter_wigner_matrices(rotation: Rotation, lmax: int) -> dict[int, np.ndarray]:
    """Real Wigner matrices exactly as the O3Augmenter stashes them.

    The augmenter stashes ``O3Transformation.wigner_D_matrix(ell)`` (the
    proper-rotation part) plus a separate per-system inversion flag; the
    inversion parity ``(-1)^l`` is applied per shell by the caller, mirroring
    how the density losses consume the stash.
    """
    transformation = O3Transformation(
        torch.tensor(rotation.as_matrix(), dtype=torch.float64), lmax
    )
    return {ell: transformation.wigner_D_matrix(ell).numpy() for ell in range(lmax + 1)}


def _pyscf_block_diagonal(
    mol, wigner: dict[int, np.ndarray], inversion: int
) -> np.ndarray:
    """Expand per-l Wigner blocks to the full AO-order block-diagonal matrix."""
    n = mol.nao
    D = np.zeros((n, n))
    offset = 0
    for shell in range(mol.nbas):
        ell = mol.bas_angular(shell)
        block = wigner[ell] * (inversion**ell)
        if ell == 1:
            p = P_SHELL_PERMUTATION
            block = block[np.ix_(p, p)]
        for _ in range(mol.bas_nctr(shell)):
            width = 2 * ell + 1
            D[offset : offset + width, offset : offset + width] = block
            offset += width
    assert offset == n
    return D


@pytest.mark.parametrize("metric", ["overlap", "coulomb"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_metric_matrix_rotates_with_augmenter_wigner_blocks(metric, seed):
    rng = np.random.default_rng(seed)
    rotation = Rotation.random(random_state=int(rng.integers(2**31)))
    inversion = -1 if seed % 2 else 1

    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.757, 0.587], [0.0, -0.757, 0.587]])
    types = [8, 1, 1]
    system = _make_system(positions, types)

    transform = rotation.as_matrix() * inversion
    system_rotated = _make_system(positions @ transform.T, types)

    M = compute_metric_matrix(system, AUX_BASIS, metric).numpy()
    M_rotated = compute_metric_matrix(system_rotated, AUX_BASIS, metric).numpy()

    mol = build_auxiliary_molecule(system, AUX_BASIS)
    lmax = max(mol.bas_angular(shell) for shell in range(mol.nbas))
    assert lmax >= 4, "test should exercise high-l shells"

    wigner = _augmenter_wigner_matrices(rotation, lmax)
    D = _pyscf_block_diagonal(mol, wigner, inversion)

    np.testing.assert_allclose(D @ M @ D.T, M_rotated, atol=1e-10, rtol=1e-8)
