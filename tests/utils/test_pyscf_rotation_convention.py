"""Convention test for the RI metric matrices under O(3) transformations.

The density losses build the two-centre metric M on the *unaugmented* geometry
and evaluate coefficients in that same frame. That scheme — and any future
optimisation that un-rotates residuals instead — rests on one identity: the
Wigner convention of metatomic's O3Transformation (which the augmenter uses)
must match how PySCF's real spherical auxiliary functions transform, i.e.
(with D the block-diagonal per-shell Wigner matrix in PySCF AO order):

    M(R x geometry) == D(R) @ M(geometry) @ D(R).T

for every supported metric spec, including the PySCF p-shell (x, y, z)
ordering and inversions (parity (-1)^l per shell).

This test checks exactly that identity for random rotations/inversions with
an auxiliary basis reaching l = 4, for the plain metrics and a spec carrying
the long-range and charge-penalty terms (which must transform the same way:
J_lr is a two-centre kernel like J, and the charge vector lives on l = 0
shells only, so its rank-1 term is rotation-invariant).
"""

import numpy as np
import pytest
import torch
from metatomic.torch import System
from metatomic.torch.o3 import O3Transformation
from scipy.spatial.transform import Rotation

from metatrain.utils.pyscf_loss import (
    build_auxiliary_molecule,
    compute_metric_matrix,
    make_metric_spec,
)


pyscf = pytest.importorskip("pyscf")

AUX_BASIS = "def2-svp-jkfit"  # reaches l = 4 for O

# PySCF orders real p functions as (x, y, z) = (m=+1, m=-1, m=0) relative to
# the m = (-1, 0, +1) ordering used by the Wigner matrices; this is the same
# permutation the coefficient-flattening helpers apply for l = 1.
P_SHELL_PERMUTATION = [2, 0, 1]


def _make_system(positions: np.ndarray, types: list) -> System:
    return System(
        positions=torch.tensor(positions, dtype=torch.float64),
        types=torch.tensor(types, dtype=torch.int32),
        cell=torch.zeros((3, 3), dtype=torch.float64),
        pbc=torch.zeros(3, dtype=torch.bool),
    )


def _pyscf_block_diagonal(mol, rotation: Rotation, inversion: int) -> np.ndarray:
    """Per-shell Wigner blocks, as the augmenter builds them, in full AO order."""
    lmax = max(mol.bas_angular(shell) for shell in range(mol.nbas))
    assert lmax >= 4, "test should exercise high-l shells"
    transformation = O3Transformation(
        torch.tensor(rotation.as_matrix(), dtype=torch.float64), lmax
    )
    wigner = {
        ell: transformation.wigner_D_matrix(ell).numpy() for ell in range(lmax + 1)
    }

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


@pytest.mark.parametrize(
    "metric",
    [
        "overlap",
        "coulomb",
        make_metric_spec("coulomb", omega=0.3, charge_weight=1.0),
        make_metric_spec("overlap", dipole_weight=1.0, quadrupole_weight=1.0),
    ],
)
@pytest.mark.parametrize("seed", [0, 1])
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
    D = _pyscf_block_diagonal(mol, rotation, inversion)

    np.testing.assert_allclose(D @ M @ D.T, M_rotated, atol=1e-10, rtol=1e-8)
