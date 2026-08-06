"""Auxiliary-basis two-centre metrics for density (RI-coefficient) losses.

A scalar field expanded on an atom-centred auxiliary basis,
:math:`\\rho(r) = \\sum_i c_i \\phi_i(r)`, has errors that are quadratic forms in the
coefficient residual, weighted by a two-centre metric :math:`M`:

* **overlap**, :math:`S_{ij} = \\int \\phi_i \\phi_j \\, dr`, giving the real-space
  L2 error :math:`\\int |\\Delta\\rho(r)|^2 dr`;
* **Coulomb**, :math:`J_{ij} = \\iint \\phi_i(r) |r - r'|^{-1} \\phi_j(r')`,
  giving the electrostatic self-energy of the residual.

Neither is preferred a priori: they weight different length scales of the error, and
which one trains better is an empirical question.

This module computes those metrics with PySCF and packs them for the dataloader to
carry to the density losses in :py:mod:`metatrain.utils.loss`.

PySCF is an optional dependency, imported lazily, so metatrain works without it
unless a density loss is actually configured.

**Positions are read in Angstrom**, matching the ``length_unit`` that metatrain
datasets declare; the collate transforms run before any unit conversion.

**Non-periodic systems only.** The metric is built from a molecular PySCF ``Mole``,
which ignores the cell; periodic systems are rejected rather than silently scored
against a molecular metric.
"""

from __future__ import annotations

import copy
import functools
import importlib
import re
from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

from .data.byte_budget_cache import (
    ByteBudgetCache,
    batch_system_ids,
    collate_cache_max_bytes,
)


if TYPE_CHECKING:
    from types import ModuleType

    from pyscf.gto import Mole


@lru_cache(maxsize=1)
def _import_pyscf() -> Tuple["ModuleType", "ModuleType"]:
    """Import PySCF lazily, with an actionable error if it is missing.

    :return: The ``pyscf.gto`` and ``pyscf.data.elements`` modules.
    """
    try:
        gto = importlib.import_module("pyscf.gto")
        elements = importlib.import_module("pyscf.data.elements")
    except ModuleNotFoundError as err:
        raise ImportError(
            "density losses require `pyscf` to compute auxiliary-basis metric "
            "matrices; install it with `pip install pyscf`."
        ) from err
    return gto, elements


def _build_etb_basis(
    ao_basis: str, atomic_numbers: Tuple[int, ...], beta: float
) -> Dict[str, object]:
    """Build an even-tempered auxiliary basis with ``pyscf.df.aug_etb``.

    Constructs a dummy molecule holding one atom of each requested element in the
    *orbital* basis ``ao_basis``, then calls ``aug_etb``. This reproduces how
    SCFBench-style RI datasets generate their auxiliary basis, so that the metric
    matches the basis the reference coefficients were fitted in.

    :param ao_basis: Orbital (not auxiliary) basis name, e.g. ``"def2-svp"``.
    :param atomic_numbers: Unique atomic numbers to build the basis for.
    :param beta: Even-tempering ratio.
    :return: Mapping from element symbol to basis specification.
    """
    gto, elements = _import_pyscf()
    df = importlib.import_module("pyscf.df")

    symbols = [elements.ELEMENTS[n] for n in atomic_numbers]
    # Spread the atoms out so the dummy molecule builds without warnings.
    mol = gto.Mole()
    mol.atom = "\n".join(f"{s} 0.0 0.0 {i * 10.0}" for i, s in enumerate(symbols))
    mol.basis = ao_basis
    mol.unit = "Angstrom"
    mol.verbose = 0
    mol.spin = None
    mol.build()

    return df.aug_etb(mol, beta=beta)


@lru_cache(maxsize=None)
def _load_auxiliary_basis(
    aux_basis: str, atomic_numbers: Tuple[int, ...]
) -> Dict[str, object]:
    """Load (and cache) the auxiliary basis for a set of elements.

    Two forms of ``aux_basis`` are supported:

    * a PySCF basis name, e.g. ``"def2-universal-jfit"``;
    * an even-tempered specification ``"etb:<ao_basis>:<beta>"``, e.g.
      ``"etb:def2-svp:2.0"``, which is built via :func:`_build_etb_basis`.

    :param aux_basis: The auxiliary basis, in either form.
    :param atomic_numbers: Unique atomic numbers to load the basis for.
    :return: Mapping from element symbol to basis specification.
    """
    gto, elements = _import_pyscf()

    parts = aux_basis.split(":")
    if len(parts) == 3 and parts[0].lower() == "etb":
        return _build_etb_basis(parts[1], atomic_numbers, float(parts[2]))

    return {
        elements.ELEMENTS[n]: gto.basis.load(aux_basis, elements.ELEMENTS[n])
        for n in atomic_numbers
    }


# ── Metric specifications ─────────────────────────────────────────────────────


#: The base two-centre metrics. A metric *spec* is either one of these bare
#: names or a string encoding optional refinements on top of them; see
#: :py:func:`make_metric_spec`.
METRICS = ("overlap", "coulomb")


#: Default weight on the plain Coulomb term when the long-range kernel is on.
#: The long-range metric alone is numerically rank-deficient (condition number
#: ~1e32 vs ~1e5 for J, because erf(wr)/r damps the compact directions away), so
#: it needs a positive-definite floor. 0.01 keeps ~95% of the long-range
#: reweighting while bringing the conditioning back to ~1e7.
DEFAULT_LR_EPS = 0.01


#: Default scaling of the van der Waals radii for the ESP surface shell.
DEFAULT_ESP_SHELL = 1.4

#: Points per atomic sphere before burial culling, for the ESP surface. A
#: module constant rather than a spec option: it sets discretisation accuracy,
#: not physics, so it should not fragment the cache key space.
ESP_POINTS_PER_ATOM = 50


def make_metric_spec(
    metric: str,
    omega: float = 0.0,
    eps: Optional[float] = None,
    charge_weight: float = 0.0,
    dipole_weight: float = 0.0,
    quadrupole_weight: float = 0.0,
    esp_weight: float = 0.0,
    esp_shell: Optional[float] = None,
) -> str:
    """Build the canonical metric-spec string shared by the loss and the transform.

    The spec fully determines the matrix, so it doubles as the ``extra_data`` key
    and the cache key. With no optional term enabled it collapses to the bare
    ``metric`` name, and each optional term appears only when enabled, keeping
    existing configs and cache keys byte-identical.

    :param metric: Base two-centre metric, ``"overlap"`` (S) or ``"coulomb"`` (J).
    :param omega: Range-separation parameter of the long-range Coulomb kernel
        ``erf(omega r)/r``. ``0`` disables it. Only valid with
        ``metric="coulomb"``.
    :param eps: Weight on the plain Coulomb term added to the long-range one,
        i.e. ``M = eps*J + J_lr``. ``None`` uses :py:data:`DEFAULT_LR_EPS`.
        Ignored when ``omega == 0``.
    :param charge_weight: Weight on the rank-1 electron-count penalty
        ``w * S_vec S_vec^T``, which adds ``w * (S_vec . dc)**2`` to the loss.
        ``0`` disables it.
    :param dipole_weight: Weight on the distributed (per-atom) dipole penalty,
        which adds ``w * sum_a |dQ_a^{l=1}|**2`` to the loss. ``0`` disables it.
    :param quadrupole_weight: The same for the per-atom quadrupoles (l=2).
    :param esp_weight: Weight on the surface-ESP penalty, an area-weighted sum
        of ``|dV(r_g)|**2`` over an accessible-surface grid (see
        :py:func:`compute_surface_points`). ``0`` disables it.
    :param esp_shell: Scaling of the van der Waals radii defining that surface.
        ``None`` uses :py:data:`DEFAULT_ESP_SHELL`. Ignored when
        ``esp_weight == 0``.
    :return: Canonical spec string.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected 'overlap' or 'coulomb'.")
    if omega < 0.0:
        raise ValueError(f"omega must be >= 0, got {omega}.")
    for weight_name, weight in (
        ("charge_weight", charge_weight),
        ("dipole_weight", dipole_weight),
        ("quadrupole_weight", quadrupole_weight),
        ("esp_weight", esp_weight),
    ):
        if weight < 0.0:
            raise ValueError(f"{weight_name} must be >= 0, got {weight}.")
    if omega > 0.0 and metric != "coulomb":
        raise ValueError(
            "the long-range kernel erf(omega r)/r is a Coulomb-metric option; "
            f"got metric='{metric}' with omega={omega}."
        )
    if omega == 0.0 and charge_weight == 0.0:
        if dipole_weight == 0.0 and quadrupole_weight == 0.0 and esp_weight == 0.0:
            return metric
    resolved_eps = DEFAULT_LR_EPS if eps is None else float(eps)
    if resolved_eps < 0.0:
        raise ValueError(f"eps must be >= 0, got {resolved_eps}.")
    spec = (
        f"{metric}|omega={float(omega):.10g}"
        f"|eps={resolved_eps:.10g}|q={float(charge_weight):.10g}"
    )
    # Appended only when enabled, so specs (= extra_data and cache keys) from
    # before these options existed stay byte-identical.
    if dipole_weight > 0.0:
        spec += f"|d={float(dipole_weight):.10g}"
    if quadrupole_weight > 0.0:
        spec += f"|Q={float(quadrupole_weight):.10g}"
    if esp_weight > 0.0:
        resolved_shell = DEFAULT_ESP_SHELL if esp_shell is None else float(esp_shell)
        if resolved_shell <= 0.0:
            raise ValueError(f"esp_shell must be > 0, got {resolved_shell}.")
        spec += f"|esp={float(esp_weight):.10g}|shell={resolved_shell:.10g}"
    return spec


def parse_metric_spec(
    spec: str,
) -> Tuple[str, float, float, float, float, float, float, float]:
    """Invert :py:func:`make_metric_spec`.

    :param spec: Canonical spec string.
    :return: ``(metric, omega, eps, charge_weight, dipole_weight,
        quadrupole_weight, esp_weight, esp_shell)``.
    """
    if "|" not in spec:
        if spec not in METRICS:
            raise ValueError(
                f"unknown metric {spec!r}; expected 'overlap' or 'coulomb'."
            )
        return spec, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, DEFAULT_ESP_SHELL
    metric, *parts = spec.split("|")
    values = dict(part.split("=") for part in parts)
    return (
        metric,
        float(values["omega"]),
        float(values["eps"]),
        float(values["q"]),
        float(values.get("d", 0.0)),
        float(values.get("Q", 0.0)),
        float(values.get("esp", 0.0)),
        float(values.get("shell", DEFAULT_ESP_SHELL)),
    )


# ── extra_data key helpers ────────────────────────────────────────────────────


def overlap_matrix_name(target_name: str) -> str:
    """Return the ``extra_data`` key holding a target's overlap matrices.

    :param target_name: Name of the RI-coefficient target.
    :return: The ``extra_data`` key.
    """
    return f"{target_name}_overlap_matrix"


def coulomb_matrix_name(target_name: str) -> str:
    """Return the ``extra_data`` key holding a target's Coulomb matrices.

    :param target_name: Name of the RI-coefficient target.
    :return: The ``extra_data`` key.
    """
    return f"{target_name}_coulomb_matrix"


def metric_matrix_name(target_name: str, metric: str) -> str:
    """Return the ``extra_data`` key for a target's two-centre metric matrices.

    :param target_name: Name of the RI-coefficient target.
    :param metric: Metric spec, as built by :py:func:`make_metric_spec`.
    :return: The ``extra_data`` key.
    """
    if metric == "overlap":
        return overlap_matrix_name(target_name)
    if metric == "coulomb":
        return coulomb_matrix_name(target_name)
    # Parse first so an unknown metric still raises the familiar error.
    parse_metric_spec(metric)
    suffix = re.sub(r"[^0-9a-zA-Z]+", "_", metric)
    return f"{target_name}_metric_{suffix}"


def ri_projections_name(target_name: str) -> str:
    """Return the ``extra_data`` key for a target's projections ``w = M c_ref``.

    :param target_name: Name of the RI-coefficient target.
    :return: The ``extra_data`` key.
    """
    return f"{target_name}_projections"


def ri_density_fit_constant_name(target_name: str) -> str:
    """Return the ``extra_data`` key for the pre-computed ``c_ref^T w`` constant.

    :param target_name: Name of the RI-coefficient target.
    :return: The ``extra_data`` key.
    """
    return f"{target_name}_density_fit_constant"


# ── Molecule / integral construction ──────────────────────────────────────────


def build_auxiliary_molecule(system: System, aux_basis: str) -> "Mole":
    """
    Build a PySCF molecule carrying the auxiliary basis, for integral evaluation.

    :param system: System whose positions are interpreted as Angstrom.
    :param aux_basis: Auxiliary basis name or ``"etb:<ao_basis>:<beta>"``.
    :return: A built molecule in spherical-harmonic (not Cartesian) form, which is
        the convention the RI coefficients follow.
    """
    gto, elements = _import_pyscf()

    if bool(system.pbc.any()):
        raise NotImplementedError(
            "density losses are implemented for non-periodic systems only: the "
            "auxiliary-basis metric is built from a molecular PySCF `Mole`, which "
            "ignores the cell, so a periodic system would silently be scored against "
            "a molecular metric. Supporting periodicity needs lattice sums (and for "
            "the Coulomb metric an Ewald treatment, since the sum is only "
            "conditionally convergent), not just passing the cell through."
        )

    types = system.types.detach().cpu().tolist()
    positions = system.positions.detach().cpu().tolist()
    atomic_numbers = tuple(sorted({int(t) for t in types}))

    mol = gto.Mole()
    mol.atom = "\n".join(
        f"{elements.ELEMENTS[int(t)]}  {x:.12f}  {y:.12f}  {z:.12f}"
        for t, (x, y, z) in zip(types, positions, strict=True)
    )
    # ``_load_auxiliary_basis`` is cached, and ``Mole.build`` mutates its basis.
    mol.basis = copy.deepcopy(_load_auxiliary_basis(aux_basis, atomic_numbers))
    mol.unit = "Angstrom"
    mol.verbose = 0
    mol.spin = None
    mol.cart = False
    mol.build()
    return mol


def compute_overlap_matrix(system: System, aux_basis: str) -> torch.Tensor:
    """Two-centre overlap matrix ``S`` (``int1e_ovlp``).

    :param system: System whose positions are interpreted as Angstrom.
    :param aux_basis: Auxiliary basis name or ``"etb:<ao_basis>:<beta>"``.
    :return: Dense ``(n_basis, n_basis)`` matrix in PySCF AO order, float64.
    """
    auxmol = build_auxiliary_molecule(system, aux_basis)
    return torch.from_numpy(auxmol.intor("int1e_ovlp")).to(torch.float64)


def compute_coulomb_matrix(system: System, aux_basis: str) -> torch.Tensor:
    """Two-centre Coulomb matrix ``J`` (``int2c2e``).

    :param system: System whose positions are interpreted as Angstrom.
    :param aux_basis: Auxiliary basis name or ``"etb:<ao_basis>:<beta>"``.
    :return: Dense ``(n_basis, n_basis)`` matrix in PySCF AO order, float64.
    """
    auxmol = build_auxiliary_molecule(system, aux_basis)
    return torch.from_numpy(auxmol.intor("int2c2e")).to(torch.float64)


def compute_long_range_coulomb_matrix(
    system: System, aux_basis: str, omega: float
) -> torch.Tensor:
    """
    Compute the long-range two-centre Coulomb matrix for kernel ``erf(omega r)/r``.

    In reciprocal space this kernel is ``4 pi/k^2 * exp(-k^2/(4 omega^2))``, i.e.
    the Coulomb metric with short-wavelength (sharp, near-nuclear) modes
    exponentially damped, leaving the smooth valence/far-field content that
    dominates the electrostatic potential outside the molecule.

    :param system: System whose positions are interpreted as Angstrom.
    :param aux_basis: Auxiliary basis name or ``"etb:<ao_basis>:<beta>"``.
    :param omega: Range-separation parameter, in inverse Bohr.
    :return: Dense long-range Coulomb matrix in PySCF AO order, float64.
    """
    auxmol = build_auxiliary_molecule(system, aux_basis)
    with auxmol.with_range_coulomb(omega):
        matrix = auxmol.intor("int2c2e")
    return torch.from_numpy(matrix).to(torch.float64)


def compute_charge_vector(system: System, aux_basis: str) -> torch.Tensor:
    """
    Compute the electron-count functional ``S_i = \\int chi_i(r) dr``.

    Only l=0 shells integrate to something nonzero, so this vector reads off the
    number of electrons carried by a coefficient vector: ``N = S_vec . c``. The
    analytic form is used rather than a numerical integral: for a contracted s
    shell with primitive exponents ``a_k`` and (PySCF-normalised) contraction
    coefficients ``d_k``, ``\\int chi dr = sum_k d_k (pi/a_k)^{3/2} /
    sqrt(4 pi)``, the ``1/sqrt(4 pi)`` coming from the ``Y_00`` factor carried by
    PySCF's spherical basis functions.

    :param system: System whose positions are interpreted as Angstrom.
    :param aux_basis: Auxiliary basis name or ``"etb:<ao_basis>:<beta>"``.
    :return: Charge functional in PySCF AO order, shape ``(naux,)``, float64.
    """
    import numpy as np

    auxmol = build_auxiliary_molecule(system, aux_basis)
    s_vector = np.zeros(auxmol.nao)
    ao_loc = auxmol.ao_loc_nr()
    for shell in range(auxmol.nbas):
        if auxmol.bas_angular(shell) != 0:
            continue
        exponents = auxmol.bas_exp(shell)
        contraction = auxmol._libcint_ctr_coeff(shell)
        primitives = (np.pi / exponents) ** 1.5 / np.sqrt(4.0 * np.pi)
        for i_contraction in range(contraction.shape[1]):
            s_vector[ao_loc[shell] + i_contraction] = float(
                np.dot(contraction[:, i_contraction], primitives)
            )
    return torch.from_numpy(s_vector).to(torch.float64)


def compute_multipole_vectors(
    system: System, aux_basis: str, angular: int
) -> torch.Tensor:
    """
    Compute the per-atom multipole functionals of order ``angular``.

    Row ``(a, m)`` reads off one component of atom ``a``'s multipole of the
    fitted density about its own centre: ``Q_a^{lm} = sum_i V[(a, m), i] c_i``.
    The operator is the Racah-normalised real solid harmonic
    ``sqrt(4 pi/(2l+1)) r^l Y_lm`` (so l=0 gives the electron count and l=1 the
    Cartesian dipole), under which only the aux functions on atom ``a`` with the
    matching ``l`` (and ``m``) contribute: for such a contracted shell with
    primitive exponents ``a_k`` and PySCF-normalised contraction coefficients
    ``d_k`` the integral is ``sqrt(4 pi/(2l+1)) sum_k d_k Gamma(l+3/2) /
    (2 a_k^(l+3/2))``.

    The ``m`` slots follow PySCF's within-shell component order, which is
    consistent across shells of the same ``l`` — all a rotation-invariant
    penalty ``sum_m |Q_a^{lm}|**2`` needs, since it does not care which real
    ``m`` each slot is.

    :param system: System whose positions are interpreted as Angstrom.
    :param aux_basis: Auxiliary basis name or ``"etb:<ao_basis>:<beta>"``.
    :param angular: Multipole order ``l``.
    :return: Dense ``(n_atoms * (2l+1), naux)`` matrix in PySCF AO order,
        float64.
    """
    import math

    import numpy as np

    auxmol = build_auxiliary_molecule(system, aux_basis)
    n_components = 2 * angular + 1
    vectors = np.zeros((auxmol.natm * n_components, auxmol.nao))
    ao_loc = auxmol.ao_loc_nr()
    angular_norm = np.sqrt(4.0 * np.pi / n_components)
    for shell in range(auxmol.nbas):
        if auxmol.bas_angular(shell) != angular:
            continue
        exponents = auxmol.bas_exp(shell)
        contraction = auxmol._libcint_ctr_coeff(shell)
        primitives = (
            angular_norm
            * math.gamma(angular + 1.5)
            / (2.0 * exponents ** (angular + 1.5))
        )
        atom = auxmol.bas_atom(shell)
        for i_contraction in range(contraction.shape[1]):
            moment = float(np.dot(contraction[:, i_contraction], primitives))
            for component in range(n_components):
                vectors[
                    atom * n_components + component,
                    ao_loc[shell] + i_contraction * n_components + component,
                ] = moment
    return torch.from_numpy(vectors).to(torch.float64)


def compute_surface_points(
    system: System, aux_basis: str, shell_scale: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build an accessible-surface point grid with area weights.

    Each atom gets a Fibonacci sphere of :py:data:`ESP_POINTS_PER_ATOM` points
    at ``shell_scale`` times its van der Waals radius; points buried inside any
    other atom's scaled sphere are culled. This is algorithmic — no per-system
    region choices — and traces cavity and pocket walls by construction, since
    that is what "accessible" means. Each point carries its share of its
    sphere's area, so the ESP penalty approximates a surface integral.

    The sphere orientations are fixed in space, so the grid is *not* exactly
    equivariant under rotations of the system (discretisation-level anisotropy
    only). That is harmless here: metrics are built on the unaugmented geometry
    and the losses evaluate in that same frame.

    :param system: System whose positions are interpreted as Angstrom.
    :param aux_basis: Auxiliary basis name or ``"etb:<ao_basis>:<beta>"`` (used
        only to build the molecule, i.e. for the atomic numbers/positions).
    :param shell_scale: Scaling of the van der Waals radii.
    :return: ``(coords, weights)``: points in Bohr, shape ``(n, 3)``, and their
        area weights in Bohr^2, shape ``(n,)``, both float64.
    """
    import numpy as np

    radii = importlib.import_module("pyscf.data.radii")

    auxmol = build_auxiliary_molecule(system, aux_basis)
    centres = auxmol.atom_coords()  # Bohr
    charges = auxmol.atom_charges()
    sphere_radii = shell_scale * radii.VDW[charges]

    n = ESP_POINTS_PER_ATOM
    # Fibonacci sphere: near-uniform, deterministic.
    golden = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - (2.0 * np.arange(n) + 1.0) / n
    rho = np.sqrt(1.0 - z * z)
    phi = golden * np.arange(n)
    unit = np.stack([rho * np.cos(phi), rho * np.sin(phi), z], axis=1)

    coords, weights = [], []
    for atom in range(auxmol.natm):
        points = centres[atom] + sphere_radii[atom] * unit
        buried = np.zeros(n, dtype=bool)
        for other in range(auxmol.natm):
            if other == atom:
                continue
            distances = np.linalg.norm(points - centres[other], axis=1)
            buried |= distances < sphere_radii[other]
        kept = points[~buried]
        coords.append(kept)
        area_per_point = 4.0 * np.pi * sphere_radii[atom] ** 2 / n
        weights.append(np.full(len(kept), area_per_point))
    coords = np.concatenate(coords)
    weights = np.concatenate(weights)
    return (
        torch.from_numpy(coords).to(torch.float64),
        torch.from_numpy(weights).to(torch.float64),
    )


def compute_esp_metric(
    system: System, aux_basis: str, shell_scale: float
) -> torch.Tensor:
    """
    Compute the surface-ESP quadratic form ``A W A^T``.

    ``A[i, g] = int chi_i(r) / |r - r_g| dr`` is the electrostatic potential of
    aux function ``i`` at surface point ``g`` (evaluated as two-centre Coulomb
    integrals against delta-like charges), and ``W`` holds the points' area
    weights, so ``dc^T (A W A^T) dc`` is the area-weighted sum of the squared
    ESP error of the fitted density over the accessible surface.

    The rank is at most the number of surface points, far below ``naux``: like
    the long-range metric, this term needs the base metric as a
    positive-definite floor.

    :param system: System whose positions are interpreted as Angstrom.
    :param aux_basis: Auxiliary basis name or ``"etb:<ao_basis>:<beta>"``.
    :param shell_scale: Scaling of the van der Waals radii for the surface.
    :return: Dense ``(n_basis, n_basis)`` matrix in PySCF AO order, float64.
    """
    import numpy as np

    gto, _ = _import_pyscf()

    auxmol = build_auxiliary_molecule(system, aux_basis)
    coords, weights = compute_surface_points(system, aux_basis, shell_scale)
    fakemol = gto.fakemol_for_charges(coords.numpy())
    integrals = gto.mole.intor_cross("int2c2e", auxmol, fakemol)  # (naux, n)
    weighted = integrals * weights.numpy()[None, :]
    return torch.from_numpy(np.ascontiguousarray(weighted @ integrals.T)).to(
        torch.float64
    )


def compute_metric_matrix(system: System, aux_basis: str, metric: str) -> torch.Tensor:
    """
    Compute a two-centre metric matrix of the auxiliary basis for one system.

    Beyond the plain ``S`` and ``J`` metrics this assembles the optional terms
    encoded in the spec (see :py:func:`make_metric_spec`)::

        M = eps * J + J_lr(omega)                  if omega > 0
        M = M + charge_weight * S_vec S_vec^T
        M = M + dipole_weight * V_1^T V_1
        M = M + quadrupole_weight * V_2^T V_2
        M = M + esp_weight * A W A^T

    The charge term is rank-1 and contributes ``charge_weight * (S_vec . dc)**2``
    to the loss, i.e. a penalty on the predicted density's electron-count error
    relative to the RI reference. The multipole terms (``V_l`` from
    :py:func:`compute_multipole_vectors`) penalise the error in each atom's
    distributed multipole of order ``l``, which is what determines the
    electrostatic potential outside the charge distribution. The ESP term
    (:py:func:`compute_esp_metric`) penalises the potential error directly on
    an accessible-surface grid.

    :param system: System whose positions are interpreted as Angstrom.
    :param aux_basis: Auxiliary basis name or ``"etb:<ao_basis>:<beta>"``.
    :param metric: Metric spec, as built by :py:func:`make_metric_spec`.
    :return: Dense ``(n_basis, n_basis)`` matrix in PySCF AO order, float64.
    """
    (
        base,
        omega,
        eps,
        charge_weight,
        dipole_weight,
        quadrupole_weight,
        esp_weight,
        esp_shell,
    ) = parse_metric_spec(metric)

    if omega > 0.0:
        matrix = compute_long_range_coulomb_matrix(system, aux_basis, omega)
        if eps > 0.0:
            matrix = matrix + eps * compute_coulomb_matrix(system, aux_basis)
    elif base == "overlap":
        matrix = compute_overlap_matrix(system, aux_basis)
    else:
        matrix = compute_coulomb_matrix(system, aux_basis)

    if charge_weight > 0.0:
        s_vector = compute_charge_vector(system, aux_basis)
        matrix = matrix + charge_weight * torch.outer(s_vector, s_vector)

    for weight, angular in ((dipole_weight, 1), (quadrupole_weight, 2)):
        if weight > 0.0:
            vectors = compute_multipole_vectors(system, aux_basis, angular)
            matrix = matrix + weight * (vectors.T @ vectors)

    if esp_weight > 0.0:
        matrix = matrix + esp_weight * compute_esp_metric(system, aux_basis, esp_shell)

    return matrix


# ── Packing ───────────────────────────────────────────────────────────────────


def pack_metric_matrices(matrices: List[torch.Tensor]) -> TensorMap:
    """
    Pack per-system metric matrices into a TensorMap, one block per system.

    Systems in a batch differ in size -- anything from a single atom to hundreds --
    so the matrices are stored **ragged**, as one block each, rather than padded into
    a single ``(n_systems, n_max, n_max)`` array. Padding costs
    ``n_systems * n_max**2`` against the ``sum_i n_i**2`` actually needed, which on a
    batch of mostly small systems with a few large ones measured ~8x more memory and
    transport, and can exhaust host memory outright. A TensorMap is already a
    collection of differently shaped blocks, so this needs no special transport.

    :param matrices: One dense square matrix per system.
    :return: TensorMap keyed by ``system``, block ``i`` holding ``(n_i, n_i)``.
    """
    if len(matrices) == 0:
        raise ValueError("expected at least one metric matrix to pack")

    device = matrices[0].device
    blocks = []
    for i_system, matrix in enumerate(matrices):
        basis = torch.arange(matrix.shape[0], dtype=torch.int32, device=device)
        blocks.append(
            TensorBlock(
                values=matrix,
                # The samples carry a "system" dimension because the O(3) augmenter
                # requires one to route rows to transformations. There are no
                # component axes, so it recognises the block as invariant and passes
                # it through untouched -- which is what allows the metric to be built
                # on the unaugmented geometry, before augmentation runs.
                samples=Labels(
                    names=["system", "basis"],
                    values=torch.stack(
                        [torch.full_like(basis, i_system), basis], dim=1
                    ),
                ),
                components=[],
                properties=Labels(names=["basis_2"], values=basis.reshape(-1, 1)),
            )
        )
    keys = Labels(
        names=["system"],
        values=torch.arange(len(matrices), dtype=torch.int32, device=device).reshape(
            -1, 1
        ),
    )
    return TensorMap(keys, blocks)


def unpack_metric_matrices(packed: TensorMap) -> List[torch.Tensor]:
    """
    Recover the per-system metric matrices.

    :param packed: Output of :func:`pack_metric_matrices`.
    :return: One ``(n_i, n_i)`` matrix per system, in batch order.
    """
    return [packed.block(i).values for i in range(len(packed))]


# ── Caching ───────────────────────────────────────────────────────────────────


_METRIC_MATRIX_CACHE: Optional[ByteBudgetCache] = None


def _metric_matrix_cache() -> ByteBudgetCache:
    """The per-process metric-matrix cache, created on first use.

    The collate transforms run inside the dataloader workers, which are kept
    alive between epochs (``persistent_workers``), so the cache survives across
    epochs. It is keyed by ``(aux_basis, metric, system id)`` on the
    *unaugmented* geometry -- the transforms run before the augmenter -- which
    is identical every epoch, so entries never go stale.

    :return: The cache.
    """
    global _METRIC_MATRIX_CACHE
    if _METRIC_MATRIX_CACHE is None:
        _METRIC_MATRIX_CACHE = ByteBudgetCache(collate_cache_max_bytes())
    return _METRIC_MATRIX_CACHE


def _batch_metric_matrices(
    systems: List[System],
    system_ids: Optional[List[int]],
    aux_basis: str,
    metric: str,
) -> List[torch.Tensor]:
    """Metric matrices for one batch, through the cache when ids are available.

    Without native system ids there is no stable cache key, so the matrices
    are recomputed -- correct, just slower.

    :param systems: The batch's systems, in batch order.
    :param system_ids: Native dataset ids of those systems, or ``None``.
    :param aux_basis: Auxiliary basis name.
    :param metric: Metric spec, as built by :py:func:`make_metric_spec`.
    :return: One dense square matrix per system, in batch order.
    """
    if system_ids is None:
        return [compute_metric_matrix(system, aux_basis, metric) for system in systems]
    cache = _metric_matrix_cache()
    matrices = []
    for system, system_id in zip(systems, system_ids, strict=True):
        # The full spec belongs in the key: the matrix depends on omega / eps /
        # charge_weight, so a hyperparameter change must not silently reuse
        # matrices built for the old one.
        key = (aux_basis, metric, system_id)
        matrix = cache.get(key)
        if matrix is None:
            matrix = compute_metric_matrix(system, aux_basis, metric)
            cache.put(key, matrix)
        matrices.append(matrix)
    return matrices


# ── Collate transforms ────────────────────────────────────────────────────────


def _metric_matrices_transform(
    target_to_aux_basis: Mapping[str, str],
    metric: str,
    systems: List[System],
    targets: Dict[str, TensorMap],
    extra: Dict[str, TensorMap],
) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
    system_ids = batch_system_ids(extra)
    packed_by_basis: Dict[str, TensorMap] = {}
    for target_name, aux_basis in target_to_aux_basis.items():
        if aux_basis not in packed_by_basis:
            packed_by_basis[aux_basis] = pack_metric_matrices(
                _batch_metric_matrices(systems, system_ids, aux_basis, metric)
            )
        extra[metric_matrix_name(target_name, metric)] = packed_by_basis[aux_basis]
    return systems, targets, extra


def get_metric_matrices_transform(
    target_to_aux_basis: Mapping[str, str],
    metric: str,
) -> Callable:
    """
    Build a collate transform attaching per-target two-centre metric matrices.

    **This transform must run before the augmenter.** The metric depends on the
    geometry, so it is computed in the dataset's own orientation, and the losses are
    evaluated in that frame (see
    :attr:`~metatrain.utils.loss.LossInterface.evaluate_in_original_frame`) rather
    than the augmented one.

    The matrices depend only on the unaugmented geometry, which is identical every
    epoch, so they are cached across epochs in a per-worker, byte-budgeted LRU cache
    (see :func:`_metric_matrix_cache`) keyed by the batch's native system ids
    (``mtt::aux::system_index``). Batches without such ids fall back to recomputing
    every time.

    Targets sharing an auxiliary basis share one computation per batch.

    :param target_to_aux_basis: Mapping from target name to auxiliary basis name.
    :param metric: Metric spec, as built by :py:func:`make_metric_spec`.
    :return: A collate transform.
    """
    parse_metric_spec(metric)  # validate eagerly, not in the first batch
    return functools.partial(
        _metric_matrices_transform, dict(target_to_aux_basis), metric
    )


def resolve_aux_basis(
    target_name: str, aux_basis: Union[str, Mapping[str, str]]
) -> str:
    """
    Resolve the auxiliary basis configured for one target.

    :param target_name: Name of the RI-coefficient target.
    :param aux_basis: Either a basis name applying to every target, or a mapping
        from target name to basis name.
    :return: The basis name for this target.
    """
    if isinstance(aux_basis, str):
        return aux_basis
    if target_name in aux_basis:
        return aux_basis[target_name]
    raise ValueError(
        f"no auxiliary basis configured for target '{target_name}'; "
        f"available targets: {', '.join(sorted(aux_basis))}."
    )
