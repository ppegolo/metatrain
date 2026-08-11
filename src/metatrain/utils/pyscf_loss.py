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
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

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

    import numpy
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


def _canonical_shells(
    esp_shell: Optional[Union[float, Sequence[float]]],
) -> Tuple[float, ...]:
    """Normalise an ESP shell specification to a sorted tuple of scalings.

    :param esp_shell: A single van der Waals scaling, a sequence of them (a
        RESP-style multi-shell surface), or ``None`` for the package default.
    :return: The shells, ascending and deduplicated.
    """
    if esp_shell is None:
        return (DEFAULT_ESP_SHELL,)
    if isinstance(esp_shell, (int, float)):
        shells: Tuple[float, ...] = (float(esp_shell),)
    else:
        shells = tuple(sorted({float(s) for s in esp_shell}))
    if len(shells) == 0:
        raise ValueError("esp_shell must name at least one shell.")
    for shell in shells:
        if shell <= 0.0:
            raise ValueError(f"esp_shell must be > 0, got {shell}.")
    return shells


def _shell_tag(esp_shell: Union[float, Sequence[float]]) -> str:
    """The canonical string form of a shell specification, for specs and keys.

    A single shell renders exactly as it always did, so specs and cache keys
    from before multi-shell surfaces existed stay byte-identical.

    :param esp_shell: A single scaling or a sequence of them.
    :return: Comma-joined ``%.10g`` values, ascending.
    """
    if isinstance(esp_shell, (int, float)):
        return f"{float(esp_shell):.10g}"
    return ",".join(f"{s:.10g}" for s in _canonical_shells(esp_shell))


def make_metric_spec(
    metric: str,
    omega: float = 0.0,
    eps: Optional[float] = None,
    charge_weight: float = 0.0,
    dipole_weight: float = 0.0,
    quadrupole_weight: float = 0.0,
    esp_weight: float = 0.0,
    esp_shell: Optional[Union[float, Sequence[float]]] = None,
    group_charge_weight: float = 0.0,
    interface_esp_weight: float = 0.0,
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
    :param esp_shell: Scaling of the van der Waals radii defining that surface —
        a single value, or a sequence of values for a RESP-style multi-shell
        surface (e.g. ``(1.0, 1.4, 2.0)``), whose grids are concatenated.
        ``None`` uses :py:data:`DEFAULT_ESP_SHELL`. Ignored when
        ``esp_weight == 0``.
    :param group_charge_weight: Weight on the per-group electron-count penalty,
        ``w * sum_g (S_g . dc)**2``, where ``S_g`` is the electron-count
        functional restricted to the atoms of group ``g`` (see
        :py:func:`compute_group_charge_factor`). ``0`` disables it. Unlike
        ``charge_weight``, this sees charge moved *between* groups, which
        leaves the total untouched.
    :param interface_esp_weight: Weight on the interface-ESP penalty, an
        area-weighted sum of ``|dV(r_g)|**2`` over the two fragments' buried
        contact patches (see :py:func:`compute_interface_esp_factor`). ``0``
        disables it. Needs the same per-atom fragment labels as the EC loss.
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
        ("group_charge_weight", group_charge_weight),
        ("interface_esp_weight", interface_esp_weight),
    ):
        if weight < 0.0:
            raise ValueError(f"{weight_name} must be >= 0, got {weight}.")
    if omega > 0.0 and metric != "coulomb":
        raise ValueError(
            "the long-range kernel erf(omega r)/r is a Coulomb-metric option; "
            f"got metric='{metric}' with omega={omega}."
        )
    if omega == 0.0 and charge_weight == 0.0:
        if (
            dipole_weight == 0.0
            and quadrupole_weight == 0.0
            and esp_weight == 0.0
            and group_charge_weight == 0.0
            and interface_esp_weight == 0.0
        ):
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
        shells = _canonical_shells(esp_shell)
        spec += f"|esp={float(esp_weight):.10g}|shell={_shell_tag(shells)}"
    if group_charge_weight > 0.0:
        spec += f"|gq={float(group_charge_weight):.10g}"
    if interface_esp_weight > 0.0:
        spec += f"|iesp={float(interface_esp_weight):.10g}"
    return spec


def parse_group_charge_weight(spec: str) -> float:
    """Read the per-group electron-count weight out of a metric spec.

    Kept apart from :py:func:`parse_metric_spec`, whose tuple is the documented
    return of a public function: this term arrived later, and widening that
    tuple would break every caller that unpacks it.

    :param spec: Canonical spec string.
    :return: The weight, ``0`` when the term is absent.
    """
    if "|" not in spec:
        return 0.0
    _, *parts = spec.split("|")
    values = dict(part.split("=") for part in parts)
    return float(values.get("gq", 0.0))


def parse_interface_esp_weight(spec: str) -> float:
    """Read the interface-ESP weight out of a metric spec.

    Kept apart from :py:func:`parse_metric_spec` for the same reason as
    :py:func:`parse_group_charge_weight`: widening that documented tuple would
    break every caller that unpacks it.

    :param spec: Canonical spec string.
    :return: The weight, ``0`` when the term is absent.
    """
    if "|" not in spec:
        return 0.0
    _, *parts = spec.split("|")
    values = dict(part.split("=") for part in parts)
    return float(values.get("iesp", 0.0))


def parse_metric_spec(
    spec: str,
) -> Tuple[
    str, float, float, float, float, float, float, Union[float, Tuple[float, ...]]
]:
    """Invert :py:func:`make_metric_spec`.

    :param spec: Canonical spec string.
    :return: ``(metric, omega, eps, charge_weight, dipole_weight,
        quadrupole_weight, esp_weight, esp_shell)``. ``esp_shell`` is a float
        for a single-shell surface — the historical shape — and a tuple of
        floats for a multi-shell one.
    """
    if "|" not in spec:
        if spec not in METRICS:
            raise ValueError(
                f"unknown metric {spec!r}; expected 'overlap' or 'coulomb'."
            )
        return spec, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, DEFAULT_ESP_SHELL
    metric, *parts = spec.split("|")
    values = dict(part.split("=") for part in parts)
    raw_shell = values.get("shell")
    if raw_shell is None:
        shell: Union[float, Tuple[float, ...]] = DEFAULT_ESP_SHELL
    else:
        shells = tuple(float(part) for part in raw_shell.split(","))
        shell = shells[0] if len(shells) == 1 else shells
    return (
        metric,
        float(values["omega"]),
        float(values["eps"]),
        float(values["q"]),
        float(values.get("d", 0.0)),
        float(values.get("Q", 0.0)),
        float(values.get("esp", 0.0)),
        shell,
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


def esp_factor_name(target_name: str, metric: str) -> str:
    """Return the ``extra_data`` key for a target's surface-ESP factors.

    :param target_name: Name of the RI-coefficient target.
    :param metric: Metric spec, as built by :py:func:`make_metric_spec`.
    :return: The ``extra_data`` key.
    """
    return f"{metric_matrix_name(target_name, metric)}_esp_factor"


def interface_esp_factor_name(target_name: str, metric: str) -> str:
    """Return the ``extra_data`` key for a target's interface-ESP factors.

    :param target_name: Name of the RI-coefficient target.
    :param metric: Metric spec, as built by :py:func:`make_metric_spec`.
    :return: The ``extra_data`` key.
    """
    return f"{metric_matrix_name(target_name, metric)}_interface_esp_factor"


def group_charge_factor_name(target_name: str, metric: str) -> str:
    """Return the ``extra_data`` key for a target's per-group charge factors.

    :param target_name: Name of the RI-coefficient target.
    :param metric: Metric spec, as built by :py:func:`make_metric_spec`.
    :return: The ``extra_data`` key.
    """
    return f"{metric_matrix_name(target_name, metric)}_group_charge_factor"


def charge_group_name(target_name: str) -> str:
    """Return the dataset field / ``extra_data`` key for per-atom charge groups.

    One integer label per atom, in the same shape as any other per-atom
    extra-data field. The labels name the pieces whose electron counts are
    penalised separately: the two partners of a complex, or the charged ends of
    a zwitterion. They need not be contiguous, and any number of groups is
    allowed.

    When this field is absent the per-atom EC fragment labels
    (:py:func:`ec_fragment_name`) are used, so a dataset prepared for the EC
    loss needs nothing added to penalise its two fragments' charges.

    :param target_name: Name of the RI-coefficient target.
    :return: The field / ``extra_data`` key.
    """
    return f"{target_name}_charge_group"


def ec_fragment_name(target_name: str) -> str:
    """Return the dataset field / ``extra_data`` key for per-atom fragment labels.

    The general spelling of the fragment specification: one 0/1 label per atom,
    stored per atom in the same shape as any other per-atom extra-data field. 0
    marks the fragment whose surface carries the interface patch — the ligand in
    a ligand-in-pocket structure — and 1 marks its partner.

    Prefer this over :py:func:`ec_fragment_split_name`, which can only describe
    a structure whose two fragments occupy contiguous ranges of the atom order.
    When both are present, this one wins.

    :param target_name: Name of the RI-coefficient target.
    :return: The field / ``extra_data`` key.
    """
    return f"{target_name}_fragment"


def ec_fragment_split_name(target_name: str) -> str:
    """Return the dataset field / ``extra_data`` key for a target's fragment split.

    The number of atoms in the first fragment of each dimer, stored per system
    in the same shape as the ``charge`` conditioning field. It is geometry
    metadata — nothing reference-derived — and it is the one per-structure
    input the EC machinery cannot compute from the system itself.

    :param target_name: Name of the RI-coefficient target.
    :return: The field / ``extra_data`` key.
    """
    return f"{target_name}_fragment_split"


def ec_machinery_name(target_name: str, part: str) -> str:
    """Return the ``extra_data`` key for one part of a target's EC machinery.

    The machinery is the geometry-only precontraction of the interface-patch
    ESP evaluation (see :py:func:`compute_ec_machinery`); it travels as three
    ragged TensorMaps — ``"moments"``, ``"vectors"`` and ``"constants"`` —
    packed like the metric matrices.

    :param target_name: Name of the RI-coefficient target.
    :param part: ``"moments"``, ``"vectors"`` or ``"constants"``.
    :return: The ``extra_data`` key.
    """
    return f"{target_name}_ec_machinery_{part}"


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


def compute_group_charge_factor(
    system: System, aux_basis: str, groups: "numpy.ndarray"
) -> torch.Tensor:
    """
    Compute the per-group electron-count factor ``F_q``.

    Row ``g`` is the electron-count functional
    :py:func:`compute_charge_vector` masked to the auxiliary functions of the
    atoms in group ``g``, so ``|F_q dc|**2 = sum_g (S_g . dc)**2`` is the summed
    squared error of the groups' electron counts.

    The whole-system ``charge_weight`` cannot see this error: charge moved from
    one group to another leaves ``S . dc`` at zero while shifting each group's
    potential, which on a binding interface is a first-order error in every
    electrostatic quantity, and in a zwitterion is the difference between the
    neutral and the charge-separated form.

    Carried as a factor rather than folded into the dense metric because it
    depends on the groups, which are per-system data the metric-matrix cache is
    not keyed on -- and because it is a rank-``K`` term with ``K`` at two or
    three, so forming ``F_q^T F_q`` would cost far more than applying it.

    :param system: System whose positions are interpreted as Angstrom.
    :param aux_basis: Auxiliary basis name or ``"etb:<ao_basis>:<beta>"``.
    :param groups: One integer label per atom; the distinct labels, in ascending
        order, index the rows.
    :return: Dense ``(n_groups, n_basis)`` matrix in PySCF AO order, float64.
    """
    import numpy as np

    auxmol = build_auxiliary_molecule(system, aux_basis)
    labels = np.asarray(groups).reshape(-1)
    if len(labels) != auxmol.natm:
        raise ValueError(
            f"the charge groups have {len(labels)} labels for a system of "
            f"{auxmol.natm} atoms; there must be exactly one label per atom."
        )

    owner = np.empty(auxmol.nao, dtype=int)
    ao_loc = auxmol.ao_loc_nr()
    for shell in range(auxmol.nbas):
        owner[ao_loc[shell] : ao_loc[shell + 1]] = auxmol.bas_atom(shell)

    charge = compute_charge_vector(system, aux_basis).numpy()
    rows = [
        charge * np.isin(owner, np.flatnonzero(labels == label))
        for label in np.unique(labels)
    ]
    return torch.from_numpy(np.ascontiguousarray(np.stack(rows))).to(torch.float64)


def compute_surface_points(
    system: System, aux_basis: str, shell_scale: Union[float, Sequence[float]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build an accessible-surface point grid with area weights.

    Each atom gets a Fibonacci sphere of :py:data:`ESP_POINTS_PER_ATOM` points
    at ``shell_scale`` times its van der Waals radius; points buried inside any
    other atom's scaled sphere are culled. This is algorithmic — no per-system
    region choices — and traces cavity and pocket walls by construction, since
    that is what "accessible" means. Each point carries its share of its
    sphere's area, so the ESP penalty approximates a surface integral.

    A sequence of scalings builds a RESP-style multi-shell surface: one grid
    per shell, each culled against its own scaled spheres, concatenated in
    ascending shell order. The outer shells sample the far field, where the
    low multipoles of the density dominate the potential, so a single tight
    shell under-constrains exactly the content that a molecule's neighbours
    feel.

    The sphere orientations are fixed in space, so the grid is *not* exactly
    equivariant under rotations of the system (discretisation-level anisotropy
    only). That is harmless here: metrics are built on the unaugmented geometry
    and the losses evaluate in that same frame.

    :param system: System whose positions are interpreted as Angstrom.
    :param aux_basis: Auxiliary basis name or ``"etb:<ao_basis>:<beta>"`` (used
        only to build the molecule, i.e. for the atomic numbers/positions).
    :param shell_scale: Scaling of the van der Waals radii — one value, or a
        sequence of values whose grids are concatenated.
    :return: ``(coords, weights)``: points in Bohr, shape ``(n, 3)``, and their
        area weights in Bohr^2, shape ``(n,)``, both float64.
    """
    import numpy as np

    radii = importlib.import_module("pyscf.data.radii")

    auxmol = build_auxiliary_molecule(system, aux_basis)
    centres = auxmol.atom_coords()  # Bohr
    charges = auxmol.atom_charges()

    n = ESP_POINTS_PER_ATOM
    # Fibonacci sphere: near-uniform, deterministic.
    golden = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - (2.0 * np.arange(n) + 1.0) / n
    rho = np.sqrt(1.0 - z * z)
    phi = golden * np.arange(n)
    unit = np.stack([rho * np.cos(phi), rho * np.sin(phi), z], axis=1)

    coords, weights = [], []
    for shell in _canonical_shells(shell_scale):
        sphere_radii = shell * radii.VDW[charges]
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


def compute_esp_factor(
    system: System, aux_basis: str, shell_scale: Union[float, Sequence[float]]
) -> torch.Tensor:
    """
    Compute the surface-ESP factor ``F = W^(1/2) A^T``.

    ``A[i, g] = int chi_i(r) / |r - r_g| dr`` is the electrostatic potential of
    aux function ``i`` at surface point ``g`` (evaluated as two-centre Coulomb
    integrals against delta-like charges), and ``W`` holds the points' area
    weights, so ``|F dc|**2`` is the area-weighted sum of the squared ESP error
    of the fitted density over the accessible surface.

    The ESP term is carried as this factor rather than folded into the dense
    metric: forming ``A W A^T`` costs ``naux^2 * n_points`` flops per system —
    minutes per large system inside a single-threaded dataloader worker — while
    the factor needs only the integrals, and the loss-side ``F dc`` product is
    no more expensive than the quadratic form it supplements.

    :param system: System whose positions are interpreted as Angstrom.
    :param aux_basis: Auxiliary basis name or ``"etb:<ao_basis>:<beta>"``.
    :param shell_scale: Scaling of the van der Waals radii for the surface —
        one value, or a sequence for a multi-shell surface.
    :return: Dense ``(n_points, n_basis)`` matrix in PySCF AO order, float64.
    """
    import numpy as np

    gto, _ = _import_pyscf()

    auxmol = build_auxiliary_molecule(system, aux_basis)
    coords, weights = compute_surface_points(system, aux_basis, shell_scale)
    fakemol = gto.fakemol_for_charges(coords.numpy())
    integrals = gto.mole.intor_cross("int2c2e", auxmol, fakemol)  # (naux, n)
    factor = np.sqrt(weights.numpy())[:, None] * integrals.T
    return torch.from_numpy(np.ascontiguousarray(factor)).to(torch.float64)


def compute_interface_esp_factor(
    system: System,
    aux_basis: str,
    split: Union[int, Sequence[int], "numpy.ndarray"],
) -> torch.Tensor:
    """
    Compute the interface-ESP factor ``F = W^(1/2) A^T`` on the contact patches.

    The points are the union of the two fragments' solvent-excluded contact
    patches (:py:func:`_ec_interface_patch`, evaluated once per orientation), so
    ``|F dc|**2`` is the area-weighted squared ESP error of the fitted density
    exactly where the accessible-surface grid of :py:func:`compute_esp_factor`
    has no points: the culling that traces "accessible" removes the buried
    interface, yet the interface is where binding electrostatics is decided and
    where the potential is a deep cancellation between the two partners'
    nuclear and electronic terms. Nothing but explicit sampling constrains it.

    Areas are raw (Bohr^2), the same convention as the surface factor, so
    ``interface_esp_weight`` trades off against ``esp_weight`` directly — a
    ratio of 3-10 upweights each buried surface element by that factor.

    :param system: System whose positions are interpreted as Angstrom.
    :param aux_basis: Auxiliary basis name or ``"etb:<ao_basis>:<beta>"``.
    :param split: Fragment specification, see :py:func:`ec_fragment_indices`.
    :return: Dense ``(n_points, n_basis)`` matrix in PySCF AO order, float64.
        Zero rows when either fragment is empty or no patch exists (monomers),
        making the structure's contribution to the penalty exactly zero.
    """
    import numpy as np

    gto, _ = _import_pyscf()

    auxmol = build_auxiliary_molecule(system, aux_basis)
    own, partner = ec_fragment_indices(split, auxmol.natm)
    empty = torch.zeros((0, auxmol.nao), dtype=torch.float64)
    if len(own) == 0 or len(partner) == 0:
        return empty
    centres = auxmol.atom_coords()
    charges = auxmol.atom_charges()

    labels = np.zeros(auxmol.natm, dtype=int)
    labels[partner] = 1
    pieces = []
    for orientation in (labels, 1 - labels):
        points, areas = _ec_interface_patch(centres, charges, orientation)
        if len(points):
            pieces.append((points, areas))
    if not pieces:
        return empty
    points = np.concatenate([p for p, _ in pieces])
    areas = np.concatenate([a for _, a in pieces])

    blocks = []
    for start in range(0, len(points), 2000):
        fake = gto.fakemol_for_charges(points[start : start + 2000])
        blocks.append(gto.mole.intor_cross("int2c2e", auxmol, fake))
    integrals = np.concatenate(blocks, axis=1)  # (naux, n)
    factor = np.sqrt(areas)[:, None] * integrals.T
    return torch.from_numpy(np.ascontiguousarray(factor)).to(torch.float64)


def compute_esp_metric(
    system: System, aux_basis: str, shell_scale: Union[float, Sequence[float]]
) -> torch.Tensor:
    """
    Compute the surface-ESP quadratic form ``A W A^T = F^T F``.

    Reference implementation for the standalone
    :py:func:`compute_metric_matrix` API and for tests; the training pipeline
    carries :py:func:`compute_esp_factor` instead and never forms this matrix.

    The rank is at most the number of surface points, far below ``naux``: like
    the long-range metric, this term needs the base metric as a
    positive-definite floor.

    :param system: System whose positions are interpreted as Angstrom.
    :param aux_basis: Auxiliary basis name or ``"etb:<ao_basis>:<beta>"``.
    :param shell_scale: Scaling of the van der Waals radii for the surface.
    :return: Dense ``(n_basis, n_basis)`` matrix in PySCF AO order, float64.
    """
    factor = compute_esp_factor(system, aux_basis, shell_scale)
    return factor.T @ factor


# ── Electrostatic-complementarity machinery ───────────────────────────────────
#
# EC compares the two fragments' electrostatic potentials on the interface
# patch: ``EC = -corr(v_0, v_1)``, area-weighted, on fragment 0's
# solvent-excluded contact patch. Both potentials are affine in the RI
# coefficients, ``v_f = n_f - V_f^T c``, so every ingredient of the correlation
# is a quadratic form in ``c`` whose geometry-dependent parts can be contracted
# once per structure. With ``w`` the normalised area weights and
# ``K = diag(w) - w w^T`` (the weighted-centring kernel), the centred weighted
# inner products are
#
#     A_fg = v_f^T K v_g
#          = s_fg - y_g . t_f - y_f . t_g + y_f^T M y_g ,   y_f = mask_f * c
#
# with the geometry-only ``M = V K V^T`` (naux x naux), ``t_f = V K n_f`` and
# ``s_fg = n_f^T K n_g``, and ``EC = -A_01 / sqrt(A_00 A_11)``. The per-step
# cost is two ``naux x naux`` matrix-vector products per structure and
# coefficient vector — no surface points, no PySCF — and everything contracted
# here depends only on the geometry and the fragment split, never on any
# coefficient vector.

#: Probe radius of the EC interface patch: a water probe, 1.4 Angstrom in Bohr.
EC_PROBE_RADIUS = 1.4 / 0.52917721092

#: Points per atomic sphere for the EC patch. Denser than the ESP loss grid
#: because the patch is a small cutout of the surface; matches the grid the EC
#: benchmark itself is evaluated on, so the trained quantity is the scored one.
EC_POINTS_PER_ATOM = 200

#: Patches with fewer points than this carry no meaningful correlation; the
#: structure then contributes zero to the EC loss (monomers, split dimers).
EC_MIN_PATCH_POINTS = 10

#: Closest approach in Angstrom that a jittered placement must respect. A shift
#: bringing any ligand-partner pair below this is drawn again; see
#: :py:func:`_ec_partner_shifts` for why the patch cannot catch this by itself.
EC_JITTER_MIN_CONTACT = 1.0

#: Draws allowed before the true placement is used instead.
EC_JITTER_ATTEMPTS = 8


def _ec_fibonacci_sphere(count: int) -> "numpy.ndarray":
    """Near-uniform points on the unit sphere, deterministic.

    :param count: Number of points.
    :return: Numpy array of shape ``(count, 3)``.
    """
    import numpy as np

    golden = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - (2.0 * np.arange(count) + 1.0) / count
    rho = np.sqrt(1.0 - z * z)
    phi = golden * np.arange(count)
    return np.stack([rho * np.cos(phi), rho * np.sin(phi), z], axis=1)


def ec_fragment_indices(
    split: Union[int, Sequence[int], "numpy.ndarray"], n_atoms: int
) -> Tuple["numpy.ndarray", "numpy.ndarray"]:
    """Resolve a fragment specification into the two groups of atom indices.

    Two spellings are accepted, and they mean the same thing whenever both can
    express it:

    * an ``int`` — the number of leading atoms that form fragment 0, the shape a
      dimer file already has, where the two molecules are stored one after the
      other;
    * a per-atom sequence of 0/1 labels — the general spelling, which does not
      require the two fragments to occupy contiguous ranges. A ligand inside a
      pocket is the case that needs it: the ligand's atoms may sit anywhere in
      the file, and the pocket's atoms around them.

    Fragment 0 is the one whose surface carries the patch — the ligand, in the
    ligand-in-pocket reading.

    :param split: Leading-atom count, or per-atom 0/1 labels.
    :param n_atoms: Number of atoms in the structure.
    :return: ``(own, partner)``, index arrays into the atoms.
    :raises ValueError: If the labels are not 0/1, or are the wrong length.
    """
    import numpy as np

    if isinstance(split, (int, np.integer)):
        labels = (np.arange(n_atoms) >= int(split)).astype(int)
    else:
        labels = np.asarray(split, dtype=int).reshape(-1)
        if len(labels) != n_atoms:
            raise ValueError(
                f"fragment labels have length {len(labels)}, expected one per "
                f"atom ({n_atoms})."
            )
        if not np.isin(labels, (0, 1)).all():
            raise ValueError(
                "fragment labels must be 0 (the patch-carrying fragment) or 1 "
                f"(its partner); got values {sorted(set(labels.tolist()))}."
            )
    return np.where(labels == 0)[0], np.where(labels == 1)[0]


def _ec_interface_patch(
    centres: "numpy.ndarray",
    charges: "numpy.ndarray",
    split: Union[int, Sequence[int], "numpy.ndarray"],
) -> Tuple["numpy.ndarray", "numpy.ndarray"]:
    """Fragment 0's solvent-excluded contact patch with a partner.

    Fragment 0's own SES is built as if the partner were absent — SAS points
    over its own atoms, projected back to the van der Waals sphere of the atom
    that generated them — and a contact point is kept when the partner buries
    the *probe centre* that generated it, since it is the probe, not the
    contact point on the vdW surface, that the partner blocks.

    :param centres: All atom positions in Bohr, shape ``(n_atoms, 3)``.
    :param charges: Atomic numbers of all atoms.
    :param split: Fragment specification, see :py:func:`ec_fragment_indices`.
    :return: ``(points, areas)`` in Bohr and Bohr^2, possibly empty.
    """
    import numpy as np

    radii = importlib.import_module("pyscf.data.radii")

    vdw = radii.VDW[charges]
    accessible = vdw + EC_PROBE_RADIUS
    own, partner = ec_fragment_indices(split, len(centres))

    # SAS of fragment 0 alone: burial judged among its own atoms only.
    unit = _ec_fibonacci_sphere(EC_POINTS_PER_ATOM)
    points, areas, owner = [], [], []
    for atom in own:
        candidates = centres[atom] + accessible[atom] * unit
        buried = np.zeros(len(candidates), dtype=bool)
        for other in own:
            if other == atom:
                continue
            distance = np.linalg.norm(candidates - centres[other], axis=1)
            buried |= distance < accessible[other]
        kept = candidates[~buried]
        points.append(kept)
        areas.append(
            np.full(len(kept), 4.0 * np.pi * accessible[atom] ** 2 / EC_POINTS_PER_ATOM)
        )
        owner.append(np.full(len(kept), atom, dtype=int))
    if not points:
        return np.zeros((0, 3)), np.zeros(0)
    probe_centres = np.concatenate(points)
    areas = np.concatenate(areas)
    owner = np.concatenate(owner)
    if len(probe_centres) == 0:
        return np.zeros((0, 3)), np.zeros(0)

    # Project onto the vdW sphere (the SES contact patch), shrink the area
    # elements accordingly, and drop points that fall inside another own atom.
    direction = probe_centres - centres[owner]
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    contact = centres[owner] + vdw[owner][:, None] * direction
    contact_areas = areas * (vdw[owner] / accessible[owner]) ** 2
    distances = np.linalg.norm(contact[:, None, :] - centres[own][None, :, :], axis=2)
    inside = distances < vdw[own][None, :] - 1e-9
    # ``owner`` holds atom indices; the column of that atom within ``own``.
    column = np.searchsorted(own, owner)
    inside[np.arange(len(contact)), column] = False
    kept = ~inside.any(axis=1)
    contact, contact_areas, probe_centres = (
        contact[kept],
        contact_areas[kept],
        probe_centres[kept],
    )

    # The patch: contact points whose probe centre the partner buries.
    distances = np.linalg.norm(
        probe_centres[:, None, :] - centres[partner][None, :, :], axis=2
    )
    buried = (distances < accessible[partner][None, :]).any(axis=1)
    return contact[buried], contact_areas[buried]


def _ec_displaced_system(
    system: System,
    split: Union[int, Sequence[int], "numpy.ndarray"],
    displacement: "numpy.ndarray",
) -> System:
    """Return a copy of ``system`` with the partner fragment rigidly shifted.

    :param system: System whose positions are interpreted as Angstrom.
    :param split: Fragment specification, see :py:func:`ec_fragment_indices`.
    :param displacement: Shift in Angstrom, shape ``(3,)``.
    :return: A new System; the input is not modified.
    """
    _, partner = ec_fragment_indices(split, len(system.positions))
    positions = system.positions.clone()
    shift = torch.as_tensor(
        displacement, dtype=positions.dtype, device=positions.device
    )
    positions[torch.as_tensor(partner, device=positions.device)] += shift
    return System(
        types=system.types,
        positions=positions,
        cell=system.cell,
        pbc=system.pbc,
    )


def ec_pointwise_pieces(
    system: System,
    aux_basis: str,
    split: Union[int, Sequence[int], "numpy.ndarray"],
    displacement: Optional["numpy.ndarray"] = None,
) -> Optional[Dict[str, Any]]:
    """The point-space ingredients of EC for one structure, before contraction.

    Everything here depends only on the geometry, the auxiliary basis and the
    fragment split. Exposed separately from :py:func:`compute_ec_machinery` so
    tests can evaluate EC pointwise and pin the contraction identity down.

    ``displacement`` rigidly moves the partner fragment — its nuclei and the
    auxiliary functions centred on them — before anything is built. This changes
    the *measurement*, not the data: the resulting EC is a different functional
    of the same coefficients, and both the prediction and the reference are read
    through it, so no new reference density is implied. See
    :py:func:`get_ec_machinery_transform` for why that is the point.

    :param system: System whose positions are interpreted as Angstrom.
    :param aux_basis: Auxiliary basis name or ``"etb:<ao_basis>:<beta>"``.
    :param split: Fragment specification, see :py:func:`ec_fragment_indices`.
    :param displacement: Rigid shift of the partner fragment in Angstrom, shape
        ``(3,)``, or ``None`` for the true placement.
    :return: ``None`` when the patch has fewer than
        :py:data:`EC_MIN_PATCH_POINTS` points; otherwise a dict with the ESP
        operator ``V[P, p]`` (naux x npoints), normalised area ``weights``,
        the two fragments' ``nuclear`` potentials at the points, and the two
        aux-function ownership ``masks``, all float64 numpy arrays.
    """
    import numpy as np

    gto, _ = _import_pyscf()

    if displacement is not None:
        system = _ec_displaced_system(system, split, displacement)
    auxmol = build_auxiliary_molecule(system, aux_basis)
    own, partner = ec_fragment_indices(split, auxmol.natm)
    if len(own) == 0 or len(partner) == 0:
        return None
    centres = auxmol.atom_coords()
    charges = auxmol.atom_charges()

    points, areas = _ec_interface_patch(centres, charges, split)
    if len(points) < EC_MIN_PATCH_POINTS:
        return None
    weights = areas / areas.sum()

    blocks = []
    for start in range(0, len(points), 2000):
        fake = gto.fakemol_for_charges(points[start : start + 2000])
        blocks.append(gto.mole.intor_cross("int2c2e", auxmol, fake))
    operator = np.concatenate(blocks, axis=1)

    nuclear = []
    for atoms in (own, partner):
        distances = np.linalg.norm(
            points[:, None, :] - centres[atoms][None, :, :], axis=2
        )
        nuclear.append((charges[atoms].astype(float)[None, :] / distances).sum(axis=1))

    owner = np.empty(auxmol.nao, dtype=int)
    ao_loc = auxmol.ao_loc_nr()
    for shell in range(auxmol.nbas):
        owner[ao_loc[shell] : ao_loc[shell + 1]] = auxmol.bas_atom(shell)
    masks = [
        np.isin(owner, own).astype(float),
        np.isin(owner, partner).astype(float),
    ]

    return {
        "operator": operator,
        "weights": weights,
        "nuclear": nuclear,
        "masks": masks,
    }


def compute_ec_machinery(
    system: System,
    aux_basis: str,
    split: Union[int, Sequence[int], "numpy.ndarray"],
    displacement: Optional["numpy.ndarray"] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Contract the EC patch evaluation into geometry-only per-structure tensors.

    See the section comment above for the algebra. The electronic potential is
    ``-V_f^T c`` (electrons lower the potential), so with ``v_f = n_f - V_f^T c``
    the cross terms of ``A_fg`` enter with a minus sign, which is what the loss
    side applies.

    :param system: System whose positions are interpreted as Angstrom.
    :param aux_basis: Auxiliary basis name or ``"etb:<ao_basis>:<beta>"``.
    :param split: Fragment specification, see :py:func:`ec_fragment_indices`.
    :param displacement: Rigid shift of the partner fragment in Angstrom, or
        ``None`` for the true placement.
    :return: ``None`` when the structure has no usable patch; otherwise
        ``(moments, vectors, constants)``: ``M = V K V^T`` of shape
        ``(naux, naux)``; the rows ``t_0, t_1, mask_0, mask_1`` of shape
        ``(4, naux)``; and ``s_fg = n_f^T K n_g`` of shape ``(2, 2)``. All
        float64.
    """
    import numpy as np

    pieces = ec_pointwise_pieces(system, aux_basis, split, displacement)
    if pieces is None:
        return None
    operator, weights = pieces["operator"], pieces["weights"]
    n0, n1 = pieces["nuclear"]

    # V K x = (V W x) - (V w)(w . x): the centring never needs the
    # (npoints x npoints) kernel formed explicitly.
    weighted = operator * weights[None, :]
    mean_column = operator @ weights
    moments = weighted @ operator.T - np.outer(mean_column, mean_column)
    t_vectors = [weighted @ n - mean_column * (weights @ n) for n in (n0, n1)]
    constants = np.array(
        [
            [weights @ (na * nb) - (weights @ na) * (weights @ nb) for nb in (n0, n1)]
            for na in (n0, n1)
        ]
    )

    vectors = np.stack(t_vectors + pieces["masks"])
    return (
        torch.from_numpy(np.ascontiguousarray(moments)).to(torch.float64),
        torch.from_numpy(vectors).to(torch.float64),
        torch.from_numpy(constants).to(torch.float64),
    )


def strip_esp_from_spec(
    spec: str,
) -> Tuple[str, float, Union[float, Tuple[float, ...]]]:
    """
    Split a metric spec into its dense part and its factored ESP part.

    The factored terms — surface ESP, per-group charge, interface ESP — all
    drop out of the dense part, which is what the metric-matrix cache is keyed
    on and shared across.

    :param spec: Metric spec, as built by :py:func:`make_metric_spec`.
    :return: ``(base_spec, esp_weight, esp_shell)``: the spec with the ESP term
        removed (everything that lives in the dense metric matrix), and the ESP
        parameters carried separately; ``esp_shell`` is a float or, for a
        multi-shell surface, a tuple of floats.
    """
    metric, omega, eps, charge, dipole, quadrupole, esp_weight, esp_shell = (
        parse_metric_spec(spec)
    )
    base_spec = make_metric_spec(
        metric, omega, eps if omega > 0.0 else None, charge, dipole, quadrupole
    )
    return base_spec, esp_weight, esp_shell


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
    an accessible-surface grid. (The training pipeline does not fold the ESP
    term into the matrix: the collate transform ships
    :py:func:`compute_esp_factor` separately and the loss adds
    ``esp_weight * |F dc|**2`` itself, avoiding the ``naux^2 * n_points``
    assembly cost. This function assembles everything for standalone use.)
    The per-group charge and interface-ESP terms are **not** assembled here:
    both depend on per-system fragment labels, which no spec string carries,
    so they travel as factors only (:py:func:`compute_group_charge_factor`,
    :py:func:`compute_interface_esp_factor`).

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
            # V^T V, but per row: V is block-sparse (a row touches only the
            # matching-l functions of one atom), so the dense gemm would waste
            # naux^2 * n_rows flops — minutes per large system inside a
            # single-threaded dataloader worker — on multiplying zeros.
            for row in vectors:
                nonzero = torch.nonzero(row).flatten()
                if len(nonzero):
                    values = row[nonzero]
                    matrix[nonzero[:, None], nonzero] += weight * torch.outer(
                        values, values
                    )

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

    :param matrices: One dense matrix per system — square for metric matrices,
        rectangular for factors such as the surface-ESP one.
    :return: TensorMap keyed by ``system``, block ``i`` holding the matrix.
    """
    if len(matrices) == 0:
        raise ValueError("expected at least one metric matrix to pack")

    device = matrices[0].device
    blocks = []
    for i_system, matrix in enumerate(matrices):
        rows = torch.arange(matrix.shape[0], dtype=torch.int32, device=device)
        columns = torch.arange(matrix.shape[1], dtype=torch.int32, device=device)
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
                    values=torch.stack([torch.full_like(rows, i_system), rows], dim=1),
                ),
                components=[],
                properties=Labels(names=["basis_2"], values=columns.reshape(-1, 1)),
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


def _batch_esp_factors(
    systems: List[System],
    system_ids: Optional[List[int]],
    aux_basis: str,
    esp_shell: Union[float, Sequence[float]],
) -> List[torch.Tensor]:
    """Surface-ESP factors for one batch, through the cache when ids exist.

    The factor depends only on ``(aux_basis, esp_shell)`` and the geometry —
    not on ``esp_weight``, which the loss applies — so the cache key omits the
    weight and entries are shared across weight settings.

    :param systems: The batch's systems, in batch order.
    :param system_ids: Native dataset ids of those systems, or ``None``.
    :param aux_basis: Auxiliary basis name.
    :param esp_shell: Scaling of the van der Waals radii for the surface.
    :return: One ``(n_points_i, n_basis_i)`` factor per system, in batch order.
    """
    if system_ids is None:
        return [compute_esp_factor(system, aux_basis, esp_shell) for system in systems]
    cache = _metric_matrix_cache()
    factors = []
    for system, system_id in zip(systems, system_ids, strict=True):
        key = (aux_basis, f"esp-factor|shell={_shell_tag(esp_shell)}", system_id)
        factor = cache.get(key)
        if factor is None:
            factor = compute_esp_factor(system, aux_basis, esp_shell)
            cache.put(key, factor)
        factors.append(factor)
    return factors


def _batch_interface_esp_factors(
    systems: List[System],
    system_ids: Optional[List[int]],
    aux_basis: str,
    splits: List[Any],
) -> List[torch.Tensor]:
    """Interface-ESP factors for one batch, through the cache when ids exist.

    The factor depends on the geometry *and* the fragment split, so the split
    goes in the cache key, exactly as for the per-group charge factors.

    :param systems: The batch's systems, in batch order.
    :param system_ids: Native dataset ids of those systems, or ``None``.
    :param aux_basis: Auxiliary basis name.
    :param splits: Fragment specification per system, see
        :py:func:`ec_fragment_indices`.
    :return: One ``(n_points_i, n_basis_i)`` factor per system, in batch order.
    """
    if system_ids is None:
        return [
            compute_interface_esp_factor(system, aux_basis, split)
            for system, split in zip(systems, splits, strict=True)
        ]
    cache = _metric_matrix_cache()
    factors = []
    for system, system_id, split in zip(systems, system_ids, splits, strict=True):
        key = (aux_basis, f"interface-esp|split={_ec_split_tag(split)}", system_id)
        factor = cache.get(key)
        if factor is None:
            factor = compute_interface_esp_factor(system, aux_basis, split)
            cache.put(key, factor)
        factors.append(factor)
    return factors


def _batch_group_charge_factors(
    systems: List[System],
    system_ids: Optional[List[int]],
    aux_basis: str,
    groups: List["numpy.ndarray"],
) -> List[torch.Tensor]:
    """Per-group charge factors for one batch, through the cache when ids exist.

    The factor depends on the geometry *and* on the grouping, so the grouping
    goes in the cache key: two runs that split the same structures differently
    must not share entries.

    :param systems: The batch's systems, in batch order.
    :param system_ids: Native dataset ids of those systems, or ``None``.
    :param aux_basis: Auxiliary basis name.
    :param groups: Per-atom group labels of each system, in batch order.
    :return: One ``(n_groups_i, n_basis_i)`` factor per system, in batch order.
    """
    if system_ids is None:
        return [
            compute_group_charge_factor(system, aux_basis, label)
            for system, label in zip(systems, groups, strict=True)
        ]
    cache = _metric_matrix_cache()
    factors = []
    for system, system_id, label in zip(systems, system_ids, groups, strict=True):
        key = (aux_basis, f"group-charge|groups={_ec_split_tag(label)}", system_id)
        factor = cache.get(key)
        if factor is None:
            factor = compute_group_charge_factor(system, aux_basis, label)
            cache.put(key, factor)
        factors.append(factor)
    return factors


def _split_per_atom_labels(
    block: "TensorBlock", n_systems: int
) -> List["numpy.ndarray"]:
    """Split a per-atom extra-data block into one label array per batch system.

    The values of the ``"system"`` sample dimension are the *original dataset*
    indices, not batch positions, so they must not be used to index the batch
    list. ``group_and_join`` concatenates the per-sample blocks in batch order,
    so the systems are the runs of equal ids, in order of first appearance.

    :param block: A per-atom block with a ``"system"`` sample dimension.
    :param n_systems: Number of systems in the batch.
    :return: One label array per system, in batch order.
    :raises RuntimeError: If the block does not cover exactly the batch.
    """
    import numpy as np

    labels = block.values.reshape(-1).to(torch.int64).cpu().numpy()
    index = block.samples.column("system").cpu().numpy()
    # Boundaries of the runs of equal ids; ``group_and_join`` never interleaves
    # two systems' rows, so a run is exactly one system.
    starts = np.flatnonzero(np.r_[True, index[1:] != index[:-1]])
    if len(starts) != n_systems:
        raise RuntimeError(
            f"per-atom extra data covers {len(starts)} systems, but the batch "
            f"has {n_systems}; the field is malformed or was reordered"
        )
    return [np.asarray(part) for part in np.split(labels, starts[1:])]


def _batch_charge_groups(
    target_name: str, systems: List[System], extra: Dict[str, TensorMap]
) -> List["numpy.ndarray"]:
    """The per-atom charge-group labels of every system in the batch.

    The target's own field wins; the EC fragment labels are the fallback, so a
    dataset prepared for the EC loss needs nothing added.

    :param target_name: Name of the RI-coefficient target.
    :param systems: The batch's systems, in batch order.
    :param extra: The batch's extra data.
    :return: One label array per system.
    :raises RuntimeError: If neither field is present.
    """
    for key in (charge_group_name(target_name), ec_fragment_name(target_name)):
        if key not in extra:
            continue
        return _split_per_atom_labels(extra[key][0], len(systems))

    raise RuntimeError(
        f"the per-group charge penalty on target '{target_name}' requires "
        f"per-atom group labels: the field '{charge_group_name(target_name)}', "
        f"or '{ec_fragment_name(target_name)}' if the groups are the EC "
        "fragments. Add one to the dataset like the 'charge' field and declare "
        "it in the options file's extra_data section."
    )


#: Placeholder machinery for structures with no usable patch: recognisably
#: wrong shapes (the loss checks the mask width against ``naux``) and zero
#: constants, so such a structure contributes exactly zero to the EC loss.
_EC_NO_PATCH = (
    torch.zeros((1, 1), dtype=torch.float64),
    torch.zeros((4, 1), dtype=torch.float64),
    torch.zeros((2, 2), dtype=torch.float64),
)


def _ec_split_tag(split: Any) -> str:
    """A short, stable cache tag for a fragment specification.

    :param split: Fragment specification, see :py:func:`ec_fragment_indices`.
    :return: A string that distinguishes it from any other specification.
    """
    if isinstance(split, int):
        return str(split)
    return "|".join(str(int(v)) for v in split)


def _batch_ec_machinery(
    systems: List[System],
    system_ids: Optional[List[int]],
    aux_basis: str,
    splits: List[Any],
    displacements: Optional[List[Optional["numpy.ndarray"]]] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """EC machinery for one batch, through the cache when ids are available.

    Without a displacement the machinery is built on first touch inside the
    dataloader worker and cached across epochs, exactly like the metric
    matrices: it depends only on the unaugmented geometry and the fragment
    split, both identical every epoch. Memory per structure is dominated by the
    ``(naux, naux)`` moment matrix, i.e. the same footprint as one metric
    matrix.

    With a displacement the machinery is **not** cached and not reused. A fresh
    partner placement is drawn for every structure at every step, so a cache
    entry would never be read a second time; caching would only evict the
    metric matrices, which are reused.

    :param systems: The batch's systems, in batch order.
    :param system_ids: Native dataset ids of those systems, or ``None``.
    :param aux_basis: Auxiliary basis name.
    :param splits: Fragment specification per system, see
        :py:func:`ec_fragment_indices`.
    :param displacements: Per-system partner shift in Angstrom, or ``None`` for
        an unaugmented batch.
    :return: One ``(moments, vectors, constants)`` triple per system.
    """
    if system_ids is None:
        system_ids = [None] * len(systems)  # type: ignore[list-item]
    if displacements is None:
        displacements = [None] * len(systems)
    # The cache holds single tensors, so the three parts live under three keys;
    # they are written together, so either all are present or none is.
    parts = ("moments", "vectors", "constants")
    cache = _metric_matrix_cache()
    machinery = []
    for system, system_id, split, shift in zip(
        systems, system_ids, splits, displacements, strict=True
    ):
        cacheable = system_id is not None and shift is None
        keys = [
            (aux_basis, f"ec-machinery-{part}|split={_ec_split_tag(split)}", system_id)
            for part in parts
        ]
        cached = [cache.get(key) for key in keys] if cacheable else [None]
        if all(tensor is not None for tensor in cached):
            machinery.append((cached[0], cached[1], cached[2]))
            continue
        entry = compute_ec_machinery(system, aux_basis, split, shift)
        if entry is None:
            entry = _EC_NO_PATCH
        if cacheable:
            for key, tensor in zip(keys, entry, strict=True):
                cache.put(key, tensor)
        machinery.append(entry)
    return machinery


# ── Collate transforms ────────────────────────────────────────────────────────


def _metric_matrices_transform(
    target_to_aux_basis: Mapping[str, str],
    metric: str,
    systems: List[System],
    targets: Dict[str, TensorMap],
    extra: Dict[str, TensorMap],
) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
    system_ids = batch_system_ids(extra)
    # The ESP term is not folded into the dense matrix (see
    # compute_esp_factor): the matrix entry — still stored under the full-spec
    # key the loss looks up — carries the base terms, and the factor travels
    # alongside it under esp_factor_name for the loss to apply itself.
    base_metric, esp_weight, esp_shell = strip_esp_from_spec(metric)
    group_charge_weight = parse_group_charge_weight(metric)
    interface_esp_weight = parse_interface_esp_weight(metric)
    packed_by_basis: Dict[str, TensorMap] = {}
    factors_by_basis: Dict[str, TensorMap] = {}
    for target_name, aux_basis in target_to_aux_basis.items():
        if aux_basis not in packed_by_basis:
            packed_by_basis[aux_basis] = pack_metric_matrices(
                _batch_metric_matrices(systems, system_ids, aux_basis, base_metric)
            )
            if esp_weight > 0.0:
                factors_by_basis[aux_basis] = pack_metric_matrices(
                    _batch_esp_factors(systems, system_ids, aux_basis, esp_shell)
                )
        extra[metric_matrix_name(target_name, metric)] = packed_by_basis[aux_basis]
        if esp_weight > 0.0:
            extra[esp_factor_name(target_name, metric)] = factors_by_basis[aux_basis]
        if group_charge_weight > 0.0:
            # Per target, not per basis: the grouping is a property of the
            # target's own field, so two targets on one basis may disagree.
            extra[group_charge_factor_name(target_name, metric)] = pack_metric_matrices(
                _batch_group_charge_factors(
                    systems,
                    system_ids,
                    aux_basis,
                    _batch_charge_groups(target_name, systems, extra),
                )
            )
        if interface_esp_weight > 0.0:
            # Also per target: the patches follow the target's fragment labels.
            extra[interface_esp_factor_name(target_name, metric)] = (
                pack_metric_matrices(
                    _batch_interface_esp_factors(
                        systems,
                        system_ids,
                        aux_basis,
                        _ec_batch_splits(target_name, systems, extra),
                    )
                )
            )
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


def _ec_batch_splits(
    target_name: str, systems: List[System], extra: Dict[str, TensorMap]
) -> List[Any]:
    """The fragment specification of every system in the batch.

    Per-atom labels win when present; otherwise the per-system leading-atom
    count is accepted, so datasets written before the general spelling existed
    keep working.

    :param target_name: Name of the RI-coefficient target.
    :param systems: The batch's systems, in batch order.
    :param extra: The batch's extra data.
    :return: One specification per system.
    :raises RuntimeError: If neither field is present.
    """
    atom_key = ec_fragment_name(target_name)
    if atom_key in extra:
        return _split_per_atom_labels(extra[atom_key][0], len(systems))

    split_key = ec_fragment_split_name(target_name)
    if split_key in extra:
        return [int(v) for v in extra[split_key][0].values.reshape(-1)]

    raise RuntimeError(
        f"the EC loss on target '{target_name}' requires a fragment "
        f"specification: either the per-atom field '{atom_key}' (0 for the "
        f"patch-carrying fragment, 1 for its partner) or the per-system field "
        f"'{split_key}' (the number of atoms in the first fragment, for "
        "structures whose fragments are contiguous). Add one to the dataset "
        "like the 'charge' field and declare it in the options file's "
        "extra_data section."
    )


def _ec_partner_shifts(
    jitter: float,
    system_ids: Optional[List[int]],
    systems: List[System],
    splits: List[Any],
) -> Optional[List[Optional["numpy.ndarray"]]]:
    """Draw one rigid partner shift per system, or ``None`` when disabled.

    A shift that drives the two fragments into each other is rejected and drawn
    again. This matters in dense structures: a protein pocket already holds the
    ligand at hydrogen-bond range, about 1.5-1.8 Angstrom, so a shift of a few
    tenths can push a pair of atoms to within a fraction of an Angstrom. The
    patch does not notice — its size stays within a few per cent of normal — so
    the term would be evaluated on overlapping nuclei with nothing reporting it.
    Measured on OMol25 pockets and interfaces, a 0.4 Angstrom shift does this to
    2-4% of draws, which is the fraction rejected here.

    The floor never demands more than the structure already achieves: a
    structure whose fragments are closer than
    :py:data:`EC_JITTER_MIN_CONTACT` to begin with is only required not to get
    worse. After :py:data:`EC_JITTER_ATTEMPTS` failures the true placement is
    used, which costs one augmented sample and never a wrong geometry.

    The draw is seeded from one fresh value consumed from the worker's own
    torch generator per batch, combined with the system id. Consuming the
    generator is what makes the shifts fresh at every step: the worker's
    *seed* is constant for the whole run whenever workers persist across
    epochs (the trainers set ``persistent_workers``) and always in the
    main process (``num_workers=0``), so anything derived from the seed
    alone would hand every system one fixed placement for all of training.
    A run remains reproducible from its global seed, because the generator
    itself is seeded from it.

    :param jitter: Standard deviation of the shift in Angstrom; 0 disables it.
    :param system_ids: Native dataset ids, or ``None`` when unavailable.
    :param systems: The batch's systems, in batch order.
    :param splits: Fragment specification per system, see
        :py:func:`ec_fragment_indices`.
    :return: One shift per system, or ``None``.
    """
    import numpy as np

    if jitter <= 0.0:
        return None
    base = int(torch.randint(0, 2**62, (1,)).item())
    ids = system_ids if system_ids is not None else list(range(len(systems)))

    shifts = []
    for system_id, system, split in zip(ids, systems, splits, strict=True):
        generator = np.random.default_rng((base, int(system_id)))
        positions = system.positions.detach().cpu().double().numpy()
        own, partner = ec_fragment_indices(split, len(positions))
        if len(own) == 0 or len(partner) == 0:
            shifts.append(generator.normal(0.0, jitter, size=3))
            continue
        here, there = positions[own], positions[partner]
        gaps = np.linalg.norm(here[:, None, :] - there[None, :, :], axis=2)
        floor = min(EC_JITTER_MIN_CONTACT, float(gaps.min()))

        chosen = np.zeros(3)
        for _ in range(EC_JITTER_ATTEMPTS):
            shift = generator.normal(0.0, jitter, size=3)
            moved = np.linalg.norm(
                here[:, None, :] - (there + shift)[None, :, :], axis=2
            ).min()
            if moved >= floor:
                chosen = shift
                break
        shifts.append(chosen)
    return shifts


def _ec_machinery_transform(
    target_to_aux_basis: Mapping[str, str],
    jitter: float,
    systems: List[System],
    targets: Dict[str, TensorMap],
    extra: Dict[str, TensorMap],
) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
    system_ids = batch_system_ids(extra)
    packed_by_basis: Dict[str, List[TensorMap]] = {}
    for target_name, aux_basis in target_to_aux_basis.items():
        splits = _ec_batch_splits(target_name, systems, extra)
        # The shift depends on the fragments, so it is drawn per target here;
        # every EC target of a batch shares one geometry, and the generator is
        # seeded the same way, so they all receive the same placement.
        shifts = _ec_partner_shifts(jitter, system_ids, systems, splits)
        share_key = f"{aux_basis}|{[_ec_split_tag(s) for s in splits]}"
        if share_key not in packed_by_basis:
            machinery = _batch_ec_machinery(
                systems, system_ids, aux_basis, splits, shifts
            )
            packed_by_basis[share_key] = [
                pack_metric_matrices([entry[i] for entry in machinery])
                for i in range(3)
            ]
        for part, packed in zip(
            ("moments", "vectors", "constants"),
            packed_by_basis[share_key],
            strict=True,
        ):
            extra[ec_machinery_name(target_name, part)] = packed
    return systems, targets, extra


def get_ec_machinery_transform(
    target_to_aux_basis: Mapping[str, str],
    jitter: float = 0.0,
) -> Callable:
    """
    Build a collate transform attaching per-target EC machinery.

    Like :py:func:`get_metric_matrices_transform`, **this must run before the
    augmenter**: the machinery is built on the unaugmented geometry, the frame
    the reference coefficients were fitted in, and the EC loss declares
    ``evaluate_in_original_frame`` accordingly. Entries are cached across
    epochs in the same worker-level byte-budgeted cache as the metric matrices.

    **Partner jitter.** With ``jitter > 0`` the partner fragment is rigidly
    displaced by a fresh random shift for every structure at every step, and the
    machinery is rebuilt rather than cached. This is augmentation of the
    *measurement*, not of the data: the displaced structure is never claimed to
    be a physical system, no new reference density is needed, and the prediction
    and the reference are read through the identical displaced functional, so
    their difference remains a true error of the model. Its purpose is to stop
    the EC term from teaching one fixed direction per structure — the direction
    is a fixed local operator contracted with the partner's field, so varying
    the field forces the model to learn the operator instead of the direction.
    The average over steps plays the role of an average over placements, at one
    placement's cost.

    :param target_to_aux_basis: Mapping from target name to auxiliary basis name.
    :param jitter: Standard deviation in Angstrom of the partner shift; 0
        disables the augmentation and restores caching.
    :return: A collate transform.
    """
    return functools.partial(
        _ec_machinery_transform, dict(target_to_aux_basis), float(jitter)
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
