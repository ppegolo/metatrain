from __future__ import annotations

import copy
import functools
import importlib
import os
import re
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

from ..data.dataset import RawExtraPayload


if TYPE_CHECKING:
    from types import ModuleType

    from pyscf.gto import Mole


RIAuxBasis = str | dict[str, str]


# ── PySCF imports ──────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _import_pyscf_modules() -> tuple[ModuleType, ModuleType]:
    try:
        gto = importlib.import_module("pyscf.gto")
        elements = importlib.import_module("pyscf.data.elements")
    except ModuleNotFoundError as err:
        raise ImportError(
            "RI overlap losses require `pyscf` to compute auxiliary overlap matrices."
        ) from err

    return gto, elements


def _build_etb_basis_via_aug_etb(
    ao_basis: str, atomic_numbers: tuple[int, ...], beta: float
) -> dict[str, object]:
    """Build an ETB auxiliary basis using ``pyscf.df.aug_etb``.

    Constructs a dummy molecule with one atom per unique element using the
    **orbital** basis ``ao_basis``, then calls ``pyscf.df.aug_etb(mol, beta)``
    — matching exactly how SCFBench and similar datasets are generated.

    :param ao_basis: Orbital (not auxiliary) basis name (e.g. ``"def2-svp"``).
    :param atomic_numbers: Tuple of unique atomic numbers present in the system.
    :param beta: Even-tempering ratio β.
    :return: Dictionary mapping element symbols to basis specifications,
        suitable for ``mol.basis``.
    """
    gto, elements = _import_pyscf_modules()
    df = importlib.import_module("pyscf.df")

    symbols = [elements.ELEMENTS[n] for n in atomic_numbers]
    # Place each element far apart so the dummy molecule builds without issues.
    atom_str = "\n".join(f"{sym} 0.0 0.0 {i * 10.0}" for i, sym in enumerate(symbols))

    mol = gto.Mole()
    mol.atom = atom_str
    mol.basis = ao_basis
    mol.unit = "Angstrom"
    mol.verbose = 0
    mol.spin = None
    mol.cart = False
    mol.build()

    return df.aug_etb(mol, beta=beta)


@lru_cache(maxsize=None)
def _load_auxiliary_basis(
    aux_basis: str, atomic_numbers: tuple[int, ...]
) -> dict[str, object]:
    """Load and cache parsed auxiliary basis data for the requested elements.

    Supports two formats for ``aux_basis``:

    - A plain PySCF basis name (e.g. ``"def2-universal-jfit"``), loaded via
      ``gto.basis.load`` for each element.
    - An even-tempered basis specification ``"etb:<ao_basis>:<beta>"``
      (e.g. ``"etb:def2-svp:2.0"``), which calls ``pyscf.df.aug_etb`` on a
      molecule built with the **orbital** basis ``<ao_basis>`` and ratio
      ``<beta>`` — the same algorithm used by SCFBench to generate RI datasets.

    :param aux_basis: Auxiliary basis name or ETB specification.
    :param atomic_numbers: Tuple of unique atomic numbers to load the basis for.
    :return: Dictionary mapping element symbols to basis specifications.
    """
    gto, elements = _import_pyscf_modules()

    etb_parts = aux_basis.split(":")
    if len(etb_parts) == 3 and etb_parts[0].lower() == "etb":
        # aug_etb returns a complete per-element dict; return it directly.
        return _build_etb_basis_via_aug_etb(
            etb_parts[1], atomic_numbers, float(etb_parts[2])
        )

    basis: dict[str, object] = {}
    for atomic_number in atomic_numbers:
        symbol = elements.ELEMENTS[atomic_number]
        basis[symbol] = gto.basis.load(aux_basis, symbol)

    return basis


# ── extra_data key helpers ─────────────────────────────────────────────────────


def overlap_matrix_name(target_name: str) -> str:
    """Return the ``extra_data`` key used for a target's overlap matrix.

    :param target_name: Name of the RI-coefficient target.
    :return: The ``extra_data`` key.
    """
    return f"{target_name}_overlap_matrix"


def coulomb_matrix_name(target_name: str) -> str:
    """Return the ``extra_data`` key used for a target's Coulomb matrix.

    :param target_name: Name of the RI-coefficient target.
    :return: The ``extra_data`` key.
    """
    return f"{target_name}_coulomb_matrix"


#: Default weight on the plain Coulomb term when the long-range kernel is on.
#: The long-range metric alone is numerically rank-deficient (condition number
#: ~1e32 vs ~1e5 for J, because erf(wr)/r damps the compact directions away), so
#: it needs a positive-definite floor.  0.01 keeps ~95% of the long-range
#: reweighting while bringing the conditioning back to ~1e7.
DEFAULT_LR_EPS = 0.01


def make_metric_spec(
    metric: str,
    omega: float = 0.0,
    eps: float | None = None,
    charge_weight: float = 0.0,
) -> str:
    """Build the canonical metric-spec string shared by the loss and the transform.

    The spec fully determines the matrix, so it doubles as the ``extra_data`` key
    and the cache key.  With no optional term enabled it collapses to the bare
    ``metric`` name, keeping existing configs and cache keys byte-identical.

    :param metric: Base two-centre metric, ``"overlap"`` (S) or ``"coulomb"`` (J).
    :param omega: Range-separation parameter of the long-range Coulomb kernel
        ``erf(omega r)/r``.  ``0`` disables it.  Only valid with
        ``metric="coulomb"``.
    :param eps: Weight on the plain Coulomb term added to the long-range one,
        i.e. ``M = eps*J + J_lr``.  ``None`` uses :py:data:`DEFAULT_LR_EPS`.
        Ignored when ``omega == 0``.
    :param charge_weight: Weight on the rank-1 electron-count penalty
        ``w * S_vec S_vec^T``, which adds ``w * (S_vec . dc)**2`` to the loss.
        ``0`` disables it.
    :return: Canonical spec string.
    """
    if metric not in ("overlap", "coulomb"):
        raise ValueError(
            f"Unknown RI metric '{metric}'. "
            "Supported values are 'overlap' and 'coulomb'."
        )
    if omega < 0.0:
        raise ValueError(f"omega must be >= 0, got {omega}.")
    if charge_weight < 0.0:
        raise ValueError(f"charge_weight must be >= 0, got {charge_weight}.")
    if omega > 0.0 and metric != "coulomb":
        raise ValueError(
            "The long-range kernel erf(omega r)/r is a Coulomb-metric option; "
            f"got metric='{metric}' with omega={omega}."
        )
    if omega == 0.0 and charge_weight == 0.0:
        return metric
    resolved_eps = DEFAULT_LR_EPS if eps is None else float(eps)
    if resolved_eps < 0.0:
        raise ValueError(f"eps must be >= 0, got {resolved_eps}.")
    return (
        f"{metric}|omega={float(omega):.10g}"
        f"|eps={resolved_eps:.10g}|q={float(charge_weight):.10g}"
    )


def parse_metric_spec(spec: str) -> tuple[str, float, float, float]:
    """Invert :py:func:`make_metric_spec`.

    :param spec: Canonical spec string.
    :return: ``(metric, omega, eps, charge_weight)``.
    """
    if "|" not in spec:
        if spec not in ("overlap", "coulomb"):
            raise ValueError(
                f"Unknown RI metric '{spec}'. "
                "Supported values are 'overlap' and 'coulomb'."
            )
        return spec, 0.0, 0.0, 0.0
    metric, *parts = spec.split("|")
    values = {}
    for part in parts:
        key, _, value = part.partition("=")
        values[key] = float(value)
    return metric, values["omega"], values["eps"], values["q"]


def metric_matrix_name(target_name: str, metric: str) -> str:
    """Return the ``extra_data`` key for a target's two-centre metric matrix.

    :param target_name: Name of the RI-coefficient target.
    :param metric: Metric spec, as built by :py:func:`make_metric_spec`.
    :return: The ``extra_data`` key.
    """
    if metric == "overlap":
        return overlap_matrix_name(target_name)
    elif metric == "coulomb":
        return coulomb_matrix_name(target_name)
    # Parse first so an unknown metric still raises the familiar error.
    parse_metric_spec(metric)
    suffix = re.sub(r"[^0-9a-zA-Z]+", "_", metric)
    return f"{target_name}_metric_{suffix}"


def ri_projections_name(target_name: str) -> str:
    """Return the ``extra_data`` key for a target's RI projections w = S c_RI.

    :param target_name: Name of the RI-coefficient target.
    :return: The ``extra_data`` key.
    """
    return f"{target_name}_projections"


def ri_density_fit_constant_name(target_name: str) -> str:
    """Return the ``extra_data`` key for the pre-computed c_RI^T w_RI constant.

    :param target_name: Name of the RI-coefficient target.
    :return: The ``extra_data`` key.
    """
    return f"{target_name}_density_fit_constant"


def resolve_ri_aux_basis(target_name: str, ri_aux_basis: RIAuxBasis) -> str:
    """Resolve the auxiliary basis configured for a given RI target.

    :param target_name: Name of the RI-coefficient target.
    :param ri_aux_basis: Global basis name, or a per-target mapping.
    :return: The auxiliary basis name for ``target_name``.
    """

    if isinstance(ri_aux_basis, str):
        return ri_aux_basis

    if target_name in ri_aux_basis:
        return ri_aux_basis[target_name]

    available_targets = ", ".join(sorted(ri_aux_basis))
    raise ValueError(
        f"No RI auxiliary basis configured for target '{target_name}'. "
        f"Available targets: {available_targets}."
    )


# ── Molecule / integral construction ──────────────────────────────────────────


def _system_to_atom_string(system: System) -> str:
    _, elements = _import_pyscf_modules()

    atomic_numbers = system.types.detach().cpu().tolist()
    positions = system.positions.detach().cpu().tolist()

    atoms = []
    for atomic_number, (x, y, z) in zip(atomic_numbers, positions, strict=True):
        symbol = elements.ELEMENTS[int(atomic_number)]
        atoms.append(f"{symbol}  {x:.12f}  {y:.12f}  {z:.12f}")

    return "\n".join(atoms)


def build_auxiliary_molecule(system: System, aux_basis: str) -> Mole:
    """
    Build a PySCF molecule carrying the auxiliary basis used for RI coefficients.

    The molecule is only used for integral evaluation.

    :param system: Atomistic system with positions in Angstrom.
    :param aux_basis: PySCF auxiliary basis name.
    :return: Built PySCF molecule in spherical-harmonic form.
    """
    gto, _ = _import_pyscf_modules()
    atomic_numbers = tuple(
        sorted({int(n) for n in system.types.detach().cpu().tolist()})
    )

    mol = gto.Mole()
    mol.atom = _system_to_atom_string(system)
    mol.basis = copy.deepcopy(_load_auxiliary_basis(aux_basis, atomic_numbers))
    mol.unit = "Angstrom"
    mol.verbose = 0
    mol.spin = None
    mol.cart = False
    mol.build()
    return mol


def compute_overlap_matrix(system: System, aux_basis: str) -> torch.Tensor:
    """
    Compute the two-centre overlap matrix for one system and auxiliary basis.

    :param system: Atomistic system with positions in Angstrom.
    :param aux_basis: PySCF auxiliary basis name.
    :return: Dense overlap matrix ``S`` in PySCF AO order, float64.
    """
    auxmol = build_auxiliary_molecule(system, aux_basis)
    return torch.from_numpy(auxmol.intor("int1e_ovlp")).to(torch.float64)


def compute_coulomb_matrix(system: System, aux_basis: str) -> torch.Tensor:
    """
    Compute the two-centre Coulomb (ERI) matrix for one system and auxiliary basis.

    :param system: Atomistic system with positions in Angstrom.
    :param aux_basis: PySCF auxiliary basis name.
    :return: Dense Coulomb matrix ``J`` in PySCF AO order, float64.
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

    :param system: Atomistic system with positions in Angstrom.
    :param aux_basis: PySCF auxiliary basis name.
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
    number of electrons carried by a coefficient vector: ``N = S_vec . c``.  The
    analytic form is used rather than a numerical integral: for a contracted s
    shell with primitive exponents ``a_k`` and (PySCF-normalised) contraction
    coefficients ``d_k``, ``\\int chi dr = sum_k d_k (pi/a_k)^{3/2} /
    sqrt(4 pi)``, the ``1/sqrt(4 pi)`` coming from the ``Y_00`` factor carried by
    PySCF's spherical basis functions.

    :param system: Atomistic system with positions in Angstrom.
    :param aux_basis: PySCF auxiliary basis name.
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


def compute_metric_matrix(system: System, aux_basis: str, metric: str) -> torch.Tensor:
    """
    Compute the two-centre metric matrix for a system.

    Beyond the plain ``S`` and ``J`` metrics this assembles the optional terms
    encoded in the spec (see :py:func:`make_metric_spec`)::

        M = eps * J + J_lr(omega)          if omega > 0
        M = M + charge_weight * S_vec S_vec^T

    The second term is rank-1 and contributes ``charge_weight * (S_vec . dc)**2``
    to the loss, i.e. a penalty on the predicted density's electron-count error
    relative to the RI reference.

    :param system: Atomistic system with positions in Angstrom.
    :param aux_basis: PySCF auxiliary basis name.
    :param metric: Metric spec, as built by :py:func:`make_metric_spec`.
    :return: Dense metric matrix in PySCF AO order, float64.
    """
    base, omega, eps, charge_weight = parse_metric_spec(metric)

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

    return matrix


# ── Ragged metric-matrix container ─────────────────────────────────────────────


@dataclass
class RaggedMetricMatrices(RawExtraPayload):
    """
    Per-system two-centre metric matrices stored ragged, with NO padding.

    The matrices are concatenated flat (``values`` = ``cat([M_i.reshape(-1)])``, length
    ``Σ N_i²``) plus their sizes ``N_i``.  This is the channel used to carry metric
    matrices through the dataloader: a single flat tensor transfers across the worker
    boundary via shared memory (one fd, not one-per-system), and — because it is a raw
    tensor rather than a :py:class:`TensorMap` — it bypasses ``save_buffer`` (float64
    only) and any batch-max padding.

    :param values: 1D tensor, concatenation of each row-major ``M_i``.
    :param sizes: basis size ``N_i`` of each system's matrix.
    """

    values: torch.Tensor
    sizes: list[int]

    def to(
        self,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        non_blocking: bool = False,
    ) -> "RaggedMetricMatrices":
        """Move/cast the flat buffer (mirrors ``Tensor.to``); sizes are metadata.

        :param dtype: Target dtype (unchanged if ``None``).
        :param device: Target device (unchanged if ``None``).
        :param non_blocking: Forwarded to ``Tensor.to``.
        :return: New :class:`RaggedMetricMatrices` with the moved buffer.
        """
        return RaggedMetricMatrices(
            self.values.to(dtype=dtype, device=device, non_blocking=non_blocking),
            self.sizes,
        )

    def matrices(self) -> list[torch.Tensor]:
        """Reconstruct the dense per-system matrices ``M_i`` (views into ``values``).

        :return: One ``(N_i, N_i)`` matrix per system.
        """
        out: list[torch.Tensor] = []
        offset = 0
        for n in self.sizes:
            out.append(self.values[offset : offset + n * n].view(n, n))
            offset += n * n
        return out

    @classmethod
    def from_matrices(cls, matrices: list[torch.Tensor]) -> "RaggedMetricMatrices":
        """Pack dense per-system matrices into the ragged layout.

        :param matrices: One square ``(N_i, N_i)`` matrix per system.
        :return: The packed ragged container.
        """
        sizes = [int(m.shape[0]) for m in matrices]
        if matrices:
            flat = torch.cat([m.reshape(-1) for m in matrices])
        else:
            flat = torch.zeros(0)
        return cls(flat, sizes)


@dataclass
class BatchRotations(RawExtraPayload):
    """Per-system rotational-augmentation info for one batch.

    Stashed into ``extra_data`` by the rotational augmenter (under
    ``"mtt::aux::rotations"``) and consumed by the density losses to un-rotate
    coefficient residuals, so that metric matrices cached on the *unrotated*
    geometries stay valid: ``Δcᵀ (D M Dᵀ) Δc == (Dᵀ Δc)ᵀ M (Dᵀ Δc)``.

    Carried raw through the dataloader worker boundary, like
    :class:`RaggedMetricMatrices`.

    :param wigner: Per-``o3_lambda`` stacked real Wigner matrices, one
        ``(n_systems, 2l+1, 2l+1)`` tensor per ``l``, in batch order.
    :param inverted: Boolean tensor ``(n_systems,)``; whether each system's
        augmentation includes an inversion.
    """

    wigner: dict[int, torch.Tensor]
    inverted: torch.Tensor

    def to(
        self,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        non_blocking: bool = False,
    ) -> "BatchRotations":
        """Move/cast the Wigner blocks (mirrors ``Tensor.to``).

        :param dtype: Target dtype (unchanged if ``None``).
        :param device: Target device (unchanged if ``None``).
        :param non_blocking: Forwarded to ``Tensor.to``.
        :return: New :class:`BatchRotations` with moved tensors.
        """
        return BatchRotations(
            {
                ell: w.to(dtype=dtype, device=device, non_blocking=non_blocking)
                for ell, w in self.wigner.items()
            },
            self.inverted.to(device=device, non_blocking=non_blocking),
        )


BATCH_ROTATIONS_NAME = "mtt::aux::rotations"


def compute_ragged_metric_matrices(
    systems: list[System],
    aux_basis: str,
    metric: str,
    dtype: torch.dtype = torch.float64,
) -> RaggedMetricMatrices:
    """
    Compute per-system metric matrices for a batch and pack them ragged (no padding).

    :param systems: Systems in one batch.
    :param aux_basis: PySCF auxiliary basis name.
    :param metric: ``"overlap"`` or ``"coulomb"``.
    :param dtype: dtype of the stored matrices. Pass the model dtype (e.g. ``float32``)
        to halve transport + memory; casting float64->float32 here is bit-identical to
        the recast that ``batch_to`` applies today.
    :return: Ragged metric matrices for the batch.
    """
    matrices = [
        compute_metric_matrix(system, aux_basis, metric).to(dtype) for system in systems
    ]
    if not matrices:
        return RaggedMetricMatrices(torch.zeros(0, dtype=dtype), [])
    return RaggedMetricMatrices.from_matrices(matrices)


# ── Collate transforms ────────────────────────────────────────────────────────


def _batch_system_ids(extra: dict) -> list[int] | None:
    """Native per-system ids of a batch, in batch order, if available.

    :param extra: The batch's extra-data dictionary.
    :return: One id per system, or ``None`` when the batch carries none.
    """
    index_map = extra.get("mtt::aux::system_index")
    if index_map is None:
        return None
    return [int(v) for v in index_map[0].values[:, 0]]


DEFAULT_METRIC_CACHE_MAX_BYTES = 2 * 1024**3


def _metric_cache_max_bytes() -> int:
    """Per-worker byte budget for the metric-matrix cache.

    Overridable via ``METATRAIN_METRIC_CACHE_MAX_BYTES``. The budget applies
    per dataloader worker process: the total host-memory footprint is
    ``budget × num_workers × ranks_per_node``.

    :return: The byte budget.
    """
    return int(
        os.environ.get(
            "METATRAIN_METRIC_CACHE_MAX_BYTES", DEFAULT_METRIC_CACHE_MAX_BYTES
        )
    )


class _ByteBudgetCache:
    """LRU tensor cache bounded by total byte size.

    Unbounded caching is not an option here: on large datasets (e.g. >1M
    systems) the per-system metric matrices exceed host memory long before an
    epoch completes. Entries larger than the whole budget are not cached.

    :param max_bytes: Total byte budget for cached tensors.
    """

    def __init__(self, max_bytes: int) -> None:
        self._data: OrderedDict[tuple, torch.Tensor] = OrderedDict()
        self._bytes = 0
        self._max_bytes = max_bytes

    def get(self, key: tuple) -> torch.Tensor | None:
        tensor = self._data.get(key)
        if tensor is not None:
            self._data.move_to_end(key)
        return tensor

    def put(self, key: tuple, tensor: torch.Tensor) -> None:
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes > self._max_bytes:
            return
        existing = self._data.pop(key, None)
        if existing is not None:
            self._bytes -= existing.numel() * existing.element_size()
        self._data[key] = tensor
        self._bytes += nbytes
        while self._bytes > self._max_bytes:
            _, evicted = self._data.popitem(last=False)
            self._bytes -= evicted.numel() * evicted.element_size()


def _metric_matrices_transform_impl(
    target_to_aux_basis: Mapping[str, str],
    metric: str,
    name_fn: Callable[[str], str],
    dtype: torch.dtype,
    persistent_cache: _ByteBudgetCache | None,
    systems: list[System],
    targets: dict[str, TensorMap],
    extra: dict,
) -> tuple[list[System], dict[str, TensorMap], dict]:
    # ``persistent_cache`` (keyed by (aux_basis, native system id)) lives inside
    # the functools.partial, so with persistent dataloader workers it survives
    # across epochs. It must only be enabled for pipelines whose systems are
    # identical every epoch (validation: no augmentation) — the trainer passes
    # None for the training transforms, whose systems are rotated per epoch.
    # Memory: bounded per worker by _metric_cache_max_bytes(); on datasets too
    # large to fit the budget the LRU degrades to per-epoch recomputation.
    system_ids = _batch_system_ids(extra) if persistent_cache is not None else None

    per_batch: dict[str, RaggedMetricMatrices] = {}
    for target_name, aux_basis in target_to_aux_basis.items():
        if aux_basis not in per_batch:
            if persistent_cache is not None and system_ids is not None:
                matrices = []
                for system, system_id in zip(systems, system_ids, strict=True):
                    # The spec must be in the key: the matrix depends on omega /
                    # eps / charge_weight, so without it a hyperparameter change
                    # would silently reuse matrices built for the old one.
                    cache_key = (aux_basis, metric, system_id)
                    matrix = persistent_cache.get(cache_key)
                    if matrix is None:
                        matrix = compute_metric_matrix(system, aux_basis, metric).to(
                            dtype
                        )
                        persistent_cache.put(cache_key, matrix)
                    matrices.append(matrix)
                per_batch[aux_basis] = (
                    RaggedMetricMatrices.from_matrices(matrices)
                    if matrices
                    else RaggedMetricMatrices(torch.zeros(0, dtype=dtype), [])
                )
            else:
                per_batch[aux_basis] = compute_ragged_metric_matrices(
                    systems, aux_basis, metric, dtype
                )
        extra[name_fn(target_name)] = per_batch[aux_basis]
    return systems, targets, extra


def _get_metric_matrices_transform_impl(
    target_to_aux_basis: Mapping[str, str],
    metric: str,
    name_fn: Callable[[str], str],
    dtype: torch.dtype,
    cache_across_epochs: bool,
) -> Callable:
    """Collate transform attaching per-target RAGGED metric matrices.

    The matrices are stored as :py:class:`RaggedMetricMatrices` (not a TensorMap), so
    they bypass ``save_buffer`` (float64-only) and padding: the CollateFn carries them
    raw through the worker boundary, and the density loss consumes them per system.

    :param target_to_aux_basis: Mapping from target name to auxiliary basis name.
    :param metric: ``"overlap"`` or ``"coulomb"``.
    :param name_fn: Maps a target name to its ``extra_data`` key.
    :param dtype: dtype of the stored matrices.
    :param cache_across_epochs: Cache per-system matrices across epochs (only
        valid when the pipeline's systems are identical every epoch).
    :return: Collate transform.
    """
    return functools.partial(
        _metric_matrices_transform_impl,
        target_to_aux_basis,
        metric,
        name_fn,
        dtype,
        _ByteBudgetCache(_metric_cache_max_bytes()) if cache_across_epochs else None,
    )


def get_overlap_matrices_transform(
    target_to_aux_basis: Mapping[str, str],
    dtype: torch.dtype = torch.float64,
    cache_across_epochs: bool = False,
) -> Callable:
    """Create a collate transform that attaches per-target overlap matrices (ragged).

    :param target_to_aux_basis: Mapping from target name to auxiliary basis name.
    :param dtype: dtype of the stored matrices.
    :param cache_across_epochs: Cache per-system matrices across epochs (only
        valid when the pipeline's systems are identical every epoch).
    :return: Collate transform.
    """
    return _get_metric_matrices_transform_impl(
        target_to_aux_basis, "overlap", overlap_matrix_name, dtype, cache_across_epochs
    )


def get_coulomb_matrices_transform(
    target_to_aux_basis: Mapping[str, str],
    dtype: torch.dtype = torch.float64,
    cache_across_epochs: bool = False,
) -> Callable:
    """Create a collate transform that attaches per-target Coulomb matrices (ragged).

    :param target_to_aux_basis: Mapping from target name to auxiliary basis name.
    :param dtype: dtype of the stored matrices.
    :param cache_across_epochs: Cache per-system matrices across epochs (only
        valid when the pipeline's systems are identical every epoch).
    :return: Collate transform.
    """
    return _get_metric_matrices_transform_impl(
        target_to_aux_basis, "coulomb", coulomb_matrix_name, dtype, cache_across_epochs
    )


def get_metric_matrices_transform(
    target_to_aux_basis: Mapping[str, str],
    metric: str,
    dtype: torch.dtype = torch.float64,
    cache_across_epochs: bool = False,
) -> Callable:
    """Create a collate transform that attaches per-target metric matrices (ragged).

    :param target_to_aux_basis: Mapping from target name to PySCF auxiliary basis name.
    :param metric: ``"overlap"`` or ``"coulomb"``.
    :param dtype: dtype of the stored matrices (pass the model dtype to save memory).
    :param cache_across_epochs: Cache per-system matrices across epochs (only
        valid when the pipeline's systems are identical every epoch, e.g. an
        augmentation-free validation set).
    :return: Collate transform.
    """
    if metric == "overlap":
        return get_overlap_matrices_transform(
            target_to_aux_basis, dtype, cache_across_epochs
        )
    elif metric == "coulomb":
        return get_coulomb_matrices_transform(
            target_to_aux_basis, dtype, cache_across_epochs
        )
    # Any spec carrying optional terms; parse_metric_spec raises on a bad one.
    parse_metric_spec(metric)
    return _get_metric_matrices_transform_impl(
        target_to_aux_basis,
        metric,
        # partial, not a lambda: the transform is carried into dataloader
        # workers and must stay picklable.
        functools.partial(metric_matrix_name, metric=metric),
        dtype,
        cache_across_epochs,
    )


def _density_fit_constant_transform_impl(
    target_to_projections_key: Mapping[str, str],
    persistent_cache: dict,
    systems: list[System],
    targets: dict[str, TensorMap],
    extra: dict[str, TensorMap],
) -> tuple[list[System], dict[str, TensorMap], dict[str, TensorMap]]:
    for target_name, proj_key in target_to_projections_key.items():
        if target_name not in targets or proj_key not in extra:
            continue

        c_map = targets[target_name]
        w_map = extra[proj_key]

        first_block = c_map.block(c_map.keys[0])
        device = first_block.values.device
        dtype = first_block.values.dtype

        # One constant per system actually present in the batch, keyed by the
        # native "system" sample id (which is not necessarily 0..B-1: e.g. for
        # a shuffled DiskDataset these are arbitrary storage entry numbers).
        # torch.unique returns the ids sorted, which the loss relies on for
        # its searchsorted lookup.
        system_ids = torch.unique(
            torch.cat([c_map.block(key).samples.values[:, 0] for key in c_map.keys])
        )

        # c_RI^T w is invariant under the rotational augmentation (c and w
        # rotate covariantly), so it can be cached per (target, system id)
        # across epochs — with persistent dataloader workers this makes the
        # transform a dict lookup from the second epoch on.
        id_list = [int(i) for i in system_ids]
        cached = [persistent_cache.get((target_name, i)) for i in id_list]
        if all(value is not None for value in cached):
            constants = torch.tensor(cached, device=device, dtype=dtype)
            extra[ri_density_fit_constant_name(target_name)] = _pack_constants(
                constants, system_ids, device
            )
            continue

        constants = torch.zeros(len(system_ids), device=device, dtype=dtype)

        for key in c_map.keys:
            c_block = c_map.block(key)
            w_block = w_map.block(key)
            positions = torch.searchsorted(
                system_ids, c_block.samples.values[:, 0].contiguous()
            ).long()

            c_vals = c_block.values  # [n_samples, n_m, n_radial]
            w_vals = w_block.values

            # NaN entries are padding; zero them out before summing.
            mask = ~(torch.isnan(c_vals) | torch.isnan(w_vals))
            contrib = torch.where(mask, c_vals * w_vals, torch.zeros_like(c_vals))
            per_sample = contrib.sum(dim=[1, 2])  # [n_samples]
            constants.scatter_add_(0, positions, per_sample)

        for i, system_id in enumerate(id_list):
            persistent_cache[(target_name, system_id)] = float(constants[i])

        extra[ri_density_fit_constant_name(target_name)] = _pack_constants(
            constants, system_ids, device
        )

    return systems, targets, extra


def _pack_constants(
    constants: torch.Tensor, system_ids: torch.Tensor, device: torch.device
) -> TensorMap:
    """Pack per-system constants as a scalar TensorMap labelled by native id.

    :param constants: One value per system, aligned with ``system_ids``.
    :param system_ids: Sorted native system ids.
    :param device: Device for the labels.
    :return: Scalar TensorMap with one sample per system.
    """
    const_block = TensorBlock(
        values=constants.unsqueeze(-1),  # [n_present_systems, 1]
        samples=Labels(
            names=["system"],
            values=system_ids.to(dtype=torch.int32, device=device).reshape(-1, 1),
        ),
        components=[],
        properties=Labels(
            names=["_"],
            values=torch.zeros((1, 1), dtype=torch.int32, device=device),
        ),
    )
    return TensorMap(Labels.single().to(device=device), [const_block])


def get_density_fit_constant_transform(
    target_to_projections_key: Mapping[str, str],
) -> Callable:
    """
    Create a collate transform that pre-computes the per-system density-fit constant.

    For each target, computes ``c_RI^T w_RI`` and stores the result in ``extra_data``
    as a scalar TensorMap. Must run before CM-removal and scale-removal transforms.

    :param target_to_projections_key: mapping from RI-coefficient target name to the
        ``extra_data`` key under which the corresponding projections ``w = M c_RI``
        are stored.
    :return: Collate transform.
    """
    return functools.partial(
        _density_fit_constant_transform_impl, target_to_projections_key, {}
    )
