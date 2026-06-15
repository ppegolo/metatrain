from __future__ import annotations

import copy
import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System


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
    """Return the ``extra_data`` key used for a target's overlap matrix."""
    return f"{target_name}_overlap_matrix"


def coulomb_matrix_name(target_name: str) -> str:
    """Return the ``extra_data`` key used for a target's Coulomb matrix."""
    return f"{target_name}_coulomb_matrix"


def metric_matrix_name(target_name: str, metric: str) -> str:
    """Return the ``extra_data`` key for a target's two-centre metric matrix.

    :param target_name: Name of the RI-coefficient target.
    :param metric: ``"overlap"`` (S) or ``"coulomb"`` (J).
    """
    if metric == "overlap":
        return overlap_matrix_name(target_name)
    elif metric == "coulomb":
        return coulomb_matrix_name(target_name)
    else:
        raise ValueError(
            f"Unknown RI metric '{metric}'. Supported values are 'overlap' and 'coulomb'."
        )


def ri_projections_name(target_name: str) -> str:
    """Return the ``extra_data`` key for a target's RI projections w = S c_RI."""
    return f"{target_name}_projections"


def ri_density_fit_constant_name(target_name: str) -> str:
    """Return the ``extra_data`` key for the pre-computed c_RI^T w_RI constant."""
    return f"{target_name}_density_fit_constant"


def resolve_ri_aux_basis(target_name: str, ri_aux_basis: RIAuxBasis) -> str:
    """Resolve the auxiliary basis configured for a given RI target."""

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


def compute_metric_matrix(system: System, aux_basis: str, metric: str) -> torch.Tensor:
    """
    Compute the two-centre metric matrix (overlap or Coulomb) for a system.

    :param system: Atomistic system with positions in Angstrom.
    :param aux_basis: PySCF auxiliary basis name.
    :param metric: ``"overlap"`` for S or ``"coulomb"`` for J.
    :return: Dense metric matrix in PySCF AO order, float64.
    """
    if metric == "overlap":
        return compute_overlap_matrix(system, aux_basis)
    elif metric == "coulomb":
        return compute_coulomb_matrix(system, aux_basis)
    else:
        raise ValueError(
            f"Unknown RI metric '{metric}'. Supported values are 'overlap' and 'coulomb'."
        )


# ── Ragged metric-matrix container ─────────────────────────────────────────────

@dataclass
class RaggedMetricMatrices:
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
        """Move/cast the flat buffer (mirrors ``Tensor.to``); sizes are metadata."""
        return RaggedMetricMatrices(
            self.values.to(dtype=dtype, device=device, non_blocking=non_blocking),
            self.sizes,
        )

    def matrices(self) -> list[torch.Tensor]:
        """Reconstruct the dense per-system matrices ``M_i`` (views into ``values``)."""
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
    """

    def transform(
        systems: list[System],
        targets: dict[str, TensorMap],
        extra: dict,
    ) -> tuple[list[System], dict[str, TensorMap], dict]:
        cache: dict[str, RaggedMetricMatrices] = {}
        for target_name, aux_basis in target_to_aux_basis.items():
            if aux_basis not in cache:
                cache[aux_basis] = compute_ragged_metric_matrices(
                    systems, aux_basis, metric, dtype
                )
            extra[name_fn(target_name)] = cache[aux_basis]
        return systems, targets, extra

    return transform


def get_overlap_matrices_transform(
    target_to_aux_basis: Mapping[str, str],
    dtype: torch.dtype = torch.float64,
    cache_across_epochs: bool = False,
) -> Callable:
    """Create a collate transform that attaches per-target overlap matrices (ragged)."""
    return _get_metric_matrices_transform_impl(
        target_to_aux_basis, "overlap", overlap_matrix_name, dtype, cache_across_epochs
    )


def get_coulomb_matrices_transform(
    target_to_aux_basis: Mapping[str, str],
    dtype: torch.dtype = torch.float64,
    cache_across_epochs: bool = False,
) -> Callable:
    """Create a collate transform that attaches per-target Coulomb matrices (ragged)."""
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
    """
    if metric == "overlap":
        return get_overlap_matrices_transform(
            target_to_aux_basis, dtype, cache_across_epochs
        )
    elif metric == "coulomb":
        return get_coulomb_matrices_transform(
            target_to_aux_basis, dtype, cache_across_epochs
        )
    else:
        raise ValueError(
            f"Unknown RI metric '{metric}'. Supported values are 'overlap' and 'coulomb'."
        )


def get_density_fit_constant_transform(
    target_to_projections_key: Mapping[str, str],
) -> Callable:
    """
    Create a collate transform that pre-computes the per-system density-fit constant.

    For each target, computes ``c_RI^T w_RI`` (= ``c_RI^T S c_RI`` when
    ``w_RI = S c_RI``) and stores the result in ``extra_data`` as a scalar
    TensorMap.  This constant bounds the density-fit loss from below at zero and
    has no gradient w.r.t. model parameters.

    **This transform must run before the CM-removal and scale-removal transforms**
    so that ``targets`` still contains raw RI coefficients in physical units.

    :param target_to_projections_key: mapping from RI-coefficient target name to the
        ``extra_data`` key under which the corresponding projections ``w = M c_RI``
        are stored.
    """

    def transform(
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
            sys_idx_col = first_block.samples.values[:, 0]
            n_systems = int(sys_idx_col.max().item()) + 1
            device = first_block.values.device
            dtype = first_block.values.dtype

            constants = torch.zeros(n_systems, device=device, dtype=dtype)

            for key in c_map.keys:
                c_block = c_map.block(key)
                w_block = w_map.block(key)
                sys_idx = c_block.samples.values[:, 0].long()

                c_vals = c_block.values  # [n_samples, n_m, n_radial]
                w_vals = w_block.values

                # NaN entries are padding; zero them out before summing.
                mask = ~(torch.isnan(c_vals) | torch.isnan(w_vals))
                contrib = torch.where(mask, c_vals * w_vals, torch.zeros_like(c_vals))
                per_sample = contrib.sum(dim=[1, 2])  # [n_samples]
                constants.scatter_add_(0, sys_idx, per_sample)

            # Pack as a scalar TensorMap: one value per system.
            const_block = TensorBlock(
                values=constants.unsqueeze(-1),  # [n_systems, 1]
                samples=Labels(
                    names=["system"],
                    values=torch.arange(
                        n_systems, dtype=torch.int32, device=device
                    ).reshape(-1, 1),
                ),
                components=[],
                properties=Labels(
                    names=["_"],
                    values=torch.zeros((1, 1), dtype=torch.int32, device=device),
                ),
            )
            extra[ri_density_fit_constant_name(target_name)] = TensorMap(
                Labels.single().to(device=device), [const_block]
            )

        return systems, targets, extra

    return transform
