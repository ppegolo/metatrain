"""Precomputed per-atom "oracle" charges as a conditioning channel.

A cheap external charge model — GFN2-xTB partial charges, computed with
`tblite <https://tblite.readthedocs.io>`_ — is evaluated once per structure
and attached to each :class:`System` as per-atom data. Models read it as a
conditioning input (see
:py:class:`metatrain.pet.modules.conditioning.AtomicChargeEmbedding`).

Why: on charged complexes, the placement of the net charge among fragments
is a global, discrete decision that local models underfit when charged
systems are a minority of the training data. GFN2 charges localise fragment
charges to ~0.01 e — near the RI reference's own floor — so they carry
exactly that bookkeeping. They are a deterministic function of the geometry,
total charge and spin, i.e. of the model's own inputs, so the same
computation can (and must) run identically at training and at inference.

Failure tolerance: a structure where the oracle fails (SCF non-convergence,
unsupported element) simply gets no ``oracle_charges`` entry; the model's
mask channel then reads "no oracle" and prediction proceeds oracle-free.
Consistent with the conditioning-dropout training mode.

``tblite`` is an optional dependency, imported lazily; GFN2 scales
superlinearly with system size, so this oracle is intended for
small-to-medium structures (sub-second below ~100 atoms). Larger systems
should run oracle-free rather than let conditioning become the bottleneck.
"""

from __future__ import annotations

import functools
import logging
from typing import Callable, Dict, List, Optional, Tuple

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

from .data.byte_budget_cache import (
    ByteBudgetCache,
    batch_system_ids,
    collate_cache_max_bytes,
)


#: Bohr radius in Angstrom; tblite takes positions in Bohr.
_BOHR = 0.52917721092

#: The System data key the oracle charges are attached under.
ORACLE_CHARGES_KEY = "mtt::oracle_charges"


def compute_gfn2_charges(
    system: System, charge: int, spin_multiplicity: int
) -> Optional[torch.Tensor]:
    """GFN2-xTB partial charges for one system.

    :param system: System whose positions are interpreted as Angstrom.
    :param charge: Total charge of the system.
    :param spin_multiplicity: Spin multiplicity (2S+1).
    :return: Per-atom charges, shape ``(n_atoms,)``, float64 — or ``None``
        when the calculation fails (the caller then attaches nothing and the
        model runs oracle-free for this structure).
    """
    try:
        from tblite.interface import Calculator
    except ImportError as err:
        raise ImportError(
            "oracle-charge conditioning requires `tblite` for GFN2-xTB "
            "charges; install it with `pip install tblite`."
        ) from err

    try:
        calculator = Calculator(
            "GFN2-xTB",
            system.types.detach().cpu().numpy(),
            system.positions.detach().cpu().double().numpy() / _BOHR,
            charge=float(charge),
            uhf=int(spin_multiplicity) - 1,
        )
        calculator.set("verbosity", 0)
        result = calculator.singlepoint()
        charges = result.get("charges")
    except Exception as err:  # tblite raises plain RuntimeErrors on SCF failure
        logging.warning(
            f"GFN2 oracle failed for a system of {len(system)} atoms "
            f"(charge {charge}, multiplicity {spin_multiplicity}): {err}; "
            "the model will run oracle-free for this structure."
        )
        return None
    return torch.from_numpy(charges).to(torch.float64)


def _per_system_scalar(
    extra: Dict[str, TensorMap], key: str, row: int, default: int
) -> int:
    """Read one system's integer scalar from a batched extra-data field.

    :param extra: The batch's extra data.
    :param key: Field to read (e.g. ``"charge"``).
    :param row: Batch row of the system.
    :param default: Value when the field is absent or NaN.
    :return: The value as an integer.
    """
    if key not in extra:
        return default
    values = extra[key].block().values
    value = values[row].reshape(-1)[0]
    if torch.isnan(value):
        return default
    return int(value)


def attach_oracle_charges(
    system: System, charges: torch.Tensor, dtype: torch.dtype
) -> None:
    """Attach per-atom oracle charges to a system, in place.

    :param system: The system; positions supply device.
    :param charges: Per-atom charges, shape ``(n_atoms,)``.
    :param dtype: dtype of the attached block (match the system's positions
        so the model can consume it without casting).
    """
    n_atoms = len(charges)
    values = charges.reshape(-1, 1).to(device=system.positions.device, dtype=dtype)
    system.add_data(
        ORACLE_CHARGES_KEY,
        TensorMap(
            keys=Labels.single(),
            blocks=[
                TensorBlock(
                    values=values,
                    samples=Labels(
                        "atom",
                        torch.arange(
                            n_atoms, dtype=torch.int32, device=values.device
                        ).reshape(-1, 1),
                    ),
                    components=[],
                    properties=Labels.range("charge", 1),
                )
            ],
        ),
    )


_ORACLE_CACHE: Optional[ByteBudgetCache] = None


def _oracle_cache() -> ByteBudgetCache:
    global _ORACLE_CACHE
    if _ORACLE_CACHE is None:
        _ORACLE_CACHE = ByteBudgetCache(collate_cache_max_bytes())
    return _ORACLE_CACHE


#: Sentinel cached for structures where the oracle failed, so the failure is
#: not retried every epoch.
_FAILED = torch.full((1,), torch.nan, dtype=torch.float64)


def _attach_precomputed(systems: List[System], packed: TensorMap) -> None:
    """Attach dataset-precomputed oracle charges to the batch's systems.

    The field arrives like any per-atom extra-data block (samples
    ``["system", "atom"]``, concatenated in batch order by
    ``group_and_join``; the ``"system"`` values are dataset indices, not
    batch positions), so the systems are the runs of equal ids, in order
    of first appearance.

    :param systems: The batch's systems, in batch order.
    :param packed: The batched per-atom oracle-charge TensorMap.
    """
    block = packed.block(0)
    index = block.samples.column("system")
    boundaries = [0]
    for i in range(1, len(index)):
        if int(index[i]) != int(index[i - 1]):
            boundaries.append(i)
    boundaries.append(len(index))
    if len(boundaries) - 1 != len(systems):
        raise RuntimeError(
            f"precomputed oracle charges cover {len(boundaries) - 1} "
            f"systems, but the batch has {len(systems)}; the field is "
            "malformed."
        )
    values = block.values.reshape(-1)
    for row, system in enumerate(systems):
        if ORACLE_CHARGES_KEY in system.known_data():
            continue
        chunk = values[boundaries[row] : boundaries[row + 1]]
        if len(chunk) != len(system):
            raise RuntimeError(
                f"precomputed oracle charges give {len(chunk)} values "
                f"for a system of {len(system)} atoms."
            )
        if torch.isnan(chunk).any():
            continue  # NaN marks "oracle failed offline": run oracle-free
        attach_oracle_charges(system, chunk, system.positions.dtype)


def _oracle_charges_transform(
    systems: List[System],
    targets: Dict[str, TensorMap],
    extra: Dict[str, TensorMap],
) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
    # Precomputed charges shipped with the dataset are authoritative for
    # every system in the batch: real values get attached, NaN entries mean
    # "oracle failed offline, run oracle-free". Never fall through to the
    # on-the-fly path here — it would recompute known failures and requires
    # tblite, which is unavailable on some training platforms.
    if ORACLE_CHARGES_KEY in extra:
        _attach_precomputed(systems, extra[ORACLE_CHARGES_KEY])
        return systems, targets, extra
    system_ids = batch_system_ids(extra)
    cache = _oracle_cache()
    for row, system in enumerate(systems):
        if ORACLE_CHARGES_KEY in system.known_data():
            continue
        charge = _per_system_scalar(extra, "charge", row, 0)
        multiplicity = _per_system_scalar(extra, "spin_multiplicity", row, 1)
        charges: Optional[torch.Tensor] = None
        key = None
        if system_ids is not None:
            key = ("oracle-gfn2", system_ids[row])
            charges = cache.get(key)
        if charges is None:
            charges = compute_gfn2_charges(system, charge, multiplicity)
            if charges is None:
                charges = _FAILED
            if key is not None:
                cache.put(key, charges)
        if len(charges) == 1 and torch.isnan(charges).all():
            continue  # oracle failed: no data attached, model runs oracle-free
        attach_oracle_charges(system, charges, system.positions.dtype)
    return systems, targets, extra


def get_oracle_charges_transform() -> Callable:
    """Build the collate transform attaching GFN2 oracle charges.

    Charges are rotation-invariant, so the transform may run before or after
    the augmenter; entries are cached across epochs in the per-worker
    byte-budgeted cache, keyed by the batch's native system ids. Structures
    without ids are recomputed every epoch (correct, just slower).

    :return: A collate transform ``(systems, targets, extra) -> same``.
    """
    return functools.partial(_oracle_charges_transform)
