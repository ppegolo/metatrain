"""Free-atom (promolecule) baseline weights for atomic-basis density targets.

The electrostatic potential outside a molecule is the residue of a near-total
cancellation between the nuclear potential and the electronic one — hundreds
of kcal/mol/e each, a few kcal/mol/e left. A model that predicts the *total*
density must get that cancellation right to fractions of a percent. A neutral,
spherically-averaged free atom removes it analytically: its far field is
exactly zero (monopole ``Z - N = 0``, dipole zero by symmetry), so a model
trained on the *deformation* density ``rho - sum_a rho_a^free`` only ever
handles the small, chemically meaningful remainder.

This module computes those free-atom densities — one per element, RI-fitted
in the target's auxiliary basis — and installs them as **fixed weights of the
existing composition model**, replacing its least-squares fit. Everything
downstream is inherited unchanged: the trainer subtracts the baseline from
the targets, the exported model adds it back, checkpoints carry it. By
spherical symmetry only the ``o3_lambda=0`` blocks are nonzero, which is
precisely the subspace the composition model fits anyway.

Every quadratic density loss is invariant under this change: the losses act
on ``Delta c``, and subtracting one baseline from both prediction and
reference leaves the difference untouched. What changes is what the network
must *represent*. Losses that evaluate a single density in absolute terms
(the EC loss's potentials, ``via_w`` against stored total-density
projections) are **not** invariant and must not be combined with this
baseline without explicit support.

Selected from an options file through the ``atomic_baseline`` hyperparameter,
with a spec string in place of the fixed-weights dict or checkpoint path::

    training:
      atomic_baseline: "free_atom:def2-svp:def2-universal-jfit"

where the two fields are the orbital basis the free-atom densities are
computed in and the auxiliary basis the target coefficients live in.
"""

import logging
from typing import TYPE_CHECKING, List, Tuple

import metatensor.torch as mts
import torch


if TYPE_CHECKING:
    from ...composition import CompositionModel


FREE_ATOM_PREFIX = "free_atom"


def is_free_atom_spec(atomic_baseline: object) -> bool:
    """Whether an ``atomic_baseline`` value selects the free-atom baseline.

    :param atomic_baseline: The raw hyperparameter value.
    :return: ``True`` for a ``"free_atom:<basis>:<aux_basis>"`` string.
    """
    return isinstance(atomic_baseline, str) and atomic_baseline.startswith(
        FREE_ATOM_PREFIX + ":"
    )


def parse_free_atom_spec(spec: str) -> Tuple[str, str]:
    """Split a free-atom spec into its orbital and auxiliary basis.

    :param spec: A ``"free_atom:<basis>:<aux_basis>"`` string. The auxiliary
        basis may itself contain colons (``"etb:def2-svp:2.0"``), so the split
        is on the first two separators only.
    :return: ``(basis, aux_basis)``.
    """
    parts = spec.split(":", 2)
    if len(parts) != 3 or parts[0] != FREE_ATOM_PREFIX or not parts[1] or not parts[2]:
        raise ValueError(
            f"invalid free-atom baseline spec {spec!r}; expected "
            "'free_atom:<orbital basis>:<auxiliary basis>', e.g. "
            "'free_atom:def2-svp:def2-universal-jfit'."
        )
    return parts[1], parts[2]


def free_atom_coefficients(
    atomic_number: int, basis: str, aux_basis: str
) -> torch.Tensor:
    """
    RI coefficients of one neutral, spherically-averaged free atom.

    The atomic density comes from PySCF's spherically-averaged atomic HF (the
    same solver behind ``init_guess_by_atom``), so open shells carry
    fractional, spherically-symmetric occupations and the density is exactly
    invariant. It is then Coulomb-metric RI-fitted in the atom's auxiliary
    basis. By symmetry only ``l=0`` functions contribute; the tiny numerical
    leakage into ``l>0`` is dropped, and the ``l=0`` coefficients are rescaled
    to integrate to exactly ``Z`` electrons, so the baseline is neutral to
    machine precision and the residual target carries pure deformation charge.

    :param atomic_number: Element to compute.
    :param basis: Orbital basis for the atomic HF density.
    :param aux_basis: Auxiliary basis (or ``"etb:<ao_basis>:<beta>"``) to fit
        in — the basis the density target's coefficients live in.
    :return: The ``l=0`` radial coefficients, in shell order, float64.
    """
    import copy

    import numpy as np

    from ..pyscf_loss import _import_pyscf, _load_auxiliary_basis

    gto, elements = _import_pyscf()
    from pyscf import df
    from pyscf.scf import atom_hf

    symbol = elements.ELEMENTS[atomic_number]
    mol = gto.Mole()
    mol.atom = f"{symbol} 0.0 0.0 0.0"
    mol.basis = basis
    mol.spin = None
    mol.verbose = 0
    mol.build()

    # PySCF's spherically-averaged atomic solver calls its own deprecated
    # ``remove_linear_dep_`` internally; that warning is theirs, not ours.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        atomic_scf = atom_hf.get_atm_nrhf(mol)[symbol]
    # (energy, mo_energies, mo_coefficients, occupations)
    _, _, mo_coefficients, occupations = atomic_scf
    dm = (mo_coefficients * occupations) @ mo_coefficients.T

    auxmol = gto.Mole()
    auxmol.atom = mol.atom
    auxmol.basis = copy.deepcopy(_load_auxiliary_basis(aux_basis, (atomic_number,)))
    auxmol.spin = None
    auxmol.verbose = 0
    auxmol.cart = False
    auxmol.build()

    j2c = auxmol.intor("int2c2e")
    j3c = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")
    coefficients = np.linalg.solve(j2c, np.einsum("ijP,ij->P", j3c, dm))

    # Only s functions survive spherical symmetry; anything else is numerical
    # noise from the fit, dropped rather than carried into the baseline.
    ao_loc = auxmol.ao_loc_nr()
    is_l0 = np.zeros(auxmol.nao, dtype=bool)
    moments = np.zeros(auxmol.nao)
    for shell in range(auxmol.nbas):
        if auxmol.bas_angular(shell) != 0:
            continue
        exponents = auxmol.bas_exp(shell)
        contraction = auxmol._libcint_ctr_coeff(shell)
        primitives = (np.pi / exponents) ** 1.5 / np.sqrt(4.0 * np.pi)
        for i in range(contraction.shape[1]):
            is_l0[ao_loc[shell] + i] = True
            moments[ao_loc[shell] + i] = float(np.dot(contraction[:, i], primitives))

    leakage = float(np.abs(coefficients[~is_l0]).max()) if (~is_l0).any() else 0.0
    if leakage > 1e-6:
        raise RuntimeError(
            f"free-atom fit for element {atomic_number} leaked {leakage:.2e} "
            "into l>0 auxiliary functions; the atomic density is not "
            "spherical, which points to a broken basis or SCF setup."
        )

    l0 = coefficients[is_l0]
    electrons = float(moments[is_l0] @ l0)
    if not 0.5 * atomic_number < electrons < 1.5 * atomic_number:
        raise RuntimeError(
            f"free-atom fit for element {atomic_number} integrates to "
            f"{electrons:.3f} electrons; expected about {atomic_number}."
        )
    l0 *= atomic_number / electrons  # neutral to machine precision

    return torch.from_numpy(l0).to(torch.float64)


def set_free_atom_composition_weights(
    composition_model: "CompositionModel", basis: str, aux_basis: str
) -> None:
    """
    Fill a composition model with free-atom baseline weights, in place.

    Replaces the least-squares composition fit: for every target of the model,
    each atomic type's row of the ``o3_lambda=0`` weight block receives that
    element's free-atom RI coefficients (padded properties stay zero, exactly
    like a fitted composition on a padded layout). The weight buffers are
    rewritten the same way the composition trainer does after ``fit``, so
    checkpointing, ``restart`` and export behave identically to a fitted
    composition model.

    :param composition_model: The model to fill. Every target must be a
        per-atom spherical (atomic-basis) target — the baseline is an RI
        density, so a scalar target here is a configuration error.
    :param basis: Orbital basis for the atomic HF densities.
    :param aux_basis: Auxiliary basis of the targets' coefficients.
    """
    base = composition_model.model
    atomic_types: List[int] = [int(t) for t in base.atomic_types]

    coefficients = {
        atomic_type: free_atom_coefficients(atomic_type, basis, aux_basis)
        for atomic_type in atomic_types
    }

    for target_name in base.target_names:
        weights = base.weights[target_name]
        if base.sample_kinds[target_name] != "per_atom" or (
            "o3_lambda" not in weights.keys.names
        ):
            raise ValueError(
                f"the free-atom baseline is an RI density, but target "
                f"'{target_name}' is not a per-atom spherical target; use the "
                "fitted composition model (or fixed weights) for it instead."
            )
        blocks = []
        for key, block in weights.items():
            values = torch.zeros_like(block.values)
            if int(key["o3_lambda"]) == 0:
                for row, atomic_type in enumerate(atomic_types):
                    element = coefficients[atomic_type]
                    if len(element) > values.shape[-1]:
                        raise ValueError(
                            f"element {atomic_type} has {len(element)} l=0 "
                            f"auxiliary functions in '{aux_basis}', but target "
                            f"'{target_name}' provides {values.shape[-1]} "
                            "l=0 properties; the target layout does not match "
                            "the auxiliary basis."
                        )
                    values[row, 0, : len(element)] = element
            blocks.append(
                mts.TensorBlock(
                    values=values,
                    samples=block.samples,
                    components=block.components,
                    properties=block.properties,
                )
            )
        base.weights[target_name] = mts.TensorMap(weights.keys, blocks)

        buffer_name = target_name + "_composition_buffer"
        device = composition_model.__getattr__(buffer_name).device
        composition_model.register_buffer(
            buffer_name,
            mts.save_buffer(
                mts.make_contiguous(base.weights[target_name].to("cpu", torch.float64))
            ).to(device),
        )

    logging.info(
        f"Free-atom baseline computed for elements {atomic_types} "
        f"(orbital basis '{basis}', auxiliary basis '{aux_basis}')."
    )
