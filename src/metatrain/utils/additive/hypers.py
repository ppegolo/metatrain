"""Hyperparameters for the ADDITIVE PRIORS, shared by every architecture.

These live here rather than in one architecture's schema because the priors are
architecture-agnostic (:mod:`metatrain.utils.additive`), and because they were for a
while declared in PET's schema alone. That meant SOAP-BPNN and SPACE could not
express them at all: the schema forbids unknown keys, so ``soft_core: true`` was not
ignored, it was REJECTED -- and the only prior those architectures could reach was
ZBL, a nuclear-charge repulsion that is meaningless on coarse-grained beads that are
not nuclei.

An architecture opts in by inheriting this class in its ``ModelHypers``.
``init_with_defaults`` walks the MRO, so the defaults come along and nothing else
changes.
"""

from typing import Dict, List

from typing_extensions import TypedDict


class AdditivePriorHypers(TypedDict):
    """Additive baselines a model learns a correction ON TOP OF (delta-learning).

    All default to off, so inheriting this class changes no existing behaviour.
    """

    zbl: bool = False
    """Use ZBL potential for short-range repulsion"""
    soft_core: bool = False
    """Use a WCA excluded-volume repulsive prior (qTIP4P/f O-O LJ core) as an
    additive baseline for delta-learning (CG coarse-bead short-range stability)."""
    soft_core_sigma_by_type: Dict[int, float] = {}
    """Per-bead-type WCA ``sigma`` (Angstrom), keyed by atomic number.

    Only meaningful when ``soft_core`` is enabled. Coarse-grained beads carry an
    atomic number as a LABEL rather than as a real element, so this is a per-bead-type
    excluded-volume radius. Types omitted here keep the global default (the
    qTIP4P/f O-O value, or the ``MTT_SOFTCORE_SIGMA_A`` environment override).
    Unlike pairs are mixed with the Lorentz-Berthelot rule
    ``sigma_ij = (sigma_i + sigma_j) / 2``, and the WCA truncation is applied per
    edge at ``2^(1/6) sigma_ij``. Example: ``{8: 3.16, 6: 3.90, 7: 3.80, 9: 3.30}``."""
    soft_core_epsilon_by_type: Dict[int, float] = {}
    """Per-bead-type WCA ``epsilon`` (eV), keyed by atomic number.

    Only meaningful when ``soft_core`` is enabled. Types omitted here keep the global
    default (the qTIP4P/f O-O value, or the ``MTT_SOFTCORE_EPSILON_EV`` environment
    override). Unlike pairs are mixed with the Lorentz-Berthelot rule
    ``epsilon_ij = sqrt(epsilon_i * epsilon_j)``."""
    soft_core_molecule_blocks: List[List[int]] = []
    """Intramolecular exclusions for the ``soft_core`` prior, as a list of
    ``[n_molecules, beads_per_molecule]`` blocks applied in order from atom 0.

    Beads of the same coarse-grained molecule are permanently bonded and sit well
    inside the WCA wall, so the prior must not act between them. Bead ordering in a
    CG frame is contiguous and fixed by the coarse-graining map, so molecule identity
    is index arithmetic rather than topology: ``[[240, 1], [120, 3]]`` means 240
    one-bead molecules followed by 120 three-bead molecules (600 beads in total).
    Every edge whose two beads share a molecule gets both its energy and its force
    zeroed. Training aborts if the total bead count does not match the systems.
    Leave empty (the default) for a single-bead-per-molecule system such as CG
    water.

    **Do not use this for molecules with more than three beads.** It masks EVERY
    same-molecule pair, which coincides with the 1-2/1-3 rule only because a
    three-bead molecule has no pair further apart. For a larger molecule -- and
    especially for a single solute in implicit solvent, where it masks the whole
    system and the prior becomes identically zero while the config still reads
    ``soft_core: true`` -- use ``soft_core_bonds`` instead."""
    soft_core_bonds: List[List[int]] = []
    """CG bond topology for the ``soft_core`` prior, as a list of ``[i, j]`` 0-based
    bead index pairs covering the whole system.

    When given, this **takes precedence over** ``soft_core_molecule_blocks`` and the
    exclusion becomes topological: every bead pair separated by at most
    ``soft_core_exclusion_depth`` bonds has its energy and force zeroed. This is the
    rule CGnet/CGSchNet use -- 1-2 and 1-3 pairs are carried by bonded prior terms,
    and the excluded-volume repulsion applies only to pairs more than two bonds
    apart. Training aborts if the bead count implied here does not match the
    systems."""
    soft_core_exclusion_depth: int = 2
    """Bond separation at or below which ``soft_core_bonds`` excludes a pair.

    The default 2 excludes 1-2 (bonded) and 1-3 (angle) pairs, matching CGnet. Only
    meaningful when ``soft_core_bonds`` is non-empty."""
    harmonic_bonded: bool = False
    """Use harmonic bond and angle terms as an additive baseline for
    delta-learning (the bonded half of the CGnet/CGSchNet prior energy).

    Complements ``soft_core``: the excluded-volume prior is switched OFF between
    1-2 and 1-3 pairs (see ``soft_core_exclusion_depth``) precisely because those
    coordinates are carried by these harmonic terms instead. CGSchNet reports the
    bonded prior as essential for capped alanine."""
    harmonic_bonded_bonds: List[List[int]] = []
    """CG bond topology for the ``harmonic_bonded`` prior, as a list of ``[i, j]``
    0-based bead index pairs.

    Bonded terms are topological, not cutoff-based: no neighbor list is used and
    the bead ordering (fixed by the coarse-graining map) is the identifier.
    Training aborts if the bead count implied here does not match the systems."""
    harmonic_bonded_bond_k: List[float] = []
    """Harmonic bond force constants in **eV/Angstrom^2**, one per entry of
    ``harmonic_bonded_bonds``.

    Obtained elsewhere by Boltzmann inversion of the all-atom reference,
    ``k = kB*T / Var[r]``; this prior only evaluates ``0.5 k (r - r0)^2``."""
    harmonic_bonded_bond_r0: List[float] = []
    """Harmonic bond equilibrium lengths in **Angstrom**, one per entry of
    ``harmonic_bonded_bonds`` (``r0 = E[r]`` from the reference)."""
    harmonic_bonded_angles: List[List[int]] = []
    """CG angle topology for the ``harmonic_bonded`` prior, as a list of
    ``[i, j, k]`` 0-based bead index triples, where ``j`` is the VERTEX."""
    harmonic_bonded_angle_k: List[float] = []
    """Harmonic angle force constants in **eV/radian^2**, one per entry of
    ``harmonic_bonded_angles`` (``k = kB*T / Var[theta]``)."""
    harmonic_bonded_angle_theta0: List[float] = []
    """Harmonic angle equilibrium values in **radian**, one per entry of
    ``harmonic_bonded_angles`` (``theta0 = E[theta]``). Degrees are rejected."""
