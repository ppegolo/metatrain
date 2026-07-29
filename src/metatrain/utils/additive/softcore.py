import logging
import os
from typing import Dict, List, Optional, Tuple

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, NeighborListOptions, System

from ..data import DatasetInfo, TargetInfo
from ..sum_over_atoms import sum_over_atoms


# qTIP4P/f oxygen Lennard-Jones parameters, taken directly from the reference
# driver (drivers/f90/pes/qtip4pf.f90: oo_sig = 5.96946 Bohr, oo_eps =
# 2.95147e-4 Hartree). These are the KNOWN excluded-volume parameters of the
# all-atom water model whose CG PMF we are learning -- NOT refit.
_SIGMA_A_DEFAULT = 5.96946 * 0.5291772108   # 3.15890 Angstrom
_EPSILON_EV_DEFAULT = 2.95147e-4 * 27.211386245988  # 0.0080308 eV

# OPT-IN OVERRIDE (defaults unchanged, so every existing campaign is unaffected).
#
# Those defaults are the OXYGEN-ATOM LJ parameters, but the CG bead is a water
# CENTROID, whose contact distance is shorter: the water-slab CG g(r) peaks at
# 2.78 A while the default WCA wall reaches r_min = 2^(1/6) sigma = 3.55 A. The
# prior then acts on EVERY first-shell neighbour rather than only on
# over-compression as intended, contributing ~+43 kbar that the network must
# cancel with ~-40 kbar of learned attraction; the residual imbalance shows up as
# a PMF pressure error. Set these to place the wall at the edge of the sampled
# data instead. Both are stored as buffers, so an exported model carries the
# values it was TRAINED with and deployment needs no environment variable.
#
# These are the GLOBAL (single-bead-type) values. A multi-component CG system
# overrides them per bead type through the `sigma_by_type` / `epsilon_by_type`
# hypers; types not listed there fall back to these globals.
SIGMA_A = float(os.environ.get("MTT_SOFTCORE_SIGMA_A", _SIGMA_A_DEFAULT))
EPSILON_EV = float(os.environ.get("MTT_SOFTCORE_EPSILON_EV", _EPSILON_EV_DEFAULT))
R_MIN = 2.0 ** (1.0 / 6.0) * SIGMA_A       # WCA cutoff (LJ minimum)

if SIGMA_A != _SIGMA_A_DEFAULT or EPSILON_EV != _EPSILON_EV_DEFAULT:
    logging.warning(
        "SoftCore prior OVERRIDDEN: sigma=%.4f A (default %.4f), eps=%.5f eV "
        "(default %.5f) -> r_min=%.3f A",
        SIGMA_A, _SIGMA_A_DEFAULT, EPSILON_EV, _EPSILON_EV_DEFAULT, R_MIN,
    )

# Hypers accepted by SoftCore. The PET-level names are prefixed with
# ``soft_core_`` (see metatrain/pet/documentation.py); PET strips the prefix.
_SUPPORTED_HYPERS = (
    "sigma_by_type",
    "epsilon_by_type",
    "molecule_blocks",
    "bonds",
    "exclusion_depth",
)


class SoftCore(torch.nn.Module):
    """Purely-repulsive Weeks-Chandler-Andersen (WCA) excluded-volume prior.

    A short-range repulsive baseline in the spirit of Clementi's CGnet prior
    energy: the network learns only the correction on top of a physical
    repulsion, so it cannot invent the spurious low-energy basins that a raw
    force-matched CG PMF develops in undersampled compressed configurations
    (the cause of NVE heating / NVT clustering-arrest).

    The form is the repulsive part of the qTIP4P/f O-O Lennard-Jones potential,
    shifted and truncated at its minimum r_min = 2^(1/6) sigma (WCA):
        u(r) = 4 eps [ (sigma/r)^12 - (sigma/r)^6 ] + eps,   r < r_min
             = 0,                                            r >= r_min
    It adds no attraction (the mean force already carries the PMF attraction);
    it only diverges as beads over-compress. Parameters are FIXED (not fit).

    **Per-type parameters.** With more than one CG bead type the single global
    (sigma, epsilon) pair is wrong: a butanol CH3 bead and a water bead have
    different excluded volumes. ``sigma_by_type`` / ``epsilon_by_type`` give a
    value per atomic number (for CG beads the atomic number is a LABEL, not a
    real element), and unlike pairs are mixed with the standard Lorentz-Berthelot
    rules::

        sigma_ij   = (sigma_i + sigma_j) / 2
        epsilon_ij = sqrt(epsilon_i * epsilon_j)

    The WCA truncation is therefore PER EDGE at ``r_min_ij = 2^(1/6) sigma_ij``,
    and the requested neighbor-list cutoff is the largest ``r_min_ij`` over the
    type pairs present in the dataset. Types not listed keep the global
    ``SIGMA_A`` / ``EPSILON_EV`` values, so a run that passes no per-type hypers
    is numerically identical to the single-type code this replaces.

    **Intramolecular masking.** Beads belonging to the same molecule are
    permanently bonded and sit at 1.5-2.5 A, far inside any WCA wall; leaving
    them unmasked would add ~1e3 eV per bonded pair to the baseline. Molecule
    identity is NOT inferred from topology (a CG bead has no bonds): it is index
    arithmetic on a block spec, which is exact because the bead ordering is
    contiguous and fixed by the coarse-graining map. ``molecule_blocks`` is a
    list of ``[n_molecules, beads_per_molecule]`` applied in order from atom 0,
    e.g. ``[[240, 1], [120, 3]]`` = 240 one-bead molecules followed by 120
    three-bead molecules = 600 beads. Every edge whose two atoms share a molecule
    id gets both its energy and its force zeroed. If the spec's total bead count
    does not match a system's atom count the model raises, rather than silently
    skipping the mask.

    (The mask is applied by molecule id alone, so an edge between a bead and the
    periodic image of another bead of the *same* molecule is masked as well. Such
    an edge is longer than the cell's half-diagonal, hence beyond every r_min for
    any cell larger than ~2 r_min, so it contributes zero either way.)

    :param hypers: dictionary, possibly empty, with any of ``sigma_by_type``
        (``{atomic_number: sigma in Angstrom}``), ``epsilon_by_type``
        (``{atomic_number: epsilon in eV}``) and ``molecule_blocks``
        (``[[n_molecules, beads_per_molecule], ...]``).
    :param dataset_info: dataset info (atomic types, angstrom units, energy eV).
    """

    def __init__(self, hypers: Dict, dataset_info: DatasetInfo):
        super().__init__()
        if not isinstance(hypers, dict):
            raise ValueError(
                f"{self.__class__.__name__} hypers takes a dictionary. "
                f"Got: {hypers}."
            )
        unknown_hypers = [key for key in hypers.keys() if key not in _SUPPORTED_HYPERS]
        if unknown_hypers:
            raise ValueError(
                f"{self.__class__.__name__} got unknown hypers "
                f"{sorted(unknown_hypers)}. Supported: {list(_SUPPORTED_HYPERS)}."
            )
        if dataset_info.length_unit != "angstrom":
            raise ValueError(
                "SoftCore only supports angstrom units, but a "
                f"{dataset_info.length_unit} unit was provided."
            )
        for target_name, target_info in dataset_info.targets.items():
            if not self.is_valid_target(target_name, target_info):
                raise ValueError(
                    f"SoftCore model does not support target {target_name}. "
                    "This is an architecture bug. Please report this issue."
                )

        self.dataset_info = dataset_info
        self.atomic_types = sorted(dataset_info.atomic_types)
        self.outputs = {
            key: ModelOutput(
                quantity=value.quantity,
                unit=value.unit,
                sample_kind="atom",
                description=value.description,
            )
            for key, value in dataset_info.targets.items()
        }

        # --- per-type (sigma, epsilon), tabulated BY ATOMIC NUMBER ------------
        # A table of length max(Z)+1 indexed directly by the atomic number keeps
        # the edge lookup a single gather and avoids a species->index indirection.
        sigma_by_type = _as_type_dict(hypers.get("sigma_by_type"), "sigma_by_type")
        epsilon_by_type = _as_type_dict(
            hypers.get("epsilon_by_type"), "epsilon_by_type"
        )
        for name, per_type in (
            ("sigma_by_type", sigma_by_type),
            ("epsilon_by_type", epsilon_by_type),
        ):
            extra_types = sorted(set(per_type.keys()) - set(self.atomic_types))
            if extra_types:
                raise ValueError(
                    f"SoftCore {name} lists atomic types {extra_types} that are "
                    f"not in the dataset (present types: {self.atomic_types})."
                )
        for z, value in sigma_by_type.items():
            if not value > 0.0:
                raise ValueError(
                    f"SoftCore sigma for type {z} must be > 0, got {value}"
                )
        for z, value in epsilon_by_type.items():
            if value < 0.0:
                raise ValueError(
                    f"SoftCore epsilon for type {z} must be >= 0, got {value}"
                )

        table_size = max(self.atomic_types) + 1
        sigma_table = torch.full((table_size,), SIGMA_A, dtype=torch.float64)
        epsilon_table = torch.full((table_size,), EPSILON_EV, dtype=torch.float64)
        for z, value in sigma_by_type.items():
            sigma_table[z] = value
        for z, value in epsilon_by_type.items():
            epsilon_table[z] = value
        # r_min is tabulated (rather than recomputed from sigma at every call) so
        # that the per-edge truncation radius is mixed as 0.5*(r_min_i + r_min_j)
        # -- identical to 2^(1/6)*sigma_ij, but bit-exact for equal types in any
        # dtype the module may have been cast to.
        r_min_table = 2.0 ** (1.0 / 6.0) * sigma_table

        # `sigma` / `epsilon` keep holding the GLOBAL values: they are the
        # fallback for unlisted types and they preserve the state-dict layout of
        # checkpoints written before the per-type extension.
        self.register_buffer("sigma", torch.tensor(SIGMA_A, dtype=torch.float64))
        self.register_buffer("epsilon", torch.tensor(EPSILON_EV, dtype=torch.float64))
        self.register_buffer("sigma_by_type", sigma_table)
        self.register_buffer("epsilon_by_type", epsilon_table)
        self.register_buffer("r_min_by_type", r_min_table)

        # neighbor-list cutoff = largest r_min over the type pairs actually present
        cutoff_radius = 0.0
        for zi in self.atomic_types:
            for zj in self.atomic_types:
                r_min_ij = 0.5 * (float(r_min_table[zi]) + float(r_min_table[zj]))
                cutoff_radius = max(cutoff_radius, r_min_ij)
        self.cutoff_radius = cutoff_radius

        # --- intramolecular mask ---------------------------------------------
        self.register_buffer(
            "molecule_ids",
            torch.tensor(
                _molecule_ids_from_blocks(hypers.get("molecule_blocks")),
                dtype=torch.int64,
            ),
        )
        # --- topological exclusion mask (takes precedence over molecule_ids) ---
        #
        # `molecule_blocks` masks EVERY same-molecule pair. That is right for a
        # 3-bead molecule, where every intramolecular pair is 1-2 or 1-3 anyway,
        # but it is wrong as soon as a molecule has more beads: for a single
        # solute in implicit solvent it masks every pair in the system and the
        # prior silently becomes identically zero while the config still reads
        # `soft_core: true`.
        #
        # `bonds` gives the CG bond topology instead, and pairs separated by at
        # most `exclusion_depth` bonds are excluded -- the rule CGnet/CGSchNet
        # use, where 1-2 and 1-3 pairs are carried by bonded prior terms and the
        # repulsion applies only to pairs more than two bonds apart.
        exclusion = _exclusion_matrix_from_bonds(
            hypers.get("bonds"),
            hypers.get("exclusion_depth", 2),
        )
        self.register_buffer("exclusion_matrix", exclusion)

    def restart(self, dataset_info: DatasetInfo) -> "SoftCore":
        for target_name, target_info in dataset_info.targets.items():
            if not self.is_valid_target(target_name, target_info):
                raise ValueError(
                    f"SoftCore model does not support target {target_name}."
                )
        self.dataset_info = self.dataset_info.union(dataset_info)
        return self

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.outputs

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Backward compatibility: checkpoints written before the per-type
        # extension only carry the scalar `sigma` / `epsilon`. Such a model is by
        # construction a single global (sigma, epsilon), so fill the per-type
        # tables from the scalars rather than failing on missing keys.
        legacy_sigma = state_dict.get(prefix + "sigma")
        legacy_epsilon = state_dict.get(prefix + "epsilon")
        if prefix + "sigma_by_type" not in state_dict and legacy_sigma is not None:
            state_dict[prefix + "sigma_by_type"] = torch.full_like(
                self.sigma_by_type, float(legacy_sigma)
            )
        if prefix + "epsilon_by_type" not in state_dict and legacy_epsilon is not None:
            state_dict[prefix + "epsilon_by_type"] = torch.full_like(
                self.epsilon_by_type, float(legacy_epsilon)
            )
        if (
            prefix + "r_min_by_type" not in state_dict
            and prefix + "sigma_by_type" in state_dict
        ):
            state_dict[prefix + "r_min_by_type"] = (
                2.0 ** (1.0 / 6.0) * state_dict[prefix + "sigma_by_type"]
            )
        if prefix + "molecule_ids" not in state_dict:
            state_dict[prefix + "molecule_ids"] = self.molecule_ids.clone()
        # Checkpoints written before the topological mask carry no exclusion
        # matrix; they used the molecule_blocks mask, so synthesising the empty
        # (0, 0) tensor restores exactly their behaviour.
        if prefix + "exclusion_matrix" not in state_dict:
            state_dict[prefix + "exclusion_matrix"] = self.exclusion_matrix.clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        neighbor_lists: List[TensorBlock] = []
        for system in systems:
            nl_options = self.requested_neighbor_lists()[0]
            neighbor_lists.append(system.get_neighbor_list(nl_options))

        rij = torch.concatenate(
            [torch.sqrt(torch.sum(nl.values**2, dim=(1, 2))) for nl in neighbor_lists]
        )
        zi, zj, intramolecular = self._edge_types_and_mask(systems, neighbor_lists)
        sigma_ij, epsilon_ij, r_min_ij = self._pair_parameters(zi, zj)
        e_pair = self._wca(
            rij, sigma_ij, epsilon_ij, r_min_ij, intramolecular
        )  # per directed edge, already halved

        indices_for_sum_list = []
        sum = 0
        for system, nl in zip(systems, neighbor_lists, strict=True):
            indices_for_sum_list.append(nl.samples.column("first_atom") + sum)
            sum += system.positions.shape[0]
        e_nodes = torch.zeros(sum, dtype=e_pair.dtype, device=e_pair.device)
        e_nodes.index_add_(0, torch.cat(indices_for_sum_list), e_pair)

        device = systems[0].positions.device
        targets_out: Dict[str, TensorMap] = {}
        for target_key, target in outputs.items():
            sample_values: List[List[int]] = []
            for i_system, system in enumerate(systems):
                sample_values += [[i_system, i_atom] for i_atom in range(len(system))]
            block = TensorBlock(
                values=e_nodes.reshape(-1, 1),
                samples=Labels(
                    ["system", "atom"],
                    torch.tensor(sample_values, device=device),
                    assume_unique=True,
                ),
                components=[],
                properties=Labels(
                    names=["energy"], values=torch.tensor([[0]], device=device)
                ),
            )
            targets_out[target_key] = TensorMap(
                keys=Labels(names=["_"], values=torch.tensor([[0]], device=device)),
                blocks=[block],
            )
            if selected_atoms is not None:
                targets_out[target_key] = mts.slice(
                    targets_out[target_key], "samples", selected_atoms
                )
            if target.sample_kind == "system":
                targets_out[target_key] = sum_over_atoms(targets_out[target_key])
        return targets_out

    # --- Analytic energy + position-gradient (no autograd) -------------------
    # Flag read by remove_additive: use analytic_contribution() instead of
    # evaluate_model()'s autograd. This is exact (WCA has a closed-form force)
    # and, being autograd-free, lets the additive force subtraction run inside
    # fork-based DataLoader workers (where torch cannot host the autograd
    # backward engine on some builds). Deployment forces still come from the PET
    # model's own autograd of the total energy, so `forward` stays energy-only.
    provides_analytic_position_gradient: bool = True

    def analytic_contribution(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        """Per-system energy with an analytic per-atom position-gradient block,
        in the same format as evaluate_model's output for an energy target that
        requires position gradients (force matching). No strain gradient."""
        device = systems[0].positions.device
        n_total = sum(system.positions.shape[0] for system in systems)

        neighbor_lists: List[TensorBlock] = []
        first_idx_list: List[torch.Tensor] = []
        rij_list: List[torch.Tensor] = []
        dvec_list: List[torch.Tensor] = []
        offset = 0
        nl_opts = self.requested_neighbor_lists()[0]
        for system in systems:
            nl = system.get_neighbor_list(nl_opts)
            neighbor_lists.append(nl)
            dvec = nl.values.reshape(-1, 3)  # first->second displacement r_j - r_i
            rij_list.append(torch.sqrt((dvec**2).sum(dim=1)))
            dvec_list.append(dvec)
            first_idx_list.append(nl.samples.column("first_atom") + offset)
            offset += system.positions.shape[0]

        rij = torch.cat(rij_list)
        dvec = torch.cat(dvec_list, dim=0)
        first_idx = torch.cat(first_idx_list)

        zi, zj, intramolecular = self._edge_types_and_mask(systems, neighbor_lists)
        sigma_ij, epsilon_ij, r_min_ij = self._pair_parameters(zi, zj)

        # already /2 for double counting
        e_pair = self._wca(rij, sigma_ij, epsilon_ij, r_min_ij, intramolecular)
        # du/dr per directed edge (NOT halved)
        dudr = self._dwca_dr(rij, sigma_ij, epsilon_ij, r_min_ij, intramolecular)

        e_nodes = torch.zeros(n_total, dtype=e_pair.dtype, device=device)
        e_nodes.index_add_(0, first_idx, e_pair)

        # dE/dr_i = sum_{edges with first=i} du/dr * (-D_e / r_e)  (see note)
        r_safe = torch.clamp(rij, min=1e-3).unsqueeze(-1)
        gv = dudr.unsqueeze(-1) * (-dvec / r_safe)   # [n_edges, 3]
        grad_nodes = torch.zeros(n_total, 3, dtype=gv.dtype, device=device)
        grad_nodes.index_add_(0, first_idx, gv)

        sys_col = torch.cat([
            torch.full((system.positions.shape[0],), i, dtype=torch.int64)
            for i, system in enumerate(systems)
        ]).to(device)
        atom_col = torch.cat([
            torch.arange(system.positions.shape[0], dtype=torch.int64)
            for system in systems
        ]).to(device)

        grad_block = TensorBlock(
            values=grad_nodes.unsqueeze(-1),  # [n_atoms, 3, 1]
            samples=Labels(["sample", "atom"],
                           torch.stack([sys_col, atom_col], dim=1)),
            components=[Labels(["xyz"], torch.tensor([[0], [1], [2]], device=device))],
            properties=Labels("energy", torch.tensor([[0]], device=device)),
        )

        out: Dict[str, TensorMap] = {}
        for target_key, target in outputs.items():
            sample_values = []
            for i_system, system in enumerate(systems):
                sample_values += [[i_system, i_atom] for i_atom in range(len(system))]
            eblock = TensorBlock(
                values=e_nodes.reshape(-1, 1),
                samples=Labels(["system", "atom"],
                               torch.tensor(sample_values, device=device),
                               assume_unique=True),
                components=[],
                properties=Labels(["energy"], torch.tensor([[0]], device=device)),
            )
            tmap = TensorMap(
                keys=Labels(["_"], torch.tensor([[0]], device=device)),
                blocks=[eblock],
            )
            if selected_atoms is not None:
                tmap = mts.slice(tmap, "samples", selected_atoms)
            if target.sample_kind == "system":
                tmap = sum_over_atoms(tmap)
            # attach the per-atom position gradient to the (now per-system) block
            blk = tmap.block().copy(deep=False)
            blk.add_gradient("positions", grad_block)
            out[target_key] = TensorMap(keys=tmap.keys, blocks=[blk])
        return out

    def _edge_types_and_mask(
        self, systems: List[System], neighbor_lists: List[TensorBlock]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per directed edge: the atomic number of the first atom, of the second
        atom, and whether the two belong to the same molecule (bonded pair, to be
        excluded from the prior). Concatenated over `systems` in the same order
        as the distances."""
        zi_list: List[torch.Tensor] = []
        zj_list: List[torch.Tensor] = []
        mask_list: List[torch.Tensor] = []
        n_spec_beads = self.molecule_ids.shape[0]
        n_excl_beads = self.exclusion_matrix.shape[0]
        for system, nl in zip(systems, neighbor_lists, strict=True):
            first = nl.samples.column("first_atom")
            second = nl.samples.column("second_atom")
            zi_list.append(system.types[first])
            zj_list.append(system.types[second])
            if n_excl_beads > 0:
                n_atoms = system.positions.shape[0]
                if n_excl_beads != n_atoms:
                    raise ValueError(
                        "SoftCore bonds describe "
                        + str(n_excl_beads)
                        + " beads but this system has "
                        + str(n_atoms)
                        + " atoms: the topological exclusion mask cannot be "
                        + "applied. Fix soft_core_bonds to match the system."
                    )
                # flat gather rather than 2-D advanced indexing: scriptable, and
                # it keeps the mask a single kernel on large edge lists
                flat = self.exclusion_matrix.reshape(-1)
                mask_list.append(flat[first * n_excl_beads + second])
            elif n_spec_beads == 0:
                mask_list.append(torch.zeros_like(first, dtype=torch.bool))
            else:
                n_atoms = system.positions.shape[0]
                if n_spec_beads != n_atoms:
                    raise ValueError(
                        "SoftCore molecule_blocks describe "
                        + str(n_spec_beads)
                        + " beads but this system has "
                        + str(n_atoms)
                        + " atoms: the intramolecular mask cannot be applied. "
                        + "Fix soft_core_molecule_blocks to match the system."
                    )
                mask_list.append(
                    self.molecule_ids[first] == self.molecule_ids[second]
                )
        return torch.cat(zi_list), torch.cat(zj_list), torch.cat(mask_list)

    def _pair_parameters(
        self, zi: torch.Tensor, zj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Lorentz-Berthelot mixed (sigma_ij, epsilon_ij, r_min_ij) per directed
        edge, evaluated in the dtype of the parameter tables."""
        sigma_i = self.sigma_by_type[zi]
        sigma_j = self.sigma_by_type[zj]
        epsilon_i = self.epsilon_by_type[zi]
        epsilon_j = self.epsilon_by_type[zj]
        sigma_ij = 0.5 * (sigma_i + sigma_j)
        # sqrt(e*e) is not bit-exact for every float; when both types carry the
        # same epsilon the mixed value IS that epsilon, so take it verbatim. This
        # is what keeps a single-type model identical to the pre-per-type code.
        epsilon_ij = torch.where(
            epsilon_i == epsilon_j, epsilon_i, torch.sqrt(epsilon_i * epsilon_j)
        )
        r_min_ij = 0.5 * (self.r_min_by_type[zi] + self.r_min_by_type[zj])
        return sigma_ij, epsilon_ij, r_min_ij

    def _dwca_dr(
        self,
        rij: torch.Tensor,
        sigma_ij: torch.Tensor,
        epsilon_ij: torch.Tensor,
        r_min_ij: torch.Tensor,
        intramolecular: torch.Tensor,
    ) -> torch.Tensor:
        """du/dr for the WCA energy per directed edge (NOT halved), zero beyond
        r_min and zero on intramolecular (bonded) edges.
        u = 4eps(sr12 - sr6) + eps, du/dr = (24 eps / r)(sr6 - 2 sr12)."""
        eps = epsilon_ij.to(rij.dtype)
        sig = sigma_ij.to(rij.dtype)
        r_min = r_min_ij.to(rij.dtype)
        r = torch.clamp(rij, min=1e-3)
        sr6 = (sig / r) ** 6
        sr12 = sr6 * sr6
        dudr = (24.0 * eps / r) * (sr6 - 2.0 * sr12)
        zero = torch.zeros_like(dudr)
        dudr = torch.where(rij < r_min, dudr, zero)
        return torch.where(intramolecular, zero, dudr)

    def _wca(
        self,
        rij: torch.Tensor,
        sigma_ij: torch.Tensor,
        epsilon_ij: torch.Tensor,
        r_min_ij: torch.Tensor,
        intramolecular: torch.Tensor,
    ) -> torch.Tensor:
        """WCA energy per directed edge (divided by 2 for double counting), zero
        beyond r_min and zero on intramolecular (bonded) edges."""
        eps = epsilon_ij.to(rij.dtype)
        sig = sigma_ij.to(rij.dtype)
        r_min = r_min_ij.to(rij.dtype)
        r = torch.clamp(rij, min=1e-3)
        sr6 = (sig / r) ** 6
        e = 4.0 * eps * (sr6 * sr6 - sr6) + eps
        zero = torch.zeros_like(e)
        e = torch.where(rij < r_min, e, zero)
        e = torch.where(intramolecular, zero, e)
        return e / 2.0

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [
            NeighborListOptions(
                cutoff=self.cutoff_radius,
                full_list=True,
                strict=True,
            )
        ]

    @staticmethod
    def is_valid_target(target_name: str, target_info: TargetInfo) -> bool:
        if target_info.quantity != "energy":
            logging.debug(f"SoftCore skips non-energy target {target_name}.")
            return False
        if not target_info.is_scalar:
            logging.debug(f"SoftCore skips non-scalar target {target_name}.")
            return False
        if len(target_info.layout.block(0).properties) > 1:
            logging.debug(f"SoftCore skips multi-property target {target_name}.")
            return False
        if target_info.unit != "eV":
            logging.debug(f"SoftCore skips non-eV target {target_name}.")
            return False
        return True


def _as_type_dict(value, name: str) -> Dict[int, float]:
    """Normalize a ``{atomic_number: value}`` hyper to ``{int: float}``.

    Accepts ``None`` (absent hyper) and tolerates string keys, which is what a
    YAML/JSON round-trip of an integer-keyed mapping can produce.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(
            f"SoftCore {name} must be a dictionary keyed by atomic number, "
            f"got {value}."
        )
    return {int(key): float(item) for key, item in value.items()}


def _exclusion_matrix_from_bonds(bonds, depth) -> torch.Tensor:
    """Boolean ``(n_beads, n_beads)`` mask: True where two beads are separated by
    at most ``depth`` bonds and the repulsion must therefore be excluded.

    ``bonds`` is a list of ``[i, j]`` 0-based bead index pairs describing the CG
    topology of the whole system. Returns a ``(0, 0)`` tensor when ``bonds`` is
    absent, which is the signal to fall back to the ``molecule_blocks`` mask.

    Depth 2 reproduces CGnet/CGSchNet: 1-2 and 1-3 pairs excluded (they are
    carried by the harmonic bond and angle priors), the repulsion applied to
    everything more than two bonds apart. A bead is always excluded from itself.
    """
    if bonds is None:
        return torch.zeros((0, 0), dtype=torch.bool)
    bond_list = [(int(i), int(j)) for i, j in bonds]
    if len(bond_list) == 0:
        raise ValueError(
            "SoftCore bonds is empty. Omit the hyper to disable the topological "
            "mask; an empty list is more likely a builder bug than an intent to "
            "exclude nothing."
        )
    depth_int = int(depth)
    if depth_int < 0:
        raise ValueError(f"SoftCore exclusion_depth must be >= 0, got {depth_int}")

    n_beads = max(max(i, j) for i, j in bond_list) + 1
    adjacency = torch.zeros((n_beads, n_beads), dtype=torch.bool)
    for i, j in bond_list:
        if i == j:
            raise ValueError(f"SoftCore bonds contains a self-bond ({i}, {j})")
        adjacency[i, j] = True
        adjacency[j, i] = True

    # reachability within `depth` hops: I | A | A^2 | ... | A^depth
    reach = torch.eye(n_beads, dtype=torch.bool)
    frontier = torch.eye(n_beads, dtype=torch.bool)
    for _ in range(depth_int):
        frontier = (frontier.to(torch.int8) @ adjacency.to(torch.int8)) > 0
        reach = reach | frontier
    return reach


def _molecule_ids_from_blocks(blocks) -> List[int]:
    """Expand a ``[[n_molecules, beads_per_molecule], ...]`` spec into a per-bead
    molecule id, assigned contiguously from bead 0.

    Bead ordering in a CG frame is fixed by the coarse-graining map, so molecule
    identity is pure index arithmetic; there is no topology to read.
    """
    if blocks is None:
        return []
    molecule_ids: List[int] = []
    n_molecules_so_far = 0
    for block in blocks:
        if len(block) != 2:
            raise ValueError(
                "SoftCore molecule_blocks entries must be "
                f"[n_molecules, beads_per_molecule] pairs, got {block}."
            )
        n_molecules, beads_per_molecule = int(block[0]), int(block[1])
        if n_molecules <= 0 or beads_per_molecule <= 0:
            raise ValueError(
                "SoftCore molecule_blocks entries must be strictly positive, "
                f"got {block}."
            )
        for _ in range(n_molecules):
            molecule_ids.extend([n_molecules_so_far] * beads_per_molecule)
            n_molecules_so_far += 1
    return molecule_ids
