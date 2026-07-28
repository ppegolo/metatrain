import logging
from typing import Dict, List, Optional, Tuple

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, NeighborListOptions, System

from ..data import DatasetInfo, TargetInfo
from ..sum_over_atoms import sum_over_atoms


# Hypers accepted by HarmonicBonded. The PET-level names are prefixed with
# ``harmonic_bonded_`` (see metatrain/pet/documentation.py); PET strips the prefix.
_SUPPORTED_HYPERS = (
    "bonds",
    "bond_k",
    "bond_r0",
    "angles",
    "angle_k",
    "angle_theta0",
)

# NOTE on the literal ``1e-30`` that appears as ``clamp(..., min=1e-30)`` below:
# it is the floor on a squared length before taking its square root. Any physical
# bead separation squared is >> 1e-30, so the floor is BIT-EXACT for every
# realistic geometry; it only fires on exactly-coincident beads, where it turns an
# infinite d(sqrt)/dx into a zero gradient instead of a NaN. It is written out at
# each use site rather than hoisted into a module constant because TorchScript
# rejects closed-over Python floats (even annotated ``Final``).


def _dtype_eps(dtype: torch.dtype) -> float:
    """Machine epsilon of the working dtype.

    Spelled out rather than read from ``torch.finfo`` because TorchScript does
    not support ``finfo``. A dtype-dependent value matters: a fixed 1e-12 guard
    on ``cos theta`` is a no-op in float32 (``1 - 1e-12 == 1.0``) and the
    ``1/sin`` factor would then divide by zero.
    """
    if dtype == torch.float64:
        return 2.220446049250313e-16
    if dtype == torch.float16:
        return 9.765625e-04
    if dtype == torch.bfloat16:
        return 7.8125e-03
    return 1.1920928955078125e-07  # float32


class HarmonicBonded(torch.nn.Module):
    r"""Harmonic bond + angle prior for coarse-grained molecules (CGSchNet Eq. 6a).

    The bonded half of Clementi's CGnet/CGSchNet prior energy. Together with the
    :class:`SoftCore` excluded-volume term it forms the fixed baseline on top of
    which the network learns only a correction, so the force-matched CG PMF
    cannot invent spurious basins along the (stiff, weakly-sampled) bonded
    coordinates. CGSchNet reports these terms as essential for capped alanine.

    Per feature :math:`f` -- a bond length (Angstrom) or a bond angle (radian) --

    .. math::

        U(f) = \tfrac{1}{2} k \, (f - f_0)^2

    so the total prior energy is the sum over all listed bonds and angles. The
    parameters are supplied ready-made, having been obtained elsewhere by
    Boltzmann inversion of the all-atom reference
    (:math:`k = k_B T / \mathrm{Var}[f]`, :math:`f_0 = \mathbb{E}[f]`). This
    module only evaluates the form; it never fits anything, and the parameters
    are stored as buffers so an exported model carries the values it was TRAINED
    with.

    **Bonded terms are TOPOLOGICAL, not cutoff-based.** There is no neighbor
    list: the features are read straight off ``system.positions`` by bead index.
    The bead ordering of a CG frame is fixed by the coarse-graining map, so an
    index is an exact and stable identifier. Consequently a system whose atom
    count differs from the bead count implied by the indices is rejected rather
    than silently partially evaluated.

    **Degenerate angles.** :math:`\theta` is obtained from
    :math:`\mathrm{atan2}(\lVert a \times b \rVert, a \cdot b)`, which is exact and
    well conditioned over the whole range :math:`[0, \pi]` and needs no clamping of
    :math:`\cos\theta`. The analytic derivative carries a :math:`1/\sin\theta`
    factor that is genuinely singular at :math:`\theta = 0, \pi`: the numerator
    :math:`\cos\theta\,\hat a - \hat b` vanishes at the same rate, so the gradient
    MAGNITUDE stays finite at :math:`\sim 1/|a|`, but that bound blows up when an arm
    is short. A *coincident* triple therefore gives a finite energy and a very LARGE
    force (:math:`\sim 10^{15}` eV/A, from the ``1e-30`` floor on :math:`|a|^2`), not a
    small one -- it signals a broken configuration and is meant to be loud.
    :math:`\sin\theta` is floored at ``1e-12`` purely to keep the intermediate finite;
    that window is :math:`\theta \sim 10^{-12}` rad, identical in float32 and float64,
    and unreachable by any thermally sampled angle.

    .. warning::

       An earlier revision clamped :math:`\cos\theta` by ``100 * finfo(dtype).eps``.
       That made the computed energy bit-constant inside the window, so autograd
       returned exactly zero there while the analytic path returned a large value, and
       the window was **dtype dependent** -- 2.1e-7 rad in float64 but 4.9e-3 rad
       (0.28 deg) in float32, which a float32 export would actually visit for an angle
       with :math:`\theta_0` near :math:`\pi`. The ``atan2`` form removes both
       problems.

    **Periodicity.** Displacements are minimum-imaged per periodic direction when a
    system declares ``pbc``, so a molecule wrapped across a cell boundary is handled
    correctly. For a non-periodic system (``pbc`` all false, the intended CG case) the
    minimum-image step is skipped entirely and the result is bit-identical to a raw
    coordinate difference. Note the convention: a bonded term longer than half the cell
    would be imaged to the wrong periodic copy, which is inherent to the minimum-image
    convention and not specific to this module.

    :param hypers: dictionary, possibly empty, with any of ``bonds``
        (``[[i, j], ...]``, 0-based bead indices), ``bond_k`` (eV/Angstrom^2),
        ``bond_r0`` (Angstrom), ``angles`` (``[[i, j, k], ...]``, ``j`` the
        vertex), ``angle_k`` (eV/radian^2) and ``angle_theta0`` (radian). An
        empty/absent set of hypers makes the module contribute exactly zero.
    :param dataset_info: dataset info (atomic types, angstrom units, energy eV).
    """

    def __init__(self, hypers: Dict, dataset_info: DatasetInfo):
        super().__init__()
        if not isinstance(hypers, dict):
            raise ValueError(
                f"{self.__class__.__name__} hypers takes a dictionary. Got: {hypers}."
            )
        unknown_hypers = [key for key in hypers.keys() if key not in _SUPPORTED_HYPERS]
        if unknown_hypers:
            raise ValueError(
                f"{self.__class__.__name__} got unknown hypers "
                f"{sorted(unknown_hypers)}. Supported: {list(_SUPPORTED_HYPERS)}."
            )
        if dataset_info.length_unit != "angstrom":
            raise ValueError(
                "HarmonicBonded only supports angstrom units, but a "
                f"{dataset_info.length_unit} unit was provided."
            )
        for target_name, target_info in dataset_info.targets.items():
            if not self.is_valid_target(target_name, target_info):
                raise ValueError(
                    f"HarmonicBonded model does not support target {target_name}. "
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

        bond_index = _index_tensor(hypers.get("bonds"), 2, "bonds")
        bond_k = _parameter_tensor(
            hypers.get("bond_k"), bond_index.shape[0], "bond_k", "bonds"
        )
        bond_r0 = _parameter_tensor(
            hypers.get("bond_r0"), bond_index.shape[0], "bond_r0", "bonds"
        )
        angle_index = _index_tensor(hypers.get("angles"), 3, "angles")
        angle_k = _parameter_tensor(
            hypers.get("angle_k"), angle_index.shape[0], "angle_k", "angles"
        )
        angle_theta0 = _parameter_tensor(
            hypers.get("angle_theta0"), angle_index.shape[0], "angle_theta0", "angles"
        )

        for name, values in (("bond_k", bond_k), ("angle_k", angle_k)):
            if bool(torch.any(values < 0.0)):
                raise ValueError(
                    f"HarmonicBonded {name} must be >= 0 (a Boltzmann-inverted "
                    f"force constant kB*T/Var[f] cannot be negative), got {values}."
                )
        if bool(torch.any(bond_r0 < 0.0)):
            raise ValueError(
                f"HarmonicBonded bond_r0 must be >= 0, got {bond_r0.tolist()}."
            )
        if bool(torch.any(angle_theta0 < 0.0)) or bool(
            torch.any(angle_theta0 > torch.pi)
        ):
            raise ValueError(
                "HarmonicBonded angle_theta0 must lie in [0, pi] radian, got "
                f"{angle_theta0.tolist()}. (Degrees are a common mistake.)"
            )

        # A bond needs two distinct beads; an angle needs three, and in
        # particular i != k (i == k is a zero-length feature, not an angle).
        for row in bond_index.tolist():
            if row[0] == row[1]:
                raise ValueError(f"HarmonicBonded bonds contains a self-bond {row}.")
        for row in angle_index.tolist():
            if row[0] == row[1] or row[1] == row[2] or row[0] == row[2]:
                raise ValueError(
                    f"HarmonicBonded angles entry {row} repeats a bead; an angle "
                    "needs three distinct beads (the vertex is the middle index)."
                )

        n_beads = 0
        if bond_index.numel() > 0:
            n_beads = max(n_beads, int(bond_index.max()) + 1)
        if angle_index.numel() > 0:
            n_beads = max(n_beads, int(angle_index.max()) + 1)
        self.n_beads = n_beads

        self.register_buffer("bond_index", bond_index)
        self.register_buffer("bond_k", bond_k)
        self.register_buffer("bond_r0", bond_r0)
        self.register_buffer("angle_index", angle_index)
        self.register_buffer("angle_k", angle_k)
        self.register_buffer("angle_theta0", angle_theta0)

    def restart(self, dataset_info: DatasetInfo) -> "HarmonicBonded":
        for target_name, target_info in dataset_info.targets.items():
            if not self.is_valid_target(target_name, target_info):
                raise ValueError(
                    f"HarmonicBonded model does not support target {target_name}."
                )
        self.dataset_info = self.dataset_info.union(dataset_info)
        return self

    def remove_output(self, target_name: str) -> None:
        """
        Remove a previously registered output target.

        :param target_name: Name of the target to remove.
        """
        self.outputs.pop(target_name, None)
        self.dataset_info.targets.pop(target_name, None)

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
        # The topology and the force constants come from the hypers, which the
        # checkpoint carries and from which this module has already been rebuilt.
        # A state dict written by a build that lacked one of these buffers is
        # therefore restored exactly by keeping the reconstructed value, rather
        # than by failing on a missing key.
        for name in (
            "bond_index",
            "bond_k",
            "bond_r0",
            "angle_index",
            "angle_k",
            "angle_theta0",
        ):
            if prefix + name not in state_dict:
                state_dict[prefix + name] = getattr(self, name).clone()
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
        positions = self._concatenated_positions(systems)
        bond_idx, angle_idx = self._global_indices(len(systems), positions.device)
        cell_rows, has_pbc, pbc_rows = self._cell_info(systems)
        e_nodes = self._per_atom_energy(
            positions, bond_idx, angle_idx, cell_rows, pbc_rows, has_pbc
        )

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
    # Same contract as SoftCore: remove_additive reads this flag and calls
    # analytic_contribution() instead of evaluate_model()'s autograd, which is
    # exact here (the harmonic form has a closed-form force) and lets the
    # additive force subtraction run inside fork-based DataLoader workers.
    # Deployment forces still come from the PET model's autograd of the total
    # energy, so `forward` stays energy-only.
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
        positions = self._concatenated_positions(systems)
        bond_idx, angle_idx = self._global_indices(len(systems), device)
        cell_rows, has_pbc, pbc_rows = self._cell_info(systems)
        e_nodes = self._per_atom_energy(
            positions, bond_idx, angle_idx, cell_rows, pbc_rows, has_pbc
        )
        grad_nodes = self._position_gradient(
            positions, bond_idx, angle_idx, cell_rows, pbc_rows, has_pbc
        )

        sys_col = torch.cat(
            [
                torch.full((system.positions.shape[0],), i, dtype=torch.int64)
                for i, system in enumerate(systems)
            ]
        ).to(device)
        atom_col = torch.cat(
            [
                torch.arange(system.positions.shape[0], dtype=torch.int64)
                for system in systems
            ]
        ).to(device)

        grad_block = TensorBlock(
            values=grad_nodes.unsqueeze(-1),  # [n_atoms, 3, 1]
            samples=Labels(["sample", "atom"], torch.stack([sys_col, atom_col], dim=1)),
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
                samples=Labels(
                    ["system", "atom"],
                    torch.tensor(sample_values, device=device),
                    assume_unique=True,
                ),
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
            blk = tmap.block().copy(deep=False)
            blk.add_gradient("positions", grad_block)
            out[target_key] = TensorMap(keys=tmap.keys, blocks=[blk])
        return out

    # --- internals -----------------------------------------------------------

    def _concatenated_positions(self, systems: List[System]) -> torch.Tensor:
        """Positions of all systems stacked along the atom axis, with the bead
        count of every system checked against the topology.

        The check is a hard error rather than a skip: a topology that does not
        describe the system means the prior would be evaluated on the wrong
        beads, which is silent and unrecoverable downstream.
        """
        positions_list: List[torch.Tensor] = []
        for system in systems:
            n_atoms = system.positions.shape[0]
            if self.n_beads > 0 and n_atoms != self.n_beads:
                raise ValueError(
                    "HarmonicBonded topology describes "
                    + str(self.n_beads)
                    + " beads but this system has "
                    + str(n_atoms)
                    + " atoms: the bonded prior cannot be applied. Fix "
                    + "harmonic_bonded_bonds / harmonic_bonded_angles to match "
                    + "the system."
                )
            positions_list.append(system.positions)
        return torch.cat(positions_list, dim=0)

    def _global_indices(
        self, n_systems: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Replicate the per-molecule bond/angle index lists over the batch,
        offset into the concatenated atom list. Every system holds exactly
        ``n_beads`` beads (enforced above), so the offset is ``i * n_beads``."""
        bond_index = self.bond_index.to(device)
        angle_index = self.angle_index.to(device)
        if self.n_beads == 0 or n_systems == 1:
            return bond_index, angle_index
        offsets = (
            torch.arange(n_systems, dtype=torch.int64, device=device) * self.n_beads
        )
        bond_idx = (bond_index.unsqueeze(0) + offsets.reshape(-1, 1, 1)).reshape(-1, 2)
        angle_idx = (angle_index.unsqueeze(0) + offsets.reshape(-1, 1, 1)).reshape(
            -1, 3
        )
        return bond_idx, angle_idx

    def _cell_info(
        self, systems: List[System]
    ) -> Tuple[torch.Tensor, bool, torch.Tensor]:
        """Per-ATOM cell rows and whether ANY system is periodic.

        Returned per atom (rather than per system) so a bonded term can look up its
        cell with the same index tensor it uses for positions; every system holds
        exactly ``n_beads`` beads, so this is a plain repeat.
        """
        cells: List[torch.Tensor] = []
        pbcs: List[torch.Tensor] = []
        has_pbc = False
        for system in systems:
            n_atoms = system.positions.shape[0]
            cells.append(system.cell.unsqueeze(0).expand(n_atoms, 3, 3))
            pbcs.append(system.pbc.unsqueeze(0).expand(n_atoms, 3))
            if bool(system.pbc.any()):
                has_pbc = True
        return torch.cat(cells, dim=0), has_pbc, torch.cat(pbcs, dim=0)

    def _minimum_image(
        self,
        d: torch.Tensor,
        cell_rows: torch.Tensor,
        pbc_rows: torch.Tensor,
        has_pbc: bool,
    ) -> torch.Tensor:
        """Minimum-image a displacement, per periodic direction.

        Skipped entirely when nothing is periodic, so the intended non-periodic CG
        case is bit-identical to a raw coordinate difference. Without this a bonded
        term evaluated on WRAPPED coordinates silently reads a bond length of order
        the cell edge -- e.g. a 1.5 A bond across a 20 A box scores ~1445 eV instead
        of 0, with no error raised anywhere.
        """
        if not has_pbc:
            return d
        inv = torch.linalg.inv(cell_rows)
        frac = torch.bmm(d.unsqueeze(1), inv).squeeze(1)
        shift = torch.round(frac)
        shift = torch.where(pbc_rows, shift, torch.zeros_like(shift))
        return d - torch.bmm(shift.unsqueeze(1), cell_rows).squeeze(1)

    def _bond_terms(
        self,
        positions: torch.Tensor,
        bond_idx: torch.Tensor,
        cell_rows: torch.Tensor,
        pbc_rows: torch.Tensor,
        has_pbc: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per bond: energy, ``r - r0``, the i->j displacement and its norm."""
        n_repeat = bond_idx.shape[0] // max(self.bond_index.shape[0], 1)
        k = self.bond_k.to(positions.dtype).repeat([n_repeat])
        r0 = self.bond_r0.to(positions.dtype).repeat([n_repeat])
        d = positions[bond_idx[:, 1]] - positions[bond_idx[:, 0]]
        d = self._minimum_image(
            d, cell_rows[bond_idx[:, 0]], pbc_rows[bond_idx[:, 0]], has_pbc
        )
        r = torch.sqrt(torch.clamp((d * d).sum(dim=1), min=1e-30))
        dr = r - r0
        return 0.5 * k * dr * dr, k * dr, d, r

    def _angle_terms(
        self,
        positions: torch.Tensor,
        angle_idx: torch.Tensor,
        cell_rows: torch.Tensor,
        pbc_rows: torch.Tensor,
        has_pbc: bool,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Per angle: energy, ``k (theta - theta0)``, the two arm unit vectors,
        their lengths, ``cos theta`` and ``sin theta``.

        ``theta = atan2(|a x b|, a . b)``: exact and well conditioned across the
        whole range, unlike ``acos`` of a clamped cosine, which loses all
        resolution near 0 and pi and made the energy bit-constant there (see the
        warning in the class docstring). ``sin theta`` is taken from the SAME
        cross-product norm, so it is consistent with the theta actually used and
        the analytic derivative is the derivative of what ``forward`` computes.
        """
        n_repeat = angle_idx.shape[0] // max(self.angle_index.shape[0], 1)
        k = self.angle_k.to(positions.dtype).repeat([n_repeat])
        theta0 = self.angle_theta0.to(positions.dtype).repeat([n_repeat])
        a = positions[angle_idx[:, 0]] - positions[angle_idx[:, 1]]
        b = positions[angle_idx[:, 2]] - positions[angle_idx[:, 1]]
        vertex_cell = cell_rows[angle_idx[:, 1]]
        vertex_pbc = pbc_rows[angle_idx[:, 1]]
        a = self._minimum_image(a, vertex_cell, vertex_pbc, has_pbc)
        b = self._minimum_image(b, vertex_cell, vertex_pbc, has_pbc)
        ra = torch.sqrt(torch.clamp((a * a).sum(dim=1), min=1e-30))
        rb = torch.sqrt(torch.clamp((b * b).sum(dim=1), min=1e-30))
        a_hat = a / ra.unsqueeze(-1)
        b_hat = b / rb.unsqueeze(-1)
        cos_theta = (a_hat * b_hat).sum(dim=1)
        cross = torch.stack(
            [
                a_hat[:, 1] * b_hat[:, 2] - a_hat[:, 2] * b_hat[:, 1],
                a_hat[:, 2] * b_hat[:, 0] - a_hat[:, 0] * b_hat[:, 2],
                a_hat[:, 0] * b_hat[:, 1] - a_hat[:, 1] * b_hat[:, 0],
            ],
            dim=1,
        )
        sin_raw = torch.sqrt(torch.clamp((cross * cross).sum(dim=1), min=1e-30))
        theta = torch.atan2(sin_raw, cos_theta)
        # floor only the 1/sin intermediate; 1e-12 rad is unreachable thermally and
        # is the SAME window in float32 and float64
        sin_theta = torch.clamp(sin_raw, min=1e-12)
        dtheta = theta - theta0
        return (
            0.5 * k * dtheta * dtheta,
            k * dtheta,
            a_hat,
            b_hat,
            ra,
            rb,
            cos_theta,
            sin_theta,
        )

    def _per_atom_energy(
        self,
        positions: torch.Tensor,
        bond_idx: torch.Tensor,
        angle_idx: torch.Tensor,
        cell_rows: torch.Tensor,
        pbc_rows: torch.Tensor,
        has_pbc: bool,
    ) -> torch.Tensor:
        """Prior energy decomposed over beads: a bond term is split evenly over
        its two beads, an angle term evenly over its three. The decomposition is
        arbitrary (a bonded term belongs to no single bead) but its sum is the
        exact total, which is what the per-system output needs."""
        n_total = positions.shape[0]
        e_nodes = torch.zeros(n_total, dtype=positions.dtype, device=positions.device)
        e_bond, _, _, _ = self._bond_terms(
            positions, bond_idx, cell_rows, pbc_rows, has_pbc
        )
        half = 0.5 * e_bond
        e_nodes = e_nodes.index_add(0, bond_idx[:, 0], half)
        e_nodes = e_nodes.index_add(0, bond_idx[:, 1], half)
        e_angle, _, _, _, _, _, _, _ = self._angle_terms(
            positions, angle_idx, cell_rows, pbc_rows, has_pbc
        )
        third = e_angle / 3.0
        e_nodes = e_nodes.index_add(0, angle_idx[:, 0], third)
        e_nodes = e_nodes.index_add(0, angle_idx[:, 1], third)
        e_nodes = e_nodes.index_add(0, angle_idx[:, 2], third)
        return e_nodes

    def _position_gradient(
        self,
        positions: torch.Tensor,
        bond_idx: torch.Tensor,
        angle_idx: torch.Tensor,
        cell_rows: torch.Tensor,
        pbc_rows: torch.Tensor,
        has_pbc: bool,
    ) -> torch.Tensor:
        """dE/dr for every bead, in closed form.

        Bond, with ``d = r_j - r_i`` and ``u = d / r``::

            dU/dr_j = k (r - r0) u,        dU/dr_i = -dU/dr_j

        Angle, with ``a = r_i - r_j``, ``b = r_k - r_j``, ``c = cos theta``::

            dtheta/dr_i = (c a_hat - b_hat) / (sin theta * |a|)
            dtheta/dr_k = (c b_hat - a_hat) / (sin theta * |b|)
            dtheta/dr_j = -(dtheta/dr_i + dtheta/dr_k)

        Building the vertex gradient as minus the sum of the other two makes each
        term's own contribution sum to exactly zero, so translational invariance
        holds up to the rounding of the per-bead accumulation only. The same
        construction makes ``a x G_i + b x G_k`` cancel, i.e. zero net torque.
        """
        grad = torch.zeros_like(positions)

        _, dudr, d, r = self._bond_terms(
            positions, bond_idx, cell_rows, pbc_rows, has_pbc
        )
        g_bond = dudr.unsqueeze(-1) * (d / r.unsqueeze(-1))
        grad = grad.index_add(0, bond_idx[:, 1], g_bond)
        grad = grad.index_add(0, bond_idx[:, 0], -g_bond)

        _, dudtheta, a_hat, b_hat, ra, rb, cos_theta, sin_theta = self._angle_terms(
            positions, angle_idx, cell_rows, pbc_rows, has_pbc
        )
        # |c a_hat - b_hat| == |sin theta| identically, so the 1/sin factor cancels
        # against the numerator and the gradient MAGNITUDE is ~1/|a|; the floor on
        # sin only bounds the intermediate. It does diverge for a coincident triple,
        # which is intended (see the class docstring).
        prefactor_i = (dudtheta / (sin_theta * ra)).unsqueeze(-1)
        prefactor_k = (dudtheta / (sin_theta * rb)).unsqueeze(-1)
        cos_col = cos_theta.unsqueeze(-1)
        f_i = prefactor_i * (cos_col * a_hat - b_hat)
        f_k = prefactor_k * (cos_col * b_hat - a_hat)
        grad = grad.index_add(0, angle_idx[:, 0], f_i)
        grad = grad.index_add(0, angle_idx[:, 2], f_k)
        grad = grad.index_add(0, angle_idx[:, 1], -(f_i + f_k))
        return grad

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        # Bonded terms are TOPOLOGICAL: the features are read from the bead
        # indices directly, so this prior needs no neighbor list at all.
        options: List[NeighborListOptions] = []
        return options

    @staticmethod
    def is_valid_target(target_name: str, target_info: TargetInfo) -> bool:
        if target_info.quantity != "energy":
            logging.debug(f"HarmonicBonded skips non-energy target {target_name}.")
            return False
        if not target_info.is_scalar:
            logging.debug(f"HarmonicBonded skips non-scalar target {target_name}.")
            return False
        if len(target_info.layout.block(0).properties) > 1:
            logging.debug(f"HarmonicBonded skips multi-property target {target_name}.")
            return False
        if target_info.unit != "eV":
            logging.debug(f"HarmonicBonded skips non-eV target {target_name}.")
            return False
        return True


def _index_tensor(value, width: int, name: str) -> torch.Tensor:
    """Normalize a list of 0-based bead index tuples to an ``(n, width)`` int64
    tensor. ``None`` and ``[]`` both mean "this term is absent"."""
    if value is None:
        return torch.zeros((0, width), dtype=torch.int64)
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"HarmonicBonded {name} must be a list of {width}-element index "
            f"lists, got {value}."
        )
    rows: List[List[int]] = []
    for entry in value:
        if len(entry) != width:
            raise ValueError(
                f"HarmonicBonded {name} entries must have {width} bead indices, "
                f"got {list(entry)}."
            )
        row = [int(index) for index in entry]
        for index in row:
            if index < 0:
                raise ValueError(
                    f"HarmonicBonded {name} contains a negative bead index in "
                    f"{row}; indices are 0-based."
                )
        rows.append(row)
    if len(rows) == 0:
        return torch.zeros((0, width), dtype=torch.int64)
    return torch.tensor(rows, dtype=torch.int64)


def _parameter_tensor(value, n_expected: int, name: str, owner: str) -> torch.Tensor:
    """Normalize a per-feature parameter list to a float64 tensor of length
    ``n_expected``, the number of features declared in ``owner``."""
    if value is None:
        values: List[float] = []
    elif isinstance(value, (list, tuple)):
        values = [float(item) for item in value]
    else:
        raise ValueError(
            f"HarmonicBonded {name} must be a list of floats, got {value}."
        )
    if len(values) != n_expected:
        raise ValueError(
            f"HarmonicBonded {name} has {len(values)} entries but {owner} "
            f"declares {n_expected} features; they must agree one-to-one."
        )
    if len(values) == 0:
        return torch.zeros((0,), dtype=torch.float64)
    return torch.tensor(values, dtype=torch.float64)
