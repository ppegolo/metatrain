"""NEP-backed model predicting per-atom GLE drift matrices.

This wires the invariant NEP descriptors to the drift head in
:mod:`metatrain.experimental.nep.modules.gle`: the descriptor half of a NEP
potential (no energy readout, no ZBL, no charges) followed by the scalar
readout and the Cartesian basis contraction.

The forward returns raw tensors rather than ``TensorMap``s, and this model is
deliberately *not* registered as a metatrain target.  The drift head is
trained against trajectory-level losses (Kalman marginal likelihood along
paths, kernel matching against binned autocorrelation functions), none of
which is expressible as metatrain's per-structure target loss, so a
first-class target would only buy collate and metrics machinery that cannot
be used.  This standalone module plus a custom training loop is the intended
architecture, not a stopgap; metatrain's role in this project is PMF training
only.  The configuration still comes from the target section of the yaml, via
:meth:`GLEDriftConfig.from_options`.

The integrator consumes ``L_blocks`` (and forms ``B = sqrt(2 kB T) L``
itself) while ``A`` is provided for inspection and for losses.
"""

# Export layout (metatomic), documented not implemented
# ----------------------------------------------------
# If this model is ever exported through metatomic — to run coarse-grained MD
# in an external engine rather than with the custom integrator — the natural
# layout for `L_blocks` is:
#
#   samples    = (system, atom)
#   components = two Cartesian axes, [3, 3]
#   properties = the flattened lower-triangular block index, n_blocks
#
# and *not* a single flat 3(n+1) x 3(n+1) Cartesian object.  The channel index
# (p, s^1, ..., s^n) is not a spatial axis: flattening it into the Cartesian
# axes would make metatomic treat it as one, and would push the unflattening
# convention onto every consumer of the exported model.

import copy
from typing import Dict, List, Optional

import torch
from metatomic.torch import NeighborListOptions, System
from torchnep.nep_descriptor import descriptor_type_ids_batched, edge_vectors_batched

from .documentation import ModelHypers
from .model import _build_nep_parameters
from .modules.gle import (
    GLEDriftConfig,
    GLEDriftHead,
    build_tensor_basis,
    edge_classes_from_bonds,
)


class GLEDriftModel(torch.nn.Module):
    """Per-atom GLE drift matrices from NEP descriptors.

    :param hypers: NEP model hyperparameters; only the descriptor ones are
        used (cutoffs, ``n_max``, ``basis_size``, ``l_max``, ``neurons``,
        ``seed``).
    :param atomic_types: Atomic numbers the model is defined for.
    :param config: Configuration of the drift head.
    """

    def __init__(
        self,
        hypers: ModelHypers,
        atomic_types: List[int],
        config: Optional[GLEDriftConfig] = None,
    ):
        super().__init__()
        self.config = config if config is not None else GLEDriftConfig()
        self.atomic_types = list(atomic_types)
        # the drift head only uses the descriptor part of a NEP model, so the
        # readout-related hypers are pinned to the plain-energy values
        self.hypers: ModelHypers = copy.deepcopy(hypers)
        self.hypers["model_type"] = 0
        self.hypers["charge_mode"] = 0

        parameters = _build_nep_parameters(self.hypers, self.atomic_types)
        self.meta = parameters.meta
        self.pbc_list: List[bool] = [True, True, True]

        # the descriptor half of a NEP potential: coefficients and
        # normalisation, without any energy readout
        self.descriptor_coefficients = torch.nn.Parameter(parameters.c.clone())
        self.register_buffer("q_scaler", parameters.q_scaler.clone())
        self.register_buffer(
            "rc_radial", torch.tensor(parameters.rc_radial, dtype=torch.float64)
        )
        self.register_buffer(
            "rc_angular", torch.tensor(parameters.rc_angular, dtype=torch.float64)
        )

        self.cutoff_radial = float(parameters.rc_radial[0])
        self.cutoff_angular = float(parameters.rc_angular[0])
        if self.config.r_cut > self.cutoff_radial:
            raise ValueError(
                f"the drift head cutoff (r_cut={self.config.r_cut}) must not "
                "exceed the NEP radial cutoff "
                f"(cutoff_radial={self.cutoff_radial})"
            )

        self.head = GLEDriftHead(
            n_features=parameters.dim,
            n_types=len(self.atomic_types),
            n_neurons=int(self.hypers["neurons"]),
            config=self.config,
        )

        lookup = torch.full((max(self.atomic_types) + 1,), -1, dtype=torch.long)
        for index, atomic_type in enumerate(self.atomic_types):
            lookup[atomic_type] = index
        self.register_buffer("species_to_type_id", lookup, persistent=False)

        self.nl_options_radial = NeighborListOptions(
            cutoff=self.cutoff_radial, full_list=True, strict=True
        )
        self.nl_options_angular = NeighborListOptions(
            cutoff=self.cutoff_angular, full_list=True, strict=True
        )

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        """Neighbor lists needed by the descriptors and by the tensor basis.

        The basis reuses the radial neighbor list and masks it at ``r_cut``,
        which is why ``r_cut`` may not exceed the radial cutoff.

        :return: The requested neighbor list options.
        """
        return [self.nl_options_radial, self.nl_options_angular]

    def forward(
        self,
        systems: List[System],
        bonds: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict the drift factor of every atom in every system.

        :param systems: Systems to predict for, with the requested neighbor
            lists already attached.
        :param bonds: Optional per-system ``[B, 2]`` tensors of bonded atom
            pairs, required when the head uses more than one neighbor class.
        :return: ``L_blocks`` of shape ``[n_atoms, n + 1, n + 1, 3, 3]`` and
            ``A`` of shape ``[n_atoms, 3(n+1), 3(n+1)]``, with the atoms of
            all systems concatenated.
        """
        if self.config.n_classes > 1 and bonds is None:
            raise ValueError(
                f"the drift head is configured with n_classes="
                f"{self.config.n_classes}, which needs a topology: pass the "
                "bonded pairs of every system as `bonds`, or configure "
                "n_classes=1"
            )

        (
            positions,
            cells,
            type_ids,
            radial_edges,
            angular_edges,
            radial_edge_batch,
            angular_edge_batch,
            radial_shifts,
            angular_shifts,
            edge_class,
        ) = self._collate(systems, bonds)

        descriptors = descriptor_type_ids_batched(
            self.meta,
            self.pbc_list,
            self.descriptor_coefficients,
            self.q_scaler,
            self.rc_radial,
            self.rc_angular,
            positions,
            cells,
            type_ids,
            radial_edges,
            angular_edges,
            radial_edge_batch,
            angular_edge_batch,
            filter_edges=True,
            radial_shifts=radial_shifts,
            angular_shifts=angular_shifts,
        )

        edge_vectors = edge_vectors_batched(
            positions,
            cells,
            radial_edge_batch,
            radial_edges,
            self.pbc_list,
            shifts=radial_shifts,
        )
        basis = build_tensor_basis(
            n_slots=positions.shape[0],
            slot_index=radial_edges[:, 0],
            edge_vectors=edge_vectors,
            edge_class=edge_class,
            config=self.config,
        )

        return self.head(descriptors, type_ids, basis)

    def _collate(self, systems: List[System], bonds: Optional[List[torch.Tensor]]):
        """Concatenate systems, their neighbor lists and their topology."""
        device = systems[0].positions.device
        dtype = systems[0].positions.dtype

        positions_list: List[torch.Tensor] = []
        cells_list: List[torch.Tensor] = []
        type_ids_list: List[torch.Tensor] = []
        radial_edges_list: List[torch.Tensor] = []
        angular_edges_list: List[torch.Tensor] = []
        radial_batch_list: List[torch.Tensor] = []
        angular_batch_list: List[torch.Tensor] = []
        radial_shifts_list: List[torch.Tensor] = []
        angular_shifts_list: List[torch.Tensor] = []
        edge_class_list: List[torch.Tensor] = []

        offset = 0
        for i_system, system in enumerate(systems):
            n_atoms = len(system)
            positions_list.append(system.positions)
            cells_list.append(system.cell)
            types = system.types
            if bool((types < 0).any()) or bool(
                (types >= self.species_to_type_id.shape[0]).any()
            ):
                raise ValueError(
                    "this system contains atomic types the model does not know about"
                )
            type_ids = self.species_to_type_id[types]
            if bool((type_ids < 0).any()):
                raise ValueError(
                    "this system contains atomic types the model does not know about"
                )
            type_ids_list.append(type_ids)

            radial_nl = system.get_neighbor_list(self.nl_options_radial)
            radial_samples = radial_nl.samples.values.to(torch.long)
            radial_edges_list.append(radial_samples[:, :2] + offset)
            radial_shifts_list.append(radial_samples[:, 2:5])
            radial_batch_list.append(
                torch.full(
                    (radial_samples.shape[0],),
                    i_system,
                    dtype=torch.long,
                    device=device,
                )
            )
            # classes are computed per system, on its own atom indices
            system_bonds = (
                torch.zeros((0, 2), dtype=torch.long, device=device)
                if bonds is None
                else bonds[i_system].to(device)
            )
            edge_class_list.append(
                edge_classes_from_bonds(
                    radial_samples[:, :2], system_bonds, self.config.n_classes
                )
            )

            angular_nl = system.get_neighbor_list(self.nl_options_angular)
            angular_samples = angular_nl.samples.values.to(torch.long)
            angular_edges_list.append(angular_samples[:, :2] + offset)
            angular_shifts_list.append(angular_samples[:, 2:5])
            angular_batch_list.append(
                torch.full(
                    (angular_samples.shape[0],),
                    i_system,
                    dtype=torch.long,
                    device=device,
                )
            )

            offset += n_atoms

        return (
            torch.cat(positions_list),
            torch.stack(cells_list).to(dtype),
            torch.cat(type_ids_list),
            torch.cat(radial_edges_list),
            torch.cat(angular_edges_list),
            torch.cat(radial_batch_list),
            torch.cat(angular_batch_list),
            torch.cat(radial_shifts_list),
            torch.cat(angular_shifts_list),
            torch.cat(edge_class_list),
        )
