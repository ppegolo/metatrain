import logging
from typing import Dict, List, Optional

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, NeighborListOptions, System
from torchnep.zbl import zbl_pair_energy_raw

from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.sum_over_atoms import sum_over_atoms


class GPUMDZBL(torch.nn.Module):
    """GPUMD-flavoured universal ZBL repulsion as an additive model.

    Unlike :py:class:`metatrain.utils.additive.ZBL` (which follows the LAMMPS
    convention: polynomial switching, per-pair covalent-radius cutoffs), this
    model uses GPUMD's form — universal screening with a cosine switching
    function between ``rc_inner = 0.5 * rc_outer`` and ``rc_outer``, shared by
    all element pairs — so that it can be written verbatim into a GPUMD
    ``nep.txt`` file (``zbl`` keyword).

    As an additive model it is removed from the targets before the scaler and
    the composition model are fitted, matching GPUMD where the ZBL term is not
    affected by any energy normalisation.

    :param rc_outer: Outer cutoff of the cosine switching function, in Å.
    :param dataset_info: Information about the dataset, including target
        quantities and atomic types.
    """

    def __init__(self, rc_outer: float, dataset_info: DatasetInfo):
        super().__init__()

        if dataset_info.length_unit.lower() not in ("angstrom", ""):
            raise ValueError(
                "GPUMD ZBL only supports angstrom units, but a "
                f"{dataset_info.length_unit} unit was provided."
            )

        self.dataset_info = dataset_info
        self.atomic_types = sorted(dataset_info.atomic_types)
        self.rc_outer = float(rc_outer)
        self.rc_inner = 0.5 * self.rc_outer
        self.cutoff_radius = self.rc_outer

        self.outputs: Dict[str, ModelOutput] = {}
        for target_name, target_info in dataset_info.targets.items():
            if self.is_valid_target(target_name, target_info):
                self.outputs[target_name] = ModelOutput(
                    quantity=target_info.quantity,
                    unit=target_info.unit,
                    sample_kind="atom",
                    description=target_info.description,
                )

        self.register_buffer(
            "species_to_index",
            torch.full((max(self.atomic_types) + 1,), -1, dtype=torch.long),
        )
        for i, t in enumerate(self.atomic_types):
            self.species_to_index[t] = i
        self.register_buffer(
            "atomic_numbers", torch.tensor(self.atomic_types, dtype=torch.long)
        )

    def restart(self, dataset_info: DatasetInfo) -> "GPUMDZBL":
        """Restart the model with a new dataset info.

        :param dataset_info: New dataset information to be used.
        :return: The restarted model.
        """
        self.dataset_info = self.dataset_info.union(dataset_info)
        for target_name, target_info in dataset_info.targets.items():
            if target_name not in self.outputs and self.is_valid_target(
                target_name, target_info
            ):
                self.outputs[target_name] = ModelOutput(
                    quantity=target_info.quantity,
                    unit=target_info.unit,
                    sample_kind="atom",
                    description=target_info.description,
                )
        return self

    def remove_output(self, target_name: str) -> None:
        """Remove a previously registered output target.

        :param target_name: Name of the target to remove.
        """
        self.outputs.pop(target_name, None)
        self.dataset_info.targets.pop(target_name, None)

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.outputs

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        """Compute the per-atom GPUMD ZBL energies.

        :param systems: List of systems to calculate the ZBL energy for.
        :param outputs: Dictionary containing the model outputs.
        :param selected_atoms: Optional selection of atoms for which to compute
            the predictions.
        :return: A dictionary with the computed predictions for each system.
        """
        device = systems[0].positions.device
        dtype = systems[0].positions.dtype
        nl_options = self.requested_neighbor_lists()[0]

        rij_list: List[torch.Tensor] = []
        ti_list: List[torch.Tensor] = []
        tj_list: List[torch.Tensor] = []
        center_list: List[torch.Tensor] = []
        total_atoms = 0
        for system in systems:
            nl = system.get_neighbor_list(nl_options)
            samples = nl.samples.values.to(torch.long)
            rij_list.append(nl.values.reshape(-1, 3))
            ti_list.append(self.species_to_index[system.types[samples[:, 0]]])
            tj_list.append(self.species_to_index[system.types[samples[:, 1]]])
            center_list.append(samples[:, 0] + total_atoms)
            total_atoms += len(system)

        rij = torch.cat(rij_list)
        e_pair = zbl_pair_energy_raw(
            rij,
            torch.cat(ti_list),
            torch.cat(tj_list),
            False,
            self.rc_inner,
            self.rc_outer,
            self.atomic_numbers,
            len(self.atomic_types),
            None,
        )
        e_zbl_nodes = torch.zeros(total_atoms, dtype=dtype, device=device)
        # each directed edge contributes half to its center atom
        e_zbl_nodes.index_add_(0, torch.cat(center_list), 0.5 * e_pair)

        targets_out: Dict[str, TensorMap] = {}
        for target_key, target in outputs.items():
            sample_values: List[List[int]] = []
            for i_system, system in enumerate(systems):
                sample_values += [[i_system, i_atom] for i_atom in range(len(system))]

            block = TensorBlock(
                values=e_zbl_nodes.reshape(-1, 1),
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

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [
            NeighborListOptions(
                cutoff=self.rc_outer,
                full_list=True,
                strict=True,
            )
        ]

    @staticmethod
    def is_valid_target(target_name: str, target_info: TargetInfo) -> bool:
        """Finds if a :py:class:`TargetInfo` object is compatible with ZBL.

        :param target_name: The name of the target to be checked.
        :param target_info: The :py:class:`TargetInfo` object to be checked.
        :return: True if the target is compatible, False otherwise.
        """
        if target_info.quantity != "energy":
            logging.debug(
                f"GPUMD ZBL does not support target {target_name} since it is "
                "not an energy."
            )
            return False
        if not target_info.is_scalar:
            logging.debug(
                f"GPUMD ZBL does not support target {target_name} since it is "
                "not a scalar."
            )
            return False
        if len(target_info.layout.block(0).properties) > 1:
            logging.debug(
                f"GPUMD ZBL does not support target {target_name} since it has "
                "more than one property."
            )
            return False
        return True
