import dataclasses
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import ase.data
import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)
from torchnep import NepParameters, NepPotential, load_nep, write_nep
from torchnep.nep_descriptor import _descriptor_type_ids_batched
from torchnep.zbl import ZBLConfig

from metatrain.composition import CompositionModel
from metatrain.utils.abc import ModelInterface
from metatrain.utils.data import TargetInfo
from metatrain.utils.data.atom_pair_helpers import check_no_atom_pair_targets
from metatrain.utils.data.dataset import DatasetInfo
from metatrain.utils.dtype import dtype_to_str
from metatrain.utils.metadata import merge_metadata
from metatrain.utils.scaler import Scaler
from metatrain.utils.sum_over_atoms import sum_over_atoms

from . import checkpoints
from .documentation import ModelHypers
from .modules import GPUMDZBL


def _build_nep_parameters(
    hypers: ModelHypers, atomic_types: List[int]
) -> NepParameters:
    """Build randomly initialised ``NepParameters`` from metatrain hypers.

    Weight initialisation follows ``torchnep``'s ``nep.in`` path:
    ``ann ~ N(0, 0.1)``, ``c ~ N(0, 1)``, ``q_scaler = 1``.
    """
    symbols = tuple(ase.data.chemical_symbols[z] for z in atomic_types)
    num_types = len(atomic_types)

    # Note: ZBL is handled as a metatrain additive model (modules.GPUMDZBL),
    # not inside the NEP potential, so that it is excluded from target scaling
    # exactly as in GPUMD.  The potential itself is built without ZBL.
    placeholder = torch.zeros(1, dtype=torch.float64)
    params = NepParameters(
        version=int(hypers["version"]),
        model_type=0,
        symbols=symbols,
        atomic_numbers=tuple(atomic_types),
        rc_radial=tuple(float(hypers["cutoff_radial"]) for _ in range(num_types)),
        rc_angular=tuple(float(hypers["cutoff_angular"]) for _ in range(num_types)),
        mn_radial=100,
        mn_angular=20,
        n_max_radial=int(hypers["n_max_radial"]),
        n_max_angular=int(hypers["n_max_angular"]),
        basis_size_radial=int(hypers["basis_size_radial"]),
        basis_size_angular=int(hypers["basis_size_angular"]),
        l_max_3body=int(hypers["l_max_3body"]),
        l_max_4body=int(hypers["l_max_4body"]),
        l_max_5body=int(hypers["l_max_5body"]),
        num_neurons1=int(hypers["neurons"]),
        ann=placeholder,
        c=placeholder,
        q_scaler=placeholder,
        zbl=None,
        charge_mode=int(hypers["charge_mode"]),
    )

    generator = torch.Generator().manual_seed(int(hypers["seed"]))
    ann = 0.1 * torch.randn(
        params.num_para_ann, generator=generator, dtype=torch.float64
    )
    c = torch.randn(
        params.num_para_descriptor, generator=generator, dtype=torch.float64
    )
    q_scaler = torch.ones(params.dim, dtype=torch.float64)
    return dataclasses.replace(params, ann=ann, c=c, q_scaler=q_scaler)


def _permute_type_params(params: NepParameters, perm: List[int]) -> NepParameters:
    """Reorder the types of ``params`` so that new type ``i`` is old ``perm[i]``.

    Reorders the per-type metadata, the per-type ANN blocks (NEP4/NEP5) and
    both type axes of the descriptor coefficients.  The shared NEP3 network,
    the global bias and ``q_scaler`` are type-independent.
    """
    num_types = params.num_types
    perm_t = torch.tensor(perm, dtype=torch.long)

    c_radial = params.c[: params.num_c_radial].reshape(
        params.dim_radial, params.basis_size_radial + 1, num_types, num_types
    )
    c_angular = params.c[params.num_c_radial :].reshape(
        params.n_max_angular + 1, params.basis_size_angular + 1, num_types, num_types
    )
    c_radial = c_radial.index_select(2, perm_t).index_select(3, perm_t)
    c_angular = c_angular.index_select(2, perm_t).index_select(3, perm_t)
    c = torch.cat([c_radial.reshape(-1), c_angular.reshape(-1)])

    ann = params.ann
    neurons = params.num_neurons1
    if params.charge_mode != 0:
        # qNEP layout: per-type [w0, b0, w1_energy|w1_charge|(w1_c6)],
        # then the global scalars [sqrt_epsilon_inf, b1]
        num_outputs = 3 if params.charge_mode == 3 else 2
        block = neurons * params.dim + neurons + neurons * num_outputs
        blocks = ann[: block * num_types].reshape(num_types, block)
        ann = torch.cat(
            [blocks.index_select(0, perm_t).reshape(-1), ann[block * num_types :]]
        )
    elif params.version in (4, 5):
        block = neurons * params.dim + 2 * neurons + (1 if params.version == 5 else 0)
        blocks = ann[: block * num_types].reshape(num_types, block)
        ann = torch.cat(
            [blocks.index_select(0, perm_t).reshape(-1), ann[block * num_types :]]
        )

    zbl = params.zbl
    if zbl is not None:
        zbl = dataclasses.replace(
            zbl, atomic_numbers=tuple(zbl.atomic_numbers[i] for i in perm)
        )

    return dataclasses.replace(
        params,
        symbols=tuple(params.symbols[i] for i in perm),
        atomic_numbers=tuple(params.atomic_numbers[i] for i in perm),
        rc_radial=tuple(params.rc_radial[i] for i in perm),
        rc_angular=tuple(params.rc_angular[i] for i in perm),
        ann=ann,
        c=c,
        zbl=zbl,
    )


def _load_nep_parameters(path: str, atomic_types: List[int]) -> NepParameters:
    """Load an existing ``nep.txt`` for fine-tuning.

    Validates that the file is a regular or NEP-Charge potential (flexible-ZBL
    files are not supported) whose elements exactly match the dataset, and
    reorders its types to the dataset's ``atomic_types`` order.
    """
    params = load_nep(path)

    if params.model_type != 0:
        raise NotImplementedError(
            "Only regular NEP potentials can be fine-tuned "
            "(dipole/polarizability/temperature models are not supported)."
        )
    if params.zbl is not None and params.zbl.enabled and params.zbl.flexible:
        raise NotImplementedError(
            "Flexible-ZBL NEP potentials cannot be fine-tuned yet."
        )
    if params.zbl is not None and params.zbl.enabled:
        if abs(params.zbl.rc_inner - 0.5 * params.zbl.rc_outer) > 1.0e-9:
            raise NotImplementedError(
                "Only ZBL with rc_inner = 0.5 * rc_outer is supported, "
                f"got rc_inner={params.zbl.rc_inner}, "
                f"rc_outer={params.zbl.rc_outer}."
            )
    if len(set(params.rc_radial)) != 1 or len(set(params.rc_angular)) != 1:
        raise NotImplementedError(
            "NEP potentials with per-type cutoffs cannot be fine-tuned yet."
        )

    if set(params.atomic_numbers) != set(atomic_types):
        raise ValueError(
            f"The dataset elements {sorted(atomic_types)} do not match the "
            f"elements of the NEP file {sorted(params.atomic_numbers)}. "
            "Fine-tuning requires the dataset to contain exactly the "
            "elements of the potential."
        )

    if list(params.atomic_numbers) != list(atomic_types):
        perm = [params.atomic_numbers.index(z) for z in atomic_types]
        params = _permute_type_params(params, perm)

    return params.to(dtype=torch.float64)


class NEP(ModelInterface[ModelHypers]):
    __checkpoint_version__ = 2
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float32, torch.float64]
    __default_metadata__ = ModelMetadata(
        references={
            "implementation": [
                "https://github.com/brucefan1983/GPUMD",
            ],
            "architecture": [
                "NEP: https://doi.org/10.1103/PhysRevB.104.104309",
            ],
        }
    )

    component_labels: Dict[str, List[List[Labels]]]  # torchscript needs this

    def __init__(self, hypers: ModelHypers, dataset_info: DatasetInfo) -> None:
        super().__init__(hypers, dataset_info, self.__default_metadata__)
        check_no_atom_pair_targets(dataset_info.targets, self.__class__.__name__)
        self.atomic_types = dataset_info.atomic_types

        # Pretrained potential loading: if nep_model is provided, load the
        # existing nep.txt instead of building from scratch.  The file's
        # architecture hyperparameters are synced back into self.hypers so
        # that checkpoints can be rebuilt without access to the file.
        nep_model = self.hypers.get("nep_model")
        self.loaded_nep = nep_model is not None
        if self.loaded_nep:
            params = _load_nep_parameters(str(nep_model), self.atomic_types)
            self.hypers["version"] = params.version
            self.hypers["cutoff_radial"] = float(params.rc_radial[0])
            self.hypers["cutoff_angular"] = float(params.rc_angular[0])
            self.hypers["n_max_radial"] = params.n_max_radial
            self.hypers["n_max_angular"] = params.n_max_angular
            self.hypers["basis_size_radial"] = params.basis_size_radial
            self.hypers["basis_size_angular"] = params.basis_size_angular
            self.hypers["l_max_3body"] = params.l_max_3body
            self.hypers["l_max_4body"] = params.l_max_4body
            self.hypers["l_max_5body"] = params.l_max_5body
            self.hypers["neurons"] = params.num_neurons1
            self.hypers["charge_mode"] = params.charge_mode
            self.hypers["zbl_outer_cutoff"] = (
                params.zbl.rc_outer
                if params.zbl is not None and params.zbl.enabled
                else None
            )
            # ZBL lives in a metatrain additive model, not in the potential
            params = dataclasses.replace(params, zbl=None)
        else:
            params = _build_nep_parameters(self.hypers, self.atomic_types)

        self.charge_mode = int(params.charge_mode)
        if self.charge_mode != 0 and params.version != 4:
            raise ValueError(
                "NEP-Charge (qNEP) requires `version: 4`, "
                f"got version {params.version}."
            )

        self.cutoff_radial = float(params.rc_radial[0])
        self.cutoff_angular = float(params.rc_angular[0])
        if self.cutoff_angular > self.cutoff_radial:
            raise ValueError(
                "The NEP angular cutoff must not be larger than the radial "
                f"cutoff, got cutoff_angular={self.cutoff_angular} > "
                f"cutoff_radial={self.cutoff_radial}."
            )

        self.nl_options_radial = NeighborListOptions(
            cutoff=self.cutoff_radial,
            full_list=True,
            strict=True,
        )
        self.nl_options_angular = NeighborListOptions(
            cutoff=self.cutoff_angular,
            full_list=True,
            strict=True,
        )
        self.targets_keys = list(dataset_info.targets.keys())[0]

        self.potential = NepPotential(params, train_q_scaler=False)

        # Lookup table from atomic numbers to NEP type indices (0..num_types-1)
        max_z = max(max(self.atomic_types), 0) + 1
        lookup = torch.full((max_z + 1,), -1, dtype=torch.long)
        for index, z in enumerate(self.atomic_types):
            lookup[z] = index
        self.register_buffer("species_to_type_id", lookup, persistent=False)

        self.scaler = Scaler(hypers={}, dataset_info=dataset_info)
        self.outputs: Dict[str, ModelOutput] = {}
        self.single_label = Labels.single()

        self.key_labels: Dict[str, Labels] = {}
        self.component_labels: Dict[str, List[List[Labels]]] = {}
        self.property_labels: Dict[str, List[Labels]] = {}
        for target_name, target in dataset_info.targets.items():
            self._add_output(target_name, target)

        composition_model = CompositionModel.from_valid_targets(
            dataset_info, self.atomic_types
        )
        additive_models: List[torch.nn.Module] = [composition_model]
        zbl_outer = self.hypers.get("zbl_outer_cutoff")
        if zbl_outer is not None:
            additive_models.append(GPUMDZBL(float(zbl_outer), dataset_info))
        self.additive_models = torch.nn.ModuleList(additive_models)

    def _add_output(self, target_name: str, target: TargetInfo) -> None:
        if not target.is_scalar:
            raise ValueError("The NEP architecture can only predict scalars.")
        self.key_labels[target_name] = target.layout.keys
        self.component_labels[target_name] = [
            block.components for block in target.layout.blocks()
        ]
        self.property_labels[target_name] = [
            block.properties for block in target.layout.blocks()
        ]
        self.outputs[target_name] = ModelOutput(
            quantity=target.quantity,
            unit=target.unit,
            sample_kind="atom",
        )

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [self.nl_options_radial, self.nl_options_angular]

    def _collate_systems(
        self, systems: List[System]
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Concatenate systems and their neighbor lists into flat batch tensors.

        Returns ``(positions, cells, type_ids, batch_index, atom_index,
        radial_edges, angular_edges, radial_edge_batch, angular_edge_batch,
        radial_shifts, angular_shifts)`` with edge atom indices offset into the
        concatenated atom axis.
        """
        device = systems[0].positions.device

        positions_list: List[torch.Tensor] = []
        cells_list: List[torch.Tensor] = []
        type_ids_list: List[torch.Tensor] = []
        batch_index_list: List[torch.Tensor] = []
        atom_index_list: List[torch.Tensor] = []
        radial_edges_list: List[torch.Tensor] = []
        angular_edges_list: List[torch.Tensor] = []
        radial_edge_batch_list: List[torch.Tensor] = []
        angular_edge_batch_list: List[torch.Tensor] = []
        radial_shifts_list: List[torch.Tensor] = []
        angular_shifts_list: List[torch.Tensor] = []

        offset = 0
        for i_system, system in enumerate(systems):
            n_atoms = len(system)
            positions_list.append(system.positions)
            cells_list.append(system.cell)
            type_ids_list.append(self.species_to_type_id[system.types])
            batch_index_list.append(
                torch.full((n_atoms,), i_system, dtype=torch.long, device=device)
            )
            atom_index_list.append(
                torch.arange(n_atoms, dtype=torch.long, device=device)
            )

            radial_nl = system.get_neighbor_list(self.nl_options_radial)
            radial_samples = radial_nl.samples.values.to(torch.long)
            radial_edges_list.append(radial_samples[:, :2] + offset)
            radial_shifts_list.append(radial_samples[:, 2:5])
            radial_edge_batch_list.append(
                torch.full(
                    (radial_samples.shape[0],),
                    i_system,
                    dtype=torch.long,
                    device=device,
                )
            )

            angular_nl = system.get_neighbor_list(self.nl_options_angular)
            angular_samples = angular_nl.samples.values.to(torch.long)
            angular_edges_list.append(angular_samples[:, :2] + offset)
            angular_shifts_list.append(angular_samples[:, 2:5])
            angular_edge_batch_list.append(
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
            torch.stack(cells_list),
            torch.cat(type_ids_list),
            torch.cat(batch_index_list),
            torch.cat(atom_index_list),
            torch.cat(radial_edges_list),
            torch.cat(angular_edges_list),
            torch.cat(radial_edge_batch_list),
            torch.cat(angular_edge_batch_list),
            torch.cat(radial_shifts_list),
            torch.cat(angular_shifts_list),
        )

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if len(outputs) == 0:
            return {}

        device = systems[0].positions.device

        if self.single_label.values.device != device:
            self.single_label = self.single_label.to(device)
            self.key_labels = {
                output_name: label.to(device)
                for output_name, label in self.key_labels.items()
            }
            self.component_labels = {
                output_name: [
                    [labels.to(device) for labels in components_block]
                    for components_block in components_tmap
                ]
                for output_name, components_tmap in self.component_labels.items()
            }
            self.property_labels = {
                output_name: [labels.to(device) for labels in properties_tmap]
                for output_name, properties_tmap in self.property_labels.items()
            }

        return_dict: Dict[str, TensorMap] = {}

        (
            positions,
            cells,
            type_ids,
            batch_index,
            atom_index,
            radial_edges,
            angular_edges,
            radial_edge_batch,
            angular_edge_batch,
            radial_shifts,
            angular_shifts,
        ) = self._collate_systems(systems)

        if self.charge_mode != 0:
            # the Ewald sum requires a periodic cell
            cell_norms = cells.abs().sum(dim=1).sum(dim=1)
            if bool((cell_norms == 0.0).any()):
                raise ValueError(
                    "NEP-Charge (qNEP) requires periodic systems: "
                    "found a system with a zero cell."
                )

        atom_energies = self.potential.atom_energies_batched(
            positions,
            cells,
            type_ids,
            batch_index,
            radial_edges,
            angular_edges,
            radial_edge_batch,
            angular_edge_batch,
            len(systems),
            filter_edges=True,
            radial_shifts=radial_shifts,
            angular_shifts=angular_shifts,
        )

        atomic_properties: Dict[str, TensorMap] = {}
        blocks: List[TensorBlock] = []

        values = torch.stack([batch_index, atom_index], dim=0).transpose(0, 1)
        sample_labels = Labels(names=["system", "atom"], values=values.to(device))

        blocks.append(
            TensorBlock(
                values=atom_energies.unsqueeze(-1),
                samples=sample_labels,
                components=self.component_labels[self.targets_keys][0],
                properties=self.property_labels[self.targets_keys][0].to(device),
            )
        )

        atomic_properties[self.targets_keys] = TensorMap(
            self.key_labels[self.targets_keys].to(device), blocks
        )

        if selected_atoms is not None:
            for output_name, tmap in atomic_properties.items():
                atomic_properties[output_name] = mts.slice(
                    tmap, axis="samples", selection=selected_atoms
                )

        for output_name, atomic_property in atomic_properties.items():
            if outputs[output_name].sample_kind == "atom":
                return_dict[output_name] = atomic_property
            else:
                # sum the atomic property to get the total property
                return_dict[output_name] = sum_over_atoms(atomic_property)

        if not self.training:
            # at evaluation, we also introduce the scaler and additive contributions
            return_dict = self.scaler(
                systems,
                return_dict,
                selected_atoms=selected_atoms,
                use_per_target_scales=True,
                use_per_property_scales=True,
            )
            for additive_model in self.additive_models:
                outputs_for_additive_model: Dict[str, ModelOutput] = {}
                for name, output in outputs.items():
                    if name in additive_model.outputs:
                        outputs_for_additive_model[name] = output
                additive_contributions = additive_model(
                    systems,
                    outputs_for_additive_model,
                    selected_atoms,
                )
                for name in additive_contributions:
                    return_dict[name] = mts.add(
                        return_dict[name],
                        additive_contributions[name],
                    )

        return return_dict

    @torch.jit.unused
    def descriptor_extrema(
        self, systems: List[System]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-dimension min/max of the un-normalised descriptors of ``systems``.

        Used by the trainer to compute the GPUMD-style ``q_scaler``
        normalisation before training starts.
        """
        with torch.no_grad():
            (
                positions,
                cells,
                type_ids,
                _batch_index,
                _atom_index,
                radial_edges,
                angular_edges,
                radial_edge_batch,
                angular_edge_batch,
                radial_shifts,
                angular_shifts,
            ) = self._collate_systems(systems)
            potential = self.potential
            ones = torch.ones(
                potential.meta.dim, dtype=positions.dtype, device=positions.device
            )
            q = _descriptor_type_ids_batched(
                potential.meta,
                potential.pbc_list,
                potential.c.detach(),
                ones,
                potential.rc_radial,
                potential.rc_angular,
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
        return q.amin(dim=0), q.amax(dim=0)

    @torch.jit.unused
    def get_fixed_composition_weights(self) -> Dict[str, Dict[int, float]]:
        """Composition weights to fix during training.

        A loaded ``nep.txt`` potential already predicts total energies, so its
        composition baseline must stay zero instead of being fitted from data.
        """
        if not self.loaded_nep:
            return {}
        return {self.targets_keys: {z: 0.0 for z in self.atomic_types}}

    @torch.jit.unused
    def get_fixed_scaling_weights(self) -> Dict[str, float]:
        """Target scales to fix during training.

        A loaded ``nep.txt`` potential already predicts unscaled energies, so
        its target scale must stay one.
        """
        if not self.loaded_nep:
            return {}
        return {self.targets_keys: 1.0}

    @torch.jit.unused
    def _energy_scales_per_type(self) -> torch.Tensor:
        """Scaler factor per NEP type (order of ``self.atomic_types``).

        A per-structure target has a single global scale (stored with
        ``atomic_type = -1``); a per-atom target stores one scale per type.
        """
        num_types = len(self.atomic_types)
        try:
            scales_tmap = self.scaler.model.scales[self.targets_keys]
        except KeyError:
            return torch.ones(num_types, dtype=torch.float64)
        if len(scales_tmap) != 1:
            raise ValueError(
                "NEP GPUMD export only supports single-block scalar targets."
            )
        block = scales_tmap.block()
        if block.values.shape[1] != 1:
            raise ValueError("NEP GPUMD export only supports single-property targets.")
        types = block.samples.values[:, 0].tolist()
        values = block.values[:, 0].to(torch.float64)
        if types == [-1]:
            return values.expand(num_types).clone()
        type_to_scale = {int(t): float(v) for t, v in zip(types, values, strict=True)}
        return torch.tensor(
            [type_to_scale.get(z, 1.0) for z in self.atomic_types],
            dtype=torch.float64,
        )

    @torch.jit.unused
    def _composition_weights_per_type(self) -> torch.Tensor:
        """Composition baseline per NEP type (order of ``self.atomic_types``)."""
        weights = self.additive_models[0].model.weights
        if self.targets_keys not in weights:
            return torch.zeros(len(self.atomic_types), dtype=torch.float64)
        block = weights[self.targets_keys].block()
        types = block.samples.values[:, 0].tolist()
        values = block.values[:, 0].to(torch.float64)
        type_to_weight = {int(t): float(v) for t, v in zip(types, values, strict=True)}
        return torch.tensor(
            [type_to_weight.get(z, 0.0) for z in self.atomic_types],
            dtype=torch.float64,
        )

    @torch.jit.unused
    def export_nep(self, path: Union[str, Path]) -> None:
        """Write a GPUMD-compatible ``nep.txt`` file for this model.

        The metatrain per-atom prediction is ``s_t * e_nep + c_t``, where
        ``s_t`` is the target scale (from the ``Scaler``) and ``c_t`` the
        per-type composition baseline.  Both are folded exactly into the NEP
        output layer so that native NEP implementations (GPUMD, NEP_CPU,
        calorine, ...) reproduce the metatrain predictions:

        - all versions: the output weights of type ``t`` are multiplied by
          ``s_t``;
        - NEP5 (``version: 5``): ``c_t`` and the scaled global bias are folded
          into the per-type bias — always exact;
        - NEP3/NEP4: there is only a global bias, so the fold is only exact
          when ``s_t * b1 - c_t`` is the same for every type (e.g. a single
          element, or uniform composition weights).  Otherwise a
          ``ValueError`` suggests using ``version: 5``.

        ZBL is handled as an additive model that is excluded from the scaler,
        exactly as in native NEP, so ZBL models export with any scale.  The
        electrostatic energy of NEP-Charge (qNEP) models is quadratic in the
        predicted charges and cannot be folded, so exporting them requires
        unit scales (train with ``scale_targets: false``).

        :param path: Output path for the ``nep.txt`` file.
        """
        params = self.potential.to_nep_parameters()
        num_types = params.num_types
        neurons = params.num_neurons1
        dim = params.dim
        version = params.version

        s = self._energy_scales_per_type()
        c = self._composition_weights_per_type()

        ann = params.ann.clone().to(torch.float64)
        if params.charge_mode != 0:
            if not torch.allclose(s, torch.ones_like(s)):
                raise ValueError(
                    "Cannot export a NEP-Charge (qNEP) model with a non-unit "
                    "target scale: the Ewald energy is quadratic in the "
                    "predicted charges and cannot be folded. Train with "
                    "`scale_targets: false` to export this model."
                )
            num_outputs = 3 if params.charge_mode == 3 else 2
            block = neurons * dim + neurons + neurons * num_outputs
            b1_index = num_types * block + 1  # skip sqrt_epsilon_inf
            b1 = float(ann[b1_index])
            # native evaluates e_t = ... - b1'; the composition fold needs the
            # per-type constant b1 - c_t to be representable by b1' alone
            d = b1 - c
            spread = float((d.max() - d.min()).abs())
            if spread > 1.0e-8 * max(1.0, float(d.abs().max())):
                raise ValueError(
                    "NEP-Charge has a single global energy bias, but the "
                    f"per-type composition weights differ (spread {spread:.3e})."
                    " Train with uniform `fixed_composition_weights` to export "
                    "this model."
                )
            ann[b1_index] = float(d.mean())
        elif version in (3, 4):
            block = neurons * dim + 2 * neurons
            b1_index = block if version == 3 else block * num_types
            b1 = float(ann[b1_index])
            if version == 3:
                # shared ANN: per-type output scaling is impossible
                if not torch.allclose(s, s[0].expand_as(s)):
                    raise ValueError(
                        "NEP3 shares one network across all types, so per-type "
                        "target scales cannot be folded. Use `version: 4` or "
                        "`version: 5`."
                    )
                w1_start = neurons * dim + neurons
                ann[w1_start : w1_start + neurons] *= s[0]
            else:
                for t in range(num_types):
                    w1_start = t * block + neurons * dim + neurons
                    ann[w1_start : w1_start + neurons] *= s[t]
            # native evaluates e_t = h @ w1_t - b1'; we need the per-type
            # constant d_t = s_t * b1 - c_t to be representable by b1' alone
            d = s * b1 - c
            spread = float((d.max() - d.min()).abs())
            if spread > 1.0e-8 * max(1.0, float(d.abs().max())):
                raise ValueError(
                    f"NEP{version} has a single global bias, but the folded "
                    f"per-type constants differ (spread {spread:.3e}). "
                    "Use `version: 5`, whose per-type bias makes the "
                    "composition fold exact for multi-element models."
                )
            ann[b1_index] = float(d.mean())
        elif version == 5:
            block = neurons * dim + 2 * neurons + 1
            b1_index = block * num_types
            b1 = float(ann[b1_index])
            for t in range(num_types):
                w1_start = t * block + neurons * dim + neurons
                bias_index = t * block + neurons * dim + 2 * neurons
                ann[w1_start : w1_start + neurons] *= s[t]
                # native: e_t = h @ w1'_t - type_bias'_t - b1
                # target: s_t * (h @ w1_t - type_bias_t - b1) + c_t
                ann[bias_index] = float(
                    s[t] * ann[bias_index] + (s[t] - 1.0) * b1 - c[t]
                )
        else:
            raise ValueError(f"unsupported NEP version for export: {version}")

        # re-attach the ZBL header from the additive model configuration
        zbl_outer = self.hypers.get("zbl_outer_cutoff")
        zbl_cfg = None
        if zbl_outer is not None:
            outer = float(zbl_outer)
            zbl_cfg = ZBLConfig(
                enabled=True,
                flexible=False,
                rc_inner=0.5 * outer,
                rc_outer=outer,
                atomic_numbers=tuple(self.atomic_types),
                flexible_params=None,
            )

        write_nep(dataclasses.replace(params, ann=ann, zbl=zbl_cfg), path)

    def restart(self, dataset_info: DatasetInfo) -> "NEP":
        # merge old and new dataset info
        merged_info = self.dataset_info.union(dataset_info)
        new_atomic_types = [
            at for at in merged_info.atomic_types if at not in self.atomic_types
        ]
        new_targets = {
            key: value
            for key, value in merged_info.targets.items()
            if key not in self.dataset_info.targets
        }
        self.has_new_targets = len(new_targets) > 0

        if len(new_atomic_types) > 0:
            raise ValueError(
                f"New atomic types found in the dataset: {new_atomic_types}. "
                "The NEP model does not support adding new atomic types."
            )

        # register new outputs as new last layers
        for target_name, target in new_targets.items():
            self._add_output(target_name, target)

        self.dataset_info = merged_info

        # restart the composition and scaler models
        self.additive_models[0].restart(
            dataset_info=DatasetInfo(
                length_unit=dataset_info.length_unit,
                atomic_types=self.atomic_types,
                targets={
                    target_name: target_info
                    for target_name, target_info in dataset_info.targets.items()
                    if CompositionModel.is_valid_target(target_name, target_info)
                },
            ),
        )
        for additive_model in self.additive_models[1:]:
            additive_model.restart(dataset_info)
        self.scaler.restart(dataset_info)

        return self

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        context: Literal["restart", "finetune", "export"],
    ) -> "NEP":
        model_data = checkpoint["model_data"]

        if context == "restart":
            logging.info(f"Using latest model from epoch {checkpoint['epoch']}")
            model_state_dict = checkpoint["model_state_dict"]
        elif context in {"finetune", "export"}:
            logging.info(f"Using best model from epoch {checkpoint['best_epoch']}")
            model_state_dict = checkpoint["best_model_state_dict"]
            if model_state_dict is None:
                model_state_dict = checkpoint["model_state_dict"]
        else:
            raise ValueError("Unknown context tag for checkpoint loading!")

        model = cls(
            hypers=model_data["model_hypers"],
            dataset_info=model_data["dataset_info"],
        )

        dtype = model_state_dict["potential.ann"].dtype
        model.to(dtype).load_state_dict(model_state_dict)
        model.additive_models[0].sync_tensor_maps()
        model.scaler.sync_tensor_maps()

        # Loading the metadata from the checkpoint
        metadata = checkpoint.get("metadata", None)
        if metadata is not None:
            model.__default_metadata__ = metadata

        return model

    def export(self, metadata: Optional[ModelMetadata] = None) -> AtomisticModel:
        dtype = next(self.parameters()).dtype
        if dtype not in self.__supported_dtypes__:
            raise ValueError(f"unsupported dtype {dtype} for NEP")

        # Make sure the model is all in the same dtype
        # For example, after training, the additive models could still be in
        # float64
        self.to(dtype)

        # Additionally, the composition model contains some `TensorMap`s that cannot
        # be registered correctly with Pytorch. This function moves them:

        self.additive_models[0].weights_to(torch.device("cpu"), torch.float64)

        interaction_ranges = [self.cutoff_radial, self.cutoff_angular]
        for additive_model in self.additive_models:
            if hasattr(additive_model, "cutoff_radius"):
                interaction_ranges.append(additive_model.cutoff_radius)
        interaction_range = max(interaction_ranges)

        capabilities = ModelCapabilities(
            outputs=self.outputs,
            atomic_types=self.atomic_types,
            interaction_range=interaction_range,
            length_unit=self.dataset_info.length_unit,
            supported_devices=self.__supported_devices__,
            dtype=dtype_to_str(dtype),
        )
        if metadata is None:
            metadata = self.__default_metadata__
        else:
            metadata = merge_metadata(self.__default_metadata__, metadata)

        return AtomisticModel(self.eval(), metadata, capabilities)

    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict) -> Dict:
        for v in range(1, cls.__checkpoint_version__):
            if checkpoint["model_ckpt_version"] == v:
                update = getattr(checkpoints, f"model_update_v{v}_v{v + 1}")
                update(checkpoint)
                checkpoint["model_ckpt_version"] = v + 1

        if checkpoint["model_ckpt_version"] != cls.__checkpoint_version__:
            raise RuntimeError(
                f"Unable to upgrade the checkpoint: the checkpoint is using model "
                f"version {checkpoint['model_ckpt_version']}, while the current model "
                f"version is {cls.__checkpoint_version__}."
            )

        return checkpoint

    def get_checkpoint(self) -> Dict:
        hypers = dict(self.hypers)
        # The architecture hypers were synced from the nep.txt file at
        # construction, so the checkpoint can be rebuilt without the file;
        # the weights live in the state dict.
        hypers.pop("nep_model", None)

        checkpoint = {
            "architecture_name": "experimental.nep",
            "model_ckpt_version": self.__checkpoint_version__,
            "metadata": self.metadata,
            "model_data": {
                "model_hypers": hypers,
                "dataset_info": self.dataset_info,
            },
            "epoch": None,
            "best_epoch": None,
            "model_state_dict": self.state_dict(),
            "best_model_state_dict": None,
        }
        return checkpoint

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.outputs
