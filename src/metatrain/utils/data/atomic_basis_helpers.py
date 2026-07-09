import functools
from typing import Callable, Dict, List, Optional, Tuple

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

from ..data import DatasetInfo
from .target_info import TargetInfo


# ===== General utilities


def get_per_atom_sample_labels(
    systems: List[System],
    system_ids: torch.Tensor,
) -> Labels:
    """
    Builds the atom sample labels for the input ``systems``, labelling each system's
    atoms with the corresponding entry of ``system_ids`` (rather than assuming the
    systems have a local batch index 0, ..., n_batch - 1), so the result can be matched
    directly against a target's own, native "system" sample values.

    :param systems: List of systems to build the per-atom sample labels for.
    :param system_ids: the "system" label value for each system, in the same order as
        ``systems``.
    :return: The per-atom sample labels.
    """
    system_indices = torch.concatenate(
        [
            torch.full(
                (len(system),),
                int(system_ids[i_system]),
                device=systems[0].device,
            )
            for i_system, system in enumerate(systems)
        ],
    )

    sample_values = torch.stack(
        [
            system_indices,
            torch.concatenate(
                [
                    torch.arange(
                        len(system),
                        device=systems[0].device,
                    )
                    for system in systems
                ],
            ),
        ],
        dim=1,
    )
    sample_labels = Labels(
        names=["system", "atom"],
        values=sample_values,
    )
    return sample_labels


# ===== Densification utilities (keys with atom types to samples)


def _densify_per_atom_atomic_basis_target(
    tensor: TensorMap,
    layout: TensorMap,
    fill_value: float = torch.nan,
) -> TensorMap:
    """
    Densify the per-atom atomic basis target by moving the "atom_type" key dimension to
    the samples, creating a padded property dimension according to the maximum property
    size for each irrep.

    :param tensor: the per-atom atomic basis target TensorMap to densify.
    :param layout: the layout TensorMap defining the global basis set.
    :param fill_value: the value to use for filling in the padded values when
        densifying.

    :return: the densified per-atom atomic basis target TensorMap.
    """

    # First ensure that the tensor has all keys present in the layout tensor (i.e. the
    # global basis set definition). If any blocks aren't present, they are added as
    # zero-sample blocks with the correct components and properties.
    blocks = []
    for key, layout_block in layout.items():
        if key in tensor.keys:
            existing_block = tensor.block(key)
            block = TensorBlock(
                values=existing_block.values,
                samples=existing_block.samples,
                components=existing_block.components,
                properties=existing_block.properties,
            )
        else:
            block = layout_block.copy(deep=False)
            assert len(block.samples) == 0
        blocks.append(block)

    tensor = TensorMap(layout.keys, blocks)

    # Now densification can be done.

    # =====
    # TODO: the following is a manual densification, but this will be replaced with
    # `keys_to_samples(..., fill_value)` once publicly available in
    # metatensor-operations.
    # return mts.keys_to_samples(tensor, fill_value=fill_value, sort_samples=True)
    # =====

    # =====
    # For now, implement a manual densification:
    # =====

    # First, identify the "atom_type"-like and non-"atom_type"-like key dimensions.
    type_indices = [
        i for i, name in enumerate(tensor.keys.names) if name.endswith("atom_type")
    ]
    non_type_indices = [
        i for i, name in enumerate(tensor.keys.names) if not name.endswith("atom_type")
    ]
    type_names = [tensor.keys.names[i] for i in type_indices]

    # Using the layout TensorMap, build the union of the property labels values across
    # all atom types
    union_properties = {}
    for key, block in layout.items():
        key_vals = tuple([key.values[i].item() for i in non_type_indices])
        if key_vals not in union_properties:
            union_properties[key_vals] = block.properties
        else:
            union_properties[key_vals] = union_properties[key_vals].union(
                block.properties
            )

    # For each block, pad the properties using the dense properties
    padded_blocks = []
    for key, block in tensor.items():
        key_vals = tuple([key.values[i].item() for i in non_type_indices])
        properties = union_properties[key_vals]

        # Create a values array filled with the fill value
        padded_values = torch.full(
            (
                len(block.samples),
                *[len(c) for c in block.components],
                len(properties),
            ),
            fill_value,
            dtype=block.values.dtype,
        )

        # Now broadcast the existing values to the new shape
        properties_mask = properties.select(block.properties)
        padded_values[..., properties_mask] = block.values
        padded_block = TensorBlock(
            values=padded_values,
            samples=block.samples,
            components=block.components,
            properties=properties,
        )
        padded_blocks.append(padded_block)

    tensor = TensorMap(tensor.keys, padded_blocks)

    # Now move the "atom_type"-like key dimension to the samples and remove them
    tensor = tensor.keys_to_samples(type_names, sort_samples=True)
    for name in type_names:
        tensor = mts.remove_dimension(tensor, "samples", name)

    return tensor


class DensifyStatics:
    """Layout-derived constants for densifying atomic-basis targets.

    Everything here depends only on the (static) layout, so computing it once
    per training run — instead of once per batch inside the collate transform —
    removes the union/select/empty-block construction from the hot path.

    :param layout: the layout ``TensorMap`` defining the global basis set.
    """

    def __init__(self, layout: TensorMap):
        keys = layout.keys
        self.layout_keys = keys
        self.type_names = [n for n in keys.names if n.endswith("atom_type")]
        non_type_indices = [
            i for i, n in enumerate(keys.names) if not n.endswith("atom_type")
        ]

        union_properties: Dict[Tuple[int, ...], Labels] = {}
        for key, block in layout.items():
            key_vals = tuple(int(key.values[i]) for i in non_type_indices)
            if key_vals not in union_properties:
                union_properties[key_vals] = block.properties
            else:
                union_properties[key_vals] = union_properties[key_vals].union(
                    block.properties
                )

        # Per layout key: the union properties for its group and the positions
        # of the block's own properties inside that union.
        self.union_properties: Dict[Tuple[int, ...], Labels] = {}
        self.property_masks: Dict[Tuple[int, ...], torch.Tensor] = {}
        self.empty_samples: Dict[Tuple[int, ...], Labels] = {}
        self.components: Dict[Tuple[int, ...], List[Labels]] = {}
        self.empty_values: Dict[Tuple[int, ...], torch.Tensor] = {}
        for key, block in layout.items():
            full_key = tuple(int(v) for v in key.values)
            key_vals = tuple(int(key.values[i]) for i in non_type_indices)
            union = union_properties[key_vals]
            self.union_properties[full_key] = union
            self.property_masks[full_key] = union.select(block.properties)
            self.empty_samples[full_key] = block.samples
            self.components[full_key] = block.components
            self.empty_values[full_key] = torch.empty(
                (0, *[len(c) for c in block.components], len(union)),
                dtype=block.values.dtype,
            )


def _densify_per_atom_with_statics(
    tensor: TensorMap,
    statics: DensifyStatics,
    fill_value: float = torch.nan,
) -> TensorMap:
    """Densify using precomputed layout statics (see :class:`DensifyStatics`).

    Semantically identical to :func:`_densify_per_atom_atomic_basis_target`,
    but all layout-only work (property unions, selection masks, empty blocks)
    is read from ``statics`` instead of being recomputed per call.

    :param tensor: the per-atom atomic basis target TensorMap to densify.
    :param statics: precomputed layout constants.
    :param fill_value: the value used for the padded entries.
    :return: the densified per-atom atomic basis target TensorMap.
    """
    present = {
        tuple(int(v) for v in key.values): block for key, block in tensor.items()
    }

    padded_blocks = []
    for key in statics.layout_keys:
        full_key = tuple(int(v) for v in key.values)
        block = present.get(full_key)
        union = statics.union_properties[full_key]
        if block is None:
            padded_blocks.append(
                TensorBlock(
                    values=statics.empty_values[full_key],
                    samples=statics.empty_samples[full_key],
                    components=statics.components[full_key],
                    properties=union,
                )
            )
            continue
        padded_values = torch.full(
            (
                len(block.samples),
                *[len(c) for c in block.components],
                len(union),
            ),
            fill_value,
            dtype=block.values.dtype,
        )
        padded_values[..., statics.property_masks[full_key]] = block.values
        padded_blocks.append(
            TensorBlock(
                values=padded_values,
                samples=block.samples,
                components=block.components,
                properties=union,
            )
        )

    out = TensorMap(statics.layout_keys, padded_blocks)
    out = out.keys_to_samples(statics.type_names, sort_samples=True)
    for name in statics.type_names:
        out = mts.remove_dimension(out, "samples", name)
    return out


def densify_atomic_basis_target(
    tensor: TensorMap,
    layout: TensorMap,
    fill_value: float = torch.nan,
) -> TensorMap:
    """
    Densify the atomic basis target by moving any "atom_type"-like key dimensions to the
    samples, creating a padded property dimension according to the maximum property size
    for each irrep.

    :param tensor: the atomic basis target TensorMap to densify.
    :param layout: the layout TensorMap defining the global basis set (i.e. the union of
        all blocks that should be present in the output).
    :param fill_value: the value to use for filling in the padded values when
        densifying.

    :return: the densified atomic basis target TensorMap.
    """
    if "atom" in tensor.sample_names:
        return _densify_per_atom_atomic_basis_target(tensor, layout, fill_value)

    raise NotImplementedError(
        "Currently only densification of per-atom atomic basis targets is implemented."
    )


def _pad_samples_per_atom_atomic_basis_target(
    systems: List[System],
    tensor: TensorMap,
    system_ids: torch.Tensor,
) -> TensorMap:
    sample_labels = get_per_atom_sample_labels(systems, system_ids)
    new_blocks = []
    for block in tensor:
        new_vals = torch.full(
            (len(sample_labels), *block.values.shape[1:]),
            fill_value=torch.nan,
            dtype=block.values.dtype,
        )
        sample_mask = sample_labels.select(block.samples)
        new_vals[sample_mask] = block.values
        new_block = TensorBlock(
            values=new_vals,
            samples=sample_labels,
            components=block.components,
            properties=block.properties,
        )
        new_blocks.append(new_block)

    return TensorMap(tensor.keys, new_blocks)


def pad_samples_atomic_basis_target(
    systems: List[System],
    tensor: TensorMap,
    system_ids: torch.Tensor,
) -> TensorMap:
    """
    Pad the samples of the atomic basis target to have the same number of samples for
    each block.

    :param systems: List of systems in the batch
    :param tensor: the atomic basis target TensorMap to pad.
    :param system_ids: the "system" label value for each system, in the same order as
        ``systems``, matching the tensor's own native "system" sample values.

    :return: the padded atomic basis target TensorMap
    """

    if "atom" in tensor.sample_names:
        return _pad_samples_per_atom_atomic_basis_target(systems, tensor, system_ids)
    raise NotImplementedError(
        "Currently only padding of per-atom atomic basis targets is implemented."
    )


def prepare_atomic_basis_targets(
    systems: List[System],
    system_ids: torch.Tensor,
    tensor: TensorMap,
    layout: TensorMap,
    fill_value: float = torch.nan,
    statics: Optional[DensifyStatics] = None,
) -> TensorMap:
    """
    Prepare the atomic basis targets for batching by densifying (moving "atom_type" key
    dimensions to the samples) and padding the samples.

    :param systems: List of systems in the batch.
    :param system_ids: Tensor containing the system ids for each sample in the input
        ``tensor``.
    :param tensor: the atomic basis target TensorMap to prepare.
    :param layout: the layout TensorMap defining the global basis set (i.e. the union of
        all blocks that should be present in the output).
    :param fill_value: the value to use for filling in the padded values when densifying
        and padding. Default is NaN, but can be set to 0 if desired (e.g. for
        classification targets).
    :param statics: optional precomputed layout constants (see
        :class:`DensifyStatics`); avoids recomputing layout-only work per batch.
    :return: the prepared atomic basis target TensorMap, densified and padded, keeping
        the tensor's own native "system" ids.
    """

    # Densify: "atom type" key dimensions -> samples
    if statics is not None and "atom" in tensor.sample_names:
        tensor = _densify_per_atom_with_statics(tensor, statics, fill_value)
    else:
        tensor = densify_atomic_basis_target(tensor, layout, fill_value)

    # Pad samples, labelling them with the target's own native "system" ids
    tensor = pad_samples_atomic_basis_target(systems, tensor, system_ids)

    return tensor


# ===== Sparsification utilities (atom types back to keys)


def _compute_sparse_properties(layout: TensorMap) -> TensorMap:
    """Compute the properties of the sparse tensor map that results from
    sparsifying a given dense tensor map.

    This can be computed once and for all for a given layout, and be reused to
    sparsify as many padded tensor maps as needed with the function
    :func:`sparsify_atomic_basis_target`.

    :param layout: the layout ``TensorMap`` defining the global basis set.
    :return: The ``TensorMap`` to be used to go from the dense to the sparse
      representation. The values of each block are the indices of the properties
      to select from a padded tensor map.
    """
    non_type_indices: list[int] = []
    for i, name in enumerate(layout.keys.names):
        if not name.endswith("atom_type"):
            non_type_indices.append(i)

    # Build union of properties across atom types for each non-type key — same
    # logic as _densify_per_atom_atomic_basis_target.
    union_properties: dict[str, Labels] = {}
    for key, block in layout.items():
        k = str([key.values[i].item() for i in non_type_indices])
        if k not in union_properties:
            union_properties[k] = block.properties
        else:
            union_properties[k] = union_properties[k].union(block.properties)

    blocks: list[TensorBlock] = []
    for key, block in layout.items():
        padded_properties = union_properties[
            str([key.values[i].item() for i in non_type_indices])
        ]

        blocks.append(
            TensorBlock(
                values=padded_properties.select(block.properties).reshape(1, -1),
                samples=Labels(["_"], torch.zeros((1, 1), dtype=torch.int64)),
                components=[],
                properties=block.properties,
            )
        )

    return TensorMap(layout.keys, blocks)


def _sparsify_per_atom_atomic_basis_target(
    systems: List[System],
    tensor: TensorMap,
    sparse_properties: TensorMap,
    atom_types_batch: Optional[torch.Tensor] = None,
) -> TensorMap:
    """
    Sparsify the per-atom atomic basis target by creating blocks with an explicit
    "atom_type" dimension. The dense blocks of the input ``tensor`` are therefore sliced
    according to atom type of each atom in the samples.

    :param systems: List of systems in the batch.
    :param tensor: the atomic basis target TensorMap to sparsify.
    :param sparse_properties: A TensorMap containing the properties to select
        from the dense tensor map to create each block in the sparse layout.
    :param atom_types_batch: Optional tensor containing the atom types for each sample
        in the batch. If not provided, these are inferred from the systems.
    :return: the sparsified atomic basis target TensorMap
    """
    if atom_types_batch is None:
        # Get the atom types for each sample in the batch from the systems
        atom_types_batch = torch.cat(
            [system.types for system in systems],
            dim=0,
        )

    device = tensor.device

    # Sparsify by moving the "atom_type" from the samples to the keys
    unique_types: list[int] = (
        torch.unique(atom_types_batch).to(torch.int64).cpu().tolist()
    )
    atom_type_masks: Dict[int, torch.Tensor] = {}
    atom_type_samples: Dict[int, Labels] = {}

    # Compute the samples masks (i.e. get the indices for the atoms
    # of each type), and build the samples labels. This could be computed
    # in the dataloader since it only depends on the atom types of the systems.
    sample_indices = torch.arange(len(atom_types_batch)).to(device=device)
    for atom_type in unique_types:
        atom_type_masks[atom_type] = sample_indices[atom_types_batch == atom_type]
        atom_type_samples[atom_type] = Labels(
            names=["system", "atom"],
            values=tensor[0].samples.values[atom_type_masks[atom_type]],
        )

    sparse_properties = sparse_properties.to(device=device)

    # Build the sparsified tensormap by looping over the dense one
    # and splitting each block into one block per atom type.
    new_keys: list[list[int]] = []
    sparse_blocks: List[TensorBlock] = []
    for key, block in tensor.items():
        key_values: list[int] = key.values.to(torch.int64).tolist()
        for atom_type in unique_types:
            new_key = key_values + [atom_type]

            # Get the corresponding layout block to know which properties to select
            block_position = sparse_properties.keys.position(new_key)
            if block_position is None:
                # This irrep doesn't exist for this atom type in the layout
                continue
            assert block_position is not None  # for torchscript
            layout_block = sparse_properties.block_by_id(block_position)

            # Select samples
            values = block.values[atom_type_masks[atom_type]]
            # Select properties
            properties_mask = layout_block.values.ravel()
            # Do block.values[..., properties_mask] in a torchscriptable way.
            if block.values.ndim == 3:
                values = values[:, :, properties_mask]
            elif block.values.ndim == 4:
                values = values[:, :, :, properties_mask]
            else:
                raise ValueError(
                    "Tensorblocks with more than 2 component dimensions can't be "
                    "sparsified with the current implementation."
                )

            sparse_block = TensorBlock(
                values=values,
                samples=atom_type_samples[atom_type],
                components=block.components,
                properties=layout_block.properties,
            )

            new_keys.append(new_key)
            sparse_blocks.append(sparse_block)

    tensor = TensorMap(
        Labels(
            names=tensor.keys.names + ["atom_type"],
            values=torch.tensor(new_keys, device=device),
        ),
        sparse_blocks,
    )

    return tensor


def sparsify_atomic_basis_target(
    systems: List[System],
    tensor: TensorMap,
    layout: TensorMap,
    atom_types_batch: Optional[torch.Tensor] = None,
    sparse_properties: Optional[TensorMap] = None,
) -> TensorMap:
    """
    Sparsify the atomic basis target by creating blocks with an explicit "atom_type"
    dimension. The dense blocks of the input ``tensor`` are therefore sliced according
    to atom type of each atom in the samples.

    :param systems: List of systems in the batch.
    :param tensor: the atomic basis target TensorMap to sparsify.
    :param layout: the layout TensorMap defining the global basis set (i.e. the union of
        all blocks that should be present in the sparsified output).
    :param atom_types_batch: Optional tensor containing the atom types for each sample
        in the batch. If not provided, these are inferred from the systems.
    :param sparse_properties: A TensorMap containing the properties to select
        from the dense tensor map to create each block in the sparse layout.
        If not provided, it is computed from ``layout``.

    :return: the sparsified atomic basis target TensorMap
    """
    if sparse_properties is None:
        sparse_properties = _compute_sparse_properties(layout)

    if "atom" in tensor.sample_names:
        return _sparsify_per_atom_atomic_basis_target(
            systems, tensor, sparse_properties, atom_types_batch
        )

    raise NotImplementedError(
        "Currently only sparsification of per-atom atomic basis targets is implemented."
    )


# ===== dataloader transforms


def _prepare_atomic_basis_targets_impl(
    target_info_dict: dict[str, TargetInfo],
    extra_data_info_dict: dict[str, TargetInfo],
    densify_statics: Dict[str, DensifyStatics],
    systems: List[System],
    targets: Dict[str, TensorMap],
    extra: Dict[str, TensorMap],
) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
    system_ids: Optional[torch.Tensor] = None
    if any(
        name in target_info_dict and target_info_dict[name].is_atomic_basis
        for name in targets
    ) or any(
        name in extra_data_info_dict and extra_data_info_dict[name].is_atomic_basis
        for name in extra
    ):
        assert "mtt::aux::system_index" in extra
        system_ids = (
            extra["mtt::aux::system_index"][0].values[:, 0].to(dtype=torch.int64)
        )

    for name, tensor in targets.items():
        if name in target_info_dict and target_info_dict[name].is_atomic_basis:
            assert system_ids is not None
            targets[name] = prepare_atomic_basis_targets(
                systems,
                system_ids,
                tensor,
                target_info_dict[name].layout,
                fill_value=torch.nan,
                statics=densify_statics.get(name),
            )

    for name, tensor in extra.items():
        if name in extra_data_info_dict and extra_data_info_dict[name].is_atomic_basis:
            assert system_ids is not None
            extra[name] = prepare_atomic_basis_targets(
                systems,
                system_ids,
                tensor,
                extra_data_info_dict[name].layout,
                fill_value=torch.nan,
                statics=densify_statics.get(name),
            )

    return systems, targets, extra


def _reverse_atomic_basis_targets_impl(
    target_info_dict: dict[str, TargetInfo],
    extra_data_info_dict: dict[str, TargetInfo],
    sparse_properties: dict[str, TensorMap],
    systems: List[System],
    targets: Dict[str, TensorMap],
    extra: Dict[str, TensorMap],
) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
    for name, tensor in targets.items():
        if name in target_info_dict and target_info_dict[name].is_atomic_basis:
            if name in sparse_properties:
                sparse_properties[name] = sparse_properties[name].to(tensor.device)
            targets[name] = sparsify_atomic_basis_target(
                systems,
                tensor,
                target_info_dict[name].layout,
                sparse_properties=sparse_properties.get(name),
            )

    for name, tensor in extra.items():
        if name in extra_data_info_dict and extra_data_info_dict[name].is_atomic_basis:
            if name in sparse_properties:
                sparse_properties[name] = sparse_properties[name].to(tensor.device)
            extra[name] = sparsify_atomic_basis_target(
                systems,
                tensor,
                extra_data_info_dict[name].layout,
                sparse_properties=sparse_properties.get(name),
            )

    return systems, targets, extra


def get_prepare_atomic_basis_targets_transform(
    target_info_dict: dict[str, TargetInfo],
    extra_data_info_dict: dict[str, TargetInfo],
) -> Tuple[Callable, Callable]:
    """
    Get a function that prepares the atomic basis targets for batching by densifying
    and padding.

    :param target_info_dict: Dictionary mapping target names to TargetInfo objects.
    :param extra_data_info_dict: Dictionary mapping extra data names to TargetInfo
        objects.

    :return: A function that takes in systems, targets and extra data, and returns the
        systems, targets and extra data with prepared atomic basis targets.
    """
    # Precompute property masks for all atomic basis targets and extra data.
    # This avoids calling layout_properties.select(block.properties) every
    # time we want to sparsify a batch.
    sparse_properties: dict[str, TensorMap] = {}
    densify_statics: Dict[str, DensifyStatics] = {}
    for name, info in target_info_dict.items():
        if info.is_atomic_basis:
            sparse_properties[name] = _compute_sparse_properties(info.layout)
            densify_statics[name] = DensifyStatics(info.layout)
    for name, info in extra_data_info_dict.items():
        if info.is_atomic_basis:
            sparse_properties[name] = _compute_sparse_properties(info.layout)
            densify_statics[name] = DensifyStatics(info.layout)

    return (
        functools.partial(
            _prepare_atomic_basis_targets_impl,
            target_info_dict,
            extra_data_info_dict,
            densify_statics,
        ),
        functools.partial(
            _reverse_atomic_basis_targets_impl,
            target_info_dict,
            extra_data_info_dict,
            sparse_properties,
        ),
    )


# ===== DatasetInfo manipulation utilities


def densify_atomic_basis_dataset_info(dataset_info: DatasetInfo) -> DatasetInfo:
    """
    Densify the atomic basis target layouts in the TargetInfos of the input
    ``dataset_info`` by moving any

    :param dataset_info: the input DatasetInfo with TargetInfos containing atomic basis
        targets to densify.
    :return: a new DatasetInfo with the same information as the input, but with the
        atomic basis targets densified.
    """

    return DatasetInfo(
        length_unit=dataset_info.length_unit,
        atomic_types=dataset_info.atomic_types,
        targets={
            target_name: (
                TargetInfo(
                    quantity=target_info.quantity,
                    unit=target_info.unit,
                    layout=densify_atomic_basis_target(
                        target_info.layout, target_info.layout
                    ),
                    description=target_info.description,
                )
                if target_info.is_atomic_basis
                else target_info
            )
            for target_name, target_info in dataset_info.targets.items()
        },
    )
