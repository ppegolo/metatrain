from typing import Dict, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

from .atomic_basis_helpers import (
    _compute_sparse_properties,
    densify_atomic_basis_target,
    sparsify_atomic_basis_target,
)


def sparse_properties_of(layout: TensorMap) -> Optional[TensorMap]:
    """The properties to select when sparsifying a dense tensor map to ``layout``.

    They depend only on the layout, and computing them takes longer than the
    sparsification itself for targets with many atomic types, so models compute them
    once and pass them to :func:`match_layout`.

    :param layout: The layout of a target.
    :return: The properties, or ``None`` if the layout is not sparse in atom types.
    """
    if any(name.endswith("atom_type") for name in layout.keys.names):
        return _compute_sparse_properties(layout)
    return None


def cached_sparse_properties(
    cache: Dict[str, TensorMap], key: str, device: torch.device
) -> Optional[TensorMap]:
    """Get properties computed by :func:`sparse_properties_of` on ``device``, moving
    them there and updating ``cache`` if needed.

    :param cache: The properties, by key.
    :param key: The key of the properties to get.
    :param device: The device of the tensor map to sparsify.
    :return: The properties, or ``None`` if ``key`` is not in ``cache``.
    """
    if key not in cache:
        return None
    properties = cache[key]
    if properties.device != device:
        properties = properties.to(device=device)
        cache[key] = properties
    return properties


def match_layout(
    tmap1: TensorMap,
    tmap2: TensorMap,
    target_info_layout: Optional[TensorMap] = None,
    systems: Optional[list[System]] = None,
    sparse_properties: Optional[TensorMap] = None,
) -> TensorMap:
    """Makes sure the layout of the two tensormaps is the same.

    It raises an error if the two tensormaps are incompatible.

    :param tmap1: The first tensormap, whose layout will be modified
      to match the second.
    :param tmap2: The second tensormap, whose layout will be used as
      the reference.
    :param target_info_layout: Layout of the data. It will only be used in
      case tmap1 needs to be densified or sparsified (in atom types)
      to match tmap2.
    :param systems: List of systems. It will only be used in case tmap1 needs
      to be sparsified (in atom types) to match tmap2.
    :param sparse_properties: The properties to select when sparsifying, computed
      from ``target_info_layout`` if not given (see
      :func:`~metatrain.utils.data.atomic_basis_helpers.sparsify_atomic_basis_target`).
    :return: The first tensormap, with its layout modified to match
      the second.
    """

    # Check whether the tensormaps are dense or sparse in atom types.
    tmap1_isdense = (
        "atom_type" not in tmap1.keys.names
        and "first_atom_type" not in tmap1.keys.names
    )
    tmap2_isdense = (
        "atom_type" not in tmap2.keys.names
        and "first_atom_type" not in tmap2.keys.names
    )

    # Check whether the two tensormaps are dense or sparse in atom types,
    # and possibly convert tmap1 to match tmap2.
    if not tmap1_isdense and tmap2_isdense:
        # We need to densify tmap1.

        if target_info_layout is None:
            # If no target_info_layout is provided, we take the layout of tmap2,
            # we just need to make sure that the blocks of tmap2 are empty.
            blocks: list[TensorBlock] = []
            for block in tmap2.blocks():
                shape = list(block.values.shape)
                shape[0] = 0
                device = block.values.device
                blocks.append(
                    TensorBlock(
                        values=torch.tensor(
                            [], dtype=block.values.dtype, device=device
                        ).reshape(shape),
                        samples=Labels(
                            names=block.samples.names,
                            values=torch.tensor(
                                [], device=device, dtype=torch.int64
                            ).reshape(0, len(block.samples.names)),
                        ),
                        components=block.components,
                        properties=block.properties,
                    )
                )
            target_info_layout = TensorMap(keys=tmap2.keys, blocks=blocks)

        tmap1 = densify_atomic_basis_target(tmap1, target_info_layout, fill_value=0.0)
    elif tmap1_isdense and not tmap2_isdense:
        # We need to sparsify tmap1.
        if systems is None:
            raise ValueError("Cannot sparsify atomic basis target without systems.")
        tmap1 = sparsify_atomic_basis_target(
            systems,
            tmap1,
            target_info_layout,
            sparse_properties=sparse_properties,
        )

    return tmap1
