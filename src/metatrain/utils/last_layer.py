"""The last-layer interface between architectures and the LLPR wrapper.

Architectures declare, per target, which last-layer feature block each target
block reads (``last_layer_feature_map``) and the size of each feature block
(``last_layer_feature_sizes``), through the two ``declare_*`` helpers below.
LLPR keeps one covariance per feature block, so a shared invariant feature
block means one covariance shared by all of a target's blocks.

For ensembles, LLPR additionally needs, for every block of a target, the
effective weight matrix of the linear readout that maps the exposed last-layer
features to that block's values. Architectures declare it in one of two ways:

- ``last_layer_parameter_names``: target name -> block key -> list of
  state-dict parameter names whose tensors, concatenated along the feature
  axis, give the block's readout weights (PET, SOAP-BPNN).
- ``last_layer_parameter_slices``: target name -> block key -> list of
  :class:`LastLayerSlice`, for architectures whose readout weights live inside
  a larger flat parameter (e.g. an e3nn ``o3.Linear``) or carry a constant
  normalization factor that is not part of the stored tensor.

The two declarations are exclusive per model: a model that declares
``last_layer_parameter_slices`` uses it for all its targets, and
``last_layer_parameter_names`` is ignored.
"""

from typing import Dict, List, NamedTuple, Tuple

import torch
from metatensor.torch import Labels, LabelsEntry, TensorBlock, TensorMap


# Feature key declared by architectures whose last-layer features are a single
# invariant block shared by every block of a target.
SHARED_FEATURE_KEY = "0"


class LastLayerSlice(NamedTuple):
    """One contiguous piece of a block's effective readout weights.

    The piece is read as ``parameter.reshape(-1)[offset : offset + n * m]``
    where ``(n, m)`` is ``shape``, reshaped to ``shape`` (transposed first if
    ``transpose_stored`` is set, for parameters stored feature-major like e3nn
    path weights), and multiplied by ``scale``. The result is an
    ``(n_properties, n_features)`` matrix; the slices of a block concatenate
    along the feature axis.
    """

    parameter_name: str
    offset: int
    shape: Tuple[int, int]
    scale: float
    transpose_stored: bool


def block_key_name(target_name: str, key: LabelsEntry) -> str:
    """Name of one block of a target, as the architectures name their last layers.

    :param target_name: name of the target the block belongs to.
    :param key: the block's entry in the target's keys.
    :return: the block key, e.g. ``mtt::foo_o3_lambda_2_o3_sigma_1``.
    """
    block_key = target_name
    for name, value in zip(key.names, key.values, strict=True):
        block_key += f"_{name}_{int(value)}"
    return block_key


def declare_shared_last_layer_features(
    model: torch.nn.Module,
    target_name: str,
    layout_keys: Labels,
    feature_size: int,
) -> None:
    """Declare that every block of a target reads one shared invariant
    last-layer feature block (exposed as the only block of the target's
    last-layer feature output).

    :param model: the model declaring its LLPR interface.
    :param target_name: name of the target.
    :param layout_keys: the keys of the target's layout.
    :param feature_size: size of the shared feature block.
    """
    model.last_layer_feature_sizes[target_name] = {SHARED_FEATURE_KEY: feature_size}
    model.last_layer_feature_map[target_name] = [SHARED_FEATURE_KEY] * len(layout_keys)


def declare_per_block_last_layer_features(
    model: torch.nn.Module,
    target_name: str,
    feature_sizes: Dict[str, int],
) -> None:
    """Declare that each block of a target reads its own last-layer feature
    block, exposed in the same order as the target's blocks.

    :param model: the model declaring its LLPR interface.
    :param target_name: name of the target.
    :param feature_sizes: block key -> feature size, in the target's block
        (and feature output block) order.
    """
    model.last_layer_feature_sizes[target_name] = feature_sizes
    model.last_layer_feature_map[target_name] = list(feature_sizes.keys())


def block_aligned_llf_tensormap(
    block_values: List[torch.Tensor],
    samples: Labels,
    keys: Labels,
    components_per_block: List[List[Labels]],
) -> TensorMap:
    """Wrap per-block last-layer feature tensors into the block-aligned
    ``TensorMap`` the LLPR wrapper consumes: the target's keys, one feature
    block per target block, ``feature`` properties, and no component axis on
    the blocks whose target block has none.

    :param block_values: one ``(n_samples, n_components, n_features)`` tensor
        per target block, in the target's block order.
    :param samples: samples labels shared by all feature blocks.
    :param keys: the target's keys.
    :param components_per_block: the target's component labels, per block.
    :return: the block-aligned feature ``TensorMap``.
    """
    blocks: List[TensorBlock] = []
    for block_index, components in enumerate(components_per_block):
        values = block_values[block_index]
        if len(components) == 0:
            values = values.squeeze(1)
        blocks.append(
            TensorBlock(
                values=values,
                samples=samples,
                components=components,
                properties=Labels(
                    names=["feature"],
                    values=torch.arange(
                        values.shape[-1], device=values.device
                    ).unsqueeze(-1),
                ),
            )
        )
    return TensorMap(keys=keys, blocks=blocks)


def resolve_last_layer_slices(
    model: torch.nn.Module, target_name: str
) -> Dict[str, List[LastLayerSlice]]:
    """Resolve the readout-weight slices of every block of a target.

    Uses the model's ``last_layer_parameter_slices`` declaration if the model
    has one (ignoring ``last_layer_parameter_names`` entirely); otherwise
    derives trivial whole-tensor slices from ``last_layer_parameter_names``.
    Blocks absent from both declarations (i.e. blocks whose values are not a
    direct linear readout of the last-layer features) are simply missing from
    the result.

    :param model: the wrapped model.
    :param target_name: name of the target.
    :return: block key -> slices whose concatenation gives that block's
        effective readout weights.
    """
    declared = getattr(model, "last_layer_parameter_slices", None)
    if declared is not None:
        resolved: Dict[str, List[LastLayerSlice]] = {}
        for block_key, slices in declared.get(target_name, {}).items():
            resolved[block_key] = [LastLayerSlice(*s) for s in slices]
        return resolved

    registered = getattr(model, "last_layer_parameter_names", {})
    state_dict = model.state_dict()
    return {
        block_key: [
            LastLayerSlice(
                parameter_name=tensor_name,
                offset=0,
                shape=(
                    state_dict[tensor_name].shape[0],
                    state_dict[tensor_name].shape[1],
                ),
                scale=1.0,
                transpose_stored=False,
            )
            for tensor_name in tensor_names
        ]
        for block_key, tensor_names in registered.get(target_name, {}).items()
    }


def assemble_block_weights(
    state_dict: Dict[str, torch.Tensor], slices: List[LastLayerSlice]
) -> torch.Tensor:
    """Assemble a block's effective readout weights from its slices.

    :param state_dict: the wrapped model's state dict.
    :param slices: the block's weight slices.
    :return: the ``(n_properties, n_features)`` effective weight matrix.
    """
    parts = []
    for s in slices:
        n_properties, n_features = s.shape
        numel = n_properties * n_features
        flat = state_dict[s.parameter_name].reshape(-1)[s.offset : s.offset + numel]
        if s.transpose_stored:
            weights = flat.reshape(n_features, n_properties).T
        else:
            weights = flat.reshape(n_properties, n_features)
        parts.append(weights * s.scale)
    return torch.concatenate(parts, dim=-1)
