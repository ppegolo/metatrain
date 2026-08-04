import logging
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    System,
    register_autograd_neighbors,
)
from torch.utils.data import DataLoader

from metatrain.utils.abc import ModelInterface
from metatrain.utils.data import (
    CollateFn,
    CombinedDataLoader,
    Dataset,
    DatasetInfo,
    unpack_batch,
)
from metatrain.utils.data.atom_pair_helpers import check_no_atom_pair_targets
from metatrain.utils.data.target_info import (
    is_auxiliary_output,
)
from metatrain.utils.io import model_from_checkpoint
from metatrain.utils.last_layer import (
    assemble_block_weights,
    block_key_name,
    resolve_last_layer_slices,
)
from metatrain.utils.metadata import merge_metadata
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists_transform,
)

from . import checkpoints
from .calibration import (
    GaussianCRPSCalibrator,
    RatioCalibrator,
)
from .documentation import ModelHypers


class LLPRUncertaintyModel(ModelInterface[ModelHypers]):
    __checkpoint_version__ = 5

    ensemble_gradient_outputs: List[str]

    # all torch devices and dtypes are supported, if they are supported by the wrapped
    # the check is performed in the trainer
    __supported_devices__ = ["cuda", "cpu", "mps"]
    __supported_dtypes__ = [torch.float32, torch.float64, torch.bfloat16, torch.float16]
    # more to be added if needed

    __default_metadata__ = ModelMetadata(
        references={
            "architecture": [
                "LLPR (uncertainty method): https://iopscience.iop.org/article/10.1088/2632-2153/ad805f",  # noqa: E501
                "LPR (if using per-atom uncertainty): https://pubs.acs.org/doi/10.1021/acs.jctc.3c00704",  # noqa: E501
            ],
        }
    )

    """A wrapper that adds LLPR uncertainties to a model.

    In order to be compatible with this class, a model needs to declare its
    last-layer feature layout per target (``last_layer_feature_sizes`` and
    ``last_layer_feature_map``, see the ``declare_*`` helpers in
    :mod:`metatrain.utils.last_layer`) and be capable of returning last-layer
    features (see auxiliary outputs in metatrain), optionally per atom to
    calculate LPRs (per-atom uncertainties) with the LLPR method.

    Optionally, in order to be compatible with the LLPR ensemble capabilities of this
    class, the wrapped model also needs to have last-layer weights accessible for each
    target block. These can be declared either as parameter names in the
    ``last_layer_parameter_names`` attribute (target name -> block key -> list of
    state-dict parameter names, concatenated along the feature axis in the order of
    the last-layer features), or, for readout weights embedded in larger flat
    parameters, as slices in ``last_layer_parameter_slices`` (see
    :mod:`metatrain.utils.last_layer`).

    All uncertainties provided by this class are standard deviations (as opposed to
    variances). Prediction rigidities (local and total) can be calculated, according to
    their definition, as the inverse of the square of the standard deviations returned
    by this class.

    :param model: The model to wrap.
    :param ensemble_weight_sizes: The sizes of the ensemble weights, only used
        internally when reloading checkpoints.
    """

    def __init__(self, hypers: ModelHypers, dataset_info: DatasetInfo) -> None:
        super().__init__(hypers, dataset_info, self.__default_metadata__)
        check_no_atom_pair_targets(dataset_info.targets, self.__class__.__name__)

        self.hypers = hypers
        self.dataset_info = dataset_info

    def set_wrapped_model(self, model: ModelInterface) -> None:
        # this function is called after initialization, as well as

        hypers = self.hypers
        dataset_info = self.dataset_info

        # ensemble weight sizes need to be extracted from the hypers

        self.model = model

        # we need the capabilities of the model to be able to infer the capabilities
        # of the LLPR model. Here, we do a trick: we call export on the model to to make
        # it handle the conversion from dataset_info to capabilities, as well as to
        # get its dtype
        old_capabilities = self.model.export().capabilities()
        dtype = getattr(torch, old_capabilities.dtype)

        # Covariance, calibration and uncertainties all need inference-mode
        # predictions (with scaler and additive contributions). Do not rely on
        # `export()` side effects for this: PET's flips the live module to eval
        # mode, SPACE's does not.
        self.model.eval()

        # checks between dataset_info and model outputs
        if dataset_info.length_unit != old_capabilities.length_unit:
            raise ValueError(
                "The length unit in the dataset info is different from the "
                "length unit of the wrapped model"
            )
        for atomic_type in dataset_info.atomic_types:
            if atomic_type not in old_capabilities.atomic_types:
                raise ValueError(
                    f"Atomic type {atomic_type} not supported by the wrapped model"
                )
        for target_name, target in dataset_info.targets.items():
            if target_name not in old_capabilities.outputs:
                raise ValueError(
                    f"Target {target_name} not supported by the wrapped model"
                )
            if target.unit != old_capabilities.outputs[target_name].unit:
                raise ValueError(
                    f"Target {target_name} has unit {target.unit}, but the "
                    f"wrapped model has unit "
                    f"{old_capabilities.outputs[target_name].unit}"
                )

        # Build the LLPR capabilities from the wrapped model's *native* outputs rather
        # than from its export-time capabilities. The latter additionally contain the
        # new-name aliases that metatomic injects for deprecated output names (e.g.
        # `non_conservative_force` for `non_conservative_forces`), while the wrapped
        # model's `forward` only responds to the native names. Building on the native
        # names keeps this wrapper transparent: it requests from and returns to the
        # wrapped model under the names the model actually understands, and metatomic's
        # `AtomisticModel` re-adds the new-name aliases (and bridges engine requests to
        # them) when this wrapper is itself exported.
        backbone_outputs = self.model.supported_outputs()

        # update capabilities: now we have additional outputs for the uncertainty
        declared_feature_sizes: Dict[str, Dict[str, int]] = getattr(
            self.model, "last_layer_feature_sizes", {}
        )
        declared_feature_map: Dict[str, List[str]] = getattr(
            self.model, "last_layer_feature_map", {}
        )
        additional_capabilities = {}
        self.outputs_list = []
        for name, output in backbone_outputs.items():
            if is_auxiliary_output(name):
                continue  # auxiliary output
            if name not in declared_feature_map:
                logging.warning(
                    f"The wrapped model declares no last-layer features for "
                    f"'{name}'; LLPR uncertainties for it are unavailable."
                )
                continue
            self.outputs_list.append(name)
            uncertainty_name = get_uncertainty_name(name)
            additional_capabilities[uncertainty_name] = ModelOutput(
                unit=output.unit,
                sample_kind=output.sample_kind,
                description=output.description,
            )

        self.capabilities = ModelCapabilities(
            outputs={**backbone_outputs, **additional_capabilities},
            atomic_types=old_capabilities.atomic_types,
            interaction_range=old_capabilities.interaction_range,
            length_unit=old_capabilities.length_unit,
            supported_devices=old_capabilities.supported_devices,
            dtype=old_capabilities.dtype,
        )

        # block keys of every target, in the model's layout order and naming convention
        self.target_block_keys: Dict[str, List[str]] = {}
        # blocks with declared readout weights: the only ones ensembles can be
        # sampled for
        self.ensemble_block_keys: Dict[str, List[str]] = {}
        # keys of the target's feature blocks, in the feature output's block order
        self.feature_keys: Dict[str, List[str]] = {}
        # feature block index each target block reads: a shared invariant feature
        # block is read by every target block, equivariant features pair one to one
        self.block_feature_index: Dict[str, List[int]] = {}
        resolved_slices = {}
        model_targets = self.model.dataset_info.targets
        for name in self.outputs_list:
            block_keys = [
                block_key_name(name, key) for key in model_targets[name].layout.keys
            ]
            self.target_block_keys[name] = block_keys

            feature_map = declared_feature_map[name]
            feature_keys = list(declared_feature_sizes[name].keys())
            # the declare_* helpers guarantee a complete, consistent declaration
            assert len(feature_map) == len(block_keys)
            assert all(key in feature_keys for key in feature_map)
            self.feature_keys[name] = feature_keys
            self.block_feature_index[name] = [
                feature_keys.index(key) for key in feature_map
            ]

            resolved_slices[name] = resolve_last_layer_slices(self.model, name)
            self.ensemble_block_keys[name] = [
                block_key
                for block_key in block_keys
                if block_key in resolved_slices[name]
            ]

        # Register covariance, Cholesky and multiplier buffers: one covariance per
        # feature block (it belongs to the features, not to the block reading them),
        # one multiplier per target block (calibrated against that block's
        # residuals).
        for name in self.outputs_list:
            uncertainty_name = get_uncertainty_name(name)
            for block_key, feature_size in declared_feature_sizes[name].items():
                self.register_buffer(
                    f"covariance_{uncertainty_name}_{block_key}",
                    torch.zeros((feature_size, feature_size), dtype=dtype),
                )
                self.register_buffer(
                    f"cholesky_{uncertainty_name}_{block_key}",
                    torch.zeros((feature_size, feature_size), dtype=dtype),
                )
            for block_key in self.target_block_keys[name]:
                self.register_buffer(
                    f"multiplier_{uncertainty_name}_{block_key}",
                    torch.tensor([1.0], dtype=dtype),
                )
                # per-property scales of the wrapped model; uncertainties and
                # ensemble spreads follow them so the multiplier stays scale-free
                self.register_buffer(
                    f"scales_{uncertainty_name}_{block_key}",
                    self._block_scales(name, block_key).to(dtype=dtype),
                )

        self.ensemble_weight_sizes = hypers["num_ensemble_members"]

        # register buffers for ensemble weights and ensemble outputs
        ensemble_outputs = {}
        # ensemble outputs `forward` can attach explicit gradients to, resolved once
        # from the wrapped model's outputs
        self.ensemble_gradient_outputs: List[str] = []
        for name in self.ensemble_weight_sizes:
            if name not in self.outputs_list:
                raise ValueError(
                    f"Output '{name}' in ensembles section is not supported by "
                    "the model"
                )
            ensemble_weights_name = (
                "mtt::aux::" + name.replace("mtt::", "") + "_ensemble_weights"
            )
            if ensemble_weights_name == "mtt::aux::energy_ensemble_weights":
                ensemble_weights_name = "energy_ensemble_weights"
            ensemble_output_name = (
                "mtt::aux::" + name.replace("mtt::", "") + "_ensemble"
            )
            if ensemble_output_name == "mtt::aux::energy_ensemble":
                ensemble_output_name = "energy_ensemble"
            explicit_gradients = self._ensemble_explicit_gradients(name)
            if len(explicit_gradients) > 0:
                self.ensemble_gradient_outputs.append(ensemble_output_name)
            ensemble_outputs[ensemble_output_name] = ModelOutput(
                unit=old_capabilities.outputs[name].unit,
                sample_kind=old_capabilities.outputs[name].sample_kind,
                explicit_gradients=explicit_gradients,
                description=f"ensemble of '{name}'",
            )
        self.capabilities = ModelCapabilities(
            outputs={**self.capabilities.outputs, **ensemble_outputs},
            atomic_types=self.capabilities.atomic_types,
            interaction_range=self.capabilities.interaction_range,
            length_unit=self.capabilities.length_unit,
            supported_devices=self.capabilities.supported_devices,
            dtype=self.capabilities.dtype,
        )
        # One ensemble layer per target block. Each block has its own last layer, with
        # its own number of outputs (a lambda=2 block emits 5 components where a
        # lambda=0 block emits 1), so they cannot share a single layer.
        self.llpr_ensemble_layers = torch.nn.ModuleDict()
        state_dict = self.model.state_dict()
        for name, value in self.ensemble_weight_sizes.items():
            missing = [
                block_key
                for block_key in self.target_block_keys[name]
                if block_key not in self.ensemble_block_keys[name]
            ]
            if len(missing) > 0:
                raise ValueError(
                    f"Cannot generate LLPR ensembles for '{name}': the wrapped model "
                    f"declares no last-layer readout weights for the block(s) "
                    f"{missing}. Uncertainties are still available for this target; "
                    f"remove it from the `num_ensemble_members` section."
                )
            for block_key in self.ensemble_block_keys[name]:
                # (n_properties, n_features) effective last layer of the block
                weights = assemble_block_weights(
                    state_dict, resolved_slices[name][block_key]
                )
                self.llpr_ensemble_layers[f"{name}::{block_key}"] = torch.nn.Linear(
                    weights.shape[1],
                    value * weights.shape[0],
                    bias=False,
                )

    def restart(self, dataset_info: DatasetInfo) -> "LLPRUncertaintyModel":
        # merge old and new dataset info
        merged_info = self.dataset_info.union(dataset_info)
        new_atomic_types = [
            at for at in merged_info.atomic_types if at not in self.model.atomic_types
        ]
        new_targets = {
            key: value
            for key, value in merged_info.targets.items()
            if key not in self.dataset_info.targets
        }
        self.has_new_targets = len(new_targets) > 0

        if self.has_new_targets:
            raise ValueError(
                f"New targets found in the dataset: {new_targets}. "
                "The LLPR ensemble calibration does not support adding new targets."
            )
        if len(new_atomic_types) > 0:
            raise ValueError(
                f"New atomic types found in the dataset: {new_atomic_types}. "
                "The LLPR ensemble calibration does not support adding new atomic "
                "types."
            )

        self.dataset_info = merged_info

        # invoke restart routine for the wrapped model
        self.model.restart(dataset_info)

        return self

    def _get_dataloader(
        self,
        datasets: List[Union[Dataset, torch.utils.data.Subset]],
        batch_size: int,
        is_distributed: bool,
    ) -> DataLoader:
        """
        Create a DataLoader for the provided datasets. As the dataloader is only used to
        accumulate the quantities needed for LLPR calibration, there is no need to
        shuffle or drop the last non-full batch. Distributed sampling can be used or
        not, based on the `is_distributed` argument, and training with double
        precision is enforced.

        :param datasets: List of datasets to create the dataloader from.
        :param batch_size: Batch size to use for the dataloader.
        :param is_distributed: Whether to use distributed sampling or not.
        :return: The created DataLoader.
        """
        # Create the collate function
        targets_keys = list(self.dataset_info.targets.keys())
        requested_neighbor_lists = get_requested_neighbor_lists(self)
        collate_fn = CollateFn(
            target_keys=targets_keys,
            callables=[
                get_system_with_neighbor_lists_transform(requested_neighbor_lists)
            ],
        )

        # Validate dtype from datasets
        if len(datasets) == 0:
            raise ValueError(
                "Cannot create dataloader from empty datasets list. "
                "Please provide non-empty datasets for LLPR calibration."
            )
        if len(datasets[0]) == 0:
            raise ValueError(
                "Cannot create dataloader from empty dataset. "
                "Please provide non-empty datasets for LLPR calibration."
            )

        # Build the dataloaders
        samplers: List[torch.utils.data.Sampler | None]
        if is_distributed:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            samplers = [
                NoPadDistributedSampler(
                    dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                    seed=0,
                )
                for dataset in datasets
            ]
        else:
            samplers = [None] * len(datasets)

        dataloaders = []
        for dataset, sampler in zip(datasets, samplers, strict=True):
            if len(dataset) < batch_size:
                raise ValueError(
                    f"A dataset has fewer samples "
                    f"({len(dataset)}) than the batch size "
                    f"({batch_size}). "
                    "Please reduce the batch size."
                )
            dataloaders.append(
                DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    sampler=sampler,
                    drop_last=False,
                    collate_fn=collate_fn,
                )
            )

        # important to keep shuffle=False for consistent calibration results
        # in distributed training
        return CombinedDataLoader(dataloaders, shuffle=False)

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        any_uncertainty_or_ensemble = False
        for output_name in outputs:
            if output_name.endswith("_uncertainty") or output_name.endswith(
                "_ensemble"
            ):
                any_uncertainty_or_ensemble = True
                break
        if not any_uncertainty_or_ensemble:
            return self.model(systems, outputs, selected_atoms)

        # Uncertainty and ensemble requests stay here: the wrapped model instead
        # receives the corresponding last-layer features and original outputs
        # (removed at the end if not requested by the user).
        outputs_for_model: Dict[str, ModelOutput] = {}
        for name, output in outputs.items():
            if name.endswith("_uncertainty") or name.endswith("_ensemble"):
                target_name = (
                    name.replace("mtt::aux::", "")
                    .replace("_uncertainty", "")
                    .replace("_ensemble", "")
                )
                outputs_for_model[f"mtt::aux::{target_name}_last_layer_features"] = (
                    ModelOutput(sample_kind=output.sample_kind)
                )
                original_name = self._get_original_name(name)
                if original_name not in outputs:
                    # the user's own request for the original output, if any,
                    # takes precedence over this derived one
                    outputs_for_model[original_name] = output
            else:
                outputs_for_model[name] = output

        # Explicit gradients of an energy ensemble require grad-enabled positions/cell
        # and autograd-registered neighbor lists *before* the wrapped model runs.
        # Engines generally hand over systems without any of this set up, so rebuild
        # them here if needed.
        for name, output in outputs.items():
            if (
                name in self.ensemble_gradient_outputs
                and len(output.explicit_gradients) > 0
            ):
                if selected_atoms is not None:
                    # the gradient samples built by `_add_energy_ensemble_gradients`
                    # assume one value sample per system, in order, and cover every
                    # atom of every system. A selection can drop systems from the
                    # value block, which leaves the two out of sync.
                    raise ValueError(
                        f"explicit gradients of '{name}' are not supported together "
                        "with 'selected_atoms'"
                    )
                systems = self._systems_with_grad(systems)
                break

        return_dict = self.model(systems, outputs_for_model, selected_atoms)

        requested_uncertainties: List[str] = []
        for name in outputs.keys():
            if name.endswith("_uncertainty"):
                requested_uncertainties.append(name)

        for uncertainty_name in requested_uncertainties:
            ll_features_name = uncertainty_name.replace(
                "_uncertainty", "_last_layer_features"
            )
            if ll_features_name == "energy_last_layer_features":
                # special case for energy_ensemble
                ll_features_name = "mtt::aux::energy_last_layer_features"
            ll_features = return_dict[ll_features_name]

            original_name = self._get_original_name(uncertainty_name)
            target = return_dict[original_name]
            block_keys = self.target_block_keys[original_name]
            feature_keys = self.feature_keys[original_name]
            feature_indices = self.block_feature_index[original_name]

            # The uncertainty mirrors the target's layout: one block per target
            # block. `1/PR` is computed once per feature block, so target blocks
            # reading a shared invariant feature block share it.
            one_over_pr_cache: Dict[int, torch.Tensor] = {}

            uncertainty_blocks: List[TensorBlock] = []
            block_index = 0
            for target_block in target.blocks():
                feature_index = feature_indices[block_index]
                feature_block = ll_features.block(feature_index)
                if feature_index not in one_over_pr_cache:
                    one_over_pr_cache[feature_index] = self._one_over_pr(
                        uncertainty_name,
                        feature_block.values,
                        feature_keys[feature_index],
                    )
                one_over_pr_values = one_over_pr_cache[feature_index]
                multiplier = self._get_multiplier(
                    uncertainty_name, block_keys[block_index]
                )
                # fold in the wrapped model's scales (see `_block_scales`)
                scales = self._get_scales(uncertainty_name, block_keys[block_index])
                number_of_components = _prod(target_block.values.shape[1:-1])
                num_prop = target_block.values.shape[-1]
                block_values = one_over_pr_values.expand(
                    -1, number_of_components, num_prop
                )
                block_values = block_values.reshape(
                    [block_values.shape[0]] + list(target_block.values.shape[1:])
                )
                # the square root converts the variance to a standard deviation
                uncertainty_blocks.append(
                    TensorBlock(
                        values=torch.sqrt(block_values) * multiplier * scales,
                        samples=feature_block.samples,
                        components=target_block.components,
                        properties=target_block.properties,
                    )
                )
                block_index += 1

            return_dict[uncertainty_name] = TensorMap(
                keys=target.keys, blocks=uncertainty_blocks
            )

        # now deal with potential ensembles (see generate_ensemble method)
        requested_ensembles: List[str] = []
        for name in outputs.keys():
            if name.endswith("_ensemble"):
                requested_ensembles.append(name)

        for ens_name in requested_ensembles:
            original_name = self._get_original_name(ens_name)

            ll_features_name = ens_name.replace("_ensemble", "_last_layer_features")
            if ll_features_name == "energy_last_layer_features":
                # special case for energy_ensemble
                ll_features_name = "mtt::aux::energy_last_layer_features"
            ll_features = return_dict[ll_features_name]

            target = return_dict[original_name]
            block_keys = self.target_block_keys[original_name]
            feature_indices = self.block_feature_index[original_name]

            ensemble_blocks: List[TensorBlock] = []
            block_index = 0
            for target_block in target.blocks():
                # TorchScript cannot build the layer name out of the target's `Labels`
                # here, hence the precomputed `target_block_keys`, paired with the
                # target's blocks positionally.
                layer_name = original_name + "::" + block_keys[block_index]
                feature_block = ll_features.block(feature_indices[block_index])
                block_index += 1

                # Loop needed due to torchscript limitations
                ensemble_values = torch.tensor([0])
                for lin_layer_name, module in self.llpr_ensemble_layers.items():
                    if lin_layer_name == layer_name:
                        # raw ens output shape is (samples, (num_ens * num_prop)) for
                        # invariant features, (samples, components, (num_ens *
                        # num_prop)) for equivariant ones
                        ensemble_values = module(feature_block.values)

                # extract shape of components and properties
                num_samples = ensemble_values.shape[0]
                components_shape = list(target_block.values.shape[1:-1])
                num_prop = target_block.values.shape[-1]

                if feature_block.values.dim() == 3:
                    # Equivariant features carry the component axis themselves and
                    # the readout weights are shared across components, so the
                    # ensemble output already has the component axis in place.
                    num_ens = ensemble_values.shape[-1] // num_prop
                    ensemble_values = ensemble_values.reshape(
                        [num_samples] + components_shape + [num_ens, num_prop]
                    )  # shape: samples, ..., num_ens, num_prop
                else:
                    # The block's ensemble layer holds `num_ens` copies of that
                    # block's last layer, whose rows run over the components first
                    # and the properties last.
                    ensemble_values = ensemble_values.reshape(
                        [num_samples, -1] + components_shape + [num_prop]
                    )  # shape: samples, num_ens, ..., num_prop
                    num_ens = ensemble_values.shape[1]
                    # move num_ens to position before num_prop (-2)
                    ensemble_values = (
                        ensemble_values.reshape(
                            num_samples,
                            num_ens,
                            _prod(components_shape),
                            num_prop,
                        )
                        .swapaxes(1, 2)
                        .reshape([num_samples] + components_shape + [num_ens, num_prop])
                    )  # shape: samples, ..., num_ens, num_prop

                # since we know the exact mean of the ensemble from the model's
                # prediction, it should be mathematically correct to use it to
                # re-center the ensemble. Besides making sure that the average is
                # always correct (so that results will always be consistent between
                # LLPR ensembles and the original model), this also takes care of
                # additive contributions that are not present in the last layer, which
                # can be composition, short-range models, a bias in the last layer,
                # etc.
                ensemble_values = (
                    ensemble_values
                    - ensemble_values.mean(dim=-2, keepdim=True)
                    + target_block.values.unsqueeze(-2)  # ens_dim
                )

                ensemble_values = ensemble_values.reshape(
                    [num_samples] + components_shape + [-1]
                )  # shape: (samples, components, (num_ens * num_prop))

                # prepare the properties Labels object for ensemble output, i.e.
                # account for the num_ens dimension
                old_prop_val = target_block.properties.values
                num_properties = old_prop_val.shape[0]
                ens_idxs = torch.arange(
                    num_ens,
                    device=old_prop_val.device,
                    dtype=old_prop_val.dtype,
                )
                ens_idxs = ens_idxs.repeat_interleave(num_properties).unsqueeze(1)
                if ens_name == "energy_ensemble":
                    # "energy_ensemble" quantity requires a single "energy"
                    # property column holding the ensemble member index
                    ens_prop = Labels(names=["energy"], values=ens_idxs)
                else:
                    exp_prop_val = old_prop_val.repeat(num_ens, 1)
                    new_prop_val = torch.cat([ens_idxs, exp_prop_val], dim=-1)
                    ens_prop = Labels(
                        names=["ensemble_member"] + target_block.properties.names,
                        values=new_prop_val,
                    )

                ensemble_block = TensorBlock(
                    values=ensemble_values,
                    samples=feature_block.samples,
                    components=target_block.components,
                    properties=ens_prop,
                )

                if ens_name in self.ensemble_gradient_outputs:
                    requested_gradients = outputs[ens_name].explicit_gradients
                    want_positions = "positions" in requested_gradients
                    want_strain = "strain" in requested_gradients
                    if want_positions or want_strain:
                        if outputs[ens_name].sample_kind != "system":
                            # `_add_energy_ensemble_gradients` assumes one sample per
                            # system. For a per-atom energy it would silently return
                            # the gradient of the summed energy, labelled as if it
                            # were per-sample, so refuse rather than produce wrong
                            # numbers.
                            raise ValueError(
                                f"explicit gradients of '{ens_name}' are only "
                                "supported for a per-system energy, but this output "
                                "was requested with sample_kind "
                                f"'{outputs[ens_name].sample_kind}'"
                            )
                        self._add_energy_ensemble_gradients(
                            ensemble_block,
                            systems,
                            ensemble_values,
                            ens_prop,
                            num_ens,
                            want_positions,
                            want_strain,
                        )

                ensemble_blocks.append(ensemble_block)

            return_dict[ens_name] = TensorMap(keys=target.keys, blocks=ensemble_blocks)

        # Remove any keys if they were not requested. This can happen for last-layer
        # features needed for uncertainty/ensemble calculation as well as for
        # the original outputs when only uncertainties/ensembles were requested
        for key in list(return_dict.keys()):
            if key not in outputs:
                return_dict.pop(key)

        return return_dict

    def _systems_with_grad(self, systems: List[System]) -> List[System]:
        """Rebuild ``systems`` so that positions and cell require grad and neighbor
        lists are registered with autograd, as needed by
        ``_add_energy_ensemble_gradients``.

        Systems whose positions and cell already require grad (e.g. because the
        engine is doing its own autograd on top of this call) are passed through
        untouched, keeping the caller's graph intact. Everything else is replaced by
        a copy built on grad-enabled positions/cell tensors, with neighbor lists
        re-registered against them.

        A tensor that already requires grad is *reused as is* rather than detached,
        even when only one of positions/cell does. Detaching it would silently cut
        the caller out of the graph: an engine that grad-enables positions only (as
        one doing forces but not stress does) would then get "one of the
        differentiated Tensors appears to not have been used in the graph" from its
        own backward pass.

        :param systems: systems to rebuild.
        :return: list with one system per input system, either the original object
            (already grad-enabled) or its grad-enabled copy.
        """
        new_systems: List[System] = []
        for system in systems:
            if system.positions.requires_grad and system.cell.requires_grad:
                new_systems.append(system)
                continue
            positions = system.positions
            if not positions.requires_grad:
                positions = positions.detach().requires_grad_(True)
            cell = system.cell
            if not cell.requires_grad:
                cell = cell.detach().requires_grad_(True)
            new_system = System(system.types, positions, cell, system.pbc)
            for nl_options in system.known_neighbor_lists():
                neighbors = mts.detach_block(system.get_neighbor_list(nl_options))
                register_autograd_neighbors(new_system, neighbors, False)
                new_system.add_neighbor_list(nl_options, neighbors)
            for data_name in system.known_data():
                new_system.add_data(data_name, system.get_data(data_name))
            new_systems.append(new_system)
        return new_systems

    def _add_energy_ensemble_gradients(
        self,
        ensemble_block: TensorBlock,
        systems: List[System],
        ensemble_values: torch.Tensor,
        ens_prop: Labels,
        num_ens: int,
        want_positions: bool,
        want_strain: bool,
    ) -> None:
        """Attach "positions"/"strain" gradients to an "energy_ensemble" block.

        Positions and cell are differentiated directly (no strain-trick deformation
        needed): since the "strain" gradient is evaluated at strain = identity, the
        virial can be recovered from the positions/cell gradients alone as
        ``positions^T @ d(output)/d(positions) + cell^T @ d(output)/d(cell)``, which is
        numerically equivalent to differentiating w.r.t. an explicit strain parameter.
        This avoids needing a dedicated strain tensor.

        :param ensemble_block: the "energy_ensemble" value block gradients are
            attached to, in place, via ``add_gradient``.
        :param systems: the systems this batch was computed for; ``positions`` and
            ``cell`` must already require grad (ensured by ``_systems_with_grad`` in
            ``forward``, or set up by the caller) for any of this to produce
            non-trivial gradients.
        :param ensemble_values: the (already recentered) ensemble values, with shape
            ``(num_samples, num_ens)`` (one row per system, one column per ensemble
            member). This is only valid for the "energy" quantity, which has no
            components and a single property.
        :param ens_prop: properties of ``ensemble_block``, reused verbatim for the
            gradient blocks, as required by metatensor.
        :param num_ens: number of ensemble members.
        :param want_positions: whether to attach a "positions" gradient.
        :param want_strain: whether to attach a "strain" gradient.
        """
        # one column per ensemble member, i.e. a single property and no components:
        # anything else interleaves the properties with the members, and the
        # member-by-member differentiation below would pick the wrong columns
        if len(ensemble_values.shape) != 2 or ensemble_values.shape[1] != num_ens:
            raise ValueError(
                "explicit gradients of an energy ensemble are only supported for a "
                "single-property energy without components"
            )

        n_systems = len(systems)
        all_positions = [system.positions for system in systems]
        all_cells = [system.cell for system in systems]
        n_atoms_per_system = [p.shape[0] for p in all_positions]
        # cells are only differentiated when the strain gradient is actually wanted:
        # asking autograd for them otherwise costs a gradient per system per member
        # that is then discarded
        grad_inputs = all_positions + all_cells if want_strain else all_positions

        dtype = ensemble_values.dtype
        device = ensemble_values.device
        n_atoms_total = sum(n_atoms_per_system)

        positions_grad_values = torch.zeros(
            (n_atoms_total, 3, num_ens), dtype=dtype, device=device
        )
        strain_grad_values = torch.zeros(
            (n_systems, 3, 3, num_ens) if want_strain else (0, 3, 3, num_ens),
            dtype=dtype,
            device=device,
        )

        for member in range(num_ens):
            grads = torch.autograd.grad(
                [ensemble_values[:, member].sum()],
                grad_inputs,
                # never False: the graph may be the caller's (see `_systems_with_grad`,
                # which passes already-grad-enabled systems straight through), and
                # freeing it here would break the caller's own backward pass with
                # "trying to backward through the graph a second time". The graph is
                # released normally once the last reference to it goes away.
                retain_graph=True,
                create_graph=False,
            )

            atom_offset = 0
            for s in range(n_systems):
                pos_grad = grads[s]
                assert pos_grad is not None

                n_atoms_s = n_atoms_per_system[s]
                # store dE/dr directly (not the force -dE/dr)
                positions_grad_values[
                    atom_offset : atom_offset + n_atoms_s, :, member
                ] = pos_grad
                if want_strain:
                    cell_grad = grads[n_systems + s]
                    assert cell_grad is not None
                    strain_grad_values[s, :, :, member] = (
                        all_positions[s].detach().t() @ pos_grad
                        + all_cells[s].detach().t() @ cell_grad
                    )
                atom_offset += n_atoms_s

        xyz = torch.tensor([[0], [1], [2]], device=device)

        if want_positions:
            sample_col = torch.repeat_interleave(
                torch.arange(n_systems, device=device),
                torch.tensor(n_atoms_per_system, device=device),
            )
            atom_col = torch.cat(
                [torch.arange(n, device=device) for n in n_atoms_per_system]
            )
            ensemble_block.add_gradient(
                "positions",
                TensorBlock(
                    values=positions_grad_values,
                    samples=Labels(
                        names=["sample", "system", "atom"],
                        values=torch.stack([sample_col, sample_col, atom_col], dim=1),
                    ),
                    components=[Labels(["xyz"], xyz)],
                    properties=ens_prop,
                ),
            )

        if want_strain:
            # build the "sample" Labels with the plain constructor, not
            # `Labels.range(...)` as the latter breaks TorchScript compatibility
            ensemble_block.add_gradient(
                "strain",
                TensorBlock(
                    values=strain_grad_values,
                    samples=Labels(
                        names=["sample"],
                        values=torch.arange(n_systems, device=device).reshape(-1, 1),
                    ),
                    components=[Labels(["xyz_1"], xyz), Labels(["xyz_2"], xyz)],
                    properties=ens_prop,
                ),
            )

    def compute_covariance(
        self,
        datasets: List[Union[Dataset, torch.utils.data.Subset]],
        batch_size: int,
        is_distributed: bool,
    ) -> None:
        """A function to compute the covariance matrix for a training set.

        The covariance is stored as a buffer in the model.

        :param datasets: List of datasets to use for covariance calculation.
        :param batch_size: Batch size to use for the dataloader.
        :param is_distributed: Whether to use distributed sampling or not.
        """
        # Create dataloader for the training datasets
        train_loader = self._get_dataloader(
            datasets, batch_size, is_distributed=is_distributed
        )

        device = next(iter(self.buffers())).device
        dtype = next(iter(self.buffers())).dtype
        with torch.no_grad():
            for batch in train_loader:
                systems, targets, _ = unpack_batch(batch)
                n_atoms = torch.tensor(
                    [len(system.positions) for system in systems], device=device
                )
                systems = [system.to(device=device, dtype=dtype) for system in systems]
                outputs_for_targets = {
                    name: ModelOutput(
                        sample_kind="atom"
                        if "atom" in target.block(0).samples.names
                        else "system"
                    )
                    for name, target in targets.items()
                }
                outputs_for_features = {
                    f"mtt::aux::{n.replace('mtt::', '')}_last_layer_features": o
                    for n, o in outputs_for_targets.items()
                }
                output = self.forward(
                    systems, {**outputs_for_targets, **outputs_for_features}
                )
                for name in targets.keys():
                    if name not in self.feature_keys:
                        # no last-layer feature declaration for this target
                        continue
                    ll_feat_tmap = output[
                        f"mtt::aux::{name.replace('mtt::', '')}_last_layer_features"
                    ]
                    uncertainty_name = get_uncertainty_name(name)
                    feature_keys = self.feature_keys[name]

                    # one covariance per feature block, accumulated from that
                    # block of the feature output
                    for feat_index in range(len(feature_keys)):
                        block_values = ll_feat_tmap.block(feat_index).values.detach()
                        # TODO: interface ll_feat calculation with the loss function,
                        # paying attention to normalization w.r.t. n_atoms
                        if outputs_for_targets[name].sample_kind == "system":
                            norm_shape = [n_atoms.shape[0]] + [1] * (
                                len(block_values.shape) - 1
                            )
                            block_values = block_values / n_atoms.reshape(norm_shape)

                        # Flatten components into samples: component-free features
                        # have no component axis (no-op), while equivariant features
                        # share one weight vector across components, whose
                        # Gauss-Newton Hessian is `sum_samples sum_components f f^T`
                        # -- exactly what flattening accumulates.
                        ll_feats = block_values.reshape(-1, block_values.shape[-1])

                        covariance = self._get_covariance(
                            uncertainty_name, feature_keys[feat_index]
                        )
                        covariance += ll_feats.T @ ll_feats

        if is_distributed:
            torch.distributed.barrier()
            # All-reduce the covariance matrices across all processes
            for name in self.outputs_list:
                uncertainty_name = get_uncertainty_name(name)
                for block_key in self.feature_keys[name]:
                    covariance = self._get_covariance(uncertainty_name, block_key)
                    torch.distributed.all_reduce(covariance)

    def compute_cholesky_decomposition(
        self, regularizer: Optional[float] = None
    ) -> None:
        """A function to compute the Cholesky decomposition of the covariance matrix.

        The Cholesky decomposition is stored as a buffer in the model.

        :param regularizer: A regularization parameter to ensure the matrix is
            positive-definite. If not provided, the function will try to compute the
            Cholesky decomposition without regularization and increase the
            regularization parameter until the matrix is positive-definite.
        """

        for name in self.outputs_list:
            uncertainty_name = get_uncertainty_name(name)
            for block_key in self.feature_keys[name]:
                covariance = self._get_covariance(uncertainty_name, block_key).to(
                    dtype=torch.float64
                )
                cholesky = self._get_cholesky(uncertainty_name, block_key)
                if regularizer is not None:
                    cholesky[:] = torch.linalg.cholesky(
                        0.5 * (covariance + covariance.T)
                        + regularizer
                        * torch.eye(
                            covariance.shape[0],
                            device=covariance.device,
                            dtype=torch.float64,
                        )
                    ).to(cholesky.dtype)
                else:
                    # Try with an increasingly high regularization parameter until
                    # the matrix is invertible
                    is_not_pd = True
                    r = 1e-20
                    while is_not_pd and r < 1e16:
                        try:
                            cholesky[:] = torch.linalg.cholesky(
                                0.5 * (covariance + covariance.T)
                                + r
                                * torch.eye(
                                    covariance.shape[0],
                                    device=covariance.device,
                                    dtype=torch.float64,
                                )
                            ).to(cholesky.dtype)
                            is_not_pd = False
                        except RuntimeError:
                            r *= 10.0
                    if is_not_pd:
                        raise RuntimeError(
                            "Could not compute Cholesky decomposition. Something "
                            "went wrong. Please contact the metatrain developers"
                        )
                    else:
                        logging.info(
                            f"Used regularization parameter of {r:.1e} to "
                            f"compute the Cholesky decomposition for `{name}`"
                        )

    def calibrate(
        self,
        datasets: List[Union[Dataset, torch.utils.data.Subset]],
        batch_size: int,
        is_distributed: bool,
        calibration_method: str,
    ) -> None:
        """
        Calibrate the LLPR model.

        This function computes the calibration constants (one for each output)
        that are used to scale the uncertainties in the LLPR model. The
        calibration is performed in a simple way by computing either the calibration
        constant as the mean of the squared residuals divided by the mean of
        the non-calibrated uncertainties (i.e., by minimizing the NLL as a function of
        the calibration constant), or by minimizing the CRPS as a function of the
        calibration constant.

        :param datasets: List of datasets to use for calibration.
        :param batch_size: Batch size to use for the dataloader.
        :param is_distributed: Whether to use distributed sampling or not.
        :param calibration_method: The method to use for calibration. Supported methods
            are "squared_residuals", "absolute_residuals", and "crps". All methods
            assume Gaussian errors. The "squared_residuals" method minimize the negative
            log-likelihood (NLL) as a function of the calibration constant.
            The "absolute_residuals" method estimates the calibration constant based on
            the mean absolute residuals, which can help reduce the effect of large
            outliers.
            The "crps" method minimizes the Continuous Ranked Probability Score (CRPS)
            as a function of the calibration constant.
        """
        valid_loader = self._get_dataloader(
            datasets, batch_size, is_distributed=is_distributed
        )

        device = next(iter(self.buffers())).device
        dtype = next(iter(self.buffers())).dtype

        calibrator: Union[RatioCalibrator, GaussianCRPSCalibrator]

        if calibration_method in ["squared_residuals", "absolute_residuals"]:
            calibrator = RatioCalibrator(method=calibration_method)  # type: ignore[arg-type]
        elif calibration_method == "crps":
            calibrator = GaussianCRPSCalibrator()
        else:
            raise ValueError(
                f"Unknown calibration method '{calibration_method}'! "
                "Supported methods are 'squared_residuals', 'absolute_residuals', and"
                " 'crps'."
            )

        # calibrator key -> its (uncertainty, block) pair; both parts may contain
        # "::", so they cannot be recovered by splitting the key
        calibrated_blocks: Dict[str, Tuple[str, str]] = {}

        with torch.no_grad():
            for batch in valid_loader:
                systems, targets, _ = unpack_batch(batch)
                systems = [system.to(device=device, dtype=dtype) for system in systems]
                targets = {
                    name: target.to(device=device, dtype=dtype)
                    for name, target in targets.items()
                }

                requested_outputs = {}
                for name in targets:
                    if name not in self.feature_keys:
                        # no last-layer feature declaration for this target
                        continue
                    sample_kind = (
                        "atom"
                        if "atom" in targets[name].block(0).samples.names
                        else "system"
                    )
                    requested_outputs[name] = ModelOutput(sample_kind=sample_kind)
                    uncertainty_name = get_uncertainty_name(name)
                    requested_outputs[uncertainty_name] = ModelOutput(
                        sample_kind=sample_kind
                    )

                outputs = self.forward(systems, requested_outputs)

                for name, target in targets.items():
                    if name not in self.feature_keys:
                        continue
                    uncertainty_name = get_uncertainty_name(name)
                    block_keys = self.target_block_keys[name]
                    prediction = outputs[name]
                    uncertainty = outputs[uncertainty_name]

                    # one multiplier per target block, fitted against that
                    # block's residuals
                    for block_index in range(len(prediction.keys)):
                        key = prediction.keys.entry(block_index)
                        pred = prediction.block(block_index).values.detach()
                        targ = target.block(key).values
                        unc = uncertainty.block(block_index).values.detach()

                        residuals = pred - targ

                        block_key = block_keys[block_index]
                        calibrator_key = f"{uncertainty_name}/{block_key}"
                        calibrated_blocks[calibrator_key] = (
                            uncertainty_name,
                            block_key,
                        )
                        calibrator.update(
                            uncertainty_name=calibrator_key,
                            residuals=residuals.reshape(-1, residuals.shape[-1]),
                            uncertainties=unc.reshape(-1, unc.shape[-1]),
                        )

        multipliers = calibrator.finalize()

        for calibrator_key, alpha in multipliers.items():
            uncertainty_name, block_key = calibrated_blocks[calibrator_key]
            multiplier = self._get_multiplier(uncertainty_name, block_key)
            multiplier[:] = alpha.to(device=device, dtype=multiplier.dtype)

    def generate_ensemble(self) -> None:
        """Generate an ensemble of weights for the model.

        The ensemble is generated by sampling from a multivariate normal
        distribution with mean given by the input weights and covariance given
        by the inverse covariance matrix.
        """
        device = next(iter(self.buffers())).device
        dtype = next(iter(self.buffers())).dtype

        state_dict = self.model.state_dict()

        for name, num_members in self.ensemble_weight_sizes.items():
            uncertainty_name = get_uncertainty_name(name)

            # each target block has its own last layer, so it is sampled separately;
            # the covariance belongs to the feature block the last layer reads (the
            # single shared one for invariant features, the block's own for
            # equivariant ones), and the multiplier is the block's own
            resolved = resolve_last_layer_slices(self.model, name)
            for block_key in self.ensemble_block_keys[name]:
                block_index = self.target_block_keys[name].index(block_key)
                feature_index = self.block_feature_index[name][block_index]
                cur_cholesky = self._get_cholesky(
                    uncertainty_name,
                    self.feature_keys[name][feature_index],
                )
                # effective weight matrix of this block's readout, assembled from
                # the slices the architecture declares; shape (n_prop, n_features)
                weights = assemble_block_weights(state_dict, resolved[block_key])
                cur_multiplier = self._get_multiplier(uncertainty_name, block_key)

                # fold in the wrapped model's scales (see `_block_scales`); the
                # re-centering in `forward` pins the ensemble mean regardless
                block_scales = self._get_scales(uncertainty_name, block_key)

                ensemble_weights = []

                for ii in range(weights.shape[0]):
                    z = torch.randn(
                        (weights.shape[1], num_members),
                        device=device,
                        dtype=dtype,
                    )
                    # using the Cholesky decomposition to sample from the multivariate
                    # normal distribution
                    ensemble_displacements = (
                        torch.linalg.solve_triangular(
                            cur_cholesky.T,
                            z,
                            upper=True,
                        )
                        * cur_multiplier.item()
                    )
                    cur_ensemble_weights = (
                        weights[ii].unsqueeze(1) + ensemble_displacements
                    )
                    # rows run over the components first and the properties last,
                    # so the property index is `ii % n_properties`
                    cur_ensemble_weights = (
                        cur_ensemble_weights
                        * block_scales[ii % block_scales.shape[0]].item()
                    )
                    ensemble_weights.append(cur_ensemble_weights)

                stacked_weights = torch.stack(
                    ensemble_weights,
                    axis=-1,
                )  # shape: (ll_feat, n_ens, n_subtarget)
                stacked_weights = stacked_weights.reshape(
                    stacked_weights.shape[0],
                    -1,
                )  # shape: (ll_feat, n_ens * n_subtarget)
                # assign the generated weights
                with torch.no_grad():
                    self.llpr_ensemble_layers[f"{name}::{block_key}"].weight.copy_(
                        stacked_weights.T
                    )

        # add the ensembles to the capabilities
        old_outputs = self.capabilities.outputs
        new_outputs = {}
        for name in self.ensemble_weight_sizes.keys():
            ensemble_name = "mtt::aux::" + name.replace("mtt::", "") + "_ensemble"
            if ensemble_name == "mtt::aux::energy_ensemble":
                ensemble_name = "energy_ensemble"
            new_outputs[ensemble_name] = ModelOutput(
                unit=old_outputs[name].unit,
                sample_kind=old_outputs[name].sample_kind,
                explicit_gradients=self._ensemble_explicit_gradients(name),
                description=f"ensemble of {name}",
            )
        self.capabilities = ModelCapabilities(
            outputs={**old_outputs, **new_outputs},
            atomic_types=self.capabilities.atomic_types,
            interaction_range=self.capabilities.interaction_range,
            length_unit=self.capabilities.length_unit,
            supported_devices=self.capabilities.supported_devices,
            dtype=self.capabilities.dtype,
        )

    def get_checkpoint(self) -> Dict[str, Any]:
        wrapped_model_checkpoint = self.model.get_checkpoint()
        state_dict = {
            k: v for k, v in self.state_dict().items() if not k.startswith("model.")
        }
        checkpoint = {
            "architecture_name": "llpr",
            "model_ckpt_version": self.__checkpoint_version__,
            "metadata": self.metadata,
            "model_data": {
                "hypers": self.hypers,
                "dataset_info": self.dataset_info,
            },
            "epoch": None,
            "best_epoch": None,
            "model_state_dict": state_dict,
            "best_model_state_dict": state_dict,
            "wrapped_model_checkpoint": wrapped_model_checkpoint,
        }
        return checkpoint

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        context: Literal["restart", "finetune", "export"],
    ) -> "LLPRUncertaintyModel":
        model = model_from_checkpoint(checkpoint["wrapped_model_checkpoint"], context)
        if context == "finetune":
            # In this case, we want to allow fine-tuning of the underlying model by
            # extracting it and returning it directly
            return model
        elif context == "restart":
            logging.info(
                "Restart for LLPRUncertaintyModel will attempt continuation of "
                "ensemble calibration"
            )
            logging.info(f"Using latest model from epoch {checkpoint['epoch']}")
            model_state_dict = checkpoint["model_state_dict"]
        elif context == "export":
            # TODO: other models print the best epoch here; consider doing the same
            # Here, it depends on whether we are exporting a model whose ensemble was
            # also trained by backpropagation or not
            model_state_dict = checkpoint["best_model_state_dict"]
            # this is None if the ensemble was not trained by backpropagation
            if model_state_dict is None:
                model_state_dict = checkpoint["model_state_dict"]
        else:
            raise ValueError("Unknown context tag for checkpoint loading!")

        llpr_model = cls(**checkpoint["model_data"])
        llpr_model.set_wrapped_model(model)

        state_dict_iter = iter(model_state_dict.values())
        next(state_dict_iter)
        dtype = next(state_dict_iter).dtype
        # TODO: find a way to refactor this to avoid strict=False
        llpr_model.to(dtype).load_state_dict(model_state_dict, strict=False)
        return llpr_model

    def export(self, metadata: Optional[ModelMetadata] = None) -> AtomisticModel:
        dtype = next(self.parameters()).dtype

        # Make sure the model is all in the same dtype
        # For example, after training, the additive models could still be in
        # float64
        self.to(dtype)

        # Additionally, the composition model contains some `TensorMap`s that cannot
        # be registered correctly with Pytorch. This function moves them:
        try:
            self.model.additive_models[0]._move_weights_to_device_and_dtype(
                torch.device("cpu"), torch.float64
            )
        except Exception:
            # no weights to move
            pass

        metadata = merge_metadata(
            merge_metadata(self.__default_metadata__, metadata),
            self.model.export().metadata(),
        )

        # some wrapped models are not scriptable as-is (e.g. SPACE carries a
        # gradient-based submodule TorchScript cannot compile) and expose a
        # destructive in-place hook to strip themselves for scripting
        if hasattr(self.model, "prepare_for_export"):
            self.model.prepare_for_export()

        return AtomisticModel(self.eval(), metadata, self.capabilities)

    def _get_covariance(self, name: str, block_key: str) -> torch.Tensor:
        name = "covariance_" + name + "_" + block_key
        requested_buffer = torch.tensor(0)
        for n, buffer in self.named_buffers():
            if n == name:
                requested_buffer = buffer
        if requested_buffer.shape == torch.Size([]):
            raise ValueError(f"Covariance for {name} not found.")
        return requested_buffer

    def _get_cholesky(self, name: str, block_key: str) -> torch.Tensor:
        name = "cholesky_" + name + "_" + block_key
        requested_buffer = torch.tensor(0)
        for n, buffer in self.named_buffers():
            if n == name:
                requested_buffer = buffer
        if requested_buffer.shape == torch.Size([]):
            raise ValueError(f"Inverse covariance for {name} not found.")
        return requested_buffer

    def _block_scales(self, name: str, block_key: str) -> torch.Tensor:
        """Per-property scales the wrapped model applies to one target block.

        :param name: name of the target.
        :param block_key: key of the target block.
        :return: the ``(n_properties,)`` scale vector, or a broadcastable one-element
            unit vector if the wrapped model has no scaler or no scales for this
            target.
        """
        scaler = getattr(self.model, "scaler", None)
        if scaler is None:
            return torch.ones(1)
        scales_maps = getattr(scaler.model, "scales", {})
        if name not in scales_maps:
            return torch.ones(1)
        block_index = self.target_block_keys[name].index(block_key)
        scale_values = scales_maps[name].block(block_index).values
        if scale_values.shape[0] > 1 and not torch.allclose(
            scale_values, scale_values[0].expand_as(scale_values)
        ):
            logging.warning(
                f"'{name}' has per-atomic-type scales; LLPR uncertainties and "
                "ensemble spreads are scaled with their average over the types."
            )
        return scale_values.mean(dim=0)

    def _one_over_pr(
        self,
        uncertainty_name: str,
        values: torch.Tensor,
        block_key: str,
    ) -> torch.Tensor:
        """Compute ``1/PR`` for one feature block; same code for PR and LPR.

        :param uncertainty_name: name of the uncertainty output.
        :param values: the feature block's values.
        :param block_key: key of the feature block's Cholesky buffer.
        :return: ``1/PR`` of shape ``(samples, components, 1)`` (components is 1
            for component-less features).
        """
        # (samples, [components...,] features) -> (samples, components, features),
        # with a single component for component-less features
        values = values.reshape(values.shape[0], -1, values.shape[-1])
        v = torch.linalg.solve_triangular(
            self._get_cholesky(uncertainty_name, block_key),
            values.reshape(-1, values.shape[-1]).T,
            upper=False,
        )
        return torch.sum(v**2, dim=0).reshape(values.shape[0], values.shape[1], 1)

    def _get_multiplier(self, name: str, block_key: str) -> torch.Tensor:
        name = "multiplier_" + name + "_" + block_key
        requested_buffer = torch.tensor(0)
        for n, buffer in self.named_buffers():
            if n == name:
                requested_buffer = buffer
        if requested_buffer.shape == torch.Size([]):
            raise ValueError(f"Multiplier for {name} not found.")
        return requested_buffer

    def _get_scales(self, name: str, block_key: str) -> torch.Tensor:
        name = "scales_" + name + "_" + block_key
        requested_buffer = torch.tensor(0)
        for n, buffer in self.named_buffers():
            if n == name:
                requested_buffer = buffer
        if requested_buffer.shape == torch.Size([]):
            raise ValueError(f"Scales for {name} not found.")
        return requested_buffer

    def _get_original_name(self, name: str) -> str:
        # hopefully a bulletproof way to get the original output name from an
        # uncertainty or ensemble name
        if name.endswith("_uncertainty"):
            original_name = name.replace("_uncertainty", "")
        elif name.endswith("_ensemble"):
            original_name = name.replace("_ensemble", "")
        else:
            raise ValueError(f"Output name {name} is neither uncertainty nor ensemble.")
        if original_name.startswith("mtt::aux::"):
            # original name could be either mtt::output or output
            # try the former, return the latter if not found
            # TODO: not sure what happens if both mtt::output and output are there
            original_name = original_name.replace("aux::", "")
            if original_name not in self.capabilities.outputs:
                original_name = original_name.replace("mtt::", "")
        return original_name

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

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.capabilities.outputs

    def _ensemble_explicit_gradients(self, name: str) -> List[str]:
        """Explicit gradients the ensemble of the ``name`` target is able to produce.

        Only energy targets are currently supported.

        :param name: name of the target the ensemble is built from.
        :return: gradient names for the corresponding ensemble output.
        """
        target = self.dataset_info.targets.get(name)
        if target is None or target.quantity != "energy":
            return []
        return ["positions", "strain"]


def get_uncertainty_name(name: str) -> str:
    """Name of the LLPR uncertainty output of a target.

    :param name: name of the target.
    :return: the uncertainty output name.
    """
    if name == "energy":
        uncertainty_name = "energy_uncertainty"
    else:
        uncertainty_name = f"mtt::aux::{name.replace('mtt::', '')}_uncertainty"
    return uncertainty_name


class NoPadDistributedSampler(torch.utils.data.Sampler[int]):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_replicas: int,
        rank: int,
        shuffle: bool = False,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        n = len(self.dataset)
        indices = torch.arange(n, dtype=torch.long)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = indices[torch.randperm(n, generator=g)]
        # no padding, no dropping
        return iter(indices[self.rank :: self.num_replicas].tolist())

    def __len__(self) -> int:
        n = len(self.dataset)
        return (n - self.rank + self.num_replicas - 1) // self.num_replicas


def _prod(list_of_int: List[int]) -> int:
    # for torchscript compatibility (math.prod is not supported)
    result = 1
    for x in list_of_int:
        result = result * x
    return result
