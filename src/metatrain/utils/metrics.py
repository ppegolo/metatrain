import copy
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch.distributed
from metatensor.torch import TensorMap
from metatomic.torch import System


def _block_missing_from_prediction(
    prediction: TensorMap, block_key: Any, key: str
) -> bool:
    """Check (and warn) whether a target block is missing from the predictions.

    The model may legitimately not predict every block of the target (e.g. a
    composition model only fits invariant blocks of a spherical target), so such
    blocks are skipped when computing the error metrics. warnings.warn (rather
    than logging.warning) so this is only reported once, instead of on every
    batch.

    :param prediction: the prediction TensorMap for the target named ``key``.
    :param block_key: the target block key to look for in the predictions.
    :param key: the name of the target, used in the warning message.
    :return: whether the block is missing from the predictions.
    """
    if block_key not in prediction.keys:
        warnings.warn(
            f"Block {block_key} of target '{key}' is not present in "
            "the model's predictions. It will be skipped when "
            "computing the error metrics, which will be computed "
            "over fewer blocks than the full target.",
            stacklevel=3,
        )
        return True
    return False


def _get_global_keys(keys: List[str]) -> List[str]:
    # Collect all keys across ranks, in case some ranks have seen different keys than
    # others
    local_keys = list(keys)
    world_size = torch.distributed.get_world_size()
    gathered_keys: List[Any] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered_keys, local_keys)
    global_keys: set[str] = set()
    for keys_from_rank in gathered_keys:
        if keys_from_rank is not None:
            global_keys.update(keys_from_rank)
    return sorted(list(global_keys))


class RMSEAccumulator:
    """Accumulates the RMSE between predictions and targets for an arbitrary
    number of keys, each corresponding to one target.

    :param separate_blocks: if true, the RMSE will be computed separately for each
        block in the target and prediction ``TensorMap`` objects.
    """

    def __init__(self, separate_blocks: bool = False) -> None:
        self.information: Dict[str, Tuple[float, int]] = {}
        """A dictionary mapping each target key to a tuple containing the sum of
        squared errors and the number of elements for which the error has been
        computed."""

        self.separate_blocks = separate_blocks
        """Whether the RMSE should be computed separately for each block in the
        target and prediction ``TensorMap`` objects."""

    def update(
        self,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Dict[str, TensorMap]] = None,
    ) -> None:
        """Updates the accumulator with new predictions and targets.

        :param predictions: A dictionary of predictions, where the keys correspond
            to the keys in the targets dictionary, and the values are the predictions.
        :param targets: A dictionary of targets, where the keys correspond to the keys
            in the predictions dictionary, and the values are the targets.
        :param extra_data: A dictionary of extra data, where the keys correspond to
            mask keys (i.e. "{target_key}_mask"), and the values are the masks to apply
            when computing the RMSE.
        """

        for key, target in targets.items():
            prediction = predictions[key]

            # Get the mask from extra data if present
            mask = None
            if extra_data is not None:
                mask_key = f"{key}_mask"
                if mask_key in extra_data:
                    mask = extra_data[mask_key]

            for block_key in target.keys:
                if _block_missing_from_prediction(prediction, block_key, key):
                    continue

                target_block = target.block(block_key)
                prediction_block = prediction.block(block_key)

                key_to_write = copy.deepcopy(key)
                if self.separate_blocks:
                    key_to_write += " ("
                    for name, value in zip(
                        block_key.names, block_key.values, strict=True
                    ):
                        key_to_write += f"{name}={int(value)},"
                    key_to_write = key_to_write[:-1]
                    key_to_write += ")"

                if key_to_write not in self.information:  # create key if not present
                    self.information[key_to_write] = (0.0, 0)

                if mask is None:
                    # Get a mask that ignores NaN values in the target
                    mask_as_tensor = ~torch.isnan(target_block.values)
                    rmse_value = (
                        (
                            (
                                prediction_block.values[mask_as_tensor]
                                - target_block.values[mask_as_tensor]
                            )
                            ** 2
                        )
                        .sum()
                        .item()
                    )
                else:
                    mask_as_tensor = mask.block(block_key).values
                    rmse_value = (
                        (
                            (
                                prediction_block.values[mask_as_tensor]
                                - target_block.values[mask_as_tensor]
                            )
                            ** 2
                        )
                        .sum()
                        .item()
                    )

                self.information[key_to_write] = (
                    self.information[key_to_write][0] + rmse_value,
                    self.information[key_to_write][1] + mask_as_tensor.sum().item(),
                )

                for gradient_name, target_gradient in target_block.gradients():
                    if (
                        f"{key_to_write}_{gradient_name}_gradients"
                        not in self.information
                    ):
                        self.information[
                            f"{key_to_write}_{gradient_name}_gradients"
                        ] = (0.0, 0)
                    prediction_gradient = prediction_block.gradient(gradient_name)

                    if mask is None:
                        # Get a mask that ignores NaN values in the target
                        mask_as_tensor = ~torch.isnan(target_gradient.values)
                        gradient_rmse_value = (
                            (
                                (
                                    prediction_gradient.values[mask_as_tensor]
                                    - target_gradient.values[mask_as_tensor]
                                )
                                ** 2
                            )
                            .sum()
                            .item()
                        )
                    else:
                        mask_as_tensor = (
                            mask.block(block_key).gradient(gradient_name).values
                        )
                        gradient_rmse_value = (
                            (
                                (
                                    prediction_gradient.values[mask_as_tensor]
                                    - target_gradient.values[mask_as_tensor]
                                )
                                ** 2
                            )
                            .sum()
                            .item()
                        )
                    self.information[f"{key_to_write}_{gradient_name}_gradients"] = (
                        self.information[f"{key_to_write}_{gradient_name}_gradients"][0]
                        + gradient_rmse_value,
                        self.information[f"{key_to_write}_{gradient_name}_gradients"][1]
                        + mask_as_tensor.sum().item(),
                    )

    def finalize(
        self,
        not_per_atom: List[str],
        is_distributed: bool = False,
        device: torch.device = None,
    ) -> Dict[str, float]:
        """Finalizes the accumulator and returns the RMSE for each key.

        All keys will be returned as "{key} RMSE (per atom)" in the output dictionary,
        unless ``key`` contains one or more of the strings in ``not_per_atom``,
        in which case "{key} RMSE" will be returned.

        :param not_per_atom: a list of strings. If any of these strings are present in
            a key, the RMSE key will not be labeled as "(per atom)".
        :param is_distributed: if true, the RMSE will be computed across all ranks
            of the distributed system.
        :param device: the local device to use for the computation. Only needed if
            ``is_distributed`` is :obj:`python:True`.

        :return: The RMSE for each key.
        """

        if is_distributed:
            # Make sure to collect all keys across ranks, in case some ranks have seen
            # different keys than others
            sorted_global_keys = _get_global_keys(list(self.information.keys()))

            for key in sorted_global_keys:
                if key in self.information:
                    sse = torch.tensor(self.information[key][0], device=device)
                    n_elems = torch.tensor(self.information[key][1], device=device)
                else:
                    sse = torch.tensor(0.0, device=device)
                    n_elems = torch.tensor(0, device=device)
                torch.distributed.all_reduce(sse)
                torch.distributed.all_reduce(n_elems)
                self.information[key] = (sse.item(), n_elems.item())

        finalized_info = {}
        for key, value in self.information.items():
            if any([s in key for s in not_per_atom]):
                out_key = f"{key} RMSE"
            else:
                out_key = f"{key} RMSE (per atom)"
            finalized_info[out_key] = (value[0] / value[1]) ** 0.5

        return finalized_info


class MAEAccumulator:
    """Accumulates the MAE between predictions and targets for an arbitrary
    number of keys, each corresponding to one target.

    :param separate_blocks: if true, the RMSE will be computed separately for each
        block in the target and prediction ``TensorMap`` objects.
    """

    information: Dict[str, Tuple[float, int]]
    separate_blocks: bool

    def __init__(self, separate_blocks: bool = False) -> None:
        self.information = {}
        """A dictionary mapping each target key to a tuple containing the sum of
        absolute errors and the number of elements for which the error has been
        computed."""

        self.separate_blocks = separate_blocks
        """Whether the MAE should be computed separately for each block in the
        target and prediction ``TensorMap`` objects."""

    def update(
        self,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Dict[str, TensorMap]] = None,
    ) -> None:
        """Updates the accumulator with new predictions and targets.

        :param predictions: A dictionary of predictions, where the keys correspond
            to the keys in the targets dictionary, and the values are the predictions.
        :param targets: A dictionary of targets, where the keys correspond to the keys
            in the predictions dictionary, and the values are the targets.
        :param extra_data: A dictionary of extra data, where the keys correspond to
            mask keys (i.e. "{target_key}_mask"), and the values are the masks to apply
            when computing the MAE.
        """

        for key, target in targets.items():
            prediction = predictions[key]

            # Get the mask from extra data if present
            mask = None
            if extra_data is not None:
                mask_key = f"{key}_mask"
                if mask_key in extra_data:
                    mask = extra_data[mask_key]

            for block_key in target.keys:
                if _block_missing_from_prediction(prediction, block_key, key):
                    continue

                target_block = target.block(block_key)
                prediction_block = prediction.block(block_key)

                key_to_write = copy.deepcopy(key)
                if self.separate_blocks:
                    key_to_write += " ("
                    for name, value in zip(
                        block_key.names, block_key.values, strict=True
                    ):
                        key_to_write += f"{name}={int(value)},"
                    key_to_write = key_to_write[:-1]
                    key_to_write += ")"

                if key_to_write not in self.information:  # create key if not present
                    self.information[key_to_write] = (0.0, 0)

                if mask is None:
                    # Get a mask that ignores NaN values in the target
                    mask_as_tensor = ~torch.isnan(target_block.values)
                    mae_value = (
                        (
                            prediction_block.values[mask_as_tensor]
                            - target_block.values[mask_as_tensor]
                        )
                        .abs()
                        .sum()
                        .item()
                    )
                else:
                    mask_as_tensor = mask.block(block_key).values
                    mae_value = (
                        (
                            prediction_block.values[mask_as_tensor]
                            - target_block.values[mask_as_tensor]
                        )
                        .abs()
                        .sum()
                        .item()
                    )

                self.information[key_to_write] = (
                    self.information[key_to_write][0] + mae_value,
                    self.information[key_to_write][1] + mask_as_tensor.sum().item(),
                )

                for gradient_name, target_gradient in target_block.gradients():
                    if (
                        f"{key_to_write}_{gradient_name}_gradients"
                        not in self.information
                    ):
                        self.information[
                            f"{key_to_write}_{gradient_name}_gradients"
                        ] = (0.0, 0)
                    prediction_gradient = prediction_block.gradient(gradient_name)

                    if mask is None:
                        # Get a mask that ignores NaN values in the target
                        mask_as_tensor = ~torch.isnan(target_gradient.values)
                        gradient_mae_value = (
                            (
                                prediction_gradient.values[mask_as_tensor]
                                - target_gradient.values[mask_as_tensor]
                            )
                            .abs()
                            .sum()
                            .item()
                        )
                    else:
                        mask_as_tensor = (
                            mask.block(block_key).gradient(gradient_name).values
                        )
                        gradient_mae_value = (
                            (
                                prediction_gradient.values[mask_as_tensor]
                                - target_gradient.values[mask_as_tensor]
                            )
                            .abs()
                            .sum()
                            .item()
                        )

                    self.information[f"{key_to_write}_{gradient_name}_gradients"] = (
                        self.information[f"{key_to_write}_{gradient_name}_gradients"][0]
                        + gradient_mae_value,
                        self.information[f"{key_to_write}_{gradient_name}_gradients"][1]
                        + mask_as_tensor.sum().item(),
                    )

    def finalize(
        self,
        not_per_atom: List[str],
        is_distributed: bool = False,
        device: torch.device = None,
    ) -> Dict[str, float]:
        """Finalizes the accumulator and returns the MAE for each key.

        All keys will be returned as "{key} MAE (per atom)" in the output dictionary,
        unless ``key`` contains one or more of the strings in ``not_per_atom``,
        in which case "{key} MAE" will be returned.

        :param not_per_atom: a list of strings. If any of these strings are present in
            a key, the MAE key will not be labeled as "(per atom)".
        :param is_distributed: if true, the MAE will be computed across all ranks
            of the distributed system.
        :param device: the local device to use for the computation. Only needed if
            ``is_distributed`` is :obj:`python:True`.

        :return: The MAE for each key.
        """

        if is_distributed:
            # Make sure to collect all keys across ranks, in case some ranks have seen
            # different keys than others
            sorted_global_keys = _get_global_keys(list(self.information.keys()))

            for key in sorted_global_keys:
                if key in self.information:
                    sae = torch.tensor(self.information[key][0], device=device)
                    n_elems = torch.tensor(self.information[key][1], device=device)
                else:
                    sae = torch.tensor(0.0, device=device)
                    n_elems = torch.tensor(0, device=device)
                torch.distributed.all_reduce(sae)
                torch.distributed.all_reduce(n_elems)
                self.information[key] = (sae.item(), n_elems.item())

        finalized_info = {}
        for key, value in self.information.items():
            if any([s in key for s in not_per_atom]):
                out_key = f"{key} MAE"
            else:
                out_key = f"{key} MAE (per atom)"
            finalized_info[out_key] = value[0] / value[1]

        return finalized_info


class EquivarianceAccumulator:
    """Accumulates the equivariance error of a model over a dataset.

    This consumes the ``<name>_var`` outputs of a metatomic
    ``SymmetrizedModel``, which contain, for every sample of a target, the
    variance of the back-rotated predictions over a rotation grid, summed over
    the component axes. Pooling these values element-wise gives exactly the
    mean squared deviation that :class:`RMSEAccumulator` would report between
    the model and a perfectly equivariant reference, so the finalized values
    are directly comparable with the corresponding accuracy RMSEs (same
    normalization, same units).

    Cartesian rank-2 outputs (stress-like) are recombined from their ``_l0``
    (trace) and ``_l2`` (symmetric traceless) parts back to element-wise
    squared errors over the full 3x3 matrix. Gradient-derived quantities are
    reported under the same keys as the accuracy metrics
    (``<target>_positions_gradients``, ``<target>_strain_gradients``); the
    strain-gradient errors are rescaled from stress units by the cell volume
    to match ``dE/dstrain``.

    :param target_names: names of the outputs requested from the symmetrized
        model.
    :param energy_gradients: gradients (among ``"positions"`` and
        ``"strain"``) that the energy target was evaluated with, i.e. for
        which the symmetrized model was called with ``compute_gradients=True``.
    :param energy_name: name of the energy target the gradients were derived
        from, used to name the gradient metrics like the accuracy ones.
    """

    def __init__(
        self,
        target_names: List[str],
        energy_gradients: Optional[List[str]] = None,
        energy_name: str = "energy",
    ) -> None:
        self.information: Dict[str, Tuple[float, int]] = {}
        """A dictionary mapping each metric key to a tuple containing the sum
        of squared deviations and the number of tensor elements pooled."""

        self._norm_information: Dict[str, Tuple[float, int]] = {}
        """Same structure, accumulating the ``<name>_norm_squared`` outputs:
        the squared magnitude of the model outputs, pooled and normalized
        exactly like the squared deviations. Used by :meth:`noise_floors`."""

        # maps each decomposed output name of the symmetrized model to the
        # metric key it contributes to and how (see update())
        self._specs: Dict[str, Tuple[str, str]] = {}
        for name in target_names:
            if name == "energy":
                self._specs["energy_l0"] = ("energy", "generic")
            elif name in (
                "forces",
                "non_conservative_force",
                "non_conservative_forces",
            ):
                self._specs[name + "_l1"] = (name, "generic")
            elif name in ("stress", "non_conservative_stress"):
                self._specs[name + "_l0"] = (name, "rank2_l0")
                self._specs[name + "_l2"] = (name, "rank2_l2")
            else:
                self._specs[name] = (name, "generic")

        for gradient in energy_gradients or []:
            if gradient == "positions":
                self._specs["forces_l1"] = (
                    f"{energy_name}_positions_gradients",
                    "generic",
                )
            elif gradient == "strain":
                self._specs["stress_l0"] = (
                    f"{energy_name}_strain_gradients",
                    "rank2_l0_volume",
                )
                self._specs["stress_l2"] = (
                    f"{energy_name}_strain_gradients",
                    "rank2_l2_volume",
                )

    def update(
        self,
        symmetrized_results: Dict[str, TensorMap],
        systems: List[System],
    ) -> None:
        """Updates the accumulator with the results of a symmetrized model.

        :param symmetrized_results: the dictionary returned by a
            ``SymmetrizedModel`` forward call on ``systems``.
        :param systems: the systems of the batch, used for per-atom
            normalization and cell volumes.
        """
        device = systems[0].positions.device
        dtype = systems[0].positions.dtype
        num_atoms = torch.tensor(
            [len(system) for system in systems], device=device, dtype=dtype
        )
        volumes = torch.stack(
            [torch.abs(torch.linalg.det(system.cell)) for system in systems]
        )

        for var_key, variance in symmetrized_results.items():
            if not var_key.endswith("_var"):
                continue
            name = var_key[: -len("_var")]
            if name not in self._specs:
                continue
            metric_key, mode = self._specs[name]
            mean = symmetrized_results[name + "_mean"]
            norm_squared = symmetrized_results[name + "_norm_squared"]

            if metric_key not in self.information:
                self.information[metric_key] = (0.0, 0)
                self._norm_information[metric_key] = (0.0, 0)

            for key, block in variance.items():
                multiplicity = 1
                for component in mean.block(key).components:
                    multiplicity *= len(component)

                values = block.values
                system = block.samples.column("system").to(
                    dtype=torch.long, device=values.device
                )
                scale = torch.ones_like(num_atoms)
                if mode.endswith("_volume"):
                    # dE/dstrain = volume * stress
                    scale = scale * volumes
                if "atom" not in block.samples.names:
                    # per-structure quantities are compared per atom, like in
                    # average_by_num_atoms
                    scale = scale / num_atoms
                scale_squared = (scale.to(values.device)[system] ** 2).view(
                    -1, *[1] * (len(values.shape) - 1)
                )
                values = values * scale_squared
                norm_values = norm_squared.block(key).values * scale_squared

                # the l0 (full trace) and l2 (half-weight symmetric traceless)
                # parts of a rank-2 Cartesian tensor recombine to the
                # element-wise squared error over the 3x3 matrix as
                # sum_ij = 2 * sum(l2) + sum(l0) / 3; elements are counted
                # once, in the l0 branch
                if mode.startswith("rank2_l0"):
                    sum_sq = values.sum().item() / 3.0
                    norm_sum = norm_values.sum().item() / 3.0
                    n_elems = len(system) * 9 * values.shape[-1]
                elif mode.startswith("rank2_l2"):
                    sum_sq = 2.0 * values.sum().item()
                    norm_sum = 2.0 * norm_values.sum().item()
                    n_elems = 0
                else:
                    sum_sq = values.sum().item()
                    norm_sum = norm_values.sum().item()
                    n_elems = len(system) * multiplicity * values.shape[-1]

                self.information[metric_key] = (
                    self.information[metric_key][0] + sum_sq,
                    self.information[metric_key][1] + n_elems,
                )
                self._norm_information[metric_key] = (
                    self._norm_information[metric_key][0] + norm_sum,
                    self._norm_information[metric_key][1] + n_elems,
                )

    def finalize(self, not_per_atom: List[str]) -> Dict[str, float]:
        """Finalizes the accumulator and returns the pooled equivariance error.

        All keys will be returned as "{key} equivariance RMSE (per atom)" in
        the output dictionary, unless ``key`` contains one or more of the
        strings in ``not_per_atom``, in which case
        "{key} equivariance RMSE" will be returned.

        :param not_per_atom: a list of strings. If any of these strings are
            present in a key, the metric key will not be labeled as
            "(per atom)".

        :return: The pooled equivariance error for each key.
        """
        finalized_info = {}
        for key, value in self.information.items():
            out_key = self._finalized_key(key, not_per_atom)
            # for a (nearly) equivariant model, the accumulated variance is
            # dominated by round-off of the second-moment estimator and can
            # come out slightly negative
            finalized_info[out_key] = (max(value[0], 0.0) / value[1]) ** 0.5

        return finalized_info

    def noise_floors(
        self, not_per_atom: List[str], resolution: float
    ) -> Dict[str, float]:
        """Estimates, for each metric of :meth:`finalize`, the error level
        corresponding to the round-off of the model outputs.

        The floor is the RMS magnitude of the output (pooled and normalized
        exactly like the errors) times ``resolution``, the relative resolution
        of the model outputs (e.g. ``torch.finfo(torch.float32).eps`` for a
        float32 model). Equivariance errors at or below this level are
        dominated by numerical round-off rather than by symmetry breaking of
        the model.

        :param not_per_atom: same as in :meth:`finalize`, so that the keys
            match.
        :param resolution: relative resolution of the model outputs.

        :return: The noise floor for each key of :meth:`finalize`.
        """
        floors = {}
        for key, value in self._norm_information.items():
            out_key = self._finalized_key(key, not_per_atom)
            floors[out_key] = resolution * (max(value[0], 0.0) / value[1]) ** 0.5
        return floors

    @staticmethod
    def _finalized_key(key: str, not_per_atom: List[str]) -> str:
        if any([s in key for s in not_per_atom]):
            return f"{key} equivariance RMSE"
        return f"{key} equivariance RMSE (per atom)"


def get_selected_metric(metric_dict: Dict[str, float], selected_metric: str) -> float:
    """
    Selects and/or calculates a (user-)selected metric from a dictionary of metrics.

    This is useful when choosing the best model from a training run.

    :param metric_dict: A dictionary of metrics, where the keys are the names of the
        metrics and the values are the corresponding values.
    :param selected_metric: The metric to return. This can be one of the following:
        - "loss": return the loss value
        - "rmse_prod": return the product of all RMSEs
        - "mae_prod": return the product of all MAEs

    :return: The value of the selected metric.
    """
    if selected_metric == "loss":
        metric = metric_dict["loss"]
    elif selected_metric == "rmse_prod":
        metric = 1
        for key in metric_dict:
            if "RMSE" in key:
                metric *= metric_dict[key]
    elif selected_metric == "mae_prod":
        metric = 1
        for key in metric_dict:
            if "MAE" in key:
                metric *= metric_dict[key]
    else:
        raise ValueError(
            f"Selected metric {selected_metric} not recognized. "
            "Please select from 'loss', 'rmse_prod', or 'mae_prod'."
        )
    return metric
