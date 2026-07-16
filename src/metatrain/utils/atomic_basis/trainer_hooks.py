"""Trainer-side orchestration for atomic-basis (RI density) training.

Everything a trainer needs to support the RI density losses lives here, behind
a two-method-family hook interface: collate transforms that attach metric
matrices / density-fit constants, physical-unit rescaling before the loss, and
the optional validation-set density metric. Trainers that don't train RI
targets get :class:`NullTrainerHooks`, whose every method is a no-op — the
trainer code itself stays agnostic of target types, and other architectures
(e.g. phace) can adopt density training by instantiating the same hooks.
"""

from typing import Any, Callable, Dict, List, Optional, cast

import torch
from metatomic.torch import System

from ..additive import add_additive
from ..loss import DensityMSELossViaC, LossSpecification


RI_LOSS_TYPES = {"density_mse_via_c", "density_mse_via_w"}


class NullTrainerHooks:
    """No-op hooks: the path taken by every non-atomic-basis training run."""

    def collate_transforms(
        self, dtype: torch.dtype
    ) -> tuple[List[Callable], List[Callable]]:
        """Collate transforms to append to the (train, validation) chains.

        :param dtype: Model dtype for any attached tensors.
        :return: ``(train_transforms, val_transforms)``, both empty here.
        """
        return [], []

    def prepare_for_loss(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
        systems: List[System],
        model: torch.nn.Module,
        scaled_predictions: Optional[Dict[str, Any]] = None,
        scaled_targets: Optional[Dict[str, Any]] = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Rescale predictions/targets to the units the loss expects (no-op).

        :param predictions: Model predictions per target.
        :param targets: Reference targets per target name.
        :param systems: Systems in the batch.
        :param model: Unwrapped model (no DDP wrapper).
        :param scaled_predictions: Optional precomputed per-target-scaled maps.
        :param scaled_targets: Optional precomputed per-target-scaled maps.
        :return: ``(predictions, targets)`` unchanged.
        """
        return predictions, targets

    def update_validation_metrics(
        self,
        scaled_predictions: Dict[str, Any],
        scaled_targets: Dict[str, Any],
        extra_data: Dict[str, Any],
    ) -> None:
        """Accumulate extra per-batch validation metrics (no-op).

        :param scaled_predictions: Per-target-scaled predictions (dense maps).
        :param scaled_targets: Per-target-scaled targets (dense maps).
        :param extra_data: The batch's extra data.
        """

    def finalize_validation_metrics(
        self, is_distributed: bool, device: torch.device
    ) -> Dict[str, float]:
        """Finalize and reset the extra validation metrics.

        :param is_distributed: Whether to reduce across ranks.
        :param device: Device for distributed reductions.
        :return: Metric name to value; empty here.
        """
        return {}


class AtomicBasisTrainerHooks(NullTrainerHooks):
    """RI density-training hooks: transforms, unit rescaling, density metric.

    :param loss_hypers: Per-target loss specifications from the training hypers.
    :param train_targets: Target infos as seen by the model during training.
    :param ri_aux_basis: The trainer's ``ri_aux_basis`` hyper.
    :param log_density_loss: The trainer's ``log_density_loss`` hyper.
    """

    def __init__(
        self,
        loss_hypers: Dict[str, LossSpecification],
        train_targets: Dict[str, Any],
        ri_aux_basis: Any,
        log_density_loss: bool,
    ) -> None:
        self._loss_hypers = loss_hypers
        self._train_targets = train_targets
        self._ri_aux_basis = ri_aux_basis
        self._log_density_loss = log_density_loss

        self._via_c_targets: set[str] = {
            name
            for name, spec in loss_hypers.items()
            if spec.get("type") == "density_mse_via_c"
        }
        self._via_w_targets: set[str] = {
            name
            for name, spec in loss_hypers.items()
            if spec.get("type") == "density_mse_via_w"
        }

        self._density_loss_fn: Optional[DensityMSELossViaC] = None
        if log_density_loss:
            # Use the first (and typically only) RI target for the density
            # metric; non-RI losses must not be picked. collate_transforms
            # raises a clear error when no RI loss is configured at all.
            ri_targets = [
                name
                for name in loss_hypers
                if name in self._via_c_targets or name in self._via_w_targets
            ]
            if ri_targets:
                self._density_loss_fn = DensityMSELossViaC(
                    name=ri_targets[0],
                    gradient=None,
                    weight=1.0,
                    reduction="mean",
                    metric="overlap",
                )
        self._density_acc: Optional[torch.Tensor] = None

    def update_validation_metrics(
        self,
        scaled_predictions: Dict[str, Any],
        scaled_targets: Dict[str, Any],
        extra_data: Dict[str, Any],
    ) -> None:
        """Accumulate the real-space density L2 metric for one batch.

        ``L = dc^T S dc`` with ``dc = c_ML - c_RI`` (CM-removed); the maps must
        be the dense (pre-sparsification) ones. Accumulated as a detached
        device tensor; one sync per epoch happens in
        :meth:`finalize_validation_metrics`.

        :param scaled_predictions: Per-target-scaled predictions (dense maps).
        :param scaled_targets: Per-target-scaled targets (dense maps).
        :param extra_data: The batch's extra data (overlap matrices etc.).
        """
        if self._density_loss_fn is None:
            return
        density_batch = self._density_loss_fn(
            scaled_predictions, scaled_targets, extra_data
        ).detach()
        self._density_acc = (
            density_batch
            if self._density_acc is None
            else self._density_acc + density_batch
        )

    def finalize_validation_metrics(
        self, is_distributed: bool, device: torch.device
    ) -> Dict[str, float]:
        """Reduce, convert and reset the density-metric accumulator.

        :param is_distributed: Whether to reduce across ranks.
        :param device: Device for distributed reductions.
        :return: ``{"density_loss": value}`` when enabled, else empty.
        """
        if self._density_loss_fn is None:
            return {}
        total = (
            self._density_acc
            if self._density_acc is not None
            else torch.zeros((), device=device)
        )
        if is_distributed:
            torch.distributed.all_reduce(total)
        self._density_acc = None
        return {"density_loss": float(total.item())}

    def collate_transforms(
        self, dtype: torch.dtype = torch.float64
    ) -> tuple[list[Callable], list[Callable]]:
        """
        Build the collate transforms required for RI density losses.

        The validation transforms additionally include the overlap-matrix
        computation for the RI targets when ``log_density_loss`` is enabled; the
        training transforms never do, since the density metric is only evaluated
        on the validation set.

        Transforms run *before* CM removal and scaling, which is necessary for the
        density-fit constant pre-computation.

        :param dtype: dtype of the metric matrices. They are stored ragged
            (:py:class:`~metatrain.utils.atomic_basis.pyscf.RaggedMetricMatrices`)
            and carried raw past ``save_buffer``, so they can be the model dtype
            (e.g. float32), halving their memory/transport.
        :return: ``(train_transforms, val_transforms)`` collate transform lists.
        """
        # Lazy import: the PySCF-side machinery only loads when RI density
        # losses are actually configured.
        from .pyscf import (
            get_density_fit_constant_transform,
            get_metric_matrices_transform,
            resolve_ri_aux_basis,
            ri_projections_name,
        )

        ri_loss_types = RI_LOSS_TYPES
        ri_aux_basis = self._ri_aux_basis
        log_density_loss = self._log_density_loss

        # Only losses of an RI type ever need metric matrices; other targets
        # (e.g. an energy MSE in the same config) must not trigger PySCF work.
        ri_target_specs = {
            name: spec
            for name, spec in self._loss_hypers.items()
            if spec.get("type", "mse") in ri_loss_types
        }

        if ri_target_specs and ri_aux_basis is None:
            raise ValueError(
                "Training with RI density losses ('density_mse_via_c', "
                "'density_mse_via_w') requires 'ri_aux_basis' to be set."
            )
        if log_density_loss and not ri_target_specs:
            raise ValueError(
                "'log_density_loss' requires at least one loss of type "
                "'density_mse_via_c' or 'density_mse_via_w'."
            )
        if not ri_target_specs:
            return [], []

        # Build per-metric, per-target → aux_basis mappings.
        metric_targets: dict[str, dict[str, str]] = {}  # metric → {target: aux_basis}
        # Maps density_mse_via_w target name → the extra_data key for its projections.
        density_fit_target_to_proj_key: dict[str, str] = {}

        for target_name, target_spec in ri_target_specs.items():
            metric = cast(str, target_spec.get("metric", "overlap"))
            aux_basis = resolve_ri_aux_basis(
                target_name,
                cast(str, ri_aux_basis)
                if isinstance(ri_aux_basis, str)
                else cast(dict, ri_aux_basis),
            )
            metric_targets.setdefault(metric, {})[target_name] = aux_basis
            if target_spec.get("type") == "density_mse_via_w":
                proj_key = cast(
                    str,
                    target_spec.get(
                        "projections_key", ri_projections_name(target_name)
                    ),
                )
                density_fit_target_to_proj_key[target_name] = proj_key

        # The density-loss validation metric additionally needs overlap matrices
        # for the RI targets. Only the validation dataloaders pay for them: the
        # training loop never evaluates the density metric.
        val_metric_targets = {
            metric: dict(targets_map) for metric, targets_map in metric_targets.items()
        }
        if log_density_loss:
            for target_name in ri_target_specs:
                aux_basis = resolve_ri_aux_basis(
                    target_name,
                    cast(str, ri_aux_basis)
                    if isinstance(ri_aux_basis, str)
                    else cast(dict, ri_aux_basis),
                )
                val_metric_targets.setdefault("overlap", {}).setdefault(
                    target_name, aux_basis
                )

        # Build transform lists: one matrix transform per metric, plus the
        # density-fit constant (needed by the loss in both train and val;
        # must run before CM removal and scaling).
        def _build_transforms(
            metric_targets: dict[str, dict[str, str]],
            cache_across_epochs: bool,
        ) -> list[Callable]:
            transforms: list[Callable] = []
            for metric, targets_map in metric_targets.items():
                transforms.append(
                    get_metric_matrices_transform(
                        targets_map, metric, dtype, cache_across_epochs
                    )
                )
            if density_fit_target_to_proj_key:
                transforms.append(
                    get_density_fit_constant_transform(density_fit_target_to_proj_key)
                )
            return transforms

        # Metric matrices are cached per system across epochs in the
        # persistent dataloader workers: the train transforms run before the
        # rotational augmenter (unrotated geometries; the density losses
        # un-rotate their residuals instead), and validation applies no
        # augmentation at all.
        return (
            _build_transforms(metric_targets, cache_across_epochs=True),
            _build_transforms(val_metric_targets, cache_across_epochs=True),
        )

    def prepare_for_loss(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
        systems: List[System],
        model: torch.nn.Module,
        scaled_predictions: Optional[Dict[str, Any]] = None,
        scaled_targets: Optional[Dict[str, Any]] = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Rescale predictions (and targets for density_mse_via_c) to the physical units
        required by the active RI loss type before passing to the loss function.

        direct-c:
            No change.  Predictions = (c_ML − CM)/σ_t, targets = (c_RI − CM)/σ_t.

        density_mse_via_c  (L = Δc^T M Δc):
            Multiply both by σ_t.
            Predictions → c_ML − CM,  targets → c_RI − CM.

        density_mse_via_w  (L = c_ML^T M c_ML − 2 c_ML^T w):
            Multiply predictions by σ_t, then add CM back.
            Predictions → c_ML.  Targets (c_RI) are not used by the loss module;
            the reference data are w_RI and the constant, both in extra_data.

        :param predictions: Model predictions per target.
        :param targets: Reference targets per target name.
        :param systems: Systems in the batch.
        :param model: Unwrapped model (no DDP wrapper).
        :param scaled_predictions: Optional precomputed ``scaler(predictions)``
            (same flags as used here); avoids rescaling twice when the caller
            already needs the scaled maps, as in the validation loop.
        :param scaled_targets: Optional precomputed ``scaler(targets)``.
        :return: ``(predictions, targets)`` rescaled as described above.
        """
        if not self._via_c_targets and not self._via_w_targets:
            return predictions, targets

        all_ri_targets = self._via_c_targets | self._via_w_targets
        loss_predictions = dict(predictions)
        loss_targets = dict(targets)

        # Apply σ_t to predictions for all RI targets that need physical-unit
        # rescaling, in one scaler call (or reuse the caller's scaled maps).
        pred_names = [t for t in all_ri_targets if t in predictions]
        if pred_names:
            rescaled = (
                scaled_predictions
                if scaled_predictions is not None
                else model.scaler(
                    systems,
                    {t: predictions[t] for t in pred_names},
                    remove=False,
                    use_per_target_scales=True,
                    use_per_property_scales=False,
                )
            )
            for target_name in pred_names:
                loss_predictions[target_name] = rescaled[target_name]

        # Apply σ_t to targets for density_mse_via_c (so both sides are CM-removed).
        targ_names = [t for t in self._via_c_targets if t in targets]
        if targ_names:
            rescaled = (
                scaled_targets
                if scaled_targets is not None
                else model.scaler(
                    systems,
                    {t: targets[t] for t in targ_names},
                    remove=False,
                    use_per_target_scales=True,
                    use_per_property_scales=False,
                )
            )
            for target_name in targ_names:
                loss_targets[target_name] = rescaled[target_name]

        # Add CM back to predictions for density_mse_via_w.
        if self._via_w_targets:
            loss_predictions = self._add_composition_model_contribution(
                model,
                systems,
                loss_predictions,
                self._via_w_targets,
            )

        return loss_predictions, loss_targets

    def _add_composition_model_contribution(
        self,
        model: torch.nn.Module,
        systems: List[System],
        predictions: Dict[str, Any],
        target_names: set[str],
    ) -> Dict[str, Any]:
        """
        Add composition-model (CM) contributions back to predictions.

        This reverses the CM-removal applied by the collate pipeline, giving
        fully-reconstructed RI coefficients c_ML = (c_ML − CM) + CM.
        CM values are detached so no gradient flows through them.

        :param model: Unwrapped model holding the composition model.
        :param systems: Systems in the batch.
        :param predictions: CM-removed model predictions per target.
        :param target_names: Targets to add the CM contribution back to.
        :return: Predictions with CM contributions added back.
        """
        additive_model = model.additive_models[0]
        cm_targets = {
            k: predictions[k]
            for k in target_names
            if k in self._train_targets
            and k in additive_model.outputs
            and k in predictions
        }
        if not cm_targets:
            return predictions

        # add_additive detaches the CM values, so no gradient flows through them.
        updated = add_additive(systems, cm_targets, additive_model, self._train_targets)
        new_predictions = dict(predictions)
        new_predictions.update(updated)
        return new_predictions


def get_trainer_hooks(
    loss_hypers: Dict[str, LossSpecification],
    train_targets: Dict[str, Any],
    ri_aux_basis: Any,
    log_density_loss: bool,
) -> NullTrainerHooks:
    """Select the trainer hooks for this training run.

    :param loss_hypers: Per-target loss specifications from the training hypers.
    :param train_targets: Target infos as seen by the model during training.
    :param ri_aux_basis: The trainer's ``ri_aux_basis`` hyper.
    :param log_density_loss: The trainer's ``log_density_loss`` hyper.
    :return: :class:`AtomicBasisTrainerHooks` when any RI density loss (or the
        density metric) is configured; the no-op :class:`NullTrainerHooks`
        otherwise, keeping non-atomic-basis training free of this machinery.
    """
    uses_ri = any(
        spec.get("type", "mse") in RI_LOSS_TYPES for spec in loss_hypers.values()
    )
    if uses_ri or log_density_loss:
        return AtomicBasisTrainerHooks(
            loss_hypers, train_targets, ri_aux_basis, log_density_loss
        )
    return NullTrainerHooks()
