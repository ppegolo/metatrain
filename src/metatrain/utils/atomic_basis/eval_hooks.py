"""Evaluation-side hooks for atomic-basis targets.

Mirrors :mod:`.trainer_hooks` for ``mtt eval``: the CLI only ever talks to the
:class:`NullEvalHooks` interface, and everything density-specific stays in
this package. Unlike the trainer, evaluation needs no scaling or
composition-model gymnastics — predictions are already in physical units with
all contributions included, so the density error is directly
``Δc^T S Δc`` with ``Δc = c_ML − c_RI``.
"""

import math
from typing import Any, Callable, Dict, List, Mapping, Optional

import torch
from metatomic.torch import System


class NullEvalHooks:
    """No-op evaluation hooks: the default for MLIP-style evaluations."""

    def collate_transforms(self, dtype: torch.dtype) -> List[Callable]:
        """Collate transforms to append to the eval dataloader's pipeline.

        :param dtype: dtype for any tensors the transforms attach.
        :return: List of collate transforms.
        """
        return []

    def update(
        self,
        systems: List[System],
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
        extra_data: Dict[str, Any],
    ) -> None:
        """Accumulate per-batch metrics.

        :param systems: Systems in the batch.
        :param predictions: Model predictions (physical units).
        :param targets: Reference targets from the dataset.
        :param extra_data: The batch's extra data.
        """
        return None

    def finalize(self) -> Dict[str, float]:
        """Return the accumulated metrics, keyed as ``"<target> <metric>"``.

        :return: Metrics to merge into the evaluation report.
        """
        return {}


class DensityErrorEvalHooks(NullEvalHooks):
    """Accumulates the real-space density RMSE for RI-coefficient targets.

    Per target: ``RMSE = sqrt(Σ_i Δc_i^T M_i Δc_i / Σ_i N_atoms,i)`` over the
    evaluated systems, where ``M`` is the two-centre metric matrix of the
    auxiliary basis: the overlap ``S`` (the root-mean integrated squared error
    of the predicted density, per atom) or the Coulomb kernel ``J`` (the
    root-mean electrostatic self-energy of the density error, per atom).

    The density loss consumes the *dense* (per-λ, species-in-samples, padded)
    map layout the training pipeline uses: both the model's sparse per-species
    predictions and the collated targets are densified in :meth:`update`.

    :param target_to_aux_basis: Mapping from RI target name to its auxiliary
        basis.
    :param target_infos: Target infos of the evaluated targets; the layouts
        drive the densification.
    :param metric: Two-centre metric — ``"overlap"`` or ``"coulomb"``.
    """

    def __init__(
        self,
        target_to_aux_basis: Mapping[str, str],
        target_infos: Mapping[str, Any],
        metric: str = "overlap",
    ):
        # Lazy import: the loss module is only needed on the density path.
        from metatrain.utils.loss import DensityMSELossViaC

        from .helpers import DensifyStatics

        self._target_to_aux_basis = dict(target_to_aux_basis)
        self._metric = metric
        self._target_infos = {
            name: target_infos[name] for name in self._target_to_aux_basis
        }
        self._densify_statics = {
            name: DensifyStatics(info.layout)
            for name, info in self._target_infos.items()
        }
        self._loss_fns = {
            name: DensityMSELossViaC(
                name=name,
                gradient=None,
                weight=1.0,
                reduction="sum",
                metric=metric,
            )
            for name in self._target_to_aux_basis
        }
        self._error_sums = dict.fromkeys(self._target_to_aux_basis, 0.0)
        self._atom_counts = dict.fromkeys(self._target_to_aux_basis, 0)

    def collate_transforms(self, dtype: torch.dtype) -> List[Callable]:
        from .pyscf import get_metric_matrices_transform

        # No cross-epoch caching: evaluation sees each system exactly once.
        return [
            get_metric_matrices_transform(
                self._target_to_aux_basis, self._metric, dtype
            )
        ]

    def _densify(
        self,
        name: str,
        tensor_map: Any,
        systems: List[System],
        system_ids_fn: Callable[[], torch.Tensor],
    ) -> Any:
        """Densify a sparse per-species map to the layout the loss expects.

        Maps already in the dense (λ-keyed) form pass through unchanged, so
        eval keeps working if the dataloader pipeline ever densifies upstream.

        :param name: Target name, used to look up the densify statics.
        :param tensor_map: The sparse per-species map to densify.
        :param systems: The batch's systems, in batch order.
        :param system_ids_fn: Returns the ``"system"`` sample values the map
            uses for the batch's systems, in batch order (batch-local
            ``0..B-1`` for model predictions, the dataset's native ids for
            collated targets). Only called on the sparse path.
        :return: The densified map (or the input, if already dense).
        """
        if all(dim in ("o3_lambda", "o3_sigma") for dim in tensor_map.keys.names):
            return tensor_map

        from .helpers import prepare_atomic_basis_targets

        return prepare_atomic_basis_targets(
            [system.to(device="cpu") for system in systems],
            system_ids_fn(),
            tensor_map,
            self._target_infos[name].layout,
            fill_value=torch.nan,
            statics=self._densify_statics[name],
        )

    def update(
        self,
        systems: List[System],
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
        extra_data: Dict[str, Any],
    ) -> None:
        from .pyscf import metric_matrix_name

        n_atoms = sum(len(system) for system in systems)
        cpu = torch.device("cpu")

        # Model predictions number the batch's systems 0..B-1; collated
        # targets keep the dataset's native ids, recovered from the system
        # index extra data. The loss aligns the two by batch order.
        def prediction_ids() -> torch.Tensor:
            return torch.arange(len(systems), dtype=torch.int64)

        def target_ids() -> torch.Tensor:
            index_map = extra_data.get("mtt::aux::system_index")
            if index_map is None:
                raise ValueError(
                    "The density error needs the `mtt::aux::system_index` extra "
                    "data; evaluate a DiskDataset-backed dataset."
                )
            return index_map[0].values[:, 0].to(dtype=torch.int64, device=cpu)

        for name, loss_fn in self._loss_fns.items():
            if name not in predictions or name not in targets:
                continue
            # The metric runs on CPU and in float64: the densify statics live
            # there (and pad in float64), and the cost is negligible next to
            # computing the overlap matrices.
            prediction = self._densify(
                name,
                predictions[name].to(device=cpu, dtype=torch.float64),
                systems,
                prediction_ids,
            )
            target = self._densify(
                name,
                targets[name].to(device=cpu, dtype=torch.float64),
                systems,
                target_ids,
            )
            matrix_key = metric_matrix_name(name, self._metric)
            batch_error = loss_fn(
                {name: prediction},
                {name: target},
                {matrix_key: extra_data[matrix_key].to(device=cpu)},
            )
            self._error_sums[name] += float(batch_error.item())
            self._atom_counts[name] += n_atoms

    def finalize(self) -> Dict[str, float]:
        return {
            f"{name} density RMSE ({self._metric}, per atom)": math.sqrt(
                error_sum / self._atom_counts[name]
            )
            for name, error_sum in self._error_sums.items()
            if self._atom_counts[name] > 0
        }


def get_eval_hooks(
    density_error: Optional[Mapping[str, Any]],
    target_infos: Mapping[str, Any],
) -> NullEvalHooks:
    """Select the evaluation hooks for this run.

    :param density_error: The eval options' ``density_error`` section
        (``{"aux_basis": <str or per-target dict>}``), or ``None``/empty when
        the density error was not requested.
    :param target_infos: Target infos of the evaluation dataset, used to
        detect which targets are atomic-basis targets.
    :return: :class:`DensityErrorEvalHooks` when requested; the no-op
        :class:`NullEvalHooks` otherwise.
    """
    if not density_error:
        return NullEvalHooks()

    atomic_basis_targets = [
        name
        for name, info in target_infos.items()
        if getattr(info, "is_atomic_basis", False)
    ]
    if not atomic_basis_targets:
        raise ValueError(
            "`density_error` was requested, but none of the evaluated targets "
            "is an atomic-basis (RI coefficient) target."
        )

    from .pyscf import resolve_ri_aux_basis

    aux_basis = density_error["aux_basis"]
    target_to_aux_basis = {
        name: resolve_ri_aux_basis(name, aux_basis) for name in atomic_basis_targets
    }
    return DensityErrorEvalHooks(
        target_to_aux_basis, target_infos, metric=density_error.get("metric", "overlap")
    )
