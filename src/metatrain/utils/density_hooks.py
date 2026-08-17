"""Trainer-side support for density (RI-coefficient) losses.

A density loss needs one thing from a trainer that a pointwise loss does not: a
two-centre metric matrix attached to every batch, built from the system's geometry.
This module collects that behind one object, so that adding density support to a
trainer is a single splice into its collate functions.

A trainer wires it in like this::

    density = get_density_hooks(self.hypers["loss"])

    CollateFn(
        target_keys,
        callables=[
            atomic_basis_transform,
            *density.training_collate_transforms(),  # before augmentation
            augmentation_callable,
            *base_callables,
        ],
    )
    CollateFn(
        target_keys,
        callables=[
            atomic_basis_transform,
            *density.validation_collate_transforms(),
            *base_callables,
        ],
    )

Without a density loss both lists are empty, so the trainer needs no conditionals and
pays nothing.

The two lists differ because a density loss may be configured as a *metric* rather
than trained on. A metric is evaluated on validation only, and building its matrices
is expensive, so the training collate must not build them.

The one ordering constraint is that these run **before** augmentation: the metric
depends on the geometry, and it is the *unaugmented* geometry the reference
coefficients were fitted in. Comparing coefficients in that same frame is then
handled generically -- the losses declare
:attr:`~metatrain.utils.loss.LossInterface.evaluate_in_original_frame`, and
:func:`~metatrain.utils.augmentation.get_augmentation_transform` picks the
augmentation workflow that honours it. None of that is density-specific, and none of
it lives here.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .pyscf_loss import (
    get_density_geometry_transform,
    get_ec_machinery_transform,
    get_metric_matrices_transform,
    make_metric_spec,
)


#: Loss types that need auxiliary-basis metric matrices attached to each batch.
DENSITY_LOSS_TYPES = ("density_mse_via_c", "density_mse_via_w")

#: Loss types that need the electrostatic-complementarity machinery instead.
EC_LOSS_TYPES = ("ec_mse",)


def _terms(specs: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Flatten a loss configuration into ``(target, specification)`` pairs.

    A target carries either one specification or a list of them (see
    :py:class:`~metatrain.utils.loss.LossAggregator`), and every term of a list
    needs its own machinery: a density term and an EC term on the same target
    ask for different things. Anything that is not a mapping — the ``"mse"``
    shorthand, say — is skipped, since no density loss can be spelled that way.

    :param specs: Loss specifications keyed by target name.
    :return: One pair per term, in configuration order.
    """
    pairs: List[Tuple[str, Dict[str, Any]]] = []
    for target_name, spec in specs.items():
        entries = spec if isinstance(spec, (list, tuple)) else [spec]
        pairs.extend(
            (target_name, entry) for entry in entries if isinstance(entry, dict)
        )
    return pairs


def _metric_transforms(
    aux_bases_by_metric: Dict[str, Dict[str, str]],
) -> List[Callable]:
    """One transform per metric; targets sharing a basis share one computation.

    :param aux_bases_by_metric: ``{metric spec: {target: aux_basis}}``.
    :return: The collate transforms.
    """
    return [
        get_metric_matrices_transform(targets_map, metric)
        for metric, targets_map in aux_bases_by_metric.items()
    ]


class DensityLossHooks:
    """
    Everything a trainer must do to support density losses.

    Build with :func:`get_density_hooks` rather than directly; it returns an inactive
    instance when no density loss is configured.

    :param trained: Mapping from metric name to the ``{target: aux_basis}`` served by
        it, for losses that are trained on.
    :param reported: The same, for losses that are only reported as metrics.
    :param ec_trained: ``{target: aux_basis}`` of the EC losses that are trained on.
    :param ec_reported: The same, for EC losses only reported as metrics.
    :param ec_jitter: Standard deviation in Angstrom of the random partner shift
        applied to training batches. Validation always uses the true placement,
        so a reported EC stays the real score.
    :param geometry_trained: Whether any trained density loss uses the torch
        metric backend, which needs the unaugmented geometry attached to
        training batches instead of the matrices themselves.
    :param geometry_reported: The same, for density metrics evaluated on
        validation only.
    """

    def __init__(
        self,
        trained: Dict[str, Dict[str, str]],
        reported: Dict[str, Dict[str, str]],
        ec_trained: Optional[Dict[str, str]] = None,
        ec_reported: Optional[Dict[str, str]] = None,
        ec_jitter: float = 0.0,
        geometry_trained: bool = False,
        geometry_reported: bool = False,
    ) -> None:
        self._trained = trained
        self._reported = reported
        self._ec_trained = ec_trained or {}
        self._ec_reported = ec_reported or {}
        self._ec_jitter = float(ec_jitter)
        self._geometry_trained = bool(geometry_trained)
        self._geometry_reported = bool(geometry_reported)

    def training_collate_transforms(self) -> List[Callable]:
        """
        Metric-matrix transforms for training batches, to run **before** augmentation.

        :return: Collate transforms; empty when nothing is trained on a density loss.
        """
        transforms = _metric_transforms(self._trained)
        if self._geometry_trained:
            transforms.append(get_density_geometry_transform())
        if self._ec_trained:
            transforms.append(
                get_ec_machinery_transform(self._ec_trained, self._ec_jitter)
            )
        return transforms

    def validation_collate_transforms(self) -> List[Callable]:
        """
        Metric-matrix transforms for validation batches.

        Covers both the trained losses -- which are also evaluated on validation --
        and any reported as metrics.

        :return: Collate transforms; empty when inactive.
        """
        combined: Dict[str, Dict[str, str]] = {
            metric: dict(targets_map) for metric, targets_map in self._trained.items()
        }
        for metric, targets_map in self._reported.items():
            combined.setdefault(metric, {}).update(targets_map)
        transforms = _metric_transforms(combined)
        if self._geometry_trained or self._geometry_reported:
            transforms.append(get_density_geometry_transform())
        ec_combined = dict(self._ec_trained)
        ec_combined.update(self._ec_reported)
        if ec_combined:
            transforms.append(get_ec_machinery_transform(ec_combined))  # jitter=0
        return transforms


def _uses_torch_backend(specs: Dict[str, Any]) -> bool:
    """Whether any density loss among ``specs`` uses the torch metric backend.

    Those losses rebuild their matrices from the geometry on the training
    device, so the collate side ships the geometry instead of matrices.

    :param specs: Loss specifications keyed by target name.
    :return: ``True`` when at least one does.
    """
    return any(
        spec.get("type") in DENSITY_LOSS_TYPES
        and spec.get("backend", "pyscf") == "torch"
        for _, spec in _terms(specs)
    )


def _aux_bases_by_metric(specs: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Group the density losses among ``specs`` by the metric spec they need.

    Losses on the torch metric backend are excluded: they need no matrices
    attached to the batch (see :py:func:`_uses_torch_backend`).

    :param specs: Loss specifications keyed by target name.
    :return: ``{metric spec: {target: aux_basis}}``, empty when none is a density
        loss.
    """
    grouped: Dict[str, Dict[str, str]] = {}
    for target_name, spec in _terms(specs):
        if spec.get("type") not in DENSITY_LOSS_TYPES:
            continue
        if spec.get("backend", "pyscf") == "torch":
            continue
        # Must be built exactly as the loss builds it: the spec is both the
        # extra_data key and the cache key, so any divergence between the two
        # sides silently loses the matrix the loss asks for.
        metric = make_metric_spec(
            spec.get("metric", "overlap"),
            spec.get("omega", 0.0),
            spec.get("eps"),
            spec.get("charge_weight", 0.0),
            spec.get("dipole_weight", 0.0),
            spec.get("quadrupole_weight", 0.0),
            spec.get("esp_weight", 0.0),
            spec.get("esp_shell"),
            spec.get("group_charge_weight", 0.0),
            spec.get("interface_esp_weight", 0.0),
        )
        grouped.setdefault(metric, {})[target_name] = spec["aux_basis"]
    return grouped


def _ec_targets(specs: Dict[str, Any]) -> Dict[str, str]:
    """The ``{target: aux_basis}`` of the EC losses among ``specs``.

    :param specs: Loss specifications keyed by target name.
    :return: Mapping for the EC machinery transform, empty when none is an EC
        loss.
    """
    return {
        target_name: spec["aux_basis"]
        for target_name, spec in _terms(specs)
        if spec.get("type") in EC_LOSS_TYPES
    }


def _ec_jitter(specs: Dict[str, Any]) -> float:
    """The partner jitter configured by the EC losses among ``specs``.

    :param specs: Loss specifications keyed by target name.
    :return: The jitter in Angstrom, 0 when unset.
    :raises ValueError: If two EC losses ask for different jitters; the shift is
        a property of the geometry, so one batch cannot honour two of them.
    """
    values = {
        float(spec.get("partner_jitter", 0.0))
        for _, spec in _terms(specs)
        if spec.get("type") in EC_LOSS_TYPES
    }
    if len(values) > 1:
        raise ValueError(
            "the EC losses ask for different 'partner_jitter' values "
            f"({sorted(values)}); the partner shift is one property of the "
            "geometry and cannot differ between targets of the same batch."
        )
    return values.pop() if values else 0.0


def get_density_hooks(
    loss_hypers: Union[str, Dict[str, Any], None],
    metrics: Optional[Dict[str, Any]] = None,
) -> DensityLossHooks:
    """
    Build the density hooks a trainer needs, or an inactive instance if none.

    :param loss_hypers: The trainer's ``loss`` hyperparameter, keyed by target name.
        A string (the shorthand for "this loss type for every target") configures no
        density loss.
    :param metrics: The ``metrics`` block, keyed by target name. A density metric is
        evaluated on validation only, so its matrices are not built for training
        batches.
    :return: The hooks for this configuration.
    """
    trained = _aux_bases_by_metric(loss_hypers) if isinstance(loss_hypers, dict) else {}
    ec_trained = _ec_targets(loss_hypers) if isinstance(loss_hypers, dict) else {}
    return DensityLossHooks(
        trained,
        _aux_bases_by_metric(metrics or {}),
        ec_trained,
        _ec_targets(metrics or {}),
        _ec_jitter(loss_hypers) if isinstance(loss_hypers, dict) else 0.0,
        geometry_trained=(
            _uses_torch_backend(loss_hypers) if isinstance(loss_hypers, dict) else False
        ),
        geometry_reported=_uses_torch_backend(metrics or {}),
    )
