"""Loss on a density expanded on an atom-centred auxiliary basis."""

from typing import Any, Dict, List, Optional, Tuple

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

from ..data.dataset import CollateTransform
from ..loss import LossInterface
from ..scaler.remove import remove_scale
from .basis import integrals
from .metric import MetricBuilder


LOSS_TYPE = "density_mse"

#: ``extra_data`` key of the atomic types and positions of the batch.
GEOMETRY_NAME = "mtt::aux::density_geometry"

#: Systems whose reference spin density integrates to less than this (in absolute
#: value) count as singlets for ``sign_invariant_singlets``.
SINGLET_TOLERANCE = 0.5


def _removed_scale_name(target_name: str) -> str:
    return f"{target_name}_removed_scale"


def _pack_geometry(systems: List[System]) -> TensorMap:
    device = systems[0].positions.device
    samples = torch.cat(
        [
            torch.stack(
                [
                    torch.full((len(system),), i, dtype=torch.int32, device=device),
                    torch.arange(len(system), dtype=torch.int32, device=device),
                ],
                dim=1,
            )
            for i, system in enumerate(systems)
        ]
    )
    values = torch.cat(
        [
            torch.cat(
                [
                    system.types.to(torch.float64).reshape(-1, 1),
                    system.positions.to(torch.float64),
                ],
                dim=1,
            )
            for system in systems
        ]
    )
    block = TensorBlock(
        values=values,
        samples=Labels(["system", "atom"], samples),
        components=[],
        properties=Labels(
            ["geometry"], torch.arange(4, dtype=torch.int32, device=device)[:, None]
        ),
    )
    return TensorMap(
        Labels(["_"], torch.zeros((1, 1), dtype=torch.int32, device=device)), [block]
    )


def _unpack_geometry(packed: TensorMap) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    block = packed.block()
    counts = torch.bincount(block.samples.values[:, 0].to(torch.int64)).tolist()
    return [
        (values[:, 0].round().to(torch.int64), values[:, 1:])
        for values in torch.split(block.values, counts)
    ]


def get_density_transforms(
    loss_hypers: Dict[str, Any], scaler: torch.nn.Module
) -> List[CollateTransform]:
    """Collate transforms adding what the density losses need to the extra data.

    These are the atomic types and positions of the batch, from which the loss
    builds the metric matrices, and the scale removed from each target. They must
    run after the augmentation, so that the positions are the ones the model sees,
    and after the densification of the atomic-basis targets.

    :param loss_hypers: The ``loss`` section of the training hyperparameters.
    :param scaler: The scaler whose scales the trainer removes from the targets.
    :return: The transforms; none if no target uses a density loss.
    """
    names = [
        name for name, hypers in loss_hypers.items() if hypers["type"] == LOSS_TYPE
    ]
    if len(names) == 0:
        return []

    def transform(
        systems: List[System],
        targets: Dict[str, TensorMap],
        extra: Dict[str, TensorMap],
    ) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
        extra[GEOMETRY_NAME] = _pack_geometry(systems)
        # ones with the NaN padding of the target, sent through the scale removal
        ones = {
            name: TensorMap(
                targets[name].keys,
                [
                    TensorBlock(
                        values=0.0 * block.values + 1.0,
                        samples=block.samples,
                        components=block.components,
                        properties=block.properties,
                    )
                    for block in targets[name].blocks()
                ],
            )
            for name in names
            if name in targets
        }
        for name, tensor in remove_scale(systems, ones, scaler).items():
            extra[_removed_scale_name(name)] = tensor
        return systems, targets, extra

    return [transform]


def _flatten(
    tensor_map: TensorMap, subtract: Optional[TensorMap] = None
) -> torch.Tensor:
    """Flatten atomic-basis coefficients into the order of the auxiliary basis.

    A densified atomic-basis target stores one block per ``o3_lambda``, with values
    ``(atom, m, n)`` and NaN wherever an element has no such function. Laying the
    blocks out as ``(atom, [l][n][m])`` and taking the non-NaN entries in row-major
    order gives the order of :py:mod:`~metatrain.utils.density.basis`.

    :param tensor_map: Coefficients to flatten, with NaN padding.
    :param subtract: If given, flatten ``tensor_map - subtract`` instead; either of
        the two can carry the padding.
    :return: The flat coefficients of all the atoms.
    """
    per_block: List[torch.Tensor] = []
    for key in sorted(tensor_map.keys, key=lambda key: int(key[0])):
        values = tensor_map.block(key).values
        if subtract is not None:
            values = values - subtract.block(key).values
        if int(key[0]) == 1:
            # m = -1, 0, 1 to x, y, z
            values = values[:, [2, 0, 1], :]
        per_block.append(values.transpose(1, 2).reshape(values.shape[0], -1))
    wide = torch.cat(per_block, dim=1)
    return wide[~torch.isnan(wide)]


def _quadratic_form(vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return torch.dot(vector, torch.mv(matrix, vector))


class DensityMSELoss(LossInterface):
    """
    Error of a density, as a quadratic form in the error of its coefficients.

    For a field :math:`\\rho(r) = \\sum_i c_i \\phi_i(r)` and a two-centre metric
    :math:`M` of the auxiliary basis, the loss of a system is
    :math:`\\Delta c^T M \\Delta c`. With ``metric="overlap"`` this is

    .. math::

        \\int |\\rho_\\mathrm{pred}(r) - \\rho_\\mathrm{ref}(r)|^2 \\, dr ,

    and with ``metric="coulomb"`` the electrostatic self-energy of
    :math:`\\rho_\\mathrm{pred} - \\rho_\\mathrm{ref}`.

    The target must be an atomic-basis target with the coefficients in the
    conventions of :py:mod:`~metatrain.utils.density.basis`, for non-periodic
    systems with positions in Angstrom. The value is the total error of each
    system, reduced over the systems of the batch.

    The scales the trainer removed from the targets are multiplied back before the
    metric is applied, so the value is the physical error.

    Two options adapt the loss to a spin density
    :math:`\\rho_\\alpha - \\rho_\\beta`. ``mask_zero_targets`` gives a zero loss
    to the systems whose reference coefficients are all exactly zero, the
    placeholder of a restricted calculation. ``sign_invariant_singlets`` scores the
    systems whose reference integrates to less than :py:data:`SINGLET_TOLERANCE`
    against both signs of the reference and keeps the smaller error, since flipping
    every spin of such a solution gives a degenerate one. Both need the reference as
    stored in the dataset, so the ``atomic_baseline`` of the target must be fixed to
    zero.

    :param name: key of the coefficient target.
    :param gradient: not supported; must be ``None``.
    :param weight: weight of this term in the aggregated loss.
    :param reduction: ``"mean"``, ``"sum"`` or ``"none"``.
    :param aux_basis: auxiliary basis the reference coefficients were fitted in, as
        accepted by :py:func:`~metatrain.utils.density.basis.load_element_basis`.
    :param metric: ``"overlap"`` or ``"coulomb"``.
    :param mask_zero_targets: whether to skip systems whose reference coefficients
        are all zero.
    :param sign_invariant_singlets: whether to score systems whose reference
        integrates to zero against the better of the two signs of the reference.
    """

    def __init__(
        self,
        name: str,
        gradient: Optional[str],
        weight: float,
        reduction: str,
        aux_basis: str,
        metric: str = "overlap",
        mask_zero_targets: bool = False,
        sign_invariant_singlets: bool = False,
    ):
        super().__init__(name, gradient, weight, reduction)
        if gradient is not None:
            raise NotImplementedError(
                f"the '{LOSS_TYPE}' loss does not support gradients of the target."
            )
        self.builder = MetricBuilder(aux_basis, metric)
        self.mask_zero_targets = mask_zero_targets
        self.sign_invariant_singlets = sign_invariant_singlets

    def _unscale(
        self, flat: torch.Tensor, extra_data: Dict[str, TensorMap]
    ) -> torch.Tensor:
        inverse_scale = _flatten(extra_data[_removed_scale_name(self.target)])
        return flat / inverse_scale

    def compute(
        self,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Any] = None,
    ) -> torch.Tensor:
        if extra_data is None or GEOMETRY_NAME not in extra_data:
            raise RuntimeError(
                f"the '{LOSS_TYPE}' loss on target '{self.target}' requires "
                f"'{GEOMETRY_NAME}' in the extra data; see `get_density_transforms`."
            )
        prediction = predictions[self.target]
        target = targets[self.target]
        geometries = _unpack_geometry(extra_data[GEOMETRY_NAME])
        values = prediction.block(0).values
        matrices = self.builder.compute_batch(geometries, values.device, values.dtype)
        sizes = [matrix.shape[0] for matrix in matrices]

        delta = _flatten(prediction, target)
        if len(delta) != sum(sizes):
            raise ValueError(
                f"target '{self.target}' has {len(delta)} coefficients in this batch, "
                f"and the '{self.builder.aux_basis}' auxiliary basis {sum(sizes)}. "
                "Check that 'aux_basis' is the basis the dataset was fitted in."
            )
        deltas = torch.split(self._unscale(delta, extra_data), sizes)
        losses = [
            _quadratic_form(delta, matrix)
            for delta, matrix in zip(deltas, matrices, strict=True)
        ]

        if self.mask_zero_targets or self.sign_invariant_singlets:
            reference = _flatten(target)
            references = torch.split(self._unscale(reference, extra_data), sizes)
        if self.sign_invariant_singlets:
            for i, (reference, (types, _)) in enumerate(
                zip(references, geometries, strict=True)
            ):
                charge = torch.dot(
                    integrals(self.builder.aux_basis, types).to(reference), reference
                )
                if abs(float(charge)) < SINGLET_TOLERANCE:
                    # c_pred + c_ref = (c_pred - c_ref) + 2 c_ref
                    flipped = _quadratic_form(deltas[i] + 2 * reference, matrices[i])
                    losses[i] = torch.minimum(losses[i], flipped)

        per_system = torch.stack(losses)
        if self.mask_zero_targets:
            keep = torch.stack([(reference != 0).any() for reference in references])
            per_system = torch.where(keep, per_system, torch.zeros_like(per_system))

        if self.reduction == "mean":
            return per_system.mean()
        elif self.reduction == "sum":
            return per_system.sum()
        elif self.reduction == "none":
            return per_system
        raise ValueError(f"unknown reduction '{self.reduction}'")
