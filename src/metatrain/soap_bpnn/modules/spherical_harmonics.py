"""Spherical harmonics whose position gradients survive export.

The custom TorchScript operators provided by ``sphericart``
(``sphericart_torch::spherical_harmonics`` and friends) do not carry an autograd
formula inside the shared library: reverse-mode differentiation is attached at
``import sphericart.torch`` time, in Python, through
``torch.library.register_autograd`` (see ``sphericart/torch/register.py``).

An exported metatrain model, however, is self-contained TorchScript. It is
deployed (i-PI, LAMMPS, plain metatomic/ASE) in processes that load the operator
library from the model's ``extensions`` directory and never import
``sphericart.torch``. In such a process the operator exists but has no autograd
rule, and PyTorch's fallback silently treats its output as a constant:
``backward()`` runs without error and drops every ``dY/dR`` term. The resulting
forces are the gradient of the energy with the angular factors frozen -- a
smooth, non-conservative force field that heats MD without bound.

The modules below compute the same harmonics but write the derivatives
explicitly into the autograd graph, using the operators' own forward-derivative
kernels (which live in the shared library and therefore *are* available at
deployment time). With ``R0 = R.detach()`` and ``delta = R - R0`` (exactly zero,
but attached to the graph), we build the second-order Taylor expansion

    Y(R) = Y(R0) + dY(R0) . delta + 1/2 delta . ddY(R0) . delta

whose value is bit-identical to the plain operator call, and whose first and
second derivatives with respect to ``R`` are the exact analytical ones. Third
and higher derivatives are zero (the Python-side registration raises an error
for those instead; they are not used anywhere in metatrain).
"""

from typing import List

import sphericart.torch
import torch


def _taylor_expand(
    values: torch.Tensor,
    gradients: torch.Tensor,
    hessians: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct a differentiable output from detached derivative kernels.

    :param values: harmonics at the detached point, shape ``[pair, m]``.
    :param gradients: their derivatives, shape ``[pair, 3, m]``.
    :param hessians: their second derivatives, shape ``[pair, 3, 3, m]``.
    :param delta: ``R - R.detach()``, shape ``[pair, 3]``; numerically zero but
        carrying the autograd graph of ``R``.
    :return: tensor of shape ``[pair, m]``, equal to ``values`` but with exact
        first and second position derivatives in its autograd graph.
    """
    first = (gradients * delta.unsqueeze(-1)).sum(dim=1)
    contracted = (hessians * delta.unsqueeze(1).unsqueeze(-1)).sum(dim=2)
    second = 0.5 * (contracted * delta.unsqueeze(-1)).sum(dim=1)
    return values + first + second


class ExportSafeSphericalHarmonics(torch.nn.Module):
    """Drop-in replacement for ``spex.angular.SphericalHarmonics``.

    Same interface and bit-identical values, but differentiable in processes
    that never ran sphericart's Python-side autograd registration (see module
    docstring).

    :param max_angular: The maximum spherical harmonic order ``l`` to compute.
    """

    def __init__(self, max_angular: int) -> None:
        super().__init__()

        self.max_angular = max_angular
        self.m_per_l = [2 * l + 1 for l in range(max_angular + 1)]
        self.calculator = sphericart.torch.SphericalHarmonics(l_max=max_angular)
        self.spec = {"max_angular": max_angular}

    def forward(self, R: torch.Tensor) -> List[torch.Tensor]:
        """Compute spherical harmonics for input vectors.

        :param R: Input vectors of shape ``[pair, 3]``.
        :return: Spherical harmonics as a list of tensors, one per ``l``,
            of shape ``[pair, 2*l + 1]``.
        """
        if R.requires_grad:
            R0 = R.detach()
            sph, dsph, ddsph = self.calculator.compute_with_hessians(R0)
            expansion = _taylor_expand(sph, dsph, ddsph, R - R0)
        else:
            expansion = self.calculator.compute(R)
        return expansion.split(self.m_per_l, dim=-1)


class ExportSafeSolidHarmonics(torch.nn.Module):
    """Drop-in replacement for ``sphericart.torch.SolidHarmonics``.

    Same values, but differentiable in processes that never ran sphericart's
    Python-side autograd registration (see module docstring).

    :param l_max: The maximum solid harmonic order ``l`` to compute.
    """

    def __init__(self, l_max: int) -> None:
        super().__init__()

        self.calculator = sphericart.torch.SolidHarmonics(l_max=l_max)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """Compute solid harmonics for input vectors.

        :param xyz: Input vectors of shape ``[pair, 3]``.
        :return: Solid harmonics of shape ``[pair, (l_max + 1)**2]``.
        """
        if xyz.requires_grad:
            xyz0 = xyz.detach()
            sph, dsph, ddsph = self.calculator.compute_with_hessians(xyz0)
            return _taylor_expand(sph, dsph, ddsph, xyz - xyz0)
        return self.calculator.compute(xyz)
