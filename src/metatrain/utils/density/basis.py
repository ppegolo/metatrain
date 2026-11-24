"""Atom-centred Gaussian auxiliary bases, as combinations of Cartesian Gaussians.

A density is expanded as :math:`\\rho(r) = \\sum_i c_i \\phi_i(r)` on contracted
real solid-harmonic Gaussians,

.. math::

    \\phi_{n l m}(r) = \\sum_k d_{k n} \\, r^l \\, Y_{l m}(\\hat r)
        \\, e^{-\\alpha_k r^2},

and the coefficients only have a meaning together with the conventions of the code
that fitted them. This module follows those of PySCF, which are also the ones of
libcint:

- the functions of an atom are ordered by angular momentum, then by shell in the
  order the basis set lists them, then by contraction, then by :math:`m`;
- :math:`m` runs from :math:`-l` to :math:`l`, except for :math:`l = 1`, which is
  ordered :math:`x, y, z`;
- :math:`Y_{l m}` are the real spherical harmonics normalised on the sphere, with
  a positive leading coefficient (:math:`Y_{1 1} \\propto x`);
- the radial part of every contracted function is normalised,
  :math:`\\int_0^\\infty R(r)^2 r^2 \\, dr = 1`.

A basis set is given either as a file in the JSON format of the `Basis Set Exchange
<https://www.basissetexchange.org>`_ (the ``molssi_bse_schema``), or by its name
there, which requires the optional ``basis_set_exchange`` package.
"""

import json
import math
import os
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import torch


def monomials(degree: int) -> torch.Tensor:
    """Exponent triples of every monomial of exactly ``degree``.

    :param degree: The homogeneous degree.
    :return: Integer tensor of shape ``(n_monomials, 3)``.
    """
    triples = [
        (i, j, degree - i - j)
        for i in range(degree, -1, -1)
        for j in range(degree - i, -1, -1)
    ]
    return torch.tensor(triples, dtype=torch.int64)


@lru_cache(maxsize=None)
def _solid_harmonics(angular: int) -> np.ndarray:
    """Real solid harmonics :math:`r^l Y_{l m}` as polynomials in ``x, y, z``.

    :param angular: The angular momentum :math:`l`.
    :return: Array of shape ``(2l + 1, n_monomials)``: one row per function, in the
        order of the module docstring, with the coefficients of the monomials of
        :py:func:`monomials`.
    """
    column = {tuple(power): i for i, power in enumerate(monomials(angular).tolist())}
    table = np.zeros((2 * angular + 1, len(column)))
    for m in range(angular + 1):
        norm = math.sqrt(
            (2 * angular + 1)
            / (4 * math.pi)
            * (2 if m > 0 else 1)
            * math.factorial(angular - m)
            / math.factorial(angular + m)
        )
        for k in range((angular - m) // 2 + 1):
            # coefficient of r^(2k) z^(l - 2k - m) in the associated Legendre part
            legendre = (
                (-1) ** k
                * math.comb(angular, k)
                * math.comb(2 * angular - 2 * k, angular)
                * math.factorial(angular - 2 * k)
                / math.factorial(angular - 2 * k - m)
                / 2**angular
            )
            for a in range(k + 1):
                for b in range(k + 1 - a):
                    # r^(2k) = (x^2 + y^2 + z^2)^k
                    radial = math.factorial(k) / (
                        math.factorial(a)
                        * math.factorial(b)
                        * math.factorial(k - a - b)
                    )
                    for p in range(m + 1):
                        # (x + iy)^m: the real part goes to +m, the imaginary to -m
                        q = m - p
                        power = (2 * a + p, 2 * b + q, angular - m - 2 * a - 2 * b)
                        value = norm * legendre * radial * math.comb(m, p)
                        table[angular + m, column[power]] += (
                            value * (1, 0, -1, 0)[q % 4]
                        )
                        if m > 0:
                            table[angular - m, column[power]] += (
                                value * (0, 1, 0, -1)[q % 4]
                            )
    if angular == 1:
        table = table[[2, 0, 1]]
    return table


def _gaussian_integral(power: int, exponent: np.ndarray) -> np.ndarray:
    # int_0^inf r^power exp(-exponent r^2) dr
    return math.gamma((power + 1) / 2) / (2 * exponent ** ((power + 1) / 2))


def _load_shells(
    aux_basis: str, atomic_number: int
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    if os.path.isfile(aux_basis):
        with open(aux_basis) as fd:
            data = json.load(fd)
    else:
        try:
            import basis_set_exchange as bse
        except ModuleNotFoundError as err:
            raise ImportError(
                f"'{aux_basis}' is not a file, and reading a basis set by name from "
                "the Basis Set Exchange requires `pip install basis_set_exchange`."
            ) from err
        data = bse.get_basis(aux_basis, elements=[atomic_number])
    shells = []
    for shell in data["elements"][str(atomic_number)]["electron_shells"]:
        exponents = np.array(shell["exponents"], dtype=np.float64)
        coefficients = np.array(shell["coefficients"], dtype=np.float64)
        angulars = shell["angular_momentum"]
        if len(angulars) == 1:
            shells.append((angulars[0], exponents, coefficients.T))
        else:
            # one contraction per angular momentum (e.g. an sp shell)
            for angular, column in zip(angulars, coefficients, strict=True):
                shells.append((angular, exponents, column[:, None]))
    return sorted(shells, key=lambda shell: shell[0])


class ElementBasis:
    """The auxiliary basis of one element, as primitive Cartesian Gaussians.

    A "unit" is one primitive Gaussian of one contracted shell: the ``2l + 1``
    functions of the shell are ``sum_units weights @ monomials * exp(-alpha r^2)``.

    :param alphas: Per angular momentum, the exponents of the units, shape
        ``(n_units,)``, in Bohr^-2.
    :param weights: Per angular momentum, the coefficients of the monomials of
        :py:func:`monomials` in each unit, shape ``(n_units, 2l + 1, n_monomials)``.
    :param offsets: Per angular momentum, the index of the first function of each
        unit's shell among the functions of the element, shape ``(n_units,)``.
    :param size: Number of functions of the element.
    """

    def __init__(
        self,
        alphas: Dict[int, torch.Tensor],
        weights: Dict[int, torch.Tensor],
        offsets: Dict[int, torch.Tensor],
        size: int,
    ) -> None:
        self.alphas = alphas
        self.weights = weights
        self.offsets = offsets
        self.size = size


@lru_cache(maxsize=None)
def load_element_basis(aux_basis: str, atomic_number: int) -> ElementBasis:
    """Load the auxiliary basis of one element.

    :param aux_basis: Name of the basis set in the Basis Set Exchange, e.g.
        ``"def2-universal-jfit"``, or path to a JSON basis set file.
    :param atomic_number: The element.
    :return: The basis of the element.
    """
    alphas: Dict[int, List[np.ndarray]] = {}
    weights: Dict[int, List[np.ndarray]] = {}
    offsets: Dict[int, List[np.ndarray]] = {}
    size = 0
    for angular, exponents, coefficients in _load_shells(aux_basis, atomic_number):
        harmonics = _solid_harmonics(angular)
        coefficients = (
            coefficients
            / np.sqrt(_gaussian_integral(2 * angular + 2, 2 * exponents))[:, None]
        )
        pairs = _gaussian_integral(2 * angular + 2, exponents[:, None] + exponents)
        coefficients = coefficients / np.sqrt(
            np.einsum("pi,pq,qi->i", coefficients, pairs, coefficients)
        )
        for contraction in coefficients.T:
            alphas.setdefault(angular, []).append(exponents)
            weights.setdefault(angular, []).append(
                contraction[:, None, None] * harmonics
            )
            offsets.setdefault(angular, []).append(np.full(len(exponents), size))
            size += 2 * angular + 1

    def stack(parts: Dict[int, List[np.ndarray]]) -> Dict[int, torch.Tensor]:
        return {
            angular: torch.from_numpy(np.concatenate(arrays))
            for angular, arrays in parts.items()
        }

    return ElementBasis(stack(alphas), stack(weights), stack(offsets), size)


def integrals(aux_basis: str, types: torch.Tensor) -> torch.Tensor:
    """Integrals ``int phi_i(r) dr`` of the auxiliary functions of one system.

    Only ``l = 0`` functions integrate to a non-zero value, so the dot product with
    the coefficients of a field is the integral of the field.

    :param aux_basis: Auxiliary basis, as accepted by :py:func:`load_element_basis`.
    :param types: Atomic numbers, shape ``(n_atoms,)``.
    :return: float64 tensor with one entry per auxiliary function.
    """
    per_element: Dict[int, torch.Tensor] = {}
    for atomic_number in sorted({int(z) for z in types.tolist()}):
        element = load_element_basis(aux_basis, atomic_number)
        values = torch.zeros(element.size, dtype=torch.float64)
        if 0 in element.alphas:
            # int exp(-alpha r^2) dr = (pi / alpha)^(3/2)
            values.index_add_(
                0,
                element.offsets[0],
                element.weights[0][:, 0, 0] * (torch.pi / element.alphas[0]) ** 1.5,
            )
        per_element[atomic_number] = values
    return torch.cat([per_element[int(z)] for z in types.tolist()])
