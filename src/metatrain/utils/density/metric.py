"""Two-centre overlap and Coulomb matrices of Gaussian auxiliary bases.

The matrices are assembled as batched torch operations on the device of the model,
from the functions of :py:mod:`~metatrain.utils.density.basis` written as
combinations of Cartesian Gaussians. The integrals between Cartesian Gaussians
follow the McMurchie-Davidson scheme: the overlap from the Hermite expansion
coefficients, the Coulomb matrix from Hermite derivatives of the Boys function,
which is an incomplete gamma function (``torch.special.gammainc``).

All the systems of a batch are assembled together, with the indices built on the host,
and gradients do not flow through the matrices.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .basis import ElementBasis, load_element_basis, monomials


_METRICS = ("overlap", "coulomb")

# Angstrom per Bohr, the value PySCF uses
_BOHR = 0.52917721092

# overlap pairs with ``mu * R^2`` beyond this contribute below 1e-20 of their prefactor
# and are dropped; the Coulomb kernel has no such decay and is never screened
_OVERLAP_SCREEN = 46.0


def _num_hermite(order: int) -> int:
    return (order + 1) * (order + 2) * (order + 3) // 6


def _hermite_indices(order: int) -> torch.Tensor:
    """All Hermite index triples ``(t, u, v)`` with ``t + u + v <= order``.

    :param order: Maximum total order.
    :return: Integer tensor of shape ``(n_indices, 3)``.
    """
    triples = [
        (t, u, v)
        for t in range(order + 1)
        for u in range(order + 1 - t)
        for v in range(order + 1 - t - u)
    ]
    return torch.tensor(triples, dtype=torch.int64)


def _boys(order: int, argument: torch.Tensor) -> torch.Tensor:
    """Boys functions ``F_0 .. F_order`` for a batch of arguments.

    Every order is evaluated directly, in one broadcast call, from the
    regularised lower incomplete gamma,
    ``F_n(T) = Gamma(n + 1/2) P(n + 1/2, T) / (2 T^(n + 1/2))``. Near ``T = 0``
    the closed form is 0/0 and a two-term Taylor series (error ``~T^2``)
    replaces it.

    :param order: Highest Boys order needed.
    :param argument: Arguments ``T``, shape ``(n,)``.
    :return: Tensor of shape ``(n, order + 1)``.
    """
    cutoff = 1e-13 if argument.dtype == torch.float64 else 3e-4
    t = argument[:, None]
    t_safe = t.clamp(min=cutoff)
    n = torch.arange(order + 1, dtype=argument.dtype, device=argument.device)
    a = n + 0.5
    closed = (
        0.5
        * torch.exp(torch.lgamma(a))
        * torch.special.gammainc(a.expand(len(argument), -1), t_safe)
    )
    closed = closed / t_safe**a
    series = 1.0 / (2.0 * n + 1.0) - t / (2.0 * n + 3.0)
    return torch.where(t < cutoff, series, closed)


def _e0_axis(
    l1: int,
    l2: int,
    pa: torch.Tensor,
    pb: torch.Tensor,
    inv_2p: torch.Tensor,
) -> torch.Tensor:
    """Hermite ``t = 0`` expansion coefficients ``E_0^{i j}`` along one axis.

    Standard McMurchie–Davidson recursion, batched over the pair dimension. The
    ``exp(-mu X^2)`` prefactor is *not* included here; the caller folds the
    full ``exp(-mu R^2)`` once.

    :param l1: Maximum monomial power on the first centre.
    :param l2: The same for the second centre.
    :param pa: ``P - A`` along the axis, shape ``(n,)``.
    :param pb: ``P - B`` along the axis, shape ``(n,)``.
    :param inv_2p: ``1 / (2 (alpha + beta))``, shape ``(n,)``.
    :return: Tensor of shape ``(n, l1 + 1, l2 + 1)``.
    """
    zero = torch.zeros_like(pa)
    table: Dict[Tuple[int, int, int], torch.Tensor] = {(0, 0, 0): torch.ones_like(pa)}

    def term(t: int, i: int, j: int) -> torch.Tensor:
        return table.get((t, i, j), zero)

    for i in range(1, l1 + 1):
        for t in range(i + 1):
            table[(t, i, 0)] = (
                inv_2p * term(t - 1, i - 1, 0)
                + pa * term(t, i - 1, 0)
                + (t + 1.0) * term(t + 1, i - 1, 0)
            )
    for j in range(1, l2 + 1):
        for i in range(l1 + 1):
            for t in range(i + j + 1):
                table[(t, i, j)] = (
                    inv_2p * term(t - 1, i, j - 1)
                    + pb * term(t, i, j - 1)
                    + (t + 1.0) * term(t + 1, i, j - 1)
                )

    return torch.stack(
        [
            torch.stack([table.get((0, i, j), zero) for j in range(l2 + 1)], dim=-1)
            for i in range(l1 + 1)
        ],
        dim=-2,
    )


def _d_table(angular: int, alphas: torch.Tensor) -> torch.Tensor:
    """Single-centre Hermite coefficients ``d_t^i`` for one exponent batch.

    ``x^i exp(-alpha x^2) = sum_t d_t^i Lambda_t(x)`` with Hermite Gaussians
    ``Lambda_t``; the recursion is the two-centre one at zero separation.
    Entries with ``t > i`` or ``i - t`` odd vanish and stay exactly zero.

    :param angular: Maximum monomial power.
    :param alphas: Exponents, shape ``(n,)``.
    :return: Tensor of shape ``(n, angular + 1, angular + 1)``, indexed
        ``[., i, t]``.
    """
    inv_2a = 0.5 / alphas
    zero = torch.zeros_like(alphas)
    rows: List[List[torch.Tensor]] = [[torch.ones_like(alphas)] + [zero] * angular]
    for i in range(1, angular + 1):
        previous = rows[i - 1]
        row = []
        for t in range(angular + 1):
            value = zero
            if t - 1 >= 0:
                value = value + inv_2a * previous[t - 1]
            if t + 1 <= i - 1:
                value = value + (t + 1.0) * previous[t + 1]
            row.append(value)
        rows.append(row)
    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)


def _hermite_program(order: int) -> List[Tuple[torch.Tensor, ...]]:
    """Level-by-level index tables of the Hermite Coulomb recursion.

    Level ``total`` builds every ``(t, u, v)`` with ``t + u + v = total`` from
    level ``total - 1`` in one vectorised step: each target is raised along its
    first non-zero axis, ``R_{..k..} = X_axis R_{..k-1..} + (k - 1) R_{..k-2..}``
    (both terms at Boys order ``n + 1``). Columns follow
    :py:func:`_hermite_indices`.

    :param order: Maximum ``t + u + v``.
    :return: Per level ``1 .. order``: ``(targets, axes, sources, factors)``,
        ``sources`` the step columns followed by the lower ones, ``factors``
        as float64.
    """
    triples = _hermite_indices(order).tolist()
    column = {tuple(triple): i for i, triple in enumerate(triples)}
    program = []
    for total in range(1, order + 1):
        targets, axes, steps, lowers, factors = [], [], [], [], []
        for triple in triples:
            if sum(triple) != total:
                continue
            axis = next(a for a in range(3) if triple[a] > 0)
            step = list(triple)
            step[axis] -= 1
            lower = list(step)
            lower[axis] -= 1
            targets.append(column[tuple(triple)])
            axes.append(axis)
            steps.append(column[tuple(step)])
            factor = triple[axis] - 1.0
            lowers.append(column[tuple(lower)] if factor > 0.0 else 0)
            factors.append(factor)
        program.append(
            (
                torch.tensor(targets),
                torch.tensor(axes),
                torch.tensor(steps + lowers),
                torch.tensor(factors, dtype=torch.float64),
            )
        )
    return program


def _hermite_coulomb(
    order: int,
    separation: torch.Tensor,
    theta: torch.Tensor,
    boys: torch.Tensor,
    program: List[Tuple[torch.Tensor, ...]],
) -> torch.Tensor:
    """Hermite Coulomb integrals ``R^0_{tuv}`` up to total ``order``.

    Seeded by ``R^n_{000} = (-2 theta)^n F_n`` and raised with the standard
    downward-in-``n`` recursion, one vectorised step per total order.

    :param order: Maximum ``t + u + v``.
    :param separation: ``A - B``, shape ``(n, 3)``.
    :param theta: Reduced exponents, shape ``(n,)``.
    :param boys: Boys values, ``(n, order + 1)``.
    :param program: :py:func:`_hermite_program` of ``order``, on the device.
    :return: Tensor of shape ``(n, n_indices)``, columns ordered as
        :py:func:`_hermite_indices`.
    """
    n_pairs = separation.shape[0]
    n_indices = _num_hermite(order)
    levels = torch.zeros(
        (n_pairs, order + 1, n_indices), dtype=boys.dtype, device=boys.device
    )
    exponents = torch.arange(order + 1, device=boys.device, dtype=boys.dtype)
    levels[:, :, 0] = (-2.0 * theta)[:, None] ** exponents * boys
    for total, (targets, axes, sources, factors) in enumerate(program, 1):
        step, lower = levels[:, 1 : order - total + 2, sources].chunk(2, dim=-1)
        levels[:, : order - total + 1, targets] = torch.addcmul(
            separation[:, None, axes] * step, lower, factors
        )
    return levels[:, 0]


def _coulomb_block(
    order: int,
    program: List[Tuple[torch.Tensor, ...]],
    combined: torch.Tensor,
    centers_1: torch.Tensor,
    centers_2: torch.Tensor,
    alphas_1: torch.Tensor,
    alphas_2: torch.Tensor,
    side_1: torch.Tensor,
    side_2: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    """Coulomb pair blocks of one chunk of primitive pairs.

    :param order: ``l1 + l2`` of the pair group.
    :param program: :py:func:`_hermite_program` of ``order``, on the device.
    :param combined: Column of each ``(hermite_1, hermite_2)`` index pair in
        the order-``l1 + l2`` Hermite integrals.
    :param centers_1: Unit centres of the first side (Bohr), ``(n_1, 3)``.
    :param centers_2: The same for the second side.
    :param alphas_1: Unit exponents of the first side, ``(n_1,)``.
    :param alphas_2: The same for the second side.
    :param side_1: Per-unit spherical-to-Hermite transforms of the first side.
    :param side_2: The same for the second side, parity sign folded in.
    :param first: Unit index of each pair on the first side.
    :param second: The same for the second side.
    :return: Tensor of shape ``(n, 2 l1 + 1, 2 l2 + 1)``.
    """
    centers_1 = centers_1[first]
    alphas_1 = alphas_1[first]
    alphas_2 = alphas_2[second]
    separation = centers_1 - centers_2[second]
    theta = alphas_1 * alphas_2 / (alphas_1 + alphas_2)
    boys = _boys(order, theta * (separation**2).sum(-1))
    prefactor = (
        2.0 * torch.pi**2.5 / (alphas_1 * alphas_2 * torch.sqrt(alphas_1 + alphas_2))
    )
    if order == 0:
        return (side_1[first, 0, 0] * side_2[second, 0, 0] * prefactor * boys[:, 0])[
            :, None, None
        ]
    hermite = _hermite_coulomb(order, separation, theta, boys, program)
    return (
        torch.einsum(
            "pmh,phk,pnk->pmn", side_1[first], hermite[:, combined], side_2[second]
        )
        * prefactor[:, None, None]
    )


def _exclusive_cumsum(values: List[int]) -> List[int]:
    """Exclusive prefix sums of a list of Python integers.

    :param values: The values.
    :return: Same length; entry ``i`` is ``sum(values[:i])``.
    """
    sums, running = [], 0
    for value in values:
        sums.append(running)
        running += value
    return sums


def _host_table(columns: List[List[int]], device: torch.device) -> torch.Tensor:
    """Per-system integer columns, sent to ``device`` in one non-blocking copy.

    :param columns: Equal-length lists of Python integers.
    :param device: Target device.
    :return: Tensor of shape ``(len(columns), n_systems)``.
    """
    return torch.tensor(columns, dtype=torch.int64).to(device, non_blocking=True)


def _pair_lists(
    counts_1: List[int], counts_2: List[int], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Within-system all-pairs indices for units concatenated across systems.

    :param counts_1: Units per system on the first side.
    :param counts_2: The same for the second side.
    :param device: Device to build the index tensors on.
    :return: ``(system, first, second)`` index tensors, one entry per pair.
    """
    pairs_per_system = [c1 * c2 for c1, c2 in zip(counts_1, counts_2, strict=True)]
    total = sum(pairs_per_system)
    pairs, pair_starts, starts_1, starts_2, widths = _host_table(
        [
            pairs_per_system,
            _exclusive_cumsum(pairs_per_system),
            _exclusive_cumsum(counts_1),
            _exclusive_cumsum(counts_2),
            counts_2,
        ],
        device,
    )
    system = torch.repeat_interleave(
        torch.arange(len(counts_1), device=device), pairs, output_size=total
    )
    rank = torch.arange(total, device=device) - pair_starts[system]
    width = widths[system]
    return (
        system,
        starts_1[system] + rank.div(width, rounding_mode="floor"),
        starts_2[system] + rank.remainder(width),
    )


def _triangle_pair_lists(
    counts: List[int], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Within-system pairs ``first <= second`` of one set of units.

    The row of each pair is inverted from its rank in closed form (float64
    square root, then a one-step correction), so no index tensor is read back.

    :param counts: Units per system.
    :param device: Device to build the index tensors on.
    :return: ``(system, first, second)`` index tensors, one entry per pair.
    """
    pairs_per_system = [c * (c + 1) // 2 for c in counts]
    total = sum(pairs_per_system)
    pairs, pair_starts, starts, widths = _host_table(
        [
            pairs_per_system,
            _exclusive_cumsum(pairs_per_system),
            _exclusive_cumsum(counts),
            counts,
        ],
        device,
    )
    system = torch.repeat_interleave(
        torch.arange(len(counts), device=device), pairs, output_size=total
    )
    rank = torch.arange(total, device=device) - pair_starts[system]
    width = widths[system]

    def row_start(row: torch.Tensor) -> torch.Tensor:
        return row * width - row * (row - 1) // 2

    span = (2 * width + 1).to(torch.float64)
    row = torch.floor(
        (span - torch.sqrt(span**2 - 8.0 * rank.to(torch.float64))) / 2.0
    ).to(torch.int64)
    row = row - (row_start(row) > rank).to(torch.int64)
    row = row + (row_start(row + 1) <= rank).to(torch.int64)
    column = row + rank - row_start(row)
    start = starts[system]
    return system, start + row, start + column


class MetricBuilder:
    """Assemble two-centre metric matrices of an auxiliary basis.

    The rows and columns of a matrix follow the order of the functions described
    in :py:mod:`~metatrain.utils.density.basis`, atom by atom.

    :param aux_basis: Auxiliary basis, as accepted by
        :py:func:`~metatrain.utils.density.basis.load_element_basis`.
    :param metric: ``"overlap"`` or ``"coulomb"``.
    """

    def __init__(self, aux_basis: str, metric: str) -> None:
        if metric not in _METRICS:
            raise ValueError(f"unknown metric {metric!r}; expected one of {_METRICS}.")
        self.aux_basis = aux_basis
        self.metric = metric
        self._elements: Dict[int, ElementBasis] = {}
        self._tables_cache: Dict[
            Tuple[int, int, torch.dtype, torch.device],
            Dict[str, torch.Tensor],
        ] = {}
        self._primitive_cache: Dict[
            Tuple[torch.device, torch.dtype, int],
            Tuple[
                Dict[int, Dict[str, torch.Tensor]],
                Dict[Tuple[int, int], Tuple[int, np.ndarray]],
            ],
        ] = {}

    def _units(
        self,
        geometries: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[
        Dict[int, Dict[str, torch.Tensor]], Dict[int, List[int]], List[int], List[int]
    ]:
        """Concatenate all systems' primitive units, grouped by ``l``.

        :param geometries: Per system, ``(types, positions)`` in Angstrom.
        :param device: Device to build on.
        :param dtype: Working dtype.
        :return: Per angular momentum a dict with keys ``centers`` (Bohr),
            ``alphas``, ``weights``, ``rows`` (system-local AO row of the first
            component); per angular momentum the units-per-system counts, as
            Python integers so pair-list construction never synchronises; and
            the number of auxiliary functions of each system; and the position of
            the matrix of each system in the flat buffer.
        """
        all_types = np.asarray(
            torch.cat([types for types, _ in geometries]).detach().cpu(),
            dtype=np.int64,
        )
        type_arrays = np.split(
            all_types, np.cumsum([len(types) for types, _ in geometries])[:-1]
        )
        elements = sorted({int(z) for z in np.unique(all_types)})
        for atomic_number in elements:
            if atomic_number not in self._elements:
                self._elements[atomic_number] = load_element_basis(
                    self.aux_basis, atomic_number
                )
        tables, slices = self._primitive_tables(device, dtype)
        angulars = sorted(
            {
                angular
                for atomic_number in elements
                for angular in self._elements[atomic_number].alphas
            }
        )

        unit_atom: Dict[int, List[np.ndarray]] = {a: [] for a in angulars}
        unit_prim: Dict[int, List[np.ndarray]] = {a: [] for a in angulars}
        unit_row: Dict[int, List[np.ndarray]] = {a: [] for a in angulars}
        unit_system: Dict[int, List[np.ndarray]] = {a: [] for a in angulars}
        counts: Dict[int, List[int]] = {a: [] for a in angulars}
        sizes: List[int] = []
        positions_parts: List[torch.Tensor] = []
        atom_base = 0

        for i_system, ((_, positions), type_array) in enumerate(
            zip(geometries, type_arrays, strict=True)
        ):
            positions_parts.append(positions.to(device=device, dtype=dtype))
            size_per_atom = np.array([self._elements[int(z)].size for z in type_array])
            atom_rows = np.concatenate([[0], np.cumsum(size_per_atom)[:-1]])
            sizes.append(int(size_per_atom.sum()))

            for angular in angulars:
                total = 0
                for z in elements:
                    entry = slices.get((z, angular))
                    if entry is None:
                        continue
                    start, local_rows = entry
                    atoms = np.nonzero(type_array == z)[0]
                    if len(atoms) == 0:
                        continue
                    n_atoms, n_prims = len(atoms), len(local_rows)
                    unit_atom[angular].append(np.repeat(atoms + atom_base, n_prims))
                    unit_prim[angular].append(
                        np.tile(np.arange(start, start + n_prims), n_atoms)
                    )
                    unit_row[angular].append(
                        np.repeat(atom_rows[atoms], n_prims)
                        + np.tile(local_rows, n_atoms)
                    )
                    unit_system[angular].append(np.full(n_atoms * n_prims, i_system))
                    total += n_atoms * n_prims
                counts[angular].append(total)
            atom_base += len(type_array)

        positions_bohr = torch.cat(positions_parts) / _BOHR
        size_array = np.array(sizes, dtype=np.int64)
        bases = _exclusive_cumsum([n * n for n in sizes])
        base_array = np.array(bases, dtype=np.int64)
        units: Dict[int, Dict[str, torch.Tensor]] = {}
        for angular in angulars:
            system = np.concatenate(unit_system[angular])
            rows = (
                np.concatenate(unit_row[angular])[:, None]
                + np.arange(2 * angular + 1)[None, :]
            )
            # Flat buffer position of matrix entry (i, j) is outer[i] + inner[j].
            outer = base_array[system][:, None] + rows * size_array[system][:, None]
            indices = torch.from_numpy(
                np.concatenate(
                    [
                        np.stack(
                            [
                                np.concatenate(unit_atom[angular]),
                                np.concatenate(unit_prim[angular]),
                            ],
                            axis=1,
                        ),
                        outer,
                        rows,
                    ],
                    axis=1,
                )
            ).to(device, non_blocking=True)
            width = 2 * angular + 1
            units[angular] = {
                "centers": positions_bohr[indices[:, 0]],
                "alphas": tables[angular]["alphas"][indices[:, 1]],
                "weights": tables[angular]["weights"][indices[:, 1]],
                "outer": indices[:, 2 : 2 + width],
                "inner": indices[:, 2 + width :],
            }
        return units, counts, sizes, bases

    def _primitive_tables(
        self, device: torch.device, dtype: torch.dtype
    ) -> Tuple[
        Dict[int, Dict[str, torch.Tensor]],
        Dict[Tuple[int, int], Tuple[int, np.ndarray]],
    ]:
        """Per-angular-momentum primitive tables of every loaded element.

        Concatenates the elements' exponents and weights per ``l`` on the
        device, once, so per-batch unit construction is a single gather.
        Rebuilt when a new element is loaded.

        :param device: Device the tables live on.
        :param dtype: Working dtype.
        :return: ``(tables, slices)``: per ``l`` the stacked ``alphas`` and
            ``weights``; per ``(element, l)`` the element's start in that stack
            and its primitives' AO row offsets (host side).
        """
        key = (device, dtype, len(self._elements))
        cached = self._primitive_cache.get(key)
        if cached is not None:
            return cached

        tables: Dict[int, Dict[str, torch.Tensor]] = {}
        slices: Dict[Tuple[int, int], Tuple[int, np.ndarray]] = {}
        per_l: Dict[int, Dict[str, list]] = {}
        for atomic_number in sorted(self._elements):
            element = self._elements[atomic_number]
            for angular in element.alphas:
                entry = per_l.setdefault(angular, {"alphas": [], "weights": []})
                start = sum(len(a) for a in entry["alphas"])
                slices[(atomic_number, angular)] = (
                    start,
                    np.asarray(element.offsets[angular]),
                )
                entry["alphas"].append(element.alphas[angular])
                entry["weights"].append(element.weights[angular])
        for angular, entry in per_l.items():
            tables[angular] = {
                "alphas": torch.cat(entry["alphas"]).to(device=device, dtype=dtype),
                "weights": torch.cat(entry["weights"]).to(device=device, dtype=dtype),
            }
        self._primitive_cache[key] = (tables, slices)
        return tables, slices

    @torch.no_grad()
    def compute_batch(
        self,
        geometries: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
        dtype: torch.dtype = torch.float64,
    ) -> List[torch.Tensor]:
        """Metric matrices of a batch of systems, assembled on ``device``.

        The primitive pairs of all the systems are processed together for each pair
        of angular momenta, and accumulated into one flat buffer holding all the
        matrices. Only ``l1 <= l2`` pairs are computed, since the matrices are
        symmetric.

        :param geometries: Per system, ``(types, positions)`` with positions in
            Angstrom.
        :param device: Device to assemble on.
        :param dtype: Working dtype of the assembly. ``torch.float32`` is
            accurate to ~1e-6, the same as casting a float64 matrix down.
        :return: One dense square matrix per system, in ``dtype``.
        """
        units, counts, sizes, bases = self._units(geometries, device, dtype)
        buffer = torch.zeros(
            sum(size * size for size in sizes), dtype=dtype, device=device
        )

        if self.metric == "coulomb":
            for angular, group in units.items():
                tables = self._group_tables(angular, angular, dtype, device)
                side = self._hermite_side(
                    angular,
                    group["alphas"],
                    group["weights"],
                    tables["powers_1"],
                    tables["hermite_1"],
                )
                group["side"] = side
                group["side_parity"] = side * tables["parity"]

        budget = 1 << 26 if dtype == torch.float32 else 1 << 25
        for l1, group_1 in units.items():
            for l2, group_2 in units.items():
                if l2 < l1:
                    continue
                if l1 == l2:
                    system, first, second = _triangle_pair_lists(counts[l1], device)
                else:
                    system, first, second = _pair_lists(counts[l1], counts[l2], device)
                tables = self._group_tables(l1, l2, dtype, device)
                n_hermite = (
                    max(
                        _num_hermite(l1) * _num_hermite(l2),
                        (l1 + l2 + 1) * _num_hermite(l1 + l2),
                    )
                    if self.metric == "coulomb"
                    else (l1 + 1) * (l2 + 1)
                )
                chunk = max(4096, budget // max(1, n_hermite))
                for start in range(0, len(system), chunk):
                    self._pair_block(
                        l1,
                        l2,
                        group_1,
                        group_2,
                        system[start : start + chunk],
                        first[start : start + chunk],
                        second[start : start + chunk],
                        buffer,
                        tables,
                    )

        matrices = []
        for start, size in zip(bases, sizes, strict=True):
            matrices.append(buffer[start : start + size * size].view(size, size))
        return matrices

    def _group_tables(
        self,
        l1: int,
        l2: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Constant index tensors of one ``(l1, l2)`` pair group, cached.

        :param l1: Angular momentum of the first side.
        :param l2: The same for the second side.
        :param dtype: Working dtype (the parity signs are float).
        :param device: Device the tensors live on.
        :return: The tables, keyed by name.
        """
        key = (l1, l2, dtype, device)
        cached = self._tables_cache.get(key)
        if cached is not None:
            return cached

        tables = {
            "powers_1": monomials(l1).to(device),
            "powers_2": monomials(l2).to(device),
        }
        if self.metric == "coulomb":
            order = l1 + l2
            indices = _hermite_indices(order).to(device)
            index_of = torch.full(
                (order + 1,) * 3, -1, dtype=torch.int64, device=device
            )
            index_of[indices[:, 0], indices[:, 1], indices[:, 2]] = torch.arange(
                len(indices), device=device
            )
            hermite_1 = _hermite_indices(l1).to(device)
            hermite_2 = _hermite_indices(l2).to(device)
            tables["hermite_1"] = hermite_1
            tables["combined"] = index_of[
                (hermite_1[:, None, :] + hermite_2[None, :, :]).unbind(-1)
            ]
            tables["parity"] = (-1.0) ** hermite_2.sum(-1).to(dtype)
            tables["program"] = [
                tuple(
                    entry.to(device=device, dtype=dtype)
                    if entry.is_floating_point()
                    else entry.to(device)
                    for entry in level
                )
                for level in _hermite_program(order)
            ]
        self._tables_cache[key] = tables
        return tables

    def _pair_block(
        self,
        l1: int,
        l2: int,
        group_1: Dict[str, torch.Tensor],
        group_2: Dict[str, torch.Tensor],
        system: torch.Tensor,
        first: torch.Tensor,
        second: torch.Tensor,
        buffer: torch.Tensor,
        tables: Dict[str, torch.Tensor],
    ) -> None:
        """Compute one chunk of primitive pairs and scatter it into the buffer.

        :param l1: Angular momentum of the first side.
        :param l2: The same for the second side.
        :param group_1: First side's unit tensors.
        :param group_2: The same for the second side.
        :param system: System index of each pair.
        :param first: Unit index of each pair on the first side.
        :param second: The same for the second side.
        :param buffer: Flat accumulation buffer of all matrices.
        :param tables: The group's constant index tensors
            (:py:meth:`_group_tables`).
        """
        if self.metric == "coulomb":
            block = _coulomb_block(
                l1 + l2,
                tables["program"],
                tables["combined"],
                group_1["centers"],
                group_2["centers"],
                group_1["alphas"],
                group_2["alphas"],
                group_1["side"],
                group_2["side_parity"],
                first,
                second,
            )
            self._scatter(group_1, group_2, first, second, l1 == l2, buffer, block)
            return

        centers_1 = group_1["centers"][first]
        centers_2 = group_2["centers"][second]
        alphas_1 = group_1["alphas"][first]
        alphas_2 = group_2["alphas"][second]
        separation = centers_1 - centers_2
        distance_sq = (separation**2).sum(-1)

        keep = (
            alphas_1 * alphas_2 / (alphas_1 + alphas_2) * distance_sq < _OVERLAP_SCREEN
        )
        if not bool(keep.all()):
            (indices,) = torch.where(keep)
            system, first, second = system[indices], first[indices], second[indices]
            centers_1, centers_2 = centers_1[indices], centers_2[indices]
            alphas_1, alphas_2 = alphas_1[indices], alphas_2[indices]
            separation, distance_sq = separation[indices], distance_sq[indices]
        if len(system) == 0:
            return

        weights_1 = group_1["weights"][first]
        weights_2 = group_2["weights"][second]
        total = alphas_1 + alphas_2
        reduced = alphas_1 * alphas_2 / total
        centre = (
            alphas_1[:, None] * centers_1 + alphas_2[:, None] * centers_2
        ) / total[:, None]
        inv_2p = 0.5 / total
        e_axes = [
            _e0_axis(
                l1,
                l2,
                (centre - centers_1)[:, axis],
                (centre - centers_2)[:, axis],
                inv_2p,
            )
            for axis in range(3)
        ]
        powers_1 = tables["powers_1"]
        powers_2 = tables["powers_2"]
        cartesian = (
            e_axes[0][:, powers_1[:, 0]][:, :, powers_2[:, 0]]
            * e_axes[1][:, powers_1[:, 1]][:, :, powers_2[:, 1]]
            * e_axes[2][:, powers_1[:, 2]][:, :, powers_2[:, 2]]
        )
        prefactor = torch.exp(-reduced * distance_sq) * (torch.pi / total) ** 1.5
        block = (
            torch.einsum("pmc,pcd,pnd->pmn", weights_1, cartesian, weights_2)
            * prefactor[:, None, None]
        )
        self._scatter(group_1, group_2, first, second, l1 == l2, buffer, block)

    @staticmethod
    def _scatter(
        group_1: Dict[str, torch.Tensor],
        group_2: Dict[str, torch.Tensor],
        first: torch.Tensor,
        second: torch.Tensor,
        same_group: bool,
        buffer: torch.Tensor,
        block: torch.Tensor,
    ) -> None:
        """Accumulate pair blocks into the flat buffer of matrices.

        :param group_1: First side's unit tensors.
        :param group_2: The same for the second side.
        :param first: Unit index of each pair on the first side.
        :param second: The same for the second side.
        :param same_group: Whether both sides are the same unit group.
        :param buffer: Flat accumulation buffer of all matrices.
        :param block: Pair blocks, shape ``(n, 2 l1 + 1, 2 l2 + 1)``.
        """
        flat = (
            group_1["outer"][first][:, :, None] + group_2["inner"][second][:, None, :]
        )
        buffer.scatter_add_(0, flat.reshape(-1), block.reshape(-1))
        # The metric is symmetric and only l1 <= l2 groups (first <= second
        # within l1 == l2) are enumerated; place the transposed block on the
        # other side of the diagonal, except for a unit paired with itself.
        transposed = block.transpose(1, 2)
        if same_group:
            transposed = transposed * (first != second).to(block.dtype)[:, None, None]
        flat_t = (
            group_2["outer"][second][:, :, None] + group_1["inner"][first][:, None, :]
        )
        buffer.scatter_add_(0, flat_t.reshape(-1), transposed.reshape(-1))

    @staticmethod
    def _hermite_side(
        angular: int,
        alphas: torch.Tensor,
        weights: torch.Tensor,
        powers: torch.Tensor,
        hermite: torch.Tensor,
    ) -> torch.Tensor:
        """One side's spherical-to-Hermite transformation.

        :param angular: The side's angular momentum.
        :param alphas: Exponents, shape ``(n,)``.
        :param weights: Monomial weights, shape ``(n, 2l + 1, n_monomials)``.
        :param powers: Monomial exponents, shape ``(n_monomials, 3)``.
        :param hermite: Hermite indices, shape ``(n_hermite, 3)``.
        :return: Tensor of shape ``(n, 2l + 1, n_hermite)``.
        """
        table = _d_table(angular, alphas)
        transform = (
            table[:, powers[:, 0][:, None], hermite[:, 0][None, :]]
            * table[:, powers[:, 1][:, None], hermite[:, 1][None, :]]
            * table[:, powers[:, 2][:, None], hermite[:, 2][None, :]]
        )
        return torch.einsum("pmc,pch->pmh", weights, transform)
