"""GLE drift matrix: a per-atom generalised-Langevin drift matrix, built the
same way a tensorial NEP builds a polarizability.

The atom-centred NEP descriptors are invariant, so the network only ever
predicts scalars; every direction comes from contracting those scalars with
Cartesian tensors built from the neighborhood geometry.  Here the tensors are

    T_i^(k,c) = sum_{j in class c} g_k(r_ij) (rhat_ij rhat_ij^T - I / 3)

with ``g_k`` the NEP radial basis (Chebyshev windows times the NEP cutoff,
which goes to zero with zero derivative at ``r_cut``), and the basis of a
block is ``[I, T^(k,c)...]``.  Since every basis element is a proper,
parity-even, symmetric tensor, so is every block the head predicts, and
equivariance under O(3) is exact by construction rather than learned.

Structure of the drift matrix
-----------------------------

For each atom ``i`` the head returns the canonical GLE drift

    A = L L^T + Omega,    Omega^T = -Omega

over the channels ``(p, s^1, ..., s^n)``, ``n = n_aux``, with

* ``L`` **block diagonal**: ``L_pp`` (instantaneous friction) and one
  ``L_ss^(c)`` per auxiliary channel (its relaxation rate).  No ``p``-``s``
  blocks.
* ``Omega`` carrying *all* the ``p``-``s`` coupling: ``Omega_ps^(c) = W_c``
  and ``Omega_sp^(c) = -W_c^T``.  Each ``W_c`` is a ``3 x 3`` Cartesian block
  built from the same parity-even basis as ``L``; channel-space antisymmetry
  is orthogonal to the Cartesian-parity rule, so no ``l = 1`` basis element
  appears anywhere.
* ``Omega_ss`` zero between the ``n_aux`` *monotone* channels, and one
  antisymmetric intra-pair rotation ``omega_c I_3`` for each of the
  ``n_pairs`` *paired* channels ``(s^{c+}, s^{c-})``.  The derivation is in
  ``glecg/docs/oscillatory_channels.md``; the summary is in the next section.

This is not a refactor of the older ``A = L L^T`` with block-lower-triangular
``L``: that class was provably too small.  The embedded memory kernel is
``K(t) = 2 A_pp delta(t) - A_ps exp(-A_ss t) A_sp``, and with ``A`` symmetric
positive semidefinite the tail ``-A_ps exp(-A_ss t) A_ps^T`` is negative
semidefinite, so only *delta-friction minus a decaying tail* was
representable and a plain positive memory kernel -- the object this framework
exists to learn -- was not.  With the antisymmetric coupling,

    K(t) = 2 A_pp delta(t) + sum_c W_c exp(-gamma_c t) W_c^T

whose tail is symmetric positive semidefinite: positive Prony kernels are in
the class.

Oscillatory pairs
-----------------

Positive Prony kernels are also *all* that class holds, and the fine-grained
kernel measured by the FG campaign is not one of them: its per-lag ratios
strictly decrease (``0.740 -> 0.483``, 4.9 sigma) where a positive sum of
exponentials forces them to increase.  ``n_pairs`` paired channels fix that.
A pair is ``(s^{c+}, s^{c-})`` with

* an antisymmetric intra-pair rotation ``Omega_{s+ s-} = omega_c I_3``,
  Cartesian-isotropic;
* its own rate per member, ``gamma_c^+`` and ``gamma_c^-``;
* a coupling to the momentum on the ``+`` member only, ``W_c^- = 0``.

The kernel is then

    K(t) = 2 A_pp delta(t) + sum_mono W_c e^{-gamma_c t} W_c^T
                           + sum_pairs f_c(t) W_c W_c^T

    f_c(t) = e^{-gbar t} [cos(nu t) + delta sin(nu t) / nu],
    gbar = (gamma^+ + gamma^-) / 2,  delta = (gamma^- - gamma^+) / 2,
    nu = sqrt(omega^2 - delta^2)

which can decay faster than any exponential and may cross zero.  ``W_c^- = 0``
is not a simplification: with matrix-valued couplings the cross terms of a
pair are *exactly* the antisymmetric part of ``K(t)``, so ``K = K^T`` -- a
time-reversal requirement -- forces them to vanish at every ``t``.  Dropping
``W^-`` enforces that by construction for any basis, and loses nothing: the
surviving ``sin`` amplitude comes from ``delta``, i.e. from the two members
having *different* rates, not from ``W^-``.

``f_c'(0) = -gamma_c^+``, so the even-in-``t`` kernel Mori-Zwanzig prescribes
is the ``gamma^+ -> 0`` boundary of the domain rather than an identity.
``omega_c = 0`` recovers a monotone channel of rate ``gamma^+`` exactly, but
``K`` depends on ``omega`` only through ``omega^2``, so ``omega = 0`` is a
stationary point of every loss: the initialisation must place ``omega_c`` at a
finite spread, which is what ``omega_min``/``omega_max`` are for.

The fluctuation-dissipation relation is unchanged and pointwise:
``Sym(A) = L L^T`` whatever ``Omega`` is -- ``Omega_ss`` included, since it is
antisymmetric like the rest of it -- so ``B = sqrt(2 kB T) L`` still holds,
``Re lambda(A) >= 0`` still holds, and the equilibrium covariance is still
``kB T`` on every channel.  Because ``L`` is block diagonal, the noise driving
``p`` and the noise driving the auxiliaries are *uncorrelated*:
``B B^T = 2 kB T diag(L_pp L_pp^T, L_ss L_ss^T)``.  What replaces the pointwise
positivity of the old kernel tail is passivity, ``Re K_hat(i Q) >= 0``, and it
follows from ``Sym(A_ss) >= 0`` alone -- for any ``Omega_ss`` whatsoever.

``markovian_block=False`` zeroes ``L_pp`` only, which is now a physically
real setting: the momentum keeps its coupling ``W_c`` to the auxiliaries and
is thermostatted purely through memory.

Gauge.  With ``L_ss`` one block per channel and ``W`` free, the ``s -> G s``
degeneracy of the old parameterisation reduces to a per-channel sign and a
permutation of the channels.  The log-spaced ``gamma_c`` at initialisation is
the remaining fix; nothing here sorts or re-orders the channels at runtime.
"""

import dataclasses
import math
from typing import Dict, List, Optional, Tuple

import torch
from torchnep.nep_descriptor import _chebyshev_basis, _cutoff


@dataclasses.dataclass(frozen=True)
class GLEDriftConfig:
    """Configuration of the GLE drift head.

    :param n_aux: Number of *monotone* auxiliary momentum channels, the ones
        with no intra-pair rotation.  With ``n_pairs = 0`` this is the whole
        auxiliary space and the class is the old positive-Prony one.
    :param n_pairs: Number of oscillatory auxiliary *pairs*, each contributing
        two channels ``(s^{c+}, s^{c-})``, one coupling block ``W_c`` (on the
        ``+`` member only) and one intra-pair frequency ``omega_c``.
    :param omega_min: Smallest intra-pair frequency at initialisation, in
        inverse time units.  Must not be zero: ``K`` depends on ``omega`` only
        through ``omega^2``, so ``omega = 0`` is gradient-dead.
    :param omega_max: Largest intra-pair frequency at initialisation.
    :param n_windows: Number of radial windows per neighbor class.
    :param r_cut: Cutoff of the tensor basis, in the length unit of the
        systems.
    :param gamma_min: Smallest auxiliary rate at initialisation, in inverse
        time units of the drift matrix.
    :param gamma_max: Largest auxiliary rate at initialisation.
    :param eps_floor: Floor added to the softplus-constrained coefficient of
        the identity in every diagonal block of ``L``.
    :param markovian_block: Whether the momentum-momentum block carries
        instantaneous dissipation.  This is a *physics* switch, not a
        numerical one: with ``False`` the ``L_pp`` block is identically zero,
        so all the dissipation has to come from the memory kernel carried by
        ``W`` (a pure-memory bath).  The resulting ``L`` is only semidefinite,
        which the OU step tolerates.
    :param n_classes: Number of neighbor classes.  The default of 2 splits
        neighbors into bonded and non-bonded and requires a topology; set it
        to 1 when no topology is available.
    """

    n_aux: int = 4
    n_pairs: int = 0
    n_windows: int = 2
    r_cut: float = 5.0
    gamma_min: float = 0.01
    gamma_max: float = 10.0
    omega_min: float = 1.0
    omega_max: float = 10.0
    eps_floor: float = 1.0e-4
    markovian_block: bool = True
    n_classes: int = 2

    def __post_init__(self) -> None:
        if self.n_aux < 0:
            raise ValueError(f"`n_aux` must not be negative, got {self.n_aux}")
        if self.n_pairs < 0:
            raise ValueError(f"`n_pairs` must not be negative, got {self.n_pairs}")
        if self.n_aux + self.n_pairs < 1:
            raise ValueError(
                "at least one auxiliary channel is required, got "
                f"n_aux={self.n_aux} and n_pairs={self.n_pairs}"
            )
        if not 0.0 < self.omega_min <= self.omega_max:
            raise ValueError(
                "`omega_min` and `omega_max` must satisfy 0 < omega_min <= "
                f"omega_max, got {self.omega_min} and {self.omega_max}"
            )
        if self.n_windows < 1:
            raise ValueError(f"`n_windows` must be positive, got {self.n_windows}")
        if self.n_classes < 1:
            raise ValueError(f"`n_classes` must be positive, got {self.n_classes}")
        if not 0.0 < self.gamma_min <= self.gamma_max:
            raise ValueError(
                "`gamma_min` and `gamma_max` must satisfy 0 < gamma_min <= "
                f"gamma_max, got {self.gamma_min} and {self.gamma_max}"
            )

    @classmethod
    def from_options(cls, options: Optional[Dict[str, object]]) -> "GLEDriftConfig":
        """Build a configuration from the target section of a metatrain yaml.

        .. code-block:: yaml

            targets:
              mtt::gle_drift:
                type:
                  gle_drift:
                    n_aux: 4
                    n_windows: 2
                    r_cut: 5.0
                    gamma_min: 0.01
                    gamma_max: 10.0
                    eps_floor: 1.0e-4
                    markovian_block: true
                    n_classes: 2

        :param options: The mapping under ``gle_drift``; ``None`` selects
            every default.
        :return: The configuration.
        """
        if options is None:
            return cls()
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = set(options) - known
        if unknown:
            raise ValueError(
                f"unknown GLE drift options {sorted(unknown)}; "
                f"expected any of {sorted(known)}"
            )
        return cls(**options)  # type: ignore[arg-type]

    @property
    def n_aux_total(self) -> int:
        """Number of auxiliary channels: ``n_aux`` monotone plus two per pair.

        This is the size of the auxiliary momentum ``s``, i.e. what
        :class:`glecg.config.SimConfig` needs as its ``n_aux``.
        """
        return self.n_aux + 2 * self.n_pairs

    @property
    def n_channels(self) -> int:
        """Number of ``3 x 3`` block rows, i.e. ``n_aux_total + 1``."""
        return self.n_aux_total + 1

    @property
    def n_blocks(self) -> int:
        """Number of diagonal blocks of ``L``, one per channel."""
        return self.n_channels

    @property
    def n_coupling(self) -> int:
        """Number of coupling blocks ``W_c``: one per monotone channel and one
        per pair.  A pair's ``-`` member has ``W^- = 0`` structurally, so it
        gets no slot at all -- see ``docs/oscillatory_channels.md`` §3.
        """
        return self.n_aux + self.n_pairs

    @property
    def n_terms(self) -> int:
        """Rows of the amplitude tensor: ``L`` blocks, ``W`` blocks, ``omega``."""
        return self.n_blocks + self.n_coupling + self.n_pairs

    @property
    def n_basis(self) -> int:
        """Number of tensor basis elements, the identity included."""
        return 1 + self.n_windows * self.n_classes

    def pair_channels(self, index: int) -> Tuple[int, int]:
        """Channel indices ``(s^{c+}, s^{c-})`` of pair ``index``.

        Channel order is ``p``, the ``n_aux`` monotone channels, then the pairs
        with their two members adjacent.  Channel ``0`` is ``p``, so an
        auxiliary channel ``a`` sits at matrix block ``a + 1``.

        :param index: Pair number, ``0 <= index < n_pairs``.
        :return: The two auxiliary channel indices, ``p`` excluded.
        """
        first = self.n_aux + 2 * index
        return first, first + 1


def edge_classes_from_bonds(
    edges: torch.Tensor, bonds: torch.Tensor, n_classes: int
) -> torch.Tensor:
    """Classify edges as bonded (class 0) or non-bonded (class 1).

    :param edges: ``[E, 2]`` tensor of ``(center, neighbor)`` atom indices.
    :param bonds: ``[B, 2]`` tensor of bonded atom pairs, in either order.
    :param n_classes: Number of neighbor classes; ``1`` puts every edge in
        the single class.
    :return: ``[E]`` tensor of class indices.
    """
    if n_classes == 1:
        return torch.zeros(edges.shape[0], dtype=torch.long, device=edges.device)
    if n_classes != 2:
        raise ValueError(
            "only 1 (all neighbors alike) and 2 (bonded / non-bonded) neighbor "
            f"classes are implemented, got n_classes={n_classes}"
        )

    bonds = bonds.to(device=edges.device, dtype=torch.long).reshape(-1, 2)
    n_atoms = int(edges.max()) + 1 if edges.numel() > 0 else 1
    n_atoms = max(n_atoms, int(bonds.max()) + 1 if bonds.numel() > 0 else 1)
    # a symmetric lookup over pair keys, so bond direction does not matter
    bonded = torch.zeros(n_atoms * n_atoms, dtype=torch.bool, device=edges.device)
    if bonds.numel() > 0:
        bonded[bonds[:, 0] * n_atoms + bonds[:, 1]] = True
        bonded[bonds[:, 1] * n_atoms + bonds[:, 0]] = True
    keys = edges[:, 0] * n_atoms + edges[:, 1]
    return torch.where(
        bonded[keys],
        torch.zeros_like(keys),
        torch.ones_like(keys),
    )


def build_tensor_basis(
    n_slots: int,
    slot_index: torch.Tensor,
    edge_vectors: torch.Tensor,
    edge_class: torch.Tensor,
    config: GLEDriftConfig,
) -> torch.Tensor:
    """Assemble the per-slot Cartesian tensor basis ``[I, T^(k,c)...]``.

    The slot index makes this generic over the set the tensors are attached
    to: pass the center atom of every edge for the per-atom head (the only
    one implemented), or an edge index for a future per-edge variant.

    :param n_slots: Number of output slots (atoms, for the per-atom head).
    :param slot_index: ``[E]`` slot each edge contributes to.
    :param edge_vectors: ``[E, 3]`` edge displacement vectors.
    :param edge_class: ``[E]`` neighbor class of each edge.
    :param config: Head configuration.
    :return: ``[n_slots, n_basis, 3, 3]`` basis tensors, elements ``1:``
        being traceless and symmetric.
    """
    dtype = edge_vectors.dtype
    device = edge_vectors.device
    distances = torch.linalg.norm(edge_vectors, dim=1)
    r_cut = torch.full_like(distances, config.r_cut)

    # `_cutoff` is NEP's own cutoff: it reaches zero with zero derivative at
    # r_cut, but keeps oscillating past it, so edges outside are masked out.
    # Both the value and its derivative are continuous across the boundary.
    inside = distances < config.r_cut
    envelope = torch.where(
        inside, _cutoff(distances, r_cut), torch.zeros_like(distances)
    )
    # NEP's radial basis: n_windows Chebyshev windows times the envelope
    windows = _chebyshev_basis(
        config.n_windows - 1, distances, r_cut, envelope
    )  # [E, n_windows]

    directions = edge_vectors / distances.clamp_min(1.0e-12).unsqueeze(1)
    identity = torch.eye(3, dtype=dtype, device=device)
    traceless = directions.unsqueeze(2) * directions.unsqueeze(1) - identity / 3.0

    one_hot = torch.nn.functional.one_hot(
        edge_class.to(torch.long), num_classes=config.n_classes
    ).to(dtype)
    # channel (k, c) is flattened as k * n_classes + c
    weights = (windows.unsqueeze(2) * one_hot.unsqueeze(1)).reshape(
        windows.shape[0], config.n_windows * config.n_classes
    )
    contributions = weights.unsqueeze(2).unsqueeze(3) * traceless.unsqueeze(1)

    accumulated = torch.zeros(
        (n_slots, config.n_windows * config.n_classes, 3, 3), dtype=dtype, device=device
    ).index_add_(0, slot_index, contributions)
    identity_element = identity.expand(n_slots, 1, 3, 3)
    return torch.cat([identity_element, accumulated], dim=1)


def contract_basis(coefficients: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Contract scalar amplitudes with the tensor basis.

    :param coefficients: ``[n_slots, n_terms, n_basis]`` scalars.
    :param basis: ``[n_slots, n_basis, 3, 3]`` tensor basis.
    :return: ``[n_slots, n_terms, 3, 3]`` Cartesian blocks.
    """
    return torch.einsum("spm,smij->spij", coefficients, basis)


def assemble_blocks(
    coefficients: torch.Tensor, basis: torch.Tensor, config: GLEDriftConfig
) -> torch.Tensor:
    """Contract the diagonal-block amplitudes into the block-diagonal ``L``.

    :param coefficients: ``[n_slots, n_blocks, n_basis]`` scalars, already
        constrained (see :func:`constrain_coefficients`).
    :param basis: ``[n_slots, n_basis, 3, 3]`` tensor basis.
    :param config: Head configuration.
    :return: ``[n_slots, n_channels, n_channels, 3, 3]`` factor; every
        off-diagonal block is exactly zero.
    """
    blocks = contract_basis(coefficients, basis)

    n_slots = blocks.shape[0]
    n_channels = config.n_channels
    channels = torch.arange(n_channels, device=blocks.device)
    factor = torch.zeros(
        (n_slots, n_channels, n_channels, 3, 3),
        dtype=blocks.dtype,
        device=blocks.device,
    )
    factor[:, channels, channels] = blocks
    return factor


def assemble_coupling(coefficients: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Contract the coupling amplitudes into the blocks ``W_c`` of ``Omega``.

    These are used raw: ``W`` enters the memory kernel quadratically, so no
    positivity constraint applies to it, and its sign is a gauge choice of the
    auxiliary channel.

    :param coefficients: ``[n_slots, n_aux, n_basis]`` raw amplitudes.
    :param basis: ``[n_slots, n_basis, 3, 3]`` tensor basis.
    :return: ``[n_slots, n_aux, 3, 3]`` coupling blocks.
    """
    return contract_basis(coefficients, basis)


def constrain_coefficients(
    outputs: torch.Tensor, diagonal_blocks: torch.Tensor, config: GLEDriftConfig
) -> torch.Tensor:
    """Apply the positivity constraint of the diagonal blocks of ``L``.

    Only the coefficient of the identity in a diagonal block is constrained;
    every other amplitude is used raw.  This is the single definition of the
    ``softplus`` plus ``eps_floor`` construction: heads must call it rather
    than repeat it.

    :param outputs: ``[n_slots, n_terms, n_basis]`` raw amplitudes, holding
        the diagonal blocks of ``L`` at the positions ``diagonal_blocks``.
    :param diagonal_blocks: Flat positions of the diagonal blocks, as returned
        by :func:`_diagonal_block_positions`.
    :param config: Head configuration.
    :return: ``[n_slots, n_terms, n_basis]`` constrained amplitudes.
    """
    constrained = (
        torch.nn.functional.softplus(outputs[:, diagonal_blocks, 0]) + config.eps_floor
    )
    amplitudes = outputs.clone()
    amplitudes[:, diagonal_blocks, 0] = constrained
    return amplitudes


def split_amplitudes(
    amplitudes: torch.Tensor, config: GLEDriftConfig
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a flat amplitude tensor into the ``L``, ``W`` and ``omega`` parts.

    The layout, shared by every head, is ``n_blocks`` diagonal-``L`` slots,
    ``n_coupling`` coupling slots, then ``n_pairs`` intra-pair frequency slots.

    A frequency is a single scalar -- ``Omega_ss`` has to be Cartesian-isotropic
    for ``K(t) = K(t)^T`` to survive an anisotropic rate block
    (``docs/oscillatory_channels.md`` §3) -- but it occupies a whole row of the
    rectangular amplitude tensor so that every head keeps one uniform readout.
    Only the identity coefficient of those rows is read; the remaining columns
    are inert and receive no gradient.

    :param amplitudes: ``[n_slots, n_terms, n_basis]`` amplitudes.
    :param config: Head configuration.
    :return: The ``[n_slots, n_blocks, n_basis]`` diagonal-``L`` part, the
        ``[n_slots, n_coupling, n_basis]`` coupling part and the
        ``[n_slots, n_pairs]`` intra-pair frequencies.
    """
    coupling_end = config.n_blocks + config.n_coupling
    return (
        amplitudes[:, : config.n_blocks],
        amplitudes[:, config.n_blocks : coupling_end],
        amplitudes[:, coupling_end:, 0],
    )


def _diagonal_block_positions(n_channels: int) -> List[int]:
    """Position of each diagonal block of ``L`` in the flat amplitude list."""
    return list(range(n_channels))


def _softplus_inverse(value: float) -> float:
    return float(math.log(math.expm1(value)))


def log_spaced(count: int, low: float, high: float) -> torch.Tensor:
    """``count`` points log-spaced over ``[low, high]``.

    ``torch.logspace(a, b, 1)`` returns the *left* endpoint, which for a single
    pair would put it at the slowest rate of the grid rather than in the middle
    of it.  This returns the geometric mean instead.

    :param count: Number of points; ``0`` gives an empty tensor.
    :param low: Lower end of the range, strictly positive.
    :param high: Upper end of the range.
    :return: ``[count]`` values, ``float64``.
    """
    if count <= 0:
        return torch.zeros(0, dtype=torch.float64)
    if count == 1:
        return torch.tensor([math.sqrt(low * high)], dtype=torch.float64)
    return torch.logspace(math.log10(low), math.log10(high), count, dtype=torch.float64)


def initial_rates(config: GLEDriftConfig) -> torch.Tensor:
    """Auxiliary rates at initialisation, in channel order.

    The ``n_aux`` monotone channels take the log-spaced grid they always did;
    each pair takes one log-spaced rate, *shared* by both of its members, so
    ``delta = 0`` and the pair starts at ``e^{-gamma t} cos(omega t)``.  A
    pair whose two members start at different rates would start with a sine
    term already switched on, which is a stronger prior than the data
    justifies.

    :param config: Head configuration.
    :return: ``[n_aux_total]`` rates, in inverse time units.
    """
    # `torch.logspace`, not `log_spaced`: the exact grid the monotone class has
    # always used, single-channel edge case included
    mono = torch.logspace(
        math.log10(config.gamma_min),
        math.log10(config.gamma_max),
        config.n_aux,
        dtype=torch.float64,
    )
    pairs = log_spaced(config.n_pairs, config.gamma_min, config.gamma_max)
    return torch.cat([mono, pairs.repeat_interleave(2)])


def initial_coupling_rates(config: GLEDriftConfig) -> torch.Tensor:
    """Rate of the channel each coupling block attaches to, at initialisation.

    ``W_c = sqrt(gamma_c) I`` needs the rate of the *coupled* channel, which
    for a pair is its ``+`` member.

    :param config: Head configuration.
    :return: ``[n_coupling]`` rates.
    """
    rates = initial_rates(config)
    return torch.cat([rates[: config.n_aux], rates[config.n_aux :][::2]])


def initial_frequencies(config: GLEDriftConfig) -> torch.Tensor:
    """Intra-pair frequencies at initialisation, log-spaced and never zero.

    ``K`` depends on ``omega`` only through ``omega^2``, so ``omega = 0`` has
    an identically vanishing gradient: a pair started there stays monotone for
    the whole fit.  See ``docs/oscillatory_channels.md`` §4.

    :param config: Head configuration.
    :return: ``[n_pairs]`` frequencies, in inverse time units.
    """
    return log_spaced(config.n_pairs, config.omega_min, config.omega_max)


class GLEDriftHead(torch.nn.Module):
    """Scalar readout of the GLE drift on top of NEP descriptors.

    The network is the usual per-element NEP network -- one hidden layer with
    ``tanh`` -- widened to ``n_terms * n_basis`` outputs: the diagonal blocks
    of ``L`` first, the coupling blocks ``W_c`` next, the intra-pair
    frequencies ``omega_c`` last (identity coefficient only).  The
    outputs are the amplitudes of the tensor basis and are used raw, except
    for the coefficient of the identity in a diagonal block of ``L``, which is
    passed through ``softplus`` and floored so those blocks stay positive
    definite.  ``W`` is unconstrained.

    The auxiliary rates must start log-spaced: the bias of the constrained
    coefficient of channel ``c`` is initialised so that ``(L_ss^(c))^2`` is
    ``gamma_c``, log-spaced over ``[gamma_min, gamma_max]``.  A random
    initialisation would collapse all the timescales into one decade and the
    model would train to nothing.  The couplings start at
    ``W_c = sqrt(gamma_c) I``, so the initial kernel
    ``sum_c gamma_c exp(-gamma_c t) I`` integrates to ``n_aux`` whatever the
    rate spectrum is, and no channel starts at the ``W = 0`` saddle where the
    gradient of a kernel-matching loss vanishes.

    ``markovian_block=False`` zeroes ``L_pp`` and nothing else: the momentum
    keeps its coupling to the auxiliaries and is thermostatted by memory
    alone.

    :param n_features: Width of the NEP descriptor.
    :param n_types: Number of atomic types (one network each).
    :param n_neurons: Width of the hidden layer.
    :param config: Head configuration.
    """

    def __init__(
        self,
        n_features: int,
        n_types: int,
        n_neurons: int,
        config: GLEDriftConfig,
    ):
        super().__init__()
        self.config = config
        self.n_terms = config.n_terms
        self.n_outputs = self.n_terms * config.n_basis

        self.input_weights = torch.nn.Parameter(
            0.1 * torch.randn(n_types, n_neurons, n_features, dtype=torch.float64)
        )
        self.input_biases = torch.nn.Parameter(
            0.1 * torch.randn(n_types, n_neurons, dtype=torch.float64)
        )
        # small output weights, so the initial prediction is set by the biases
        self.output_weights = torch.nn.Parameter(
            1.0e-3
            * torch.randn(n_types, n_neurons, self.n_outputs, dtype=torch.float64)
        )
        self.output_biases = torch.nn.Parameter(
            self._initial_output_biases(n_types, config)
        )

        diagonal = _diagonal_block_positions(config.n_channels)
        self.register_buffer(
            "diagonal_blocks",
            torch.tensor(diagonal, dtype=torch.long),
            persistent=False,
        )

    @staticmethod
    def _initial_output_biases(n_types: int, config: GLEDriftConfig) -> torch.Tensor:
        """Biases placing the auxiliary rates log-spaced at initialisation.

        The momentum-momentum block starts at the floor instead, at a rate of
        ``eps_floor`` rather than anywhere near ``gamma_min``.  The Markovian
        block and the fastest memory channel are near-degenerate for any
        trajectory-based loss sampled at a finite interval -- a channel with
        ``gamma >~ 1 / dt_dump`` cannot be told apart from delta-correlated
        friction.  Starting ``L_pp`` at a competitive amplitude therefore lets
        the optimizer dump kernel weight into it while the memory channels
        undertrain, which then looks exactly like a system with little memory.
        At the floor, instantaneous friction has to be earned against the
        memory representation, and ``markovian_block=False`` becomes a small
        perturbation of the default initialisation rather than a different
        regime.
        """
        n_terms = config.n_terms
        biases = torch.zeros(n_types, n_terms, config.n_basis, dtype=torch.float64)
        diagonal_rates = torch.cat(
            [
                torch.tensor([config.eps_floor], dtype=torch.float64),
                initial_rates(config),
            ]
        )

        for position, block in enumerate(_diagonal_block_positions(config.n_channels)):
            # A^(aa) = L^(aa) (L^(aa))^T ~ gamma_a I, so the identity
            # coefficient has to come out of the softplus at sqrt(gamma_a)
            target = math.sqrt(float(diagonal_rates[position])) - config.eps_floor
            biases[:, block, 0] = _softplus_inverse(max(target, 1.0e-12))

        # W_c = sqrt(gamma_c) I: unconstrained, so the bias is the value
        coupling_end = config.n_blocks + config.n_coupling
        biases[:, config.n_blocks : coupling_end, 0] = (
            initial_coupling_rates(config).sqrt().unsqueeze(0)
        )
        # omega_c: unconstrained too, and never zero (see `initial_frequencies`)
        biases[:, coupling_end:, 0] = initial_frequencies(config).unsqueeze(0)
        return biases.reshape(n_types, n_terms * config.n_basis)

    def coefficients(
        self, descriptors: torch.Tensor, type_ids: torch.Tensor
    ) -> torch.Tensor:
        """Constrained basis amplitudes, ``[n_atoms, n_blocks + n_aux, n_basis]``."""
        dtype = descriptors.dtype
        type_ids = type_ids.to(device=descriptors.device, dtype=torch.long)

        input_weights = self.input_weights.to(dtype)[type_ids]
        input_biases = self.input_biases.to(dtype)[type_ids]
        hidden = torch.tanh(
            torch.einsum("nhd,nd->nh", input_weights, descriptors) - input_biases
        )
        output_weights = self.output_weights.to(dtype)[type_ids]
        output_biases = self.output_biases.to(dtype)[type_ids]
        outputs = torch.einsum("nho,nh->no", output_weights, hidden) + output_biases

        config = self.config
        outputs = outputs.reshape(-1, self.n_terms, config.n_basis)
        return constrain_coefficients(outputs, self.diagonal_blocks, config)

    def forward(
        self,
        descriptors: torch.Tensor,
        type_ids: torch.Tensor,
        basis: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict the drift factor, the couplings and the drift matrix.

        :param descriptors: ``[n_atoms, n_features]`` NEP descriptors.
        :param type_ids: ``[n_atoms]`` atomic type indices.
        :param basis: ``[n_atoms, n_basis, 3, 3]`` tensor basis.
        :return: ``L_blocks`` of shape ``[n_atoms, C, C, 3, 3]``, ``W_blocks``
            of shape ``[n_atoms, n_coupling, 3, 3]``, ``pair_frequencies`` of
            shape ``[n_atoms, n_pairs]`` and the drift matrix
            ``A = L L^T + Omega`` of shape ``[n_atoms, 3C, 3C]``, with
            ``C = n_aux_total + 1``.
        """
        amplitudes = self.coefficients(descriptors, type_ids)
        l_amplitudes, w_amplitudes, frequencies = split_amplitudes(
            amplitudes, self.config
        )
        factor = assemble_blocks(l_amplitudes, basis, self.config)
        couplings = assemble_coupling(w_amplitudes, basis)

        if not self.config.markovian_block:
            # zero L_pp: the momentum keeps its coupling `W` and is
            # thermostatted purely through the memory kernel
            mask = torch.ones_like(factor)
            mask[:, 0, 0] = 0.0
            factor = factor * mask

        return {
            "L_blocks": factor,
            "W_blocks": couplings,
            "pair_frequencies": frequencies,
            "A": drift_matrix(factor, couplings, frequencies),
        }


def omega_matrix(
    couplings: torch.Tensor, pair_frequencies: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """The antisymmetric drift ``Omega`` from its coupling blocks.

    ``Omega_ps^(c) = W_c``, ``Omega_sp^(c) = -W_c^T``, and ``Omega_ss`` the
    block-diagonal sum of one intra-pair rotation per pair, in the
    channel-major layout ``3 a + i``.

    The auxiliary channels are ordered ``n_mono`` monotone ones first, then the
    pairs with their two members adjacent; the coupling slots are ordered the
    same way, with *one* slot per pair (the ``+`` member -- a pair's ``-``
    member has ``W^- = 0`` structurally).  ``n_mono`` is therefore
    ``couplings.shape[1] - n_pairs``, which is what this reads.

    :param couplings: ``[n_slots, n_mono + n_pairs, 3, 3]`` blocks ``W_c``.
    :param pair_frequencies: ``[n_slots, n_pairs]`` intra-pair frequencies
        ``omega_c``, or ``None`` for no pairs.
    :return: ``[n_slots, 3(n_aux_total + 1), 3(n_aux_total + 1)]``
        antisymmetric matrices.
    """
    n_slots, n_coupling = couplings.shape[0], couplings.shape[1]
    n_pairs = 0 if pair_frequencies is None else pair_frequencies.shape[1]
    n_mono = n_coupling - n_pairs
    if n_mono < 0:
        raise ValueError(
            f"got {n_pairs} pair frequencies for only {n_coupling} coupling blocks"
        )
    n_aux = n_mono + 2 * n_pairs

    size = 3 * (n_aux + 1)
    omega = torch.zeros(
        (n_slots, size, size), dtype=couplings.dtype, device=couplings.device
    )
    # [n_slots, 3, 3 n_aux]: the whole (p, s) row of Omega at once, with the
    # zero blocks of the pairs' `-` members left in place
    row = torch.zeros(
        (n_slots, 3, n_aux, 3), dtype=couplings.dtype, device=couplings.device
    )
    row[:, :, :n_mono] = couplings[:, :n_mono].permute(0, 2, 1, 3)
    if n_pairs > 0:
        plus = n_mono + 2 * torch.arange(n_pairs, device=couplings.device)
        row[:, :, plus] = couplings[:, n_mono:].permute(0, 2, 1, 3)
    row = row.reshape(n_slots, 3, 3 * n_aux)
    omega[:, :3, 3:] = row
    omega[:, 3:, :3] = -row.transpose(-1, -2)

    if n_pairs > 0 and pair_frequencies is not None:
        identity = torch.eye(3, dtype=couplings.dtype, device=couplings.device)
        rotations = pair_frequencies.reshape(n_slots, n_pairs, 1, 1) * identity
        for index in range(n_pairs):
            first = 3 * (1 + n_mono + 2 * index)
            second = first + 3
            omega[:, first:second, second : second + 3] = rotations[:, index]
            omega[:, second : second + 3, first:second] = -rotations[:, index]
    return omega


def drift_matrix(
    factor: torch.Tensor,
    couplings: torch.Tensor,
    pair_frequencies: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``A = L L^T + Omega`` from the block forms of ``L``, ``W`` and ``omega``.

    :param factor: ``[n_slots, n_channels, n_channels, 3, 3]`` blocks of ``L``.
    :param couplings: ``[n_slots, n_mono + n_pairs, 3, 3]`` blocks ``W_c``.
    :param pair_frequencies: ``[n_slots, n_pairs]`` intra-pair frequencies, or
        ``None`` for no pairs.
    :return: ``[n_slots, 3 * n_channels, 3 * n_channels]`` drift matrices.
    """
    n_slots, n_channels = factor.shape[0], factor.shape[1]
    flat = factor.permute(0, 1, 3, 2, 4).reshape(
        n_slots, 3 * n_channels, 3 * n_channels
    )
    return flat @ flat.transpose(-1, -2) + omega_matrix(couplings, pair_frequencies)
