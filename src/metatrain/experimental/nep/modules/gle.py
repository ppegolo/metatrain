"""GLE drift matrix: the Cholesky factor of a per-atom generalised-Langevin
drift matrix, built the same way a tensorial NEP builds a polarizability.

The atom-centred NEP descriptors are invariant, so the network only ever
predicts scalars; every direction comes from contracting those scalars with
Cartesian tensors built from the neighborhood geometry.  Here the tensors are

    T_i^(k,c) = sum_{j in class c} g_k(r_ij) (rhat_ij rhat_ij^T - I / 3)

with ``g_k`` the NEP radial basis (Chebyshev windows times the NEP cutoff,
which goes to zero with zero derivative at ``r_cut``), and the basis of a
block is ``[I, T^(k,c)...]``.  Since every basis element is a proper,
parity-even, symmetric tensor, so is every block of ``L``, and equivariance
under O(3) is exact by construction rather than learned.

For each atom ``i`` the head returns the block-lower-triangular factor

    L_i in R^{3(n+1) x 3(n+1)},   n = n_aux

over the channels ``(p, s^1, ..., s^n)``.  Blocks above the diagonal are
identically zero and are never parameterized.  The drift matrix is
``A_i = L_i L_i^T`` (symmetric positive semidefinite by construction); the
noise map ``B_i = sqrt(2 kB T) L_i`` is left to the integrator, which never
needs a Cholesky decomposition at runtime.

The block-lower-triangular form with positive-definite diagonal blocks *is*
the gauge fix: nothing here diagonalizes or sorts the auxiliary rates.
"""

import dataclasses
import math
from typing import Dict, List, Optional

import torch
from torchnep.nep_descriptor import _chebyshev_basis, _cutoff


@dataclasses.dataclass(frozen=True)
class GLEDriftConfig:
    """Configuration of the GLE drift head.

    :param n_aux: Number of auxiliary momentum channels ``n``.  The factor
        ``L`` is an ``(n + 1) x (n + 1)`` grid of ``3 x 3`` blocks.
    :param n_windows: Number of radial windows per neighbor class.
    :param r_cut: Cutoff of the tensor basis, in the length unit of the
        systems.
    :param gamma_min: Smallest auxiliary rate at initialisation, in inverse
        time units of the drift matrix.
    :param gamma_max: Largest auxiliary rate at initialisation.
    :param eps_floor: Floor added to the softplus-constrained coefficient of
        the identity in every diagonal block.
    :param markovian_block: Whether the momentum-momentum block carries
        dissipation.  This is a *physics* switch, not a numerical one: with
        ``False`` the ``(p, p)`` block is identically zero, so all the
        dissipation has to come from the auxiliary channels (a pure-memory
        test).  The resulting ``L`` is only semidefinite, which the OU step
        tolerates.
    :param n_classes: Number of neighbor classes.  The default of 2 splits
        neighbors into bonded and non-bonded and requires a topology; set it
        to 1 when no topology is available.
    """

    n_aux: int = 4
    n_windows: int = 2
    r_cut: float = 5.0
    gamma_min: float = 0.01
    gamma_max: float = 10.0
    eps_floor: float = 1.0e-4
    markovian_block: bool = True
    n_classes: int = 2

    def __post_init__(self) -> None:
        if self.n_aux < 1:
            raise ValueError(f"`n_aux` must be positive, got {self.n_aux}")
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
    def n_channels(self) -> int:
        """Number of ``3 x 3`` block rows, i.e. ``n_aux + 1``."""
        return self.n_aux + 1

    @property
    def n_blocks(self) -> int:
        """Number of lower-triangular blocks ``(a, b)``, ``a >= b``."""
        return self.n_channels * (self.n_channels + 1) // 2

    @property
    def n_basis(self) -> int:
        """Number of tensor basis elements, the identity included."""
        return 1 + self.n_windows * self.n_classes


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


def assemble_blocks(
    coefficients: torch.Tensor, basis: torch.Tensor, config: GLEDriftConfig
) -> torch.Tensor:
    """Contract the scalar coefficients with the tensor basis.

    :param coefficients: ``[n_slots, n_blocks, n_basis]`` scalars, already
        constrained (see :class:`GLEDriftHead`).
    :param basis: ``[n_slots, n_basis, 3, 3]`` tensor basis.
    :param config: Head configuration.
    :return: ``[n_slots, n_channels, n_channels, 3, 3]`` block-lower-
        triangular factor; blocks above the diagonal are exactly zero.
    """
    blocks = torch.einsum("spm,smij->spij", coefficients, basis)

    n_slots = blocks.shape[0]
    n_channels = config.n_channels
    rows, columns = _tril_indices(n_channels, blocks.device)
    factor = torch.zeros(
        (n_slots, n_channels, n_channels, 3, 3),
        dtype=blocks.dtype,
        device=blocks.device,
    )
    factor[:, rows, columns] = blocks
    return factor


def _tril_indices(n_channels: int, device: torch.device) -> torch.Tensor:
    indices = torch.tril_indices(n_channels, n_channels, device=device)
    return indices[0], indices[1]


def _diagonal_block_positions(n_channels: int) -> List[int]:
    """Position of each diagonal block ``(a, a)`` in the flat block list."""
    rows, columns = torch.tril_indices(n_channels, n_channels)
    return [index for index in range(rows.shape[0]) if rows[index] == columns[index]]


def _softplus_inverse(value: float) -> float:
    return float(math.log(math.expm1(value)))


class GLEDriftHead(torch.nn.Module):
    """Scalar readout of the GLE drift factor on top of NEP descriptors.

    The network is the usual per-element NEP network — one hidden layer with
    ``tanh`` — widened to ``n_blocks * n_basis`` outputs.  Its outputs are the
    amplitudes of the tensor basis and are used raw, except for the
    coefficient of the identity in a diagonal block, which is passed through
    ``softplus`` and floored so the diagonal blocks stay positive definite.

    The auxiliary rates must start log-spaced: the bias of those constrained
    coefficients is initialised so that ``(lambda_id^(aa))^2`` is
    ``gamma_a``, log-spaced over ``[gamma_min, gamma_max]``, and every other
    output starts at zero.  A random initialisation would collapse all the
    timescales into one decade and the model would train to nothing.

    ``markovian_block=False`` zeroes the whole momentum-momentum block: a
    physics switch that forces all dissipation through the auxiliary
    channels, not a numerical option.

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
        self.n_outputs = config.n_blocks * config.n_basis

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
        trajectory-based loss sampled at a finite interval — a channel with
        ``gamma >~ 1 / dt_dump`` cannot be told apart from delta-correlated
        friction.  Starting ``(p, p)`` at a competitive amplitude therefore
        lets the optimizer dump kernel weight into it while the memory
        channels undertrain, which then looks exactly like a system with
        little memory.  At the floor, instantaneous friction has to be earned
        against the memory representation, and ``markovian_block=False``
        becomes a small perturbation of the default initialisation rather
        than a different regime.
        """
        biases = torch.zeros(
            n_types, config.n_blocks, config.n_basis, dtype=torch.float64
        )
        rates = torch.logspace(
            math.log10(config.gamma_min),
            math.log10(config.gamma_max),
            config.n_aux,
            dtype=torch.float64,
        )
        rates = torch.cat(
            [torch.tensor([config.eps_floor], dtype=torch.float64), rates]
        )

        for position, block in enumerate(_diagonal_block_positions(config.n_channels)):
            # A^(aa) = L^(aa) (L^(aa))^T ~ gamma_a * I, so the identity
            # coefficient has to come out of the softplus at sqrt(gamma_a)
            target = math.sqrt(float(rates[position])) - config.eps_floor
            biases[:, block, 0] = _softplus_inverse(max(target, 1.0e-12))
        return biases.reshape(n_types, config.n_blocks * config.n_basis)

    def coefficients(
        self, descriptors: torch.Tensor, type_ids: torch.Tensor
    ) -> torch.Tensor:
        """Constrained basis amplitudes, ``[n_atoms, n_blocks, n_basis]``."""
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
        outputs = outputs.reshape(-1, config.n_blocks, config.n_basis)
        # only the identity coefficient of the diagonal blocks is constrained
        constrained = (
            torch.nn.functional.softplus(outputs[:, self.diagonal_blocks, 0])
            + config.eps_floor
        )
        amplitudes = outputs.clone()
        amplitudes[:, self.diagonal_blocks, 0] = constrained
        return amplitudes

    def forward(
        self,
        descriptors: torch.Tensor,
        type_ids: torch.Tensor,
        basis: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict the drift factor and the drift matrix.

        :param descriptors: ``[n_atoms, n_features]`` NEP descriptors.
        :param type_ids: ``[n_atoms]`` atomic type indices.
        :param basis: ``[n_atoms, n_basis, 3, 3]`` tensor basis.
        :return: ``L_blocks`` of shape ``[n_atoms, n + 1, n + 1, 3, 3]`` and
            the drift matrix ``A`` of shape ``[n_atoms, 3(n+1), 3(n+1)]``.
        """
        amplitudes = self.coefficients(descriptors, type_ids)
        factor = assemble_blocks(amplitudes, basis, self.config)

        if not self.config.markovian_block:
            # zero the (p, p) block: all dissipation goes through the
            # auxiliary channels
            mask = torch.ones_like(factor)
            mask[:, 0, 0] = 0.0
            factor = factor * mask

        return {"L_blocks": factor, "A": drift_matrix(factor)}


def drift_matrix(factor: torch.Tensor) -> torch.Tensor:
    """``A = L L^T`` from the block form of ``L``.

    :param factor: ``[n_slots, n_channels, n_channels, 3, 3]`` blocks.
    :return: ``[n_slots, 3 * n_channels, 3 * n_channels]`` drift matrices.
    """
    n_slots, n_channels = factor.shape[0], factor.shape[1]
    flat = factor.permute(0, 1, 3, 2, 4).reshape(
        n_slots, 3 * n_channels, 3 * n_channels
    )
    return flat @ flat.transpose(-1, -2)
