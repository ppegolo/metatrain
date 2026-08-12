"""System-level conditioning embeddings for charge and spin multiplicity."""

from typing import List

import torch


class SystemConditioningEmbedding(torch.nn.Module):
    """Embeds per-system charge and spin multiplicity into per-atom features.

    Each system in a batch can have a different total charge and spin multiplicity.
    These are embedded via learned lookup tables, concatenated, and projected down
    to the model's feature dimension. The resulting per-system embedding is broadcast
    to all atoms belonging to that system.

    The module declares :attr:`required_data_keys` so that the training pipeline
    can automatically set up the data transforms that attach the required values
    to each ``System``, without the trainer needing to know the key names.

    :param d_out: Output embedding dimension (should match d_node).
    :param max_charge: Maximum absolute charge value. Supports charges in
        the range ``[-max_charge, +max_charge]``.
    :param max_spin_multiplicity: Maximum spin multiplicity (2S+1). Supports values
        in the range ``[1, max_spin_multiplicity]``.
    """

    required_data_keys: List[str] = ["charge", "spin_multiplicity"]

    def __init__(
        self, d_out: int, max_charge: int = 10, max_spin_multiplicity: int = 10
    ):
        super().__init__()
        self.max_charge = max_charge
        self.max_spin_multiplicity = max_spin_multiplicity
        d_inner = d_out
        self.charge_embedding = torch.nn.Embedding(2 * max_charge + 1, d_inner)
        self.spin_multiplicity_embedding = torch.nn.Embedding(
            max_spin_multiplicity, d_inner
        )
        # Zero-init the output gate so the module starts as a no-op (adds zero
        # to node features).  This stabilises early training: the base model
        # learns first and the conditioning branch activates gradually as the
        # gate weights move off zero.
        gate = torch.nn.Linear(d_inner, d_out)
        torch.nn.init.zeros_(gate.weight)
        torch.nn.init.zeros_(gate.bias)
        self.project = torch.nn.Sequential(
            torch.nn.Linear(2 * d_inner, d_inner),
            torch.nn.SiLU(),
            gate,
        )

    def validate(self, charge: torch.Tensor, spin_multiplicity: torch.Tensor) -> None:
        """Check that charge and spin_multiplicity values are within the supported
        range.

        Call this outside of ``torch.compile`` regions to get descriptive errors.

        :param charge: Per-system total charge, shape ``[n_systems]``.
        :param spin_multiplicity: Per-system spin multiplicity, shape ``[n_systems]``.
        """
        if (charge < -self.max_charge).any() or (charge > self.max_charge).any():
            raise ValueError(
                f"charge values must be in [{-self.max_charge}, "
                f"{self.max_charge}], got min={charge.min().item()}, "
                f"max={charge.max().item()}. Increase max_charge in "
                f"model hypers to support wider charge ranges."
            )
        if (spin_multiplicity < 1).any() or (
            spin_multiplicity > self.max_spin_multiplicity
        ).any():
            raise ValueError(
                f"spin_multiplicity values must be in [1, "
                f"{self.max_spin_multiplicity}], got "
                f"min={spin_multiplicity.min().item()}, "
                f"max={spin_multiplicity.max().item()}. Increase "
                f"max_spin_multiplicity in model hypers to support higher "
                f"spin multiplicities."
            )

    def forward(
        self,
        charge: torch.Tensor,
        spin_multiplicity: torch.Tensor,
        system_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-atom conditioning features from per-system charge and
        spin_multiplicity.

        :param charge: Per-system total charge, shape ``[n_systems]``, integer.
        :param spin_multiplicity: Per-system spin multiplicity (2S+1), shape
            ``[n_systems]``, integer >= 1.
        :param system_indices: Maps each atom to its system index,
            shape ``[n_atoms]``.
        :return: Per-atom conditioning features, shape ``[n_atoms, d_out]``.
        """
        c_emb = self.charge_embedding(charge + self.max_charge)  # [n_systems, d_out]
        s_emb = self.spin_multiplicity_embedding(spin_multiplicity - 1)
        system_emb = self.project(torch.cat([c_emb, s_emb], dim=-1))  # [n_systems, d]
        return system_emb[system_indices]  # [n_atoms, d_out]


class AtomicChargeEmbedding(torch.nn.Module):
    """Embeds precomputed per-atom charges ("oracle charges") into node features.

    The charges come from a cheap external calculation (GFN2-xTB partial
    charges; see :py:mod:`metatrain.utils.oracle_charges`) and act as a
    physics-informed conditioning channel: they carry the global charge
    bookkeeping (which fragment of a charged complex holds the ion) that a
    local model must otherwise infer from minority training data. They are a
    deterministic function of the model's own inputs, so they add inductive
    bias rather than information — the model is free to learn on top of, or
    ignore, the channel.

    Each atom contributes ``(q, mask)``: ``mask`` is 1 where an oracle value
    is present and 0 where it is not, so "the oracle says zero charge" and
    "no oracle available" are distinct inputs. Missing oracles (``mask=0``,
    ``q=0``) are the graceful-degradation path for systems where the oracle
    is unavailable or too expensive.

    **Conditioning dropout.** With ``dropout > 0``, during training each
    *system* in the batch independently has its oracle replaced by the
    missing state (``q=0, mask=0``) with that probability, redrawn at every
    forward pass (a fresh Bernoulli per system per step, like standard
    dropout — a fixed subset would let the model memorise which systems are
    oracle-free). This trains the oracle-free operating mode alongside the
    conditioned one, in the spirit of classifier-free-guidance dropout.

    Like :py:class:`SystemConditioningEmbedding`, the output gate is
    zero-initialised so the module starts as a no-op and activates gradually.

    :param d_out: Output embedding dimension (should match d_node).
    :param dropout: Per-system probability of dropping the oracle channel
        during training. ``0`` disables dropout.
    """

    required_data_keys: List[str] = ["mtt::oracle_charges"]

    def __init__(self, d_out: int, dropout: float = 0.0):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.dropout = dropout
        d_inner = d_out
        gate = torch.nn.Linear(d_inner, d_out)
        torch.nn.init.zeros_(gate.weight)
        torch.nn.init.zeros_(gate.bias)
        self.project = torch.nn.Sequential(
            torch.nn.Linear(2, d_inner),
            torch.nn.SiLU(),
            torch.nn.Linear(d_inner, d_inner),
            torch.nn.SiLU(),
            gate,
        )

    def forward(
        self,
        charges: torch.Tensor,
        mask: torch.Tensor,
        system_indices: torch.Tensor,
        n_systems: int,
    ) -> torch.Tensor:
        """Embed per-atom oracle charges.

        :param charges: Oracle charges per atom, shape ``(n_atoms,)``; zero
            where absent.
        :param mask: 1 where an oracle value is present, 0 where absent,
            shape ``(n_atoms,)``.
        :param system_indices: System index of each atom, shape
            ``(n_atoms,)``.
        :param n_systems: Number of systems in the batch.
        :return: Per-atom conditioning embedding, shape ``(n_atoms, d_out)``.
        """
        if self.training and self.dropout > 0.0:
            keep = (torch.rand(n_systems, device=charges.device) >= self.dropout).to(
                charges.dtype
            )[system_indices]
            charges = charges * keep
            mask = mask * keep
        features = torch.stack([charges, mask], dim=-1)
        return self.project(features)
