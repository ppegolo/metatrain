import copy
import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import ase.data
import ase.units
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DistributedSampler

from metatrain.composition import train_or_load_composition_model
from metatrain.pet.modules.finetuning import apply_finetuning_strategy
from metatrain.utils.abc import ModelInterface, TrainerInterface
from metatrain.utils.additive import get_remove_additive_transform
from metatrain.utils.augmentation import O3Augmenter
from metatrain.utils.data import (
    CollateFn,
    CombinedDataLoader,
    Dataset,
    build_train_dataloaders,
    build_val_dataloaders,
    get_num_workers,
    unpack_batch,
    validate_num_workers,
)
from metatrain.utils.data.atomic_basis_helpers import (
    get_prepare_atomic_basis_targets_transform,
)
from metatrain.utils.distributed.distributed_data_parallel import (
    DistributedDataParallel,
)
from metatrain.utils.distributed.slurm import (
    initialize_slurm_nccl_process_group,
    resolve_distributed,
)
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.io import check_file_extension
from metatrain.utils.logging import ROOT_LOGGER, MetricLogger
from metatrain.utils.metrics import MAEAccumulator, RMSEAccumulator, get_selected_metric
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists_transform,
)
from metatrain.utils.per_atom import average_by_num_atoms
from metatrain.utils.scaler import get_remove_scale_transform
from metatrain.utils.system_data import get_system_data_transform
from metatrain.utils.transfer import batch_to

from . import checkpoints
from .covariant import make_A_covariant
from .documentation import TrainerHypers
from .model import GLE, gle_target_info


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    train_hypers: TrainerHypers,
    steps_per_epoch: int,
) -> LambdaLR:
    """
    Get a CosineAnnealing learning-rate scheduler with warmup

    :param optimizer: The optimizer for which to create the scheduler.
    :param train_hypers: The training hyperparameters.
    :param steps_per_epoch: The number of steps per epoch.
    :return: The learning rate scheduler.
    """
    total_steps = train_hypers["num_epochs"] * steps_per_epoch
    warmup_steps = int(train_hypers["warmup_fraction"] * total_steps)
    min_lr_ratio = 0.0  # hardcoded for now, could be made configurable in the future

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            progress = (current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler


def make_A(theta: torch.Tensor, n: int, eps: float = 1e-6) -> torch.Tensor:
    """Map unconstrained network output to a stable GLE drift matrix."""
    if theta.shape[-1] != n * n:
        raise ValueError(f"Expected theta.shape[-1] = {n * n}, got {theta.shape[-1]}")

    batch_shape = theta.shape[:-1]
    num_L = n * (n + 1) // 2

    theta_L = theta[..., :num_L]
    theta_K = theta[..., num_L:]

    L = torch.zeros(*batch_shape, n, n, dtype=theta.dtype, device=theta.device)
    tril_i, tril_j = torch.tril_indices(n, n, offset=0, device=theta.device)
    L[..., tril_i, tril_j] = theta_L

    diag_mask = tril_i == tril_j
    idx = torch.arange(n, device=theta.device)
    L[..., idx, idx] = torch.nn.functional.softplus(theta_L[..., diag_mask]) + eps

    symmetric = L @ L.transpose(-1, -2)

    K = torch.zeros(*batch_shape, n, n, dtype=theta.dtype, device=theta.device)
    lower_i, lower_j = torch.tril_indices(n, n, offset=-1, device=theta.device)
    K[..., lower_i, lower_j] = theta_K
    K[..., lower_j, lower_i] = -theta_K

    return 0.5 * symmetric + K


def mvn_loss_cholesky(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    target: torch.Tensor,
    jitter: float,
) -> torch.Tensor:
    """Negative log likelihood for a batched multivariate normal."""
    _, dimension = mean.shape
    eye = torch.eye(dimension, dtype=covariance.dtype, device=covariance.device)
    # Guarantee positive-definiteness. The transition covariance
    # m kT (I - Tpp Tpp^T) uses only the 3x3 momentum block Tpp of the full OU
    # propagator exp(-A dt); this marginal is an approximation and can lose PSD-ness
    # when the memory kernel makes the momentum block expansive (strong caging /
    # backscatter at the training lag), so a fixed jitter is not enough. Shift each
    # batch element up by just enough to clear its smallest eigenvalue, then add the
    # jitter floor. No-op (only +jitter) when already well-conditioned, so systems
    # that never trip the approximation are unchanged.
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    # detached: the floor is a numerical conditioner (like jitter), not a term whose
    # magnitude the model should be differentiated through (avoids eigvalsh backward
    # instability at near-degenerate eigenvalues).
    min_eig = torch.linalg.eigvalsh(covariance.detach())[..., 0]
    shift = torch.clamp_min(jitter - min_eig, 0.0)
    covariance = covariance + (shift.unsqueeze(-1).unsqueeze(-1) + jitter) * eye

    L = torch.linalg.cholesky(covariance)
    diff = (target - mean).unsqueeze(-1)
    y = torch.linalg.solve_triangular(L, diff, upper=False)

    mahalanobis = y.squeeze(-1).pow(2).sum(dim=-1)
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)

    nll = 0.5 * (dimension * math.log(2.0 * math.pi) + logdet + mahalanobis)
    # reduced chi-square (mean whitened-residual norm / dof): calibration diagnostic,
    # target 1.0 (>1 covariance too small / under-damped, <1 too large). Unit- and
    # dt-independent, unlike the NLL itself.
    chi2_red = (mahalanobis / dimension).mean()
    return nll.mean(), chi2_red


class GLELoss:
    """Loss functions for the GLE drift matrix.

    The model emits an unconstrained ``theta`` per bead; :func:`make_A` maps it to a
    stable drift matrix ``A``, and the noise follows from the fluctuation-dissipation
    theorem. Two targets are supported: the momentum transition density (per-bead or
    pairwise / momentum-conserving), and direct matching of ``A``'s analytic memory
    kernel to a precomputed per-bead-type Volterra kernel.
    """

    def __init__(
        self,
        num_auxiliary_variables: int,
        bead_mass_by_z: Dict[int, float],
        temperature: float,
        jitter: float,
        pairwise: bool = False,
        neighbor_list_options: Optional[Any] = None,
        target_kind: str = "transition",
        kernel_target: Optional[Dict[str, Any]] = None,
        kernel_app_weight: float = 0.0,
        theta_baseline: Optional[torch.Tensor] = None,
        kernel_reg_weight: float = 0.0,
        theta_reg_weight: float = 0.0,
        gamma0_reg_weight: float = 0.0,
        gamma0_pin_freqs_thz: Optional[List[float]] = None,
        covariant: bool = False,
    ) -> None:
        # COVARIANT mode: the auxiliaries are 3-VECTORS rather than scalars, so the state
        # is 3 (1 + n_aux) and the drift is assembled from equivariant l = 0 + l = 2
        # Cartesian blocks (`metatrain.gle.covariant`). Scalar auxiliaries admit no
        # covariant GLE with memory at all -- covariance would force the p <-> s coupling
        # to zero -- so this is a different state space, not a reparametrisation, and the
        # two are deliberately not checkpoint-compatible.
        #
        # Everything downstream is unchanged by construction: the state is ordered
        # block-major with the momentum FIRST, so the `[:3, :3]` momentum block of the
        # propagator and the `[3:, 3:]` auxiliary block still slice correctly.
        self.covariant = covariant
        self.n_gle_variables = (
            3 * (1 + num_auxiliary_variables)
            if covariant
            else 3 + num_auxiliary_variables
        )
        self.bead_mass_by_z = bead_mass_by_z
        self.temperature = temperature
        self.jitter = jitter
        self.pairwise = pairwise
        self.neighbor_list_options = neighbor_list_options
        if pairwise and neighbor_list_options is None:
            raise ValueError(
                "pairwise GLE loss requires the model neighbor-list options"
            )
        # PMF-aware "memory_kernel" target: match A's analytic memory kernel
        # -tr(A_ps e^{-A_ss t} A_sp)/3 to a precomputed per-bead-type residual-force
        # kernel (mean force + slow model-error floor already removed offline).
        self.target_kind = target_kind
        self.kernel_app_weight = kernel_app_weight
        # Delta-learning transport-aware regularizers for the transition target:
        # theta_baseline[Z] is the frozen per-type baseline the model adds to mtt::A;
        # kernel_reg_weight pins the full-drift transport friction to the Volterra
        # kernel; theta_reg_weight keeps the correction theta_net small.
        self.theta_baseline = theta_baseline
        self.kernel_reg_weight = kernel_reg_weight
        self.theta_reg_weight = theta_reg_weight
        # gamma0 pin: hold each bead's zero-frequency friction (Schur complement of
        # the drift) at the frozen baseline's per-type value. The single-lag
        # transition NLL cannot distinguish dissipative friction from
        # energy-conserving antisymmetric rotation into the auxiliaries and actively
        # erodes gamma0 to zero; this term blocks that degenerate direction while
        # leaving the finite-frequency response free.
        self.gamma0_reg_weight = gamma0_reg_weight
        self.gamma0_target: Optional[torch.Tensor] = None

    def _make_A(self, theta: torch.Tensor) -> torch.Tensor:
        """Assemble the drift, by whichever construction this loss was configured with.

        Covariant mode uses equivariant l = 0 + l = 2 Cartesian blocks and a positive
        semi-definite ``M M^T`` symmetric part; the scalar mode keeps the Cholesky-style
        ``1/2 L L^T + K``. They are NOT interchangeable -- the state spaces differ
        (3 (1 + n_aux) against 3 + n_aux) -- so the choice is made once, here, rather than
        being inferred from a tensor shape somewhere downstream.
        """
        if self.covariant:
            n_aux = self.n_gle_variables // 3 - 1
            return make_A_covariant(theta, n_aux)
        return make_A(theta, self.n_gle_variables)
        # Frequency grid for the pin (THz). [0.0] = the plain gamma0 point-pin.
        # A point-pin at omega=0 is evadable: a model can keep gamma0 fixed and dig a
        # hole in Re Sigma(omega) in the cage-hopping band 0.05-1 THz. Pinning a grid
        # across the transport band closes that escape.
        if gamma0_pin_freqs_thz is None:
            gamma0_pin_freqs_thz = [0.0]
        # f [THz] -> angular frequency in ASE time units
        self._pin_omegas = torch.tensor(
            [2.0 * math.pi * f * 1e-3 / ase.units.fs for f in gamma0_pin_freqs_thz],
            dtype=torch.float64,
        )
        if gamma0_reg_weight > 0.0:
            if theta_baseline is None:
                raise ValueError(
                    "gamma0_reg_weight > 0 requires a theta_baseline (the pin "
                    "target is the baseline's own friction response)"
                )
            with torch.no_grad():
                A_base = self._make_A(theta_baseline.to(torch.float64))
                self.gamma0_target = self._band_of(A_base)  # [Z, F]
        if kernel_reg_weight > 0.0 and kernel_target is None:
            raise ValueError(
                "kernel_reg_weight > 0 requires kernel_target_file (the Volterra "
                "per-type kernels) to be set"
            )
        if kernel_target is not None:
            self._kt_ase = torch.tensor(kernel_target["t_ase"], dtype=torch.float64)
            self._kt_w = torch.tensor(kernel_target["w"], dtype=torch.float64)
            self._kt_by_z = {
                int(z): torch.tensor(v, dtype=torch.float64)
                for z, v in kernel_target["by_z"].items()
            }
        self.last_chi2_red = float("nan")  # calibration diagnostic, set each call

    def _gamma0_of(self, A: torch.Tensor) -> torch.Tensor:
        """Zero-frequency friction of drift matrices A [..., n, n]: the Schur
        complement tr(A_pp - A_ps A_ss^-1 A_sp)/3 = Markov friction + int_0^inf K(t) dt.
        A small diagonal shift on A_ss (scaled by its mean diagonal) guards the solve.
        """
        App = A[..., :3, :3]
        Aps = A[..., :3, 3:]
        Asp = A[..., 3:, :3]
        Ass = A[..., 3:, 3:]
        eye = torch.eye(Ass.shape[-1], dtype=A.dtype, device=A.device)
        shift = 1e-10 * Ass.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-30)
        Ass = Ass + shift[..., None, None] * eye
        schur = App - Aps @ torch.linalg.solve(Ass, Asp)
        return schur.diagonal(dim1=-2, dim2=-1).mean(-1)

    def _band_of(self, A: torch.Tensor) -> torch.Tensor:
        """Dissipative response Re tr(A_pp - A_ps (A_ss + i omega)^-1 A_sp)/3 of drift
        matrices A [B, n, n] on the pin frequency grid. Returns [B, F]. omega = 0
        reduces to `_gamma0_of` (the zero-frequency friction)."""
        omegas = self._pin_omegas.to(device=A.device)
        cdtype = torch.complex128 if A.dtype == torch.float64 else torch.complex64
        App = A[:, :3, :3].to(cdtype)
        Aps = A[:, :3, 3:].to(cdtype)
        Asp = A[:, 3:, :3].to(cdtype)
        Ass = A[:, 3:, 3:].to(cdtype)
        eye = torch.eye(Ass.shape[-1], dtype=cdtype, device=A.device)
        shift = 1e-10 * A[:, 3:, 3:].diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(
            1e-30
        )
        out = []
        for omega in omegas:
            Ass_w = (
                Ass + (shift.to(cdtype) + 1j * omega.to(A.dtype))[..., None, None] * eye
            )
            sigma = App - Aps @ torch.linalg.solve(Ass_w, Asp)
            out.append(sigma.diagonal(dim1=-2, dim2=-1).mean(-1).real)
        return torch.stack(out, dim=-1).to(A.dtype)

    def _bead_masses(self, systems: List[Any], device: Any, dtype: Any) -> torch.Tensor:
        """Per-bead masses gathered by atomic-number tag, in `systems` order."""
        types = torch.concatenate([s.types for s in systems]).to(device)
        masses = torch.zeros(types.shape[0], device=device, dtype=dtype)
        covered = torch.zeros(types.shape[0], dtype=torch.bool, device=device)
        for z, m in self.bead_mass_by_z.items():
            mask = types == z
            masses[mask] = m
            covered |= mask
        if not bool(covered.all()):
            missing = sorted(set(types[~covered].tolist()))
            raise ValueError(
                f"no bead mass configured for atomic types {missing}; "
                "add them to bead_mass_by_symbol"
            )
        return masses

    def __call__(
        self,
        systems: List[Any],
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        theta_total = predictions["mtt::A"].block().values
        A = self._make_A(theta_total)
        if self.target_kind == "memory_kernel":
            return self._kernel_loss(systems, A)
        if self.pairwise:
            nll = self._pairwise_nll(systems, A, targets)
            return nll + self._hybrid_regularizers(systems, A, theta_total)

        momenta = torch.concatenate(
            [
                system.get_data("momentum").block().values.squeeze(-1)
                for system in systems
            ]
        )
        future_momenta = targets["momentum"].block().values.squeeze(-1)

        time_lags = torch.concatenate(
            [
                torch.full(
                    (len(system),),
                    system.get_data("mtt::time_lag").block().values.item(),
                    device=system.positions.device,
                    dtype=system.positions.dtype,
                )
                for system in systems
            ]
        )
        time_lags = time_lags * ase.units.fs

        propagator = torch.matrix_exp(-A * time_lags.unsqueeze(-1).unsqueeze(-1))[
            :, :3, :3
        ]
        mean = (propagator @ momenta.unsqueeze(-1)).squeeze(-1)
        masses = self._bead_masses(systems, A.device, A.dtype)
        covariance = (
            masses[:, None, None]
            * ase.units.kB
            * self.temperature
            * (
                torch.eye(3, dtype=A.dtype, device=A.device)
                - propagator @ propagator.transpose(-1, -2)
            )
        )

        nll, chi2_red = mvn_loss_cholesky(mean, covariance, future_momenta, self.jitter)
        self.last_chi2_red = float(chi2_red.detach())
        return nll + self._hybrid_regularizers(systems, A, theta_total)

    def _hybrid_regularizers(
        self, systems: List[Any], A: torch.Tensor, theta_total: torch.Tensor
    ) -> torch.Tensor:
        """Delta-learning transport-aware regularizers for the transition target.

        ``kernel_reg_weight``: match the full drift's analytic memory kernel to the
        per-type Volterra target (pins the transport friction / zero-frequency kernel
        integral, which the transition NLL alone does not constrain).
        ``theta_reg_weight``: L2-penalize the correction
        ``theta_net = mtt::A - theta_base[Z]`` so the frozen per-type baseline stays
        the dominant prior. Both default to zero (no-op). The transition calibration
        diagnostic ``last_chi2_red`` is preserved.
        """
        extra = torch.zeros((), dtype=A.dtype, device=A.device)
        if self.kernel_reg_weight > 0.0:
            saved_chi2 = self.last_chi2_red
            extra = extra + self.kernel_reg_weight * self._kernel_loss(systems, A)
            self.last_chi2_red = saved_chi2  # keep the transition diagnostic
        if self.theta_reg_weight > 0.0 and self.theta_baseline is not None:
            types = torch.concatenate([s.types for s in systems]).to(A.device)
            theta_base = self.theta_baseline.to(
                device=A.device, dtype=theta_total.dtype
            ).index_select(0, types)
            theta_net = theta_total - theta_base
            extra = extra + self.theta_reg_weight * theta_net.pow(2).sum(-1).mean()
        if self.gamma0_reg_weight > 0.0 and self.gamma0_target is not None:
            types = torch.concatenate([s.types for s in systems]).to(A.device)
            band = self._band_of(A)  # [B, F]
            target = self.gamma0_target.to(
                device=A.device, dtype=band.dtype
            ).index_select(0, types)  # [B, F]
            # normalize per type by the largest response on the grid, not per
            # frequency: individual Re Sigma(omega) values may pass through ~0
            scale = target.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
            rel = (band - target) / scale
            extra = extra + self.gamma0_reg_weight * rel.pow(2).mean()
        return extra

    def _pairwise_nll(
        self,
        systems: List[Any],
        A: torch.Tensor,
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Momentum-conserving transition loss.

        The friction acts on the relative velocity of neighbor bead pairs
        ``w_e = P_I / M_I - P_J / M_J`` with the symmetric edge drift
        ``A_e = 0.5 (A_I + A_J)``. Because the OU transition on ``w_e`` is pushed
        back to beads ``I, J`` with opposite sign, total coarse momentum is
        conserved (``sum_I K_IJ = 0``). The auxiliary variables are marginalized
        exactly as in the per-bead case, so the observed relative-velocity
        transition is Gaussian:
            w_e^+ | w_e^-  ~  N( T_{e,PP} w_e^-,  Sigma_w (I - T_{e,PP} T_{e,PP}^T) )
        with the equilibrium relative-velocity covariance
        ``Sigma_w = k_B T (1/M_I + 1/M_J)``.
        """
        device, dtype = A.device, A.dtype
        masses = self._bead_masses(systems, device, dtype)

        # per-bead input / target momenta, concatenated in `systems` order to match
        # the per-atom row order of the `mtt::A` prediction block.
        P_in = torch.concatenate(
            [s.get_data("momentum").block().values.squeeze(-1) for s in systems]
        )
        P_out = targets["momentum"].block().values.squeeze(-1)

        # build the global (directed) edge list from the model neighbor lists,
        # dropping periodic self-images (i == j) whose relative velocity is zero.
        centers: List[torch.Tensor] = []
        neighbors: List[torch.Tensor] = []
        edge_lags: List[torch.Tensor] = []
        offset = 0
        for s in systems:
            nl_values = s.get_neighbor_list(self.neighbor_list_options).samples.values
            ci = nl_values[:, 0]
            nj = nl_values[:, 1]
            keep = ci != nj
            ci = ci[keep] + offset
            nj = nj[keep] + offset
            centers.append(ci)
            neighbors.append(nj)
            lag = s.get_data("mtt::time_lag").block().values.item()
            edge_lags.append(
                torch.full((ci.shape[0],), lag, device=device, dtype=dtype)
            )
            offset += len(s)

        ci = torch.concatenate(centers)
        nj = torch.concatenate(neighbors)
        if ci.numel() == 0:
            raise ValueError("pairwise GLE loss: no neighbor pairs in batch")
        lags = torch.concatenate(edge_lags) * ase.units.fs

        w_in = P_in[ci] / masses[ci, None] - P_in[nj] / masses[nj, None]
        w_out = P_out[ci] / masses[ci, None] - P_out[nj] / masses[nj, None]
        A_edge = 0.5 * (A[ci] + A[nj])

        propagator = torch.matrix_exp(-A_edge * lags.unsqueeze(-1).unsqueeze(-1))[
            :, :3, :3
        ]
        mean = (propagator @ w_in.unsqueeze(-1)).squeeze(-1)
        # k_B T (1/M_I + 1/M_J), per edge
        sigma_w = (
            ase.units.kB * self.temperature * (1.0 / masses[ci] + 1.0 / masses[nj])
        )
        covariance = sigma_w[:, None, None] * (
            torch.eye(3, dtype=dtype, device=device)
            - propagator @ propagator.transpose(-1, -2)
        )

        nll, chi2_red = mvn_loss_cholesky(mean, covariance, w_out, self.jitter)
        self.last_chi2_red = float(chi2_red.detach())
        return nll

    def _kernel_loss(self, systems: List[Any], A: torch.Tensor) -> torch.Tensor:
        """PMF-aware memory-kernel matching loss.

        A learns friction only (the mean force is applied separately at runtime),
        so its analytic memory kernel must reproduce the residual-force kernel:
            K_model(t) = -tr( A_ps exp(-A_ss t) A_sp ) / 3
        matched per bead type to the precomputed target ``K_target`` (residual-force
        ACF with the mean force and the slow model-error floor removed). No momenta,
        mass, or temperature enter here: the target carries all of that. The 3x3
        instantaneous block A_pp is unconstrained by the kernel, so it is lightly
        regularized toward zero (friction should be memory-carried).
        """
        dev, dt = A.device, A.dtype
        Aps = A[:, :3, 3:]
        Asp = A[:, 3:, :3]
        Ass = A[:, 3:, 3:]
        App = A[:, :3, :3]
        t = self._kt_ase.to(device=dev, dtype=dt)  # [K] lags in ASE time units
        # analytic memory kernel per bead over the lag grid
        E = torch.matrix_exp(-Ass.unsqueeze(1) * t.view(1, -1, 1, 1))  # [N,K,ns,ns]
        Km = -torch.einsum("nab,nkbc,ncd->nkad", Aps, E, Asp)  # [N,K,3,3]
        Km = Km.diagonal(dim1=-2, dim2=-1).sum(-1) / 3.0  # [N,K] scalar (isotropic)
        # gather the per-bead-type target by atomic number
        types = torch.concatenate([s.types for s in systems]).to(dev)  # [N]
        tgt = torch.zeros_like(Km)
        k0 = torch.ones(Km.shape[0], device=dev, dtype=dt)
        for z, Kz in self._kt_by_z.items():
            mask = types == z
            if bool(mask.any()):
                Kz = Kz.to(device=dev, dtype=dt)
                tgt[mask] = Kz
                k0[mask] = Kz[0].clamp_min(1e-12)
        # per-type-normalized squared kernel error (so all types weigh equally, while
        # the friction magnitude relative to K(0) -- hence D -- is still fit)
        resid = (Km - tgt) / k0.view(-1, 1)
        w = self._kt_w.to(device=dev, dtype=dt).view(1, -1)
        loss = (resid**2 * w).sum(dim=1).mean()
        if self.kernel_app_weight > 0.0:
            loss = loss + self.kernel_app_weight * (App**2).sum(dim=(-1, -2)).mean()
        self.last_chi2_red = float((resid**2).mean().detach())
        return loss


def _load_kernel_target(path: str) -> Dict[str, Any]:
    """Read the per-bead-type Volterra memory kernels used as a GLE target.

    :param path: Path to the ``.npz`` holding the ``t_match_ase`` lag grid and one
        ``K_z<Z>`` array per bead type (keyed by atomic-number tag).
    :return: Dictionary with the lag grid, uniform lag weights, and the per-type
        kernels keyed by atomic number.
    """
    data = np.load(path)
    by_z = {
        int(name[len("K_z") :]): data[name]
        for name in data.files
        if name.startswith("K_z")
    }
    if not by_z:
        raise ValueError(
            f"kernel_target_file {path} contains no K_z<Z> arrays; regenerate it"
        )
    n_lags = len(data["t_match_ase"])
    return {
        "t_ase": data["t_match_ase"],
        "w": np.ones(n_lags) / n_lags,
        "by_z": by_z,
    }


class Trainer(TrainerInterface[TrainerHypers]):
    __checkpoint_version__ = 15

    def __init__(self, hypers: TrainerHypers) -> None:
        super().__init__(hypers)

        self.optimizer_state_dict: Optional[Dict[str, Any]] = None
        self.scheduler_state_dict: Optional[Dict[str, Any]] = None
        self.epoch: Optional[int] = None
        self.best_epoch: Optional[int] = None
        self.best_metric: Optional[float] = None
        self.best_model_state_dict: Optional[Dict[str, Any]] = None
        self.best_optimizer_state_dict: Optional[Dict[str, Any]] = None

    def train(
        self,
        model: GLE,
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ) -> None:
        assert dtype in GLE.__supported_dtypes__

        is_distributed = resolve_distributed(self.hypers.get("distributed"))
        is_finetune = self.hypers["finetune"]["read_from"] is not None

        if is_distributed:
            if len(devices) > 1:
                raise ValueError(
                    "Requested distributed training with the `multi-gpu` device. "
                    " If you want to run distributed training with GLE, please "
                    "set `device` to cuda."
                )
            # the calculation of the device number works both when GPUs on different
            # processes are not visible to each other and when they are
            device, world_size, rank = initialize_slurm_nccl_process_group(
                self.hypers["distributed_port"]
            )
        else:
            rank = 0
            world_size = 1
            device = devices[0]
            # only one device, as we don't support non-distributed multi-gpu for now

        if is_distributed:
            logging.info(f"Training on {world_size} devices with dtype {dtype}")
        else:
            logging.info(f"Training on device {device} with dtype {dtype}")

        # Apply fine-tuning strategy if provided
        if is_finetune:
            assert self.hypers["finetune"]["read_from"] is not None  # for mypy
            # ``inherit_heads`` is a one-time weight-copy initialization step that
            # must only run when finetuning first starts (a fresh ``Trainer``).
            is_fresh_finetune_start = self.optimizer_state_dict is None
            model = apply_finetuning_strategy(
                model,
                self.hypers["finetune"],
                apply_inherit_heads=is_fresh_finetune_start,
            )
            method = self.hypers["finetune"]["method"]
            num_params = sum(p.numel() for p in model.parameters())
            num_trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

            logging.info(f"Applied finetuning strategy: {method}")
            logging.info(
                f"Number of trainable parameters: {num_trainable_params} "
                f"[{num_trainable_params / num_params:.2%} %]"
            )
            inherit_heads = self.hypers["finetune"]["inherit_heads"]
            if inherit_heads and is_fresh_finetune_start:
                logging.info(
                    "Inheriting initial weights for heads and last layers for targets: "
                    f"from {list(inherit_heads.values())} to "
                    f"{list(inherit_heads.keys())}"
                )

        # Move the model to the device and dtype:
        model.to(device=device, dtype=dtype)
        # The additive models are always in float64 (to avoid numerical errors in
        # the composition weights, which can be very large).
        for additive_model in model.additive_models:
            additive_model.to(dtype=torch.float64)
        model.scaler.to(dtype=torch.float64)

        # Set up transformations
        dataset_info = model.dataset_info
        train_targets = dataset_info.targets
        extra_data_info = dataset_info.extra_data
        rotational_augmenter = O3Augmenter(
            target_info_dict=train_targets, extra_data_info_dict=extra_data_info
        )
        requested_neighbor_lists = get_requested_neighbor_lists(model)
        max_atoms = self.hypers["max_atoms_per_batch"]
        atomic_basis_transform, atomic_basis_reverse_transform = (
            get_prepare_atomic_basis_targets_transform(train_targets, extra_data_info)
        )

        train_or_load_composition_model(
            composition_model=model.additive_models[0],
            atomic_baseline=self.hypers["atomic_baseline"],
            train_datasets=train_datasets,
            other_additive_models=list(model.additive_models[1:]),
            batch_size=self.hypers["batch_size"],
            is_distributed=is_distributed,
            checkpoint_dir=checkpoint_dir,
        )

        if self.hypers["scale_targets"]:
            logging.info("Calculating scaling weights")
            model.scaler.train_model(
                train_datasets,
                model.additive_models,
                self.hypers["batch_size"],
                is_distributed,
                self.hypers["fixed_scaling_weights"],
                initial_transforms=[atomic_basis_transform],
                per_structure_targets=self.hypers["per_structure_targets"],
            )

        logging.info("Setting up data loaders")

        if is_distributed:
            train_samplers = [
                DistributedSampler(
                    train_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    drop_last=True,
                )
                for train_dataset in train_datasets
            ]
            val_samplers = [
                DistributedSampler(
                    val_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                    drop_last=False,
                )
                for val_dataset in val_datasets
            ]
        else:
            train_samplers = [None] * len(train_datasets)
            val_samplers = [None] * len(val_datasets)

        # Extract additive models and scaler and move them to CPU/float64 so they
        # can be used in the collate function
        model.additive_models[0].weights_to(device="cpu", dtype=torch.float64)
        additive_models = copy.deepcopy(
            model.additive_models.to(dtype=torch.float64, device="cpu")
        )
        model.additive_models.to(device)
        model.additive_models[0].weights_to(device=device, dtype=torch.float64)
        model.scaler.scales_to(device="cpu", dtype=torch.float64)
        scaler = copy.deepcopy(model.scaler.to(dtype=torch.float64, device="cpu"))
        model.scaler.to(device)
        model.scaler.scales_to(device=device, dtype=torch.float64)

        # Create collate functions
        conditioning_keys = list(model.requested_inputs().keys())
        conditioning_callables = (
            [get_system_data_transform(conditioning_keys)] if conditioning_keys else []
        )

        target_keys = list(train_targets.keys())
        # Shared callables that run after `atomic_basis_transform` (and after
        # rotational augmentation in training).
        base_callables: List[Callable[..., Any]] = [
            get_system_with_neighbor_lists_transform(requested_neighbor_lists),
            *conditioning_callables,
            get_remove_additive_transform(additive_models, train_targets),
            get_remove_scale_transform(scaler),
        ]
        collate_fn_train = CollateFn(
            target_keys=target_keys,
            callables=[
                atomic_basis_transform,
                rotational_augmenter.apply_random_augmentations,
                *base_callables,
            ],
        )
        collate_fn_val = CollateFn(
            target_keys=target_keys,
            callables=[  # no augmentation for validation
                atomic_basis_transform,
                *base_callables,
            ],
        )

        if self.hypers["num_workers"] is None:
            num_workers = get_num_workers()
            logging.info(
                "Number of workers for data-loading not provided and chosen "
                f"automatically. Using {num_workers} workers."
            )
        else:
            num_workers = self.hypers["num_workers"]
            validate_num_workers(num_workers)

        # Create dataloader for the training datasets:
        train_dataloaders, epoch_samplers = build_train_dataloaders(
            train_datasets=train_datasets,
            train_distributed_samplers=train_samplers,
            collate_fn_train=collate_fn_train,
            batch_size=self.hypers["batch_size"],
            max_atoms_per_batch=max_atoms,
            min_atoms_per_batch=self.hypers["min_atoms_per_batch"],
            num_workers=num_workers,
        )
        train_dataloader = CombinedDataLoader(train_dataloaders, shuffle=True)

        # Create dataloader for the validation datasets:
        val_dataloaders = build_val_dataloaders(
            val_datasets=val_datasets,
            val_distributed_samplers=val_samplers,
            collate_fn_val=collate_fn_val,
            batch_size=self.hypers["batch_size"],
            max_atoms_per_batch=max_atoms,
            num_workers=num_workers,
        )
        val_dataloader = CombinedDataLoader(val_dataloaders, shuffle=False)

        if is_distributed:
            model = DistributedDataParallel(model, device_ids=[device])

        outputs_list = []
        for target_name, target_info in train_targets.items():
            outputs_list.append(target_name)
            for gradient_name in target_info.gradients:
                outputs_list.append(f"{target_name}_{gradient_name}_gradients")

        # Create the GLE loss function. Unlike PET, GLE does not use the generic
        # LossAggregator: the supervised quantity (the future bead momenta, or a
        # precomputed memory kernel) is related to the model output `mtt::A` through
        # the GLE transition density rather than by a pointwise comparison.
        core_model = model.module if is_distributed else model
        pairwise = bool(self.hypers["pairwise"])
        target_kind = self.hypers["target_kind"]
        kernel_reg_weight = float(self.hypers["kernel_reg_weight"])
        # the Volterra per-type kernels are needed both by the memory_kernel target and
        # by the transport-aware transition regularizer (kernel_reg_weight > 0)
        kernel_target = None
        if target_kind == "memory_kernel" or kernel_reg_weight > 0.0:
            kernel_target = _load_kernel_target(self.hypers["kernel_target_file"])
        bead_mass_by_z = {
            ase.data.atomic_numbers[symbol]: float(mass)
            for symbol, mass in self.hypers["bead_mass_by_symbol"].items()
        }
        loss_fn = GLELoss(
            num_auxiliary_variables=core_model.n_gle_variables - 3,
            bead_mass_by_z=bead_mass_by_z,
            temperature=float(self.hypers["temperature"]),
            jitter=float(self.hypers["transition_jitter"]),
            pairwise=pairwise,
            neighbor_list_options=core_model.requested_nl if pairwise else None,
            target_kind=target_kind,
            kernel_target=kernel_target,
            kernel_app_weight=float(self.hypers["kernel_app_weight"]),
            theta_baseline=core_model.theta_baseline,
            kernel_reg_weight=kernel_reg_weight,
            theta_reg_weight=float(self.hypers["theta_reg_weight"]),
            gamma0_reg_weight=float(self.hypers["gamma0_reg_weight"]),
            gamma0_pin_freqs_thz=[
                float(f) for f in self.hypers["gamma0_pin_freqs_thz"]
            ],
        )
        if target_kind == "memory_kernel":
            loss_description = "memory-kernel (PMF-aware)"
        elif pairwise:
            loss_description = "pairwise (momentum-conserving)"
        else:
            loss_description = "per-bead"
        logging.info(
            f"Using {loss_description} GLE loss with {core_model.n_gle_variables} "
            f"variables, masses={self.hypers['bead_mass_by_symbol']} Da, "
            f"T={self.hypers['temperature']} K"
        )

        # ``mtt::A`` is not a dataset target: it is the model's own output that the
        # GLE loss turns into a transition density. It therefore has to be requested
        # explicitly alongside the dataset's targets.
        gle_output = gle_target_info(core_model.n_gle_variables)

        def requested_outputs(target_names) -> Dict[str, Any]:
            requested = {name: train_targets[name] for name in target_names}
            requested["mtt::A"] = gle_output
            return requested

        if self.hypers["weight_decay"] is not None:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.hypers["learning_rate"],
                weight_decay=self.hypers["weight_decay"],
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.hypers["learning_rate"]
            )

        if self.optimizer_state_dict is not None:
            # try to load the optimizer state dict, but this is only possible
            # if there are no new targets in the model (new parameters)
            if not (model.module if is_distributed else model).has_new_targets:
                optimizer.load_state_dict(self.optimizer_state_dict)

        # Create a learning rate scheduler
        lr_scheduler = get_scheduler(optimizer, self.hypers, len(train_dataloader))

        if self.scheduler_state_dict is not None:
            # same as the optimizer, try to load the scheduler state dict
            if not (model.module if is_distributed else model).has_new_targets:
                lr_scheduler.load_state_dict(self.scheduler_state_dict)

        per_structure_targets = self.hypers["per_structure_targets"]

        # Log the initial learning rate:
        logging.info(f"Base learning rate: {self.hypers['learning_rate']}")

        start_epoch = 0 if self.epoch is None else self.epoch + 1

        # Train the model:
        if self.best_metric is None:
            self.best_metric = float("inf")
        logging.info("Starting training")
        epoch = start_epoch

        for epoch in range(start_epoch, self.hypers["num_epochs"]):
            for sampler in epoch_samplers:
                sampler.set_epoch(epoch)
            train_rmse_calculator = RMSEAccumulator(self.hypers["log_separate_blocks"])
            val_rmse_calculator = RMSEAccumulator(self.hypers["log_separate_blocks"])
            if self.hypers["log_mae"]:
                train_mae_calculator = MAEAccumulator(
                    self.hypers["log_separate_blocks"]
                )
                val_mae_calculator = MAEAccumulator(self.hypers["log_separate_blocks"])

            train_loss = 0.0
            train_chi2 = 0.0
            train_batches = 0
            for batch in train_dataloader:
                optimizer.zero_grad()

                systems, targets, extra_data = unpack_batch(batch)
                systems, targets, extra_data = batch_to(
                    systems, targets, extra_data, dtype=dtype, device=device
                )
                predictions = evaluate_model(
                    model,
                    systems,
                    requested_outputs(targets.keys()),
                    is_training=True,
                )

                # average by the number of atoms
                predictions = average_by_num_atoms(
                    predictions, systems, per_structure_targets
                )
                targets = average_by_num_atoms(targets, systems, per_structure_targets)

                # Apply per-property scales to the predictions before loss computation.
                # The targets from the dataloader have only been scaled per-target, and
                # not per-property. This transformation only applies to targets with
                # per-property scales (i.e. multiple blocks or multiple properties), and
                # leaves the others unchanged.
                predictions = (model.module if is_distributed else model).scaler(
                    systems,
                    predictions,
                    remove=False,
                    use_per_target_scales=False,  # never before loss
                    use_per_property_scales=True,
                )

                train_loss_batch = loss_fn(systems, predictions, targets)

                if is_distributed:
                    # make sure all parameters contribute to the gradient calculation
                    # to make torch DDP happy
                    train_loss_batch += 0.0 * sum(
                        p.sum() for p in model.parameters() if p.requires_grad
                    )

                train_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.hypers["grad_clip_norm"]
                )
                optimizer.step()
                lr_scheduler.step()

                if is_distributed:
                    # sum the loss over all processes
                    torch.distributed.all_reduce(train_loss_batch)
                train_loss += train_loss_batch.item()
                train_chi2 += loss_fn.last_chi2_red
                train_batches += 1

                # Reapply scales and accumulate quantities for computing train metrics,
                # but only if this is an epoch to log
                if epoch == start_epoch or epoch % self.hypers["log_interval"] == 0:
                    scaled_predictions = (
                        model.module if is_distributed else model
                    ).scaler(
                        systems,
                        predictions,
                        remove=False,
                        use_per_target_scales=True,
                        use_per_property_scales=False,
                    )
                    scaled_targets = (model.module if is_distributed else model).scaler(
                        systems,
                        targets,
                        remove=False,
                        use_per_target_scales=True,
                        use_per_property_scales=False,
                    )

                    if self.hypers["log_separate_blocks"]:
                        # if any atomic basis outputs are present and metrics are to be
                        # reported per-block, reverse the transform (i.e. sparsify)
                        # before calculating metrics
                        systems, scaled_targets, extra_data = (
                            atomic_basis_reverse_transform(
                                systems, scaled_targets, extra_data
                            )
                        )
                        systems, scaled_predictions, _ = atomic_basis_reverse_transform(
                            systems, scaled_predictions, {}
                        )

                    train_rmse_calculator.update(
                        scaled_predictions, scaled_targets, extra_data
                    )
                    if self.hypers["log_mae"]:
                        train_mae_calculator.update(
                            scaled_predictions, scaled_targets, extra_data
                        )

            # Compute train metrics if they are to be logged this epoch:
            if epoch == start_epoch or epoch % self.hypers["log_interval"] == 0:
                finalized_train_info = train_rmse_calculator.finalize(
                    not_per_atom=["positions_gradients"] + per_structure_targets,
                    is_distributed=is_distributed,
                    device=device,
                )
                if self.hypers["log_mae"]:
                    finalized_train_info.update(
                        train_mae_calculator.finalize(
                            not_per_atom=["positions_gradients"]
                            + per_structure_targets,
                            is_distributed=is_distributed,
                            device=device,
                        )
                    )

            with torch.set_grad_enabled(
                any(target_info.gradients for target_info in train_targets.values())
            ):  # keep gradients on if any of the targets require them
                val_loss = 0.0
                val_chi2 = 0.0
                val_batches = 0
                for batch in val_dataloader:
                    systems, targets, extra_data = unpack_batch(batch)
                    systems, targets, extra_data = batch_to(
                        systems, targets, extra_data, dtype=dtype, device=device
                    )
                    predictions = evaluate_model(
                        model,
                        systems,
                        requested_outputs(targets.keys()),
                        is_training=False,
                    )

                    # average by the number of atoms
                    predictions = average_by_num_atoms(
                        predictions, systems, per_structure_targets
                    )
                    targets = average_by_num_atoms(
                        targets, systems, per_structure_targets
                    )

                    # Apply per-property scales to the predictions before loss
                    # computation. The targets from the dataloader have only been scaled
                    # per-target, and not per-property. This transformation only applies
                    # to targets with per-property scales (i.e. multiple blocks or
                    # multiple properties), and leaves the others unchanged.
                    predictions = (model.module if is_distributed else model).scaler(
                        systems,
                        predictions,
                        remove=False,
                        use_per_target_scales=False,
                        use_per_property_scales=True,
                    )

                    val_loss_batch = loss_fn(systems, predictions, targets)

                    if is_distributed:
                        # sum the loss over all processes
                        torch.distributed.all_reduce(val_loss_batch)
                    val_loss += val_loss_batch.item()
                    val_chi2 += loss_fn.last_chi2_red
                    val_batches += 1

                    # Reapply scales and accumulate quantities for computing val
                    # metrics. This is done for every epoch as validation metrics are
                    # needed for model selection
                    scaled_predictions = (
                        model.module if is_distributed else model
                    ).scaler(
                        systems,
                        predictions,
                        remove=False,
                        use_per_target_scales=True,
                        use_per_property_scales=False,
                    )
                    scaled_targets = (model.module if is_distributed else model).scaler(
                        systems,
                        targets,
                        remove=False,
                        use_per_target_scales=True,
                        use_per_property_scales=False,
                    )

                    if self.hypers["log_separate_blocks"]:
                        # if any atomic basis outputs are present and metrics are to be
                        # reported per-block, reverse the transform (i.e. sparsify)
                        # before calculating metrics
                        systems, scaled_targets, extra_data = (
                            atomic_basis_reverse_transform(
                                systems, scaled_targets, extra_data
                            )
                        )
                        systems, scaled_predictions, _ = atomic_basis_reverse_transform(
                            systems, scaled_predictions, {}
                        )

                    val_rmse_calculator.update(
                        scaled_predictions, scaled_targets, extra_data
                    )
                    if self.hypers["log_mae"]:
                        val_mae_calculator.update(
                            scaled_predictions, scaled_targets, extra_data
                        )

            # Compute val metrics:
            finalized_val_info = val_rmse_calculator.finalize(
                not_per_atom=["positions_gradients"] + per_structure_targets,
                is_distributed=is_distributed,
                device=device,
            )
            if self.hypers["log_mae"]:
                finalized_val_info.update(
                    val_mae_calculator.finalize(
                        not_per_atom=["positions_gradients"] + per_structure_targets,
                        is_distributed=is_distributed,
                        device=device,
                    )
                )

            # The GLE loss is a per-batch mean negative log likelihood, so the epoch
            # figure only means something as a mean over batches (PET's LossAggregator
            # figure is a sum).
            train_loss = train_loss / max(train_batches, 1)
            val_loss = val_loss / max(val_batches, 1)
            # chi2_red calibration diagnostic: target 1.0; >1 covariance too small /
            # under-damped, <1 too large. Per-rank mean, then averaged across ranks.
            train_chi2 = train_chi2 / max(train_batches, 1)
            val_chi2 = val_chi2 / max(val_batches, 1)
            if is_distributed:
                chi2_tensor = torch.tensor([train_chi2, val_chi2], device=device)
                torch.distributed.all_reduce(chi2_tensor)
                train_chi2, val_chi2 = (chi2_tensor / world_size).tolist()

            # Now we log the information:
            if epoch == start_epoch or epoch % self.hypers["log_interval"] == 0:
                finalized_train_info = {
                    "loss": train_loss,
                    "chi2_red": train_chi2,
                    **finalized_train_info,
                }
            finalized_val_info = {
                "loss": val_loss,
                "chi2_red": val_chi2,
                **finalized_val_info,
            }

            if epoch == start_epoch:
                metric_logger = MetricLogger(
                    log_obj=ROOT_LOGGER,
                    dataset_info=(
                        model.module if is_distributed else model
                    ).dataset_info,
                    initial_metrics=[finalized_train_info, finalized_val_info],
                    names=["training", "validation"],
                )
            if epoch % self.hypers["log_interval"] == 0:
                metric_logger.log(
                    metrics=[finalized_train_info, finalized_val_info],
                    epoch=epoch,
                    rank=rank,
                    learning_rate=optimizer.param_groups[0]["lr"],
                )

            val_metric = get_selected_metric(
                finalized_val_info, self.hypers["best_model_metric"]
            )
            if val_metric < self.best_metric:
                self.best_metric = val_metric
                self.best_model_state_dict = copy.deepcopy(
                    (model.module if is_distributed else model).state_dict()
                )
                self.best_epoch = epoch
                self.best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())

            if epoch % self.hypers["checkpoint_interval"] == 0:
                if is_distributed:
                    torch.distributed.barrier()
                self.optimizer_state_dict = optimizer.state_dict()
                self.scheduler_state_dict = lr_scheduler.state_dict()
                self.epoch = epoch
                if rank == 0:
                    self.save_checkpoint(
                        (model.module if is_distributed else model),
                        Path(checkpoint_dir) / f"model_{epoch}.ckpt",
                    )

        # prepare for the checkpoint that will be saved outside the function
        self.epoch = epoch
        self.optimizer_state_dict = optimizer.state_dict()
        self.scheduler_state_dict = lr_scheduler.state_dict()

        if is_distributed:
            torch.distributed.destroy_process_group()

    def save_checkpoint(self, model: ModelInterface, path: Union[str, Path]) -> None:
        checkpoint = model.get_checkpoint()
        if self.best_model_state_dict is not None:
            self.best_model_state_dict["finetune_config"] = model.finetune_config
        checkpoint.update(
            {
                "trainer_ckpt_version": self.__checkpoint_version__,
                "train_hypers": self.hypers,
                "epoch": self.epoch,
                "optimizer_state_dict": self.optimizer_state_dict,
                "scheduler_state_dict": self.scheduler_state_dict,
                "best_epoch": self.best_epoch,
                "best_metric": self.best_metric,
                "best_model_state_dict": self.best_model_state_dict,
                "best_optimizer_state_dict": self.best_optimizer_state_dict,
            }
        )
        torch.save(
            checkpoint,
            check_file_extension(path, ".ckpt"),
        )

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        hypers: TrainerHypers,
        context: Literal["restart", "finetune"],
    ) -> "Trainer":
        trainer = cls(hypers)
        trainer.optimizer_state_dict = checkpoint["optimizer_state_dict"]
        trainer.scheduler_state_dict = checkpoint["scheduler_state_dict"]
        if context == "restart":
            trainer.epoch = checkpoint["epoch"]
        else:
            assert context == "finetune"
            trainer.epoch = None  # interpreted as zero in the training loop
        trainer.best_epoch = checkpoint["best_epoch"]
        trainer.best_metric = checkpoint["best_metric"]
        trainer.best_model_state_dict = checkpoint["best_model_state_dict"]
        trainer.best_optimizer_state_dict = checkpoint["best_optimizer_state_dict"]

        return trainer

    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict) -> Dict:
        for v in range(1, cls.__checkpoint_version__):
            if checkpoint["trainer_ckpt_version"] == v:
                update = getattr(checkpoints, f"trainer_update_v{v}_v{v + 1}")
                update(checkpoint)
                checkpoint["trainer_ckpt_version"] = v + 1

        if checkpoint["trainer_ckpt_version"] != cls.__checkpoint_version__:
            raise RuntimeError(
                f"Unable to upgrade the checkpoint: the checkpoint is using "
                f"trainer version {checkpoint['trainer_ckpt_version']}, while the "
                f"current trainer version is {cls.__checkpoint_version__}."
            )
        return checkpoint
