"""Does the covariant GLE loss actually RECOVER a drift it was trained on?

Everything else on this branch tests construction and algebra: covariance,
positive-definiteness, irrep round trips, FDT identities. None of it shows that the loss
identifies anything. This does the one experiment that can fail for a real reason -- generate
momentum transitions from a KNOWN covariant drift, fit a fresh drift to them, and check the
observable dynamics comes back.

**What is identifiable, and what is not.** From single-lag ``(p0, p1)`` pairs the auxiliaries
are never observed, so the likelihood constrains only the momentum block of the propagator,
``T_pp = [exp(-A dt)]_{:3,:3}`` -- not ``A`` itself. Different drifts share a ``T_pp``: the
trainer's own comments record that "the single-lag transition NLL cannot distinguish
dissipative friction from energy-conserving antisymmetric rotation into the auxiliaries",
which is exactly why the production loss carries a gamma0 pin. So the success criterion here
is recovery of ``T_pp``, and claiming recovery of ``A`` would be claiming something the data
cannot support.

The transition law being fitted is the exact marginal over the unobserved auxiliaries at
equilibrium,

    p1 | p0  ~  N( T_pp p0 ,  m kT (I - T_pp T_pp^T) ) ,

which is what `GLELoss` uses.
"""

import torch

from ..covariant import make_A_covariant, theta_size


def _sample_transitions(A, dt, kT, n_samples, generator):
    """Exact OU sampling from equilibrium; returns the OBSERVED (p0, p1) only."""
    d = A.shape[-1]
    eye = torch.eye(d, dtype=A.dtype)
    T = torch.matrix_exp(-A * dt)
    sigma = kT * (eye - T @ T.T)
    sigma = 0.5 * (sigma + sigma.T)
    # equilibrium start, then one exact step
    y0 = torch.randn(n_samples, d, dtype=A.dtype, generator=generator) * kT**0.5
    chol = torch.linalg.cholesky(sigma + 1e-12 * eye)
    noise = torch.randn(n_samples, d, dtype=A.dtype, generator=generator) @ chol.T
    y1 = y0 @ T.T + noise
    return y0[:, :3], y1[:, :3]


def _nll(theta, n_aux, dt, kT, p0, p1):
    """The transition NLL, matching `GLELoss`'s per-bead branch."""
    A = make_A_covariant(theta.unsqueeze(0), n_aux)[0]
    Tpp = torch.matrix_exp(-A * dt)[:3, :3]
    mean = p0 @ Tpp.T
    cov = kT * (torch.eye(3, dtype=theta.dtype) - Tpp @ Tpp.T)
    cov = 0.5 * (cov + cov.T) + 1e-9 * torch.eye(3, dtype=theta.dtype)
    chol = torch.linalg.cholesky(cov)
    diff = (p1 - mean).unsqueeze(-1)
    y = torch.linalg.solve_triangular(chol, diff, upper=False).squeeze(-1)
    logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()
    return 0.5 * (y.pow(2).sum(-1) + logdet).mean()


def test_covariant_loss_recovers_the_momentum_propagator():
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(1)
    n_aux, dt, kT, n_samples = 1, 0.4, 1.0, 40000

    truth = torch.randn(theta_size(n_aux), dtype=torch.float64, generator=generator) * 0.6
    A_true = make_A_covariant(truth.unsqueeze(0), n_aux)[0]
    p0, p1 = _sample_transitions(A_true, dt, kT, n_samples, generator)
    Tpp_true = torch.matrix_exp(-A_true * dt)[:3, :3]

    theta = (
        torch.randn(theta_size(n_aux), dtype=torch.float64, generator=generator) * 0.6
    ).requires_grad_(True)
    before = _nll(theta, n_aux, dt, kT, p0, p1).item()
    optimizer = torch.optim.Adam([theta], lr=0.02)
    for _ in range(1500):
        optimizer.zero_grad()
        loss = _nll(theta, n_aux, dt, kT, p0, p1)
        loss.backward()
        optimizer.step()
    after = _nll(theta, n_aux, dt, kT, p0, p1).item()

    A_fit = make_A_covariant(theta.detach().unsqueeze(0), n_aux)[0]
    Tpp_fit = torch.matrix_exp(-A_fit * dt)[:3, :3]
    # the loss at the TRUE parameters: the fit cannot do better than this on average, so it
    # is the meaningful reference rather than zero
    reference = _nll(truth, n_aux, dt, kT, p0, p1).item()

    error = (Tpp_fit - Tpp_true).abs().max().item()
    scale = Tpp_true.abs().max().item()
    print(f"  NLL  before {before:.5f} -> after {after:.5f}   (at truth {reference:.5f})")
    print(f"  max |Tpp_fit - Tpp_true| = {error:.4f}   (scale {scale:.4f})")
    print(f"  symmetric part of A_fit positive definite: "
          f"{torch.linalg.eigvalsh(0.5 * (A_fit + A_fit.T)).min():.3e} > 0")

    assert after < before, "the loss did not decrease"
    assert after <= reference + 0.01, "the fit did not reach the likelihood at the truth"
    assert error < 0.05 * max(scale, 1.0), f"T_pp not recovered: {error:.4f}"


if __name__ == "__main__":
    test_covariant_loss_recovers_the_momentum_propagator()
    print("PASS")
