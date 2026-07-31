"""Does including the conservative impulse actually remove the over-damping bias?

This is the experiment behind route (a+). `test_covariant_recovery.py` shows the loss
recovers a drift from UNDRIVEN transitions; the question here is what happens when the
transitions were produced by a drift AND a conservative force, which is the real situation
(the PMF force acts throughout every training window).

The mechanism the test has to expose is not subtle once stated. The likelihood ties the
transition mean and covariance together through one matrix,

    p1 | p0  ~  N( T_pp p0 ,  m kT (I - T_pp T_pp^T) ) ,

so a residual that is larger than ``I - T_pp T_pp^T`` allows can only be explained by
shrinking ``T_pp``. Dropping the impulse leaves exactly such a residual -- the momentum the
force imparted -- in the noise. The fit therefore buys the extra variance by shrinking
``T_pp``, i.e. by inventing friction. The deployed dynamics then applies the force again,
ON TOP of that invented friction, and comes out over-damped.

Note this bias does NOT require the force to correlate with ``p0``: the synthetic force here
is independent of the momentum, and the bias appears anyway, through the variance channel
alone. Real data has the correlation channel as well, so the measurement below is a floor on
the effect, not an estimate of it.

The generating process is the exact driven OU recursion, the same one the loss assumes, so
what is being tested is the ESTIMATOR and not a discretisation error.
"""

import torch

from ..covariant import make_A_covariant, propagated_impulse, propagator_and_integral, theta_size


def _driven_transitions(A, forces, dt, kT, generator):
    """Exact driven OU: ``y_{k+1} = E y_k + G F_k + noise``, started from equilibrium.

    :param forces: ``[n_samples, n_steps, d]`` driving term per sample and sub-interval.
    :return: the OBSERVED momentum blocks ``(p0, p_tau)``.
    """
    d = A.shape[-1]
    eye = torch.eye(d, dtype=A.dtype)
    E, G = propagator_and_integral(A, dt)
    step_cov = kT * (eye - E @ E.T)
    chol = torch.linalg.cholesky(0.5 * (step_cov + step_cov.T) + 1e-12 * eye)

    n_samples, n_steps, _ = forces.shape
    y = torch.randn(n_samples, d, dtype=A.dtype, generator=generator) * kT**0.5
    y0 = y.clone()
    for k in range(n_steps):
        noise = torch.randn(n_samples, d, dtype=A.dtype, generator=generator) @ chol.T
        y = y @ E.T + forces[:, k, :] @ G.T + noise
    return y0[:, :3], y[:, :3]


def _nll(theta, n_aux, forces, dt, kT, p0, p1, with_impulse):
    """The transition NLL, with and without the propagated impulse in the mean."""
    A = make_A_covariant(theta.unsqueeze(0), n_aux)[0]
    n_steps = forces.shape[1]
    Tpp = torch.matrix_exp(-A * (n_steps * dt))[:3, :3]
    mean = p0 @ Tpp.T
    if with_impulse:
        mean = mean + propagated_impulse(A, forces, dt)[:, :3]
    cov = kT * (torch.eye(3, dtype=theta.dtype) - Tpp @ Tpp.T)
    cov = 0.5 * (cov + cov.T) + 1e-9 * torch.eye(3, dtype=theta.dtype)
    chol = torch.linalg.cholesky(cov)
    diff = (p1 - mean).unsqueeze(-1)
    y = torch.linalg.solve_triangular(chol, diff, upper=False).squeeze(-1)
    logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()
    return 0.5 * (y.pow(2).sum(-1) + logdet).mean()


def _fit(n_aux, forces, dt, kT, p0, p1, with_impulse, generator, steps=1200):
    theta = (
        torch.randn(theta_size(n_aux), dtype=torch.float64, generator=generator) * 0.6
    ).requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=0.02)
    for _ in range(steps):
        optimizer.zero_grad()
        _nll(theta, n_aux, forces, dt, kT, p0, p1, with_impulse).backward()
        optimizer.step()
    A_fit = make_A_covariant(theta.detach().unsqueeze(0), n_aux)[0]
    return torch.matrix_exp(-A_fit * (forces.shape[1] * dt))[:3, :3]


def _correlated_force(n_samples, n_steps, d, dt, tau, scale, generator):
    """A smooth force path: an OU process in time, INDEPENDENT of the momentum.

    Correlated in time because a real PMF force is (it is a smooth function of a diffusing
    configuration), independent of ``p`` because the bias under test does not need that
    channel and leaving it out makes the measurement a floor rather than a best case.
    """
    forces = torch.zeros(n_samples, n_steps, d, dtype=torch.float64)
    a = torch.exp(torch.tensor(-dt / tau, dtype=torch.float64))
    f = torch.randn(n_samples, 3, dtype=torch.float64, generator=generator) * scale
    for k in range(n_steps):
        kick = torch.randn(n_samples, 3, dtype=torch.float64, generator=generator)
        f = a * f + (1 - a**2).sqrt() * scale * kick
        forces[:, k, :3] = f  # the conservative force acts on the momentum block only
    return forces


def test_impulse_removes_the_over_damping_bias():
    generator = torch.Generator().manual_seed(1)
    n_aux, dt, kT, n_steps, n_samples = 1, 0.1, 1.0, 8, 30000

    truth = torch.randn(theta_size(n_aux), dtype=torch.float64, generator=generator) * 0.6
    A_true = make_A_covariant(truth.unsqueeze(0), n_aux)[0]
    forces = _correlated_force(n_samples, n_steps, A_true.shape[-1], dt, 0.3, 1.5, generator)
    p0, p1 = _driven_transitions(A_true, forces, dt, kT, generator)
    Tpp_true = torch.matrix_exp(-A_true * (n_steps * dt))[:3, :3]

    Tpp_aware = _fit(n_aux, forces, dt, kT, p0, p1, True, torch.Generator().manual_seed(7))
    Tpp_naive = _fit(n_aux, forces, dt, kT, p0, p1, False, torch.Generator().manual_seed(7))

    err_aware = (Tpp_aware - Tpp_true).abs().max().item()
    err_naive = (Tpp_naive - Tpp_true).abs().max().item()
    # tr(T_pp)/3 is the momentum autocorrelation over the window: SMALLER means the fitted
    # dynamics decorrelates faster, i.e. more friction. That is the over-damping.
    corr_true = (Tpp_true.trace() / 3).item()
    corr_aware = (Tpp_aware.trace() / 3).item()
    corr_naive = (Tpp_naive.trace() / 3).item()

    print(f"  max |Tpp - truth|:  with impulse {err_aware:.4f}   without {err_naive:.4f}")
    print(
        f"  momentum autocorrelation over the window: truth {corr_true:.4f}, "
        f"with impulse {corr_aware:.4f}, without {corr_naive:.4f}"
    )

    assert err_aware < 0.05, f"the impulse-aware fit did not recover T_pp: {err_aware:.4f}"
    assert err_naive > 4 * err_aware, (
        "dropping the impulse was not measurably worse, so this test is not exercising "
        f"the bias it exists for: {err_naive:.4f} against {err_aware:.4f}"
    )
    # the direction matters as much as the size: the bias must be towards MORE friction
    assert corr_naive < corr_true - 0.02, (
        f"the impulse-free fit was not over-damped ({corr_naive:.4f} against a true "
        f"{corr_true:.4f}); a bias in the other direction would mean a different mechanism"
    )


def test_left_padding_a_short_window_is_exact():
    """Batching windows of different lengths relies on this, and the wrong choice is silent.

    The recursion weights each sub-interval by the time REMAINING to the end of the window,
    so zeros PREPENDED contribute nothing and every real step keeps its weight. Zeros
    APPENDED keep propagating past the end of the window and evaluate the impulse for a
    longer lag -- which is the natural way to write it, produces a perfectly smooth result,
    and is wrong. The second assertion is what stops that mistake from passing.
    """
    generator = torch.Generator().manual_seed(3)
    n_aux = 1
    d = 3 * (1 + n_aux)
    theta = torch.randn(theta_size(n_aux), dtype=torch.float64, generator=generator)
    A = make_A_covariant(theta.unsqueeze(0), n_aux)[0]

    dt, n_real, n_pad = 0.15, 5, 4
    forces = torch.randn(2, n_real, d, dtype=torch.float64, generator=generator)
    zeros = torch.zeros(2, n_pad, d, dtype=torch.float64)

    reference = propagated_impulse(A, forces, dt)
    left = propagated_impulse(A, torch.cat([zeros, forces], dim=1), dt)
    right = propagated_impulse(A, torch.cat([forces, zeros], dim=1), dt)

    left_error = (left - reference).abs().max().item()
    right_error = (right - reference).abs().max().item()
    print(f"  left-padded error {left_error:.3e}   right-padded error {right_error:.3e}")

    assert left_error < 1e-12, f"left padding changed the impulse: {left_error:.3e}"
    assert right_error > 1e-3, (
        "right padding did NOT change the impulse, so this test cannot catch the mistake "
        "it exists for -- the propagator is probably too close to the identity here"
    )


if __name__ == "__main__":
    test_impulse_removes_the_over_damping_bias()
    test_left_padding_a_short_window_is_exact()
    print("PASS")
