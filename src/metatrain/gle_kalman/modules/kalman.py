"""Frozen-A windowed Kalman-filter negative log-likelihood for GLE training.

The extended-Markovian (GLE) state z = [p, s] (physical momentum p in R^3, hidden
auxiliary momenta s in R^n) follows, with A = A(q) ~ constant over a short window:

    z_{k+1} = T z_k + b_k + eta_k,   T = exp(-A dt),  eta_k ~ N(0, Q),  Q = Sigma - T Sigma T^T
    y_k     = H z_k                  (H = [I3 0],  exact observation of p)

with stationary covariance Sigma = m k_B T * I_d (FDT), and conservative input
b_k = [F(q_k) dt ; 0] (the PMF mean force kicks only the physical momentum).

We observe the window of physical momenta y_0..y_{L-1} and evaluate the exact Gaussian
likelihood via the Kalman filter (prediction-error decomposition). Hidden aux are
propagated as a persistent state -> constrains the multi-time memory (kernel shape),
unlike the pairwise transition-density loss.
"""
import torch


def kalman_gle_nll(A, p_seq, f_seq, dt, sigma2, eps=1e-8):
    """Windowed Kalman NLL, batched over windows/atoms (dim B), sequential over time L.

    Args:
        A:      (B, d, d) drift matrices, d = 3 + n.
        p_seq:  (B, L, 3) observed physical momenta over the window.
        f_seq:  (B, L, 3) PMF forces over the window (None -> no conservative input).
        dt:     scalar timestep (same units as 1/A).
        sigma2: (B,) stationary variance m*k_B*T per window/atom.
    Returns:
        scalar NLL summed over all windows and time steps (mean optional by caller).
    """
    B, L, _ = p_seq.shape
    d = A.shape[-1]
    dtype, device = A.dtype, A.device

    # A may be frozen (B,d,d) -> one T for all steps, or per-frame (B,L,d,d) -> T_k.
    per_frame = A.dim() == 4
    if per_frame:
        T_all = torch.matrix_exp(-A * dt)                          # (B,L,d,d)
    else:
        T_all = torch.matrix_exp(-A * dt)                          # (B,d,d)
    eye_d = torch.eye(d, dtype=dtype, device=device)
    Sigma = sigma2[:, None, None] * eye_d                           # (B,d,d)

    def step_TQ(k):
        T = T_all[:, k] if per_frame else T_all
        Q = Sigma - T @ Sigma @ T.transpose(-1, -2)
        return T, Q

    twopi = torch.tensor(2.0 * torch.pi, dtype=dtype, device=device)

    # --- init: prior z ~ N(0, Sigma); condition on y_0 (exact obs of physical block) ---
    y0 = p_seq[:, 0, :]                                             # (B,3)
    S0 = sigma2[:, None, None] * torch.eye(3, dtype=dtype, device=device)  # (B,3,3)
    nll = 0.5 * (torch.logdet(twopi * S0)
                 + (y0.unsqueeze(-2) @ torch.linalg.solve(S0, y0.unsqueeze(-1))).squeeze(-1).squeeze(-1))
    nll = nll.sum()
    # posterior: physical pinned to y0, aux mean 0, cov = sigma2 * diag(0_3, I_n)
    z = torch.zeros(B, d, dtype=dtype, device=device)
    z[:, :3] = y0
    P = Sigma.clone()
    P[:, :3, :] = 0.0
    P[:, :, :3] = 0.0                                               # physical block & cross -> 0

    H_idx = slice(0, 3)
    for k in range(1, L):
        T, Q = step_TQ(k - 1)                                      # frozen or per-frame
        # BAOAB one step: B(half-kick @ q_{k-1}) -> O(T,Q) -> B(half-kick @ q_k).
        # The PMF force lives in the B-steps; the O-step (A) is friction-only -> A learns
        # only the bath. B-steps shift the mean, do not change P (deterministic).
        if f_seq is not None:
            b1 = torch.zeros(B, d, dtype=dtype, device=device)
            b1[:, :3] = f_seq[:, k - 1, :] * (dt * 0.5)
            z = z + b1                                             # B half-kick (start config)
        z = (T @ z.unsqueeze(-1)).squeeze(-1)                      # O drift (mean)
        if f_seq is not None:
            b2 = torch.zeros(B, d, dtype=dtype, device=device)
            b2[:, :3] = f_seq[:, k, :] * (dt * 0.5)
            z = z + b2                                             # B half-kick (end config)
        P = T @ P @ T.transpose(-1, -2) + Q                        # O noise
        # innovation (exact obs of physical block)
        S = P[:, H_idx, H_idx]                                     # (B,3,3)
        S = S + eps * torch.eye(3, dtype=dtype, device=device)
        nu = p_seq[:, k, :] - z[:, :3]                             # (B,3)
        Sinv_nu = torch.linalg.solve(S, nu.unsqueeze(-1))         # (B,3,1)
        nll = nll + 0.5 * (torch.logdet(twopi * S)
                           + (nu.unsqueeze(-2) @ Sinv_nu).squeeze(-1).squeeze(-1)).sum()
        # update (Joseph not strictly needed; standard form)
        PH = P[:, :, H_idx]                                        # (B,d,3)
        K = torch.linalg.solve(S, PH.transpose(-1, -2)).transpose(-1, -2)  # (B,d,3) = PH S^-1
        z = z + (K @ nu.unsqueeze(-1)).squeeze(-1)
        KH = K @ P[:, H_idx, :]                                    # (B,d,d) = K H P
        P = P - KH
        P = 0.5 * (P + P.transpose(-1, -2))                       # symmetrize
    return nll
