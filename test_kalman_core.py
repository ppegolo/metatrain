"""Validate the windowed-Kalman NLL: recover a known A's momentum-VACF from synthetic
OU trajectories by minimizing the Kalman loss. The latent aux are non-unique, so we
compare the OBSERVABLE physical-momentum VACF [exp(-A t)]_pp, not A itself."""
import torch

from metatrain.gle_kalman.modules.kalman import kalman_gle_nll
from metatrain.gle_kalman.trainer import make_A

torch.manual_seed(0)
dtype = torch.float64
d, n_aux = 6, 3          # 3 physical + 3 aux
dt = 0.1
L, B = 40, 4000          # window length, #windows

# --- ground-truth A (stable: 0.5 LL^T + K) ---
theta_true = 0.3 * torch.randn(d * d, dtype=dtype)
A_true = make_A(theta_true.unsqueeze(0), d).squeeze(0)            # (d,d)


def simulate(A, L, B):
    """Generate B OU windows (free, b=0), observe physical momenta. Sigma = I_d."""
    T = torch.matrix_exp(-A * dt)
    Sigma = torch.eye(d, dtype=dtype)
    Q = Sigma - T @ Sigma @ T.T
    Lq = torch.linalg.cholesky(0.5 * (Q + Q.T) + 1e-10 * torch.eye(d, dtype=dtype))
    z = torch.randn(B, d, dtype=dtype)                            # z0 ~ N(0, I)
    ps = [z[:, :3]]
    for _ in range(1, L):
        z = z @ T.T + torch.randn(B, d, dtype=dtype) @ Lq.T
        ps.append(z[:, :3])
    return torch.stack(ps, dim=1)                                 # (B,L,3)


p_seq = simulate(A_true, L, B)
sigma2 = torch.ones(B, dtype=dtype)


def vacf_pp(A, ts):
    """Physical-momentum VACF: mean diagonal of [exp(-A t)]_{:3,:3}."""
    out = []
    for t in ts:
        Tt = torch.matrix_exp(-A * t)
        out.append(Tt[:3, :3].diagonal().mean())
    return torch.stack(out)


ts = torch.linspace(0.0, 3.0, 30, dtype=dtype)
vacf_true = vacf_pp(A_true, ts)

# --- baselines: NLL at truth vs random ---
nll_true = kalman_gle_nll(A_true.unsqueeze(0).expand(B, d, d), p_seq, None, dt, sigma2).item()
theta_rand = 0.3 * torch.randn(d * d, dtype=dtype)
A_rand = make_A(theta_rand.unsqueeze(0), d).squeeze(0)
nll_rand = kalman_gle_nll(A_rand.unsqueeze(0).expand(B, d, d), p_seq, None, dt, sigma2).item()
print(f"NLL(A_true)={nll_true/B:.4f}  NLL(random)={nll_rand/B:.4f}  (per window)")

# --- fit a free A by minimizing the Kalman NLL ---
theta = torch.nn.Parameter(0.3 * torch.randn(d * d, dtype=dtype))
opt = torch.optim.Adam([theta], lr=2e-2)
for it in range(400):
    opt.zero_grad()
    A = make_A(theta.unsqueeze(0), d).squeeze(0)
    loss = kalman_gle_nll(A.unsqueeze(0).expand(B, d, d), p_seq, None, dt, sigma2) / B
    loss.backward()
    opt.step()
    if it % 100 == 0 or it == 399:
        with torch.no_grad():
            verr = (vacf_pp(make_A(theta.unsqueeze(0), d).squeeze(0), ts) - vacf_true).abs().max().item()
        print(f"it {it:3d}  NLL/win {loss.item():.4f}  max|VACF_fit-VACF_true| {verr:.4f}")

A_fit = make_A(theta.detach().unsqueeze(0), d).squeeze(0)
vacf_fit = vacf_pp(A_fit, ts)
err = (vacf_fit - vacf_true).abs().max().item()
print(f"\nRESULT: NLL(fit)/win={loss.item():.4f} (truth {nll_true/B:.4f}); "
      f"max VACF error={err:.4f}")
print("PASS" if err < 0.05 and loss.item() < nll_rand / B else "FAIL")
