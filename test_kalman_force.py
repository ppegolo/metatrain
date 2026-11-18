"""Validate the CONSERVATIVE-FORCE path of the windowed Kalman (untested by the b=0
synthetic test). Generate trajectories with a known harmonic cage force F=-k q, then:
 (1) PLUMBING: generate with the SAME b_k=F*dt the Kalman assumes -> must recover A.
 (2) APPROX BIAS: generate with the EXACT b_k=A^-1(I-T)[F;0] -> infer with F*dt -> measure
     how biased the recovered friction (VACF) is (does F*dt make A too weak?)."""
import torch

from metatrain.gle_kalman.modules.kalman import kalman_gle_nll
from metatrain.gle_kalman.trainer import make_A

torch.manual_seed(1)
dtype = torch.float64
d, n_aux = 6, 3
dt = 2.0 * 0.0982          # ~2 fs in 'ase-ish' units so A*dt is O(0.1)
L, B = 40, 4000
mass = 18.0
kappa = 0.02               # cage stiffness
sigma2 = torch.full((B,), 3.0, dtype=dtype)

theta_true = 0.3 * torch.randn(d * d, dtype=dtype)
A_true = make_A(theta_true.unsqueeze(0), d).squeeze(0)
T_true = torch.matrix_exp(-A_true * dt)
Sig = sigma2[0] * torch.eye(d, dtype=dtype)
Q = Sig - T_true @ Sig @ T_true.T
Lq = torch.linalg.cholesky(0.5 * (Q + Q.T) + 1e-10 * torch.eye(d, dtype=dtype))
Ainv_ImT = torch.linalg.solve(A_true, torch.eye(d, dtype=dtype) - T_true)  # A^-1(I-T)


def simulate(exact_force):
    q = torch.zeros(B, 3, dtype=dtype)
    z = torch.randn(B, d, dtype=dtype) * sigma2[0].sqrt()
    ps, fs = [z[:, :3].clone()], []
    for k in range(1, L):
        F = -kappa * q                                   # cage force at current q
        c = torch.zeros(B, d, dtype=dtype); c[:, :3] = F
        if exact_force:
            drive = (Ainv_ImT @ c.unsqueeze(-1)).squeeze(-1)
        else:
            drive = c * dt
        z = (T_true @ z.unsqueeze(-1)).squeeze(-1) + drive + torch.randn(B, d, dtype=dtype) @ Lq.T
        q = q + z[:, :3] / mass * dt
        ps.append(z[:, :3].clone()); fs.append(F.clone())
    fs.append(-kappa * q)
    return torch.stack(ps, 1), torch.stack(fs, 1)


def vacf(A, ts):
    return torch.stack([torch.matrix_exp(-A * t)[:3, :3].diagonal().mean() for t in ts])


ts = torch.linspace(0, 3.0, 30, dtype=dtype)
vt = vacf(A_true, ts)


def fit(p_seq, f_seq, label):
    th = torch.nn.Parameter(0.3 * torch.randn(d * d, dtype=dtype))
    opt = torch.optim.Adam([th], lr=2e-2)
    for it in range(400):
        opt.zero_grad()
        A = make_A(th.unsqueeze(0), d).squeeze(0)
        loss = kalman_gle_nll(A.unsqueeze(0).expand(B, d, d), p_seq, f_seq, dt, sigma2) / B
        loss.backward(); opt.step()
    A = make_A(th.detach().unsqueeze(0), d).squeeze(0)
    err = (vacf(A, ts) - vt).abs().max().item()
    print(f"{label}: max|VACF_fit - VACF_true| = {err:.4f}  {'PASS' if err<0.05 else 'FAIL/BIAS'}")
    return err


# (1) plumbing: generate + infer both with F*dt -> must recover
p1, f1 = simulate(exact_force=False)
fit(p1, f1, "(1) plumbing  [gen F*dt, infer F*dt]")
# (2) approx bias: generate EXACT, infer F*dt
p2, f2 = simulate(exact_force=True)
fit(p2, f2, "(2) approx bias[gen exact,  infer F*dt]")
# control: generate exact, infer exact would need exact b_k in kalman (not implemented) -> skip
