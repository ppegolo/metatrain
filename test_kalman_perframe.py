"""Validate the PER-FRAME Kalman loss: when A varies over the window (as real A(q(t))
does), a single frozen A cannot fit the data but a per-frame A_seq can. Demonstrates
why per-frame is the fix and that the per-frame recursion + gradients are correct."""
import torch

from metatrain.gle_kalman.modules.kalman import kalman_gle_nll
from metatrain.gle_kalman.trainer import make_A

torch.manual_seed(2)
dtype = torch.float64
d = 6
dt = 0.15
L, B = 30, 4000
sig2 = torch.full((B,), 2.0, dtype=dtype)

# varying ground-truth A over the window: interpolate between two stable A's
th0 = 0.3 * torch.randn(d * d, dtype=dtype)
th1 = 0.3 * torch.randn(d * d, dtype=dtype)
A_seq_true = torch.stack([make_A((th0 + (k / (L - 1)) * (th1 - th0)).unsqueeze(0), d).squeeze(0)
                          for k in range(L)])                      # (L,d,d)
T_true = torch.matrix_exp(-A_seq_true * dt)                        # (L,d,d)
Sig = sig2[0] * torch.eye(d, dtype=dtype)


def simulate():
    z = torch.randn(B, d, dtype=dtype) * sig2[0].sqrt()
    ps = [z[:, :3].clone()]
    for k in range(1, L):
        Tk = T_true[k - 1]
        Q = Sig - Tk @ Sig @ Tk.T
        Lq = torch.linalg.cholesky(0.5 * (Q + Q.T) + 1e-9 * torch.eye(d, dtype=dtype))
        z = z @ Tk.T + torch.randn(B, d, dtype=dtype) @ Lq.T
        ps.append(z[:, :3].clone())
    return torch.stack(ps, 1)


p_seq = simulate()
nll_true = kalman_gle_nll(A_seq_true.unsqueeze(0).expand(B, L, d, d), p_seq, None, dt, sig2).item() / B
print(f"NLL(true per-frame A_seq) = {nll_true:.4f}  (target)")


def fit(per_frame, label, iters=500):
    if per_frame:
        th = torch.nn.Parameter(0.3 * torch.randn(L, d * d, dtype=dtype))
    else:
        th = torch.nn.Parameter(0.3 * torch.randn(d * d, dtype=dtype))
    opt = torch.optim.Adam([th], lr=2e-2)
    for it in range(iters):
        opt.zero_grad()
        if per_frame:
            A = make_A(th, d).unsqueeze(0).expand(B, L, d, d)
        else:
            A = make_A(th.unsqueeze(0), d).squeeze(0).unsqueeze(0).expand(B, d, d)
        loss = kalman_gle_nll(A, p_seq, None, dt, sig2) / B
        loss.backward(); opt.step()
    print(f"{label}: final NLL/win = {loss.item():.4f}")
    return loss.item()


nll_pf = fit(True, "per-frame fit  ")
nll_fr = fit(False, "frozen fit     ")
gap = nll_fr - nll_pf
print(f"\nfrozen - perframe NLL gap = {gap:.4f}  "
      f"(per-frame should reach ~true {nll_true:.3f}; frozen stuck higher)")
print("PASS" if (abs(nll_pf - nll_true) < 0.05 and gap > 0.05) else "CHECK")
