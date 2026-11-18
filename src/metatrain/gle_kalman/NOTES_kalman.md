# gle_kalman architecture (windowed Kalman GLE training)

Goal: fit A(q) by the EXACT Gaussian likelihood of CG momentum windows under the
extended-Markovian (GLE) model, with persistent hidden aux -> constrains the multi-time
memory (kernel/cage shape), unlike the pairwise transition-density loss in `gle`.

Design: FROZEN-A windowed Kalman. A(q)~const over a short window (~10-15 fs); eval A once
per window-anchor config, run the Kalman recursion over the window's observed momenta.
One network eval per window (cheap); inference (run_gle) unchanged.

## Done
- modules/kalman.py: kalman_gle_nll(A,p_seq,f_seq,dt,sigma2) — windowed Kalman NLL,
  batched over windows/atoms, sequential over L. VALIDATED (test_kalman_core.py):
  recovers known A's momentum-VACF to <0.005 from synthetic OU data. PASS.
- ase.py reader: additively attaches per-atom momenta_seq/forces_seq (N,3L) + info dt_fs
  -> system.get_data("mtt::momenta_seq"/"mtt::forces_seq"/"mtt::dt"). Guarded (no effect
  on gle jobs).
- gle_kalman package imports OK (copy of gle).

## TODO
1. Wire loss: gle_kalman/trainer.py GLELoss.__call__ -> read mtt::momenta_seq/forces_seq
   (reshape N,3L -> N,L,3) + dt; sigma2 = m*kB*T per atom; call kalman_gle_nll. Keep a
   one-step "momenta" target (momenta_after) so RMSE metric plumbing is unchanged.
2. Data build: from continuous CG trajs (gle-data/run{t}), per anchor (every stride):
   window of L consecutive CG momenta -> momenta_seq, PMF forces at each -> forces_seq,
   anchor positions, momenta_after (1-step, for metric), info dt_fs. Save extxyz.
   L*dt should span the cage (~80-150 fs): L~12-15 at dt=10fs.
3. Test tiny train locally (few epochs), verify loss decreases + exports.
4. Simple training on lyra; run via run_gle; compare D + VACF to fixed-gle (2.07e-5).
