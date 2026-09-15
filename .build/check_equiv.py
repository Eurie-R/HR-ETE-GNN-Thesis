"""Prove the batched walk-forward equals the old one-day-at-a-time loop.

The refactor only changes *when* forward passes are grouped, never what the
model sees, so predictions must agree to floating-point noise. This replays the
original loop against the current implementation on synthetic data.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import hetegnn_base as hb


def reference_run(wf, model_name, edge_kind, regime_mode, seed):
    """The pre-refactor loop: decide, then predict a single day, every step."""
    cfg = wf.cfg
    hb.set_seed(seed)
    N = cfg.train_period
    model, A_t, lo, hi = None, None, None, None
    n_refits, last_full = 0, -10 ** 9
    preds, actuals, dates = [], [], []

    for e in range(N, len(wf.X), cfg.stride):
        tgt = wf.target_dates[e]
        need_full = model is None or (regime_mode == "periodic" and e - last_full >= N)
        need_ft = regime_mode == "hurst" and model is not None and tgt in wf.change_dates
        if need_full or need_ft:
            Xtr_raw, ytr = wf.X[e - N:e], wf.y[e - N:e]
            lo, hi = hb.minmax_fit(Xtr_raw)
            Xtr = hb.minmax_apply(Xtr_raw, lo, hi)
            if edge_kind is not None:
                win_end = wf.end_idx[e - 1] + 1
                win = wf.ret.values[max(0, win_end - N):win_end]
                A = hb.EDGE_BUILDERS[edge_kind](win, cfg, seed=seed)
                if cfg.normalize_adjacency:
                    A = hb.row_normalize(A)
                A_t = torch.tensor(A, dtype=torch.float32, device=torch.device(cfg.device))
            if need_full:
                model = hb.train(hb.build_model(model_name, cfg), A_t, Xtr, ytr, cfg)
                last_full = e
            else:
                model = hb.train(model, A_t, Xtr, ytr, cfg, epochs=cfg.finetune_epochs)
            n_refits += 1
        preds.append(hb.predict(model, A_t, hb.minmax_apply(wf.X[e:e + 1], lo, hi), cfg)[0])
        actuals.append(wf.y[e])
        dates.append(tgt)
    return np.asarray(preds), np.asarray(actuals), pd.DatetimeIndex(dates), n_refits


rng = np.random.default_rng(0)
T, N_NODES = 1500, 5
idx = pd.bdate_range("2010-01-04", periods=T)
r = rng.standard_normal((T, N_NODES)) * 0.01
for j in range(1, N_NODES):
    r[1:, j] += 0.4 * r[:-1, 0]
cols = tuple(f"N{j}" for j in range(N_NODES))
ret = pd.DataFrame(r, index=idx, columns=list(cols))
rv = hb.realized_volatility(ret, 20)
proxy = pd.Series(rng.standard_normal(T) * 0.01, index=idx)

cfg = hb.Config(tickers=cols, train_period=250, epochs=6, n_shuffles=8,
                patience=3, finetune_epochs=3, stride=1, device="cpu")
hurst = hb.rolling_hurst(proxy, 120, cfg.hurst_correction)
changes = hb.regime_change_dates(
    hb.regime_labels(hurst, cfg.regime_lookback, cfg.regime_sensitivity))

fails = []
for model_name, edge in (("GNN", "ETE"), ("GNN", "Pearson"), ("LSTM", None)):
    for regime in ("hurst", "periodic"):
        wf = hb.WalkForward(cfg, ret, rv, changes)
        t0 = time.time(); ref_p, ref_y, ref_d, ref_n = reference_run(
            wf, model_name, edge, regime, seed=0); t_ref = time.time() - t0

        wf2 = hb.WalkForward(cfg, ret, rv, changes)
        t0 = time.time(); got = wf2.run(model_name, edge, regime, seed=0)
        t_new = time.time() - t0

        same_n = ref_n == got.n_refits
        same_d = ref_d.equals(got.dates)
        same_y = np.array_equal(ref_y, got.y_true)
        dmax = float(np.abs(ref_p - got.y_pred).max())
        ok = same_n and same_d and same_y and dmax < 1e-5
        if not ok:
            fails.append((model_name, edge, regime))
        label = f"{edge or model_name}-{regime}"
        print(f"{'PASS' if ok else 'FAIL'}  {label:20s} "
              f"refits {ref_n}=={got.n_refits}  dates={same_d}  targets={same_y}  "
              f"max|dpred|={dmax:.2e}   {t_ref:.1f}s -> {t_new:.1f}s "
              f"({t_ref / max(t_new, 1e-9):.1f}x)")

print("\n" + ("EQUIVALENT" if not fails else f"MISMATCH: {fails}"))

# --- stride must subsample the evaluation, never the retraining schedule -----
# The property that matters is coverage: every regime change in the evaluated
# span must be followed by a refit at the first evaluated day at or after it.
# Two changes inside one strided span legitimately collapse into one refit, so
# the raw count may fall -- what must never happen is a change being ignored.
print("\nregime-change coverage vs stride")
stride_fail = False
for s in (1, 2, 5, 10, 20):
    c = hb.Config(**{**cfg.__dict__, "stride": s})
    wf = hb.WalkForward(c, ret, rv, changes)
    idx = np.arange(c.train_period, len(wf.X), s)
    sched = wf.refit_schedule(idx, "hurst")
    refit_at = {int(idx[pos]) for pos, _ in sched}

    change_e = [e for e in range(c.train_period, len(wf.X)) if wf.is_change[e]]
    uncovered = []
    for e in change_e:
        nxt = idx[idx >= e]
        if len(nxt) and int(nxt[0]) not in refit_at:
            uncovered.append(e)

    # what the old exact-date rule would have produced, for contrast
    naive = 1 + sum(1 for e in idx[1:] if wf.is_change[int(e)])

    ok = not uncovered
    stride_fail |= not ok
    print(f"  {'PASS' if ok else 'FAIL'}  stride={s:<3d} forecasts={len(idx):5d}  "
          f"changes={len(change_e):3d}  refits={len(sched):3d}  "
          f"uncovered={len(uncovered):3d}   (exact-date rule would give "
          f"{naive:3d})")

sys.exit(1 if (fails or stride_fail) else 0)
