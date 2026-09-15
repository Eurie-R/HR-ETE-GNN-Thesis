"""Time one full-sample walk-forward per model so the full experiment can be budgeted."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import hetegnn_base as hb

cfg = hb.Config()
print(f"device={cfg.device}  epochs={cfg.epochs}  N={cfg.train_period}  stride={cfg.stride}")

prices, ret, rv, proxy, src, hurst, labels, changes = hb.prepare(cfg)
print(f"panel {prices.shape}  regime changes {len(changes)}")

wf = hb.WalkForward(cfg, ret, rv, changes)
print(f"forecast days per run: {len(wf.X) - cfg.train_period}\n")

total = 0.0
for label, kind, edge in hb.DEFAULT_MODELS:
    for regime in ("hurst", "periodic"):
        t0 = time.time()
        r = wf.run(kind, edge, regime, seed=0)
        dt = time.time() - t0
        total += dt
        m = hb.metrics(r.y_true, r.y_pred)
        print(f"{label:12s} [{regime:8s}] {dt:7.1f}s  refits={r.n_refits:3d}  "
              f"RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} MAPE={m['MAPE']:6.3f} "
              f"corr={m['Correlation']:.4f} hit={m['HitRatio']:.4f}")

print(f"\none seed, all 12 cells: {total/60:.1f} min")
print(f"projected 10 repeats:   {total*10/3600:.1f} h")
