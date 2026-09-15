"""Download the real panel once and report how the under-specified choices land.

Answers the two questions that decide whether the replication is on track:
  * does the ETF panel cover the paper's 2003-02-07 .. 2024-05-29 window?
  * which Hurst estimator gets the regime-change count near the paper's 47?
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import hetegnn_base as hb

cfg = hb.Config()
t0 = time.time()
prices, proxy_ret, proxy_src = hb.download_prices(cfg)
print(f"download: {time.time() - t0:.1f}s")
print(f"prices  {prices.shape}  {prices.index[0].date()} -> {prices.index[-1].date()}"
      f"   (paper: 2003-02-07 -> 2024-05-29)")

print("\nfirst trade date per ETF:")
for t in cfg.tickers:
    s = prices[t].dropna()
    print(f"  {t}  {s.index[0].date()}")

print("\nregime-proxy splice:")
for tkr, n in proxy_src.value_counts().items():
    span = proxy_src[proxy_src == tkr].index
    print(f"  {tkr:5s} {n:5d} days  {span.min().date()} -> {span.max().date()}")

ret = hb.log_returns(prices)
rv = hb.realized_volatility(ret, cfg.rv_window)
print(f"\nlog-returns {ret.shape}   RV {rv.shape}")
print("\nRV descriptive stats (paper Table 5, bottom row per ETF):")
print(pd.DataFrame({"mean": rv.mean(), "max": rv.max(), "min": rv.min(),
                    "std": rv.std(), "skew": rv.skew(),
                    "kurt": rv.kurtosis()}).round(4).to_string())

print("\nHurst estimator vs the paper's 47 regime changes:")
for corr in ("none", "anis_lloyd"):
    t0 = time.time()
    h = hb.rolling_hurst(proxy_ret, cfg.hurst_window, corr)
    lab = hb.regime_labels(h, cfg.regime_lookback, cfg.regime_sensitivity,
                           cfg.hurst_threshold)
    ch = hb.regime_change_dates(lab)
    counts = lab.value_counts().to_dict()
    print(f"  {corr:11s} mean H={h.mean():.4f}  frac>0.5={(h > 0.5).mean():6.1%}  "
          f"changes={len(ch):3d}  labels={counts}  ({time.time() - t0:.0f}s)")

print("\nedge matrices on the first N=750 block:")
win = ret.values[:cfg.train_period]
for kind in ("ETE", "TE", "Granger", "Pearson"):
    t0 = time.time()
    A = hb.EDGE_BUILDERS[kind](win, cfg, seed=0)
    nz = A[A > 0]
    print(f"  {kind:8s} {int((A > 0).sum()):3d}/90 edges  "
          f"mean weight={nz.mean() if nz.size else 0:.4f}  ({time.time() - t0:.2f}s)")
