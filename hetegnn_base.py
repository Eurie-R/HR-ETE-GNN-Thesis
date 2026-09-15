"""
Faithful replication of the BASE model from

    Lee, S. & Cho, P. (2025). "Graph-Based Stock Volatility Forecasting with
    Effective Transfer Entropy and Hurst-Based Regime Adaptation."
    Fractal Fract. 9, 339.  https://doi.org/10.3390/fractalfract9060339

The paper calls the proposed model **H-ETE-GNN** (Hurst Exponent + Effective
Transfer Entropy + Graph Neural Network).  This module reproduces it, together
with every benchmark the paper reports, so that the Renyi extension
(HR-ETE-GNN) can later be compared against a like-for-like baseline.

What the paper specifies, and where it is implemented
-----------------------------------------------------
  Sec 2.1  Discretized TE (3 bins, k=l=1)          -> transfer_entropy()
           Effective TE = TE - mean(shuffled TE)   -> effective_te()   Eq (4)
           Z = (TE - mu_shuf)/sd_shuf, keep Z>1.96 -> effective_te()   Eq (5)
  Sec 2.1  Granger (binarised at 5%), |Pearson|    -> build_granger/pearson_matrix
  Sec 2.2  Multi-scale conv, kernels {3,5,7} x 12  -> MultiScaleConv   Eq (8)
           3-layer message passing                 -> GCNLayer/ETEGNN  Eq (9)
  Sec 2.3  R/S Hurst exponent                      -> hurst_rs()       Eq (10-16)
  Sec 3.1  250-day rolling Hurst on a World proxy  -> rolling_hurst()
           Regime = >=s of last 10 days above/below 0.5 -> regime_labels()
           Walk-forward, refit edges + fine-tune on regime change
                                                   -> WalkForward.run()
  Sec 3.2  10 country ETFs, 2003-02-07..2024-05-29, RV over 20 days  -> Config
  Sec 4    RMSE/MAE/MAPE/Correlation/Hit ratio     -> metrics()        Eq (19-23)

Tuned hyper-parameters the paper reports (Table 3): M=20, N=750, s=6.
Learning rate, batch size, hidden sizes and the shuffle count m were tuned by
Bayesian optimisation but their selected values are NOT reported; the defaults
below sit in the middle of the published search spaces (Table 1) and are
exposed on Config so they can be re-tuned.

Known deviations from the paper are collected in DEVIATIONS at the bottom of
this file.  Read them before quoting any replicated number.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-12

# Forecasts between two refits share a model, so they are produced in batches of
# this size rather than one day at a time.  Purely a speed knob - the model sees
# each sample independently either way, so results are unchanged.
_PREDICT_CHUNK = 512


# =============================================================================
# Configuration
# =============================================================================

# Table 4: the ten country ETFs whose realized volatility is forecast.
TICKERS: Tuple[str, ...] = (
    "EWA",  # Australia
    "EWC",  # Canada
    "EWG",  # Germany
    "EWJ",  # Japan
    "EWT",  # Taiwan
    "EWU",  # United Kingdom
    "EWW",  # Mexico
    "EWY",  # South Korea
    "EWZ",  # Brazil
    "EZA",  # South Africa
)

# The paper uses "iShares MSCI World ETF" as the regime proxy over 2003-2024,
# but URTH only began trading in 2012.  We splice, newest-first: URTH where it
# exists, ACWI before that, IOO before that.  See DEVIATIONS[1].
REGIME_PROXY_CHAIN: Tuple[str, ...] = ("URTH", "ACWI", "IOO")


@dataclass
class Config:
    # ---- data (Sec 3.2) ----
    tickers: Tuple[str, ...] = TICKERS
    proxy_chain: Tuple[str, ...] = REGIME_PROXY_CHAIN
    start: str = "2003-02-07"
    end: str = "2024-05-30"         # yfinance end is exclusive; paper ends 05-29
    rv_window: int = 20              # Eq (18): RV over the past 20 log-returns

    # ---- tuned in the paper (Table 3) ----
    input_size: int = 20             # M
    train_period: int = 750          # N
    regime_sensitivity: int = 6      # s

    # ---- fixed by the paper's text (Sec 2.1 / 3.1) ----
    n_bins: int = 3                  # ternary discretization for TE
    te_lag: int = 1                  # k = l = 1
    z_threshold: float = 1.96        # keep only Z > 1.96 edges
    hurst_window: int = 250          # rolling window for H
    regime_lookback: int = 10        # "during the past 10 days"
    hurst_threshold: float = 0.5
    hurst_correction: str = "none"   # "none" (literal paper) | "anis_lloyd"
    conv_kernels: Tuple[int, ...] = (3, 5, 7)
    conv_channels: int = 12
    epochs: int = 100
    val_fraction: float = 0.3        # training:validation = 7:3

    # ---- Bayesian-optimised in the paper, values unreported (Table 1) ----
    lr: float = 1e-3
    batch_size: int = 8
    hidden1: int = 64
    hidden2: int = 32
    n_shuffles: int = 50             # m, search space [10, 100]

    # ---- replication knobs (see DEVIATIONS) ----
    conv_readout: str = "flatten"    # "flatten" (literal Eq 8) | "avgpool"
    # True by default: on raw weights the ETE graph contributes 0.7% of the self
    # term in Eq (9) and ETE-GNN collapses into a per-node MLP, which makes the
    # four edge types a comparison of weight scale rather than topology.  Set
    # False (CLI: --raw-adjacency) for the literal paper.  See DEVIATIONS[4].
    normalize_adjacency: bool = True
    final_layer_relu: bool = False
    # Epochs without validation improvement before stopping.  None = run the
    # full budget.  train() always restores the best-validation weights, so
    # stopping early is purely a compute saving and never protects against
    # overfitting - while setting it too low silently biases model comparison.
    # Measured on the real panel, the longest run of non-improving epochs before
    # the eventual best was 50 for LSTM, 19 for GRU and 17 for the GNN, so the
    # old default of 10 truncated all three: it left both RNNs worse than
    # predicting the training mean on held-out data (0.374 / 0.378 vs a 0.355
    # baseline) while barely touching the GNN, which would have inflated the
    # paper's headline GNN-vs-RNN gap.  See .build/check_rnn_earlystop.py.
    patience: Optional[int] = None
    finetune_epochs: int = 30        # regime-change fine-tune budget
    rnn_hidden: int = 64
    rnn_layers: int = 2
    rnn_dropout: float = 0.1

    # ---- experiment control ----
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    n_repeats: int = 10              # "All experiments were repeated ten times"
    seed: int = 0
    # >1 forecasts every k-th day, for fast rehearsals.  It subsamples only the
    # evaluation: every regime change still triggers its refit, so the
    # Hurst-vs-Periodic contrast is not an artefact of the stride.
    stride: int = 1
    # The net is small (576->64->32->1) and batched at 8, so torch's default of
    # one thread per core loses far more to synchronisation than it gains: on a
    # 12-core box a full fit takes 34 s at 12 threads and 9.6 s at 2.  None
    # leaves torch alone.
    torch_threads: Optional[int] = 4

    def n_nodes(self) -> int:
        return len(self.tickers)

    def apply_runtime(self) -> "Config":
        """Apply process-wide settings this config implies.  Safe to call twice."""
        if self.torch_threads and self.device == "cpu":
            torch.set_num_threads(min(self.torch_threads, os.cpu_count() or 1))
        return self


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# 1. Data  (Sec 3.2, Eq 17-18)
# =============================================================================

def _download_close(tickers: Sequence[str], start: str, end: str,
                    attempts: int = 3) -> pd.DataFrame:
    """Adjusted close for `tickers`, retrying tickers that come back empty.

    yfinance fails per-ticker rather than raising: a rate limit or a lock on its
    sqlite timezone cache yields an all-NaN column, which then silently empties
    the panel at the first dropna().  So download sequentially (threads=False,
    which is what trips the cache lock) and retry only what is still missing.
    """
    import yfinance as yf

    frames: Dict[str, pd.Series] = {}
    todo = list(tickers)
    for attempt in range(attempts):
        if not todo:
            break
        if attempt:
            time.sleep(2 * attempt)
        raw = yf.download(todo, start=start, end=end, auto_adjust=True,
                          progress=False, threads=False)
        if raw is None or raw.empty:
            continue
        close = raw["Close"]
        if isinstance(close, pd.Series):                # single ticker
            close = close.to_frame(todo[0])
        for t in list(todo):
            if t in close.columns and close[t].notna().sum() > 0:
                frames[t] = close[t].dropna()
                todo.remove(t)

    if not frames:
        raise RuntimeError(f"yfinance returned nothing for {list(tickers)}")
    return pd.DataFrame(frames)


def download_prices(cfg: Config) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Adjusted close for the ten ETFs plus a spliced World-ETF proxy.

    Returns (prices, proxy_log_returns, proxy_source).  The proxy is returned as
    log-returns rather than a price level because the splice across
    URTH/ACWI/IOO is only meaningful in return space; proxy_source names, for
    each date, which ticker supplied the return, so the splice is auditable.
    """
    raw = _download_close(list(cfg.tickers) + list(cfg.proxy_chain),
                          cfg.start, cfg.end)

    missing = [t for t in cfg.tickers
               if t not in raw.columns or raw[t].notna().sum() == 0]
    if missing:
        raise RuntimeError(
            f"no price data for {missing}. yfinance returns empty columns on "
            f"transient errors (rate limits, its sqlite cache locking); "
            f"re-run, and if it persists check the tickers are still listed.")

    prices = raw[list(cfg.tickers)].dropna()
    if prices.empty:
        raise RuntimeError("no dates where all ten ETFs trade simultaneously")

    proxy_ret, source = None, None
    for tkr in cfg.proxy_chain:                     # most preferred first
        if tkr not in raw.columns:
            continue
        r = np.log(raw[tkr] / raw[tkr].shift(1)).dropna()
        s = pd.Series(tkr, index=r.index)
        if proxy_ret is None:
            proxy_ret, source = r, s
        else:
            proxy_ret = proxy_ret.combine_first(r)
            source = source.combine_first(s)
    if proxy_ret is None:
        raise RuntimeError(f"none of {cfg.proxy_chain} downloaded")

    proxy_ret = proxy_ret.reindex(prices.index).dropna()
    source = source.reindex(proxy_ret.index)
    return prices, proxy_ret, source


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Eq (17):  r_t = ln(p_t / p_{t-1})."""
    return np.log(prices / prices.shift(1)).dropna()


def realized_volatility(log_ret: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Eq (18):  RV_t = sqrt( (1/n) * sum_{i=1..n} (100 * r_{t-i})^2 ).

    Implemented as a trailing n-day window ending at t, so RV_t is observable
    at the close of day t and RV_{t+1} is a genuine one-day-ahead target.
    """
    return (100.0 * log_ret).pow(2).rolling(n).mean().pow(0.5).dropna()


# =============================================================================
# 2. Edge matrices
# =============================================================================
# Convention used everywhere in this module:
#     M[i, j] == strength of the directed link  source i  ->  target j.
# Eq (9) aggregates A_ji h_j into node i, i.e. sums over sources j, which is
# exactly M.T @ H.  See GCNLayer.

def discretize(x: np.ndarray, n_bins: int = 3) -> np.ndarray:
    """Equal-width symbolisation over [min, max] (Sec 2.1).

    With n_bins=3 the symbols read as {significant loss, neutral, significant
    gain}, the ternary partition of Marschinski & Kantz that the paper adopts.
    """
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or hi <= lo:
        return np.zeros(x.shape, dtype=np.int64)
    edges = np.linspace(lo, hi, n_bins + 1)[1:-1]
    return np.digitize(x, edges).astype(np.int64)


def _te_from_symbols(sy: np.ndarray, sx: np.ndarray, n_bins: int,
                     lag: int = 1) -> float:
    """Eq (3):  TE_{Y->X} with k = l = lag, from pre-symbolised series.

    TE = sum p(x_{t+1}, x_t, y_t) * log[ p(x_{t+1}|x_t,y_t) / p(x_{t+1}|x_t) ]
    """
    xf = sx[lag + 1:]                     # x_{t+1}
    n = xf.size
    if n < 10:
        return np.nan
    xp = sx[lag:-1][:n]                   # x_t
    yp = sy[lag:-1][:n]                   # y_t

    b = n_bins
    counts = np.bincount(xf * b * b + xp * b + yp, minlength=b ** 3)
    p = counts.reshape(b, b, b).astype(float) / n          # p[x_{t+1}, x_t, y_t]

    p_xp_yp = p.sum(axis=0)                                # p(x_t, y_t)
    p_xf_xp = p.sum(axis=2)                                # p(x_{t+1}, x_t)
    p_xp = p.sum(axis=(0, 2))                              # p(x_t)

    num = p / (p_xp_yp[None, :, :] + _EPS)                 # p(x_{t+1}|x_t,y_t)
    den = p_xf_xp / (p_xp[None, :] + _EPS)                 # p(x_{t+1}|x_t)
    ratio = num / (den[:, :, None] + _EPS)

    mask = p > 0
    return float(np.sum(p[mask] * np.log(ratio[mask] + _EPS)))


def transfer_entropy(y: np.ndarray, x: np.ndarray, n_bins: int = 3,
                     lag: int = 1) -> float:
    """Discretized transfer entropy from Y to X, in nats."""
    return _te_from_symbols(discretize(y, n_bins), discretize(x, n_bins),
                            n_bins, lag)


def effective_te(y: np.ndarray, x: np.ndarray, n_bins: int = 3, lag: int = 1,
                 m_shuffles: int = 50, rng: Optional[np.random.Generator] = None
                 ) -> Tuple[float, float]:
    """Eq (4) and Eq (5).

    Shuffling the *symbols* of Y is identical to shuffling Y and re-binning,
    because a permutation leaves the marginal - and therefore the bin edges -
    unchanged.  Returns (ETE, Z).
    """
    rng = rng or np.random.default_rng(0)
    sy = discretize(y, n_bins)
    sx = discretize(x, n_bins)

    te = _te_from_symbols(sy, sx, n_bins, lag)
    if not np.isfinite(te):
        return np.nan, np.nan

    surr = np.array([_te_from_symbols(rng.permutation(sy), sx, n_bins, lag)
                     for _ in range(m_shuffles)])
    mu, sd = np.nanmean(surr), np.nanstd(surr)
    return te - mu, (te - mu) / (sd + _EPS)


def build_ete_matrix(returns: np.ndarray, cfg: Config, seed: int = 0
                     ) -> np.ndarray:
    """ETE adjacency: keep ETE where Z > 1.96, zero elsewhere (Sec 2.1)."""
    n = returns.shape[1]
    A = np.zeros((n, n))
    rng = np.random.default_rng(seed)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ete, z = effective_te(returns[:, i], returns[:, j], cfg.n_bins,
                                  cfg.te_lag, cfg.n_shuffles, rng)
            if np.isfinite(z) and z > cfg.z_threshold:
                A[i, j] = max(ete, 0.0)
    return A


def build_te_matrix(returns: np.ndarray, cfg: Config, seed: int = 0
                    ) -> np.ndarray:
    """Plain TE adjacency - no surrogate correction, no significance filter."""
    n = returns.shape[1]
    syms = [discretize(returns[:, i], cfg.n_bins) for i in range(n)]
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            te = _te_from_symbols(syms[i], syms[j], cfg.n_bins, cfg.te_lag)
            A[i, j] = max(te, 0.0) if np.isfinite(te) else 0.0
    return A


def granger_pvalue(y: np.ndarray, x: np.ndarray, p: int = 1) -> float:
    """p-value of the sum-of-squared-residuals F-test that Y Granger-causes X.

    This is statsmodels' `grangercausalitytests(...)['ssr_ftest']`, computed
    directly.  The library call fits four tests per pair and dominates the
    walk-forward's runtime (~10 s for a 6-node matrix); this is the same
    statistic two orders of magnitude faster, which matters because the matrix
    is rebuilt at every refit.  Eq (6) with lag order p.
    """
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    n = x.size - p
    if n <= 2 * p + 1:
        return 1.0

    target = x[p:]
    const = np.ones((n, 1))
    x_lags = np.column_stack([x[p - k - 1: -k - 1] for k in range(p)])
    y_lags = np.column_stack([y[p - k - 1: -k - 1] for k in range(p)])

    def _rss(design: np.ndarray) -> float:
        beta, *_ = np.linalg.lstsq(design, target, rcond=None)
        resid = target - design @ beta
        return float(resid @ resid)

    rss_r = _rss(np.hstack([const, x_lags]))            # restricted
    rss_u = _rss(np.hstack([const, x_lags, y_lags]))    # unrestricted
    df_denom = n - (2 * p + 1)
    if df_denom <= 0 or rss_u <= 0:
        return 1.0

    f_stat = ((rss_r - rss_u) / p) / (rss_u / df_denom)
    if not np.isfinite(f_stat) or f_stat < 0:
        return 1.0
    return float(stats.f.sf(f_stat, p, df_denom))


def build_granger_matrix(returns: np.ndarray, cfg: Config, seed: int = 0
                         ) -> np.ndarray:
    """Granger causality, binarised at the 5% level (Sec 3.1)."""
    n = returns.shape[1]
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if granger_pvalue(returns[:, i], returns[:, j], cfg.te_lag) < 0.05:
                A[i, j] = 1.0
    return A


def build_pearson_matrix(returns: np.ndarray, cfg: Config, seed: int = 0
                         ) -> np.ndarray:
    """Absolute Pearson correlation, zero diagonal (Sec 3.1)."""
    C = np.abs(np.corrcoef(returns, rowvar=False))
    np.fill_diagonal(C, 0.0)
    return np.nan_to_num(C)


EDGE_BUILDERS = {
    "ETE": build_ete_matrix,
    "TE": build_te_matrix,
    "Granger": build_granger_matrix,
    "Pearson": build_pearson_matrix,
}


def row_normalize(A: np.ndarray) -> np.ndarray:
    s = A.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return A / s


# =============================================================================
# 3. Hurst exponent and regime detection  (Sec 2.3, Eq 10-16)
# =============================================================================

def expected_rs(w: int) -> float:
    """Anis & Lloyd (1976) expectation of (R/S)_w for an i.i.d. series.

    Peters' small-sample form is used for w <= 340, the asymptotic form above.
    Only needed by the "anis_lloyd" correction in hurst_rs().
    """
    from scipy.special import gammaln

    i = np.arange(1, w)
    tail = float(np.sum(np.sqrt((w - i) / i)))
    if w > 340:
        front = (w - 0.5) / w * (w * np.pi / 2) ** -0.5
    else:
        front = (w - 0.5) / w * np.exp(
            gammaln((w - 1) / 2) - 0.5 * np.log(np.pi) - gammaln(w / 2))
    return float(front * tail)


def hurst_rs(series: np.ndarray, min_scale: int = 10, n_scales: int = 10,
             correction: str = "none") -> float:
    """Rescaled-range Hurst exponent, following Eq (10)-(16).

    Non-overlapping segments of length w; R_i from the cumulative deviation
    (Eq 11-12); S_i is the population standard deviation (Eq 13 divides by w,
    not w-1); (R/S)_w averaged over segments (Eq 14); H is the OLS slope of
    log(R/S)_w on log(w) (Eq 16).

    `correction="none"` is the literal paper.  Be aware that the raw R/S slope
    is badly biased upward on short windows: on 250 i.i.d. normal draws it
    averages ~0.57 and lands above 0.5 in ~84% of windows, which would push the
    Sec 3.1 classifier into a near-permanent "Above" regime.  `"anis_lloyd"`
    divides out the expected R/S of an i.i.d. series of the same length
    (mean ~0.47, ~31% above 0.5 on the same test).  See DEVIATIONS[10].
    """
    x = np.asarray(series, dtype=float)
    n = x.size
    if n < 2 * min_scale:
        return np.nan

    scales = np.unique(np.logspace(np.log10(min_scale), np.log10(n // 2),
                                   n_scales).astype(int))
    logw, logrs, used = [], [], []
    for w in scales:
        m = n // w
        if m < 1:
            continue
        seg = x[: m * w].reshape(m, w)
        dev = np.cumsum(seg - seg.mean(axis=1, keepdims=True), axis=1)
        R = dev.max(axis=1) - dev.min(axis=1)
        S = seg.std(axis=1, ddof=0)                       # Eq (13)
        ok = S > 0
        if not ok.any():
            continue
        logw.append(np.log(w))
        logrs.append(np.log((R[ok] / S[ok]).mean()))      # Eq (14)
        used.append(int(w))

    if len(logw) < 2:
        return np.nan
    H = float(np.polyfit(logw, logrs, 1)[0])              # Eq (16)

    if correction == "anis_lloyd":
        H_exp = float(np.polyfit(logw, [np.log(expected_rs(w)) for w in used], 1)[0])
        H = H - H_exp + 0.5
    return H


def rolling_hurst(proxy_returns: pd.Series, window: int = 250,
                  correction: str = "none") -> pd.Series:
    """Sliding-window daily Hurst exponent on the World-ETF log-returns."""
    return (proxy_returns.rolling(window)
            .apply(lambda a: hurst_rs(a, correction=correction), raw=True)
            .dropna())


def regime_labels(hurst: pd.Series, lookback: int = 10, s: int = 6,
                  threshold: float = 0.5) -> pd.Series:
    """Sec 3.1 regime rule.

    Over the trailing `lookback` days: if at least `s` daily Hurst values are
    above `threshold` the period is "Above"; if at least `s` are below it is
    "Below".  When neither count reaches s the previous label is carried
    forward, so a regime persists until it is positively displaced.
    """
    h = hurst.values
    idx = hurst.index
    out: List[Optional[str]] = []
    current: Optional[str] = None
    for t in range(len(h)):
        if t + 1 < lookback:
            out.append(current)
            continue
        w = h[t + 1 - lookback: t + 1]
        n_above = int((w > threshold).sum())
        n_below = int((w < threshold).sum())
        if n_above >= s:
            current = "Above"
        elif n_below >= s:
            current = "Below"
        out.append(current)
    return pd.Series(out, index=idx, name="regime")


def regime_change_dates(labels: pd.Series) -> pd.DatetimeIndex:
    """Dates on which the regime label differs from the previous day's."""
    lab = labels.dropna()
    if len(lab) == 0:
        return pd.DatetimeIndex([])
    changed = lab != lab.shift(1)
    changed.iloc[0] = False                # the first label is not a "change"
    return lab.index[changed]


# =============================================================================
# 4. Model  (Sec 2.2, Eq 8-9)
# =============================================================================

class MultiScaleConv(nn.Module):
    """Eq (8):  f = concat_i ReLU(K_i * v + b_i).

    Three 1-D filters with receptive fields {3, 5, 7} and 12 channels each,
    applied to every node's M-day input series independently.  `readout`
    controls how the per-kernel feature maps become the node embedding:
      "flatten" - concatenate channels x time, the literal reading of Eq (8);
      "avgpool" - global average pool over time first (see DEVIATIONS[3]).
    """

    def __init__(self, in_len: int, channels: int = 12,
                 kernels: Sequence[int] = (3, 5, 7), readout: str = "flatten"):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(1, channels, kernel_size=k) for k in kernels])
        self.readout = readout
        if readout == "flatten":
            self.out_dim = sum(channels * (in_len - k + 1) for k in kernels)
        elif readout == "avgpool":
            self.out_dim = channels * len(kernels)
        else:
            raise ValueError(f"unknown readout {readout!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, L) -> (B, N, out_dim)
        B, N, L = x.shape
        z = x.reshape(B * N, 1, L)
        feats = [F.relu(c(z)) for c in self.convs]
        if self.readout == "avgpool":
            feats = [f.mean(dim=2) for f in feats]
        else:
            feats = [f.flatten(1) for f in feats]
        return torch.cat(feats, dim=1).view(B, N, -1)


class GCNLayer(nn.Module):
    """Eq (9):  h_i' = ReLU( h_i W1 + sum_j A_ji h_j W2 ).

    A is held in [source, target] layout, so A_ji is A[j, i] and the
    neighbourhood sum for node i is einsum('ji,bjf->bif', A, H) == A.T @ H.
    """

    def __init__(self, in_dim: int, out_dim: int, activation: bool = True):
        super().__init__()
        self.W1 = nn.Linear(in_dim, out_dim)
        self.W2 = nn.Linear(in_dim, out_dim, bias=False)
        self.activation = activation

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        out = self.W1(H) + torch.einsum("ji,bjf->bif", A, self.W2(H))
        return F.relu(out) if self.activation else out


class ETEGNN(nn.Module):
    """Multi-scale conv encoder + three message-passing layers, one RV per node."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.feat = MultiScaleConv(cfg.input_size, cfg.conv_channels,
                                   cfg.conv_kernels, cfg.conv_readout)
        d = self.feat.out_dim
        self.layers = nn.ModuleList([
            GCNLayer(d, cfg.hidden1),
            GCNLayer(cfg.hidden1, cfg.hidden2),
            GCNLayer(cfg.hidden2, 1, activation=cfg.final_layer_relu),
        ])

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        h = self.feat(x)
        for layer in self.layers:
            h = layer(h, A)
        return h.squeeze(-1)                              # (B, N)


class RNNBaseline(nn.Module):
    """LSTM / GRU benchmark: the M-day log-returns of all ETFs -> all RVs.

    No causality matrix is used, per Sec 3.1.
    """

    def __init__(self, cfg: Config, kind: str = "LSTM"):
        super().__init__()
        rnn = nn.LSTM if kind.upper() == "LSTM" else nn.GRU
        self.rnn = rnn(input_size=cfg.n_nodes(), hidden_size=cfg.rnn_hidden,
                       num_layers=cfg.rnn_layers, batch_first=True,
                       dropout=cfg.rnn_dropout if cfg.rnn_layers > 1 else 0.0)
        self.head = nn.Linear(cfg.rnn_hidden, cfg.n_nodes())

    def forward(self, x: torch.Tensor, A: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        # x: (B, N, L) -> (B, L, N) for the RNN
        out, _ = self.rnn(x.transpose(1, 2))
        return self.head(out[:, -1, :])


def build_model(name: str, cfg: Config) -> nn.Module:
    if name in ("LSTM", "GRU"):
        return RNNBaseline(cfg, name)
    return ETEGNN(cfg)


# =============================================================================
# 5. Sample construction and training
# =============================================================================

def make_windows(ret_mat: np.ndarray, rv_mat: np.ndarray, M: int
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slide an M-day log-return window and pair it with next-day RV.

    ret_mat and rv_mat are aligned on the same dates.  Sample e ends at index
    end_idx[e], uses returns [end-M+1 .. end] and targets RV[end+1].
    """
    T = ret_mat.shape[0]
    X, y, end_idx = [], [], []
    for t in range(M - 1, T - 1):
        X.append(ret_mat[t - M + 1: t + 1].T)             # (N, M)
        y.append(rv_mat[t + 1])                           # (N,)
        end_idx.append(t)
    return np.asarray(X), np.asarray(y), np.asarray(end_idx)


def minmax_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-node min/max over a training block; Sec 3.1 rescales inputs to [0,1]."""
    return X.min(axis=(0, 2)), X.max(axis=(0, 2))


def minmax_apply(X: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    rng = np.where(hi - lo < _EPS, 1.0, hi - lo)
    return (X - lo[None, :, None]) / rng[None, :, None]


def train(model: nn.Module, A: Optional[torch.Tensor],
          X: np.ndarray, y: np.ndarray, cfg: Config,
          epochs: Optional[int] = None) -> nn.Module:
    """MSE + Adam + early stopping on a 7:3 train/validation split (Sec 3.1)."""
    epochs = cfg.epochs if epochs is None else epochs
    dev = torch.device(cfg.device)
    model = model.to(dev)

    n_val = max(1, int(round(cfg.val_fraction * len(X))))
    split = len(X) - n_val                                # validation is the tail
    Xtr = torch.tensor(X[:split], dtype=torch.float32, device=dev)
    ytr = torch.tensor(y[:split], dtype=torch.float32, device=dev)
    Xva = torch.tensor(X[split:], dtype=torch.float32, device=dev)
    yva = torch.tensor(y[split:], dtype=torch.float32, device=dev)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    patience = cfg.patience if cfg.patience else epochs + 1
    best, best_state, bad = math.inf, None, 0

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=dev)
        for b in range(0, len(perm), cfg.batch_size):
            idx = perm[b: b + cfg.batch_size]
            opt.zero_grad()
            loss = F.mse_loss(model(Xtr[idx], A), ytr[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vloss = F.mse_loss(model(Xva, A), yva).item()
        if vloss < best - 1e-6:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model: nn.Module, A: Optional[torch.Tensor], X: np.ndarray,
            cfg: Config) -> np.ndarray:
    model.eval()
    dev = torch.device(cfg.device)
    xt = torch.tensor(X, dtype=torch.float32, device=dev)
    return model(xt, A).cpu().numpy()


# =============================================================================
# 6. Walk-forward experiment  (Sec 3.1, Figures 1-2)
# =============================================================================

@dataclass
class RunResult:
    model: str
    regime: str
    dates: pd.DatetimeIndex
    y_true: np.ndarray
    y_pred: np.ndarray
    n_refits: int
    seconds: float


class WalkForward:
    """One-day-ahead walk-forward evaluation under either retraining policy.

    regime_mode="hurst"    refit the edge matrix and fine-tune whenever the
                           Hurst regime label flips (the proposed H-*-GNN).
    regime_mode="periodic" full retrain every N days (the paper's control).
    """

    def __init__(self, cfg: Config, ret: pd.DataFrame, rv: pd.DataFrame,
                 change_dates: pd.DatetimeIndex):
        self.cfg = cfg
        common = ret.index.intersection(rv.index)
        self.ret = ret.loc[common]
        self.rv = rv.loc[common]
        self.dates = common
        self.change_dates = set(pd.DatetimeIndex(change_dates))

        self.X, self.y, self.end_idx = make_windows(
            self.ret.values, self.rv.values, cfg.input_size)
        self.sample_dates = self.dates[self.end_idx]      # date the window ends
        self.target_dates = self.dates[self.end_idx + 1]  # date being forecast
        self.is_change = np.array([d in self.change_dates
                                   for d in self.target_dates])

    def refit_schedule(self, eval_idx: np.ndarray, regime_mode: str
                       ) -> List[Tuple[int, str]]:
        """Positions in `eval_idx` where the model is rebuilt, and how.

        The schedule depends only on the calendar - a periodic retrain every N
        samples, or a flip in the Hurst regime label - never on the data or the
        fitted weights.  Resolving it up front lets the forecasts between two
        refits be produced in one batched forward pass instead of one per day,
        which is where nearly all of the walk-forward's runtime used to go.

        With cfg.stride > 1 a regime change can fall on a day that is not being
        forecast.  Skipping the refit in that case would quietly give the Hurst
        arm fewer retrainings than the paper specifies - at stride 5 only 8 of
        the 50 changes survived - and would make the Hurst-vs-Periodic contrast
        an artefact of the stride.  So a change anywhere in the skipped span
        still triggers the refit: stride subsamples the evaluation, never the
        retraining schedule.
        """
        N = self.cfg.train_period
        out: List[Tuple[int, str]] = []
        last_full = -10 ** 9
        prev_e = -1
        for pos, e in enumerate(eval_idx):
            e = int(e)
            if pos == 0:
                out.append((pos, "full"))
                last_full = e
            elif regime_mode == "periodic" and e - last_full >= N:
                out.append((pos, "full"))
                last_full = e
            elif (regime_mode == "hurst"
                  and self.is_change[prev_e + 1: e + 1].any()):
                out.append((pos, "finetune"))
            prev_e = e
        return out

    def run(self, model_name: str, edge_kind: Optional[str],
            regime_mode: str, seed: int) -> RunResult:
        cfg = self.cfg
        set_seed(seed)
        t0 = time.time()

        N = cfg.train_period
        n_samples = len(self.X)
        if N >= n_samples:                                # need N prior samples
            raise ValueError("training period longer than the sample")

        eval_idx = np.arange(N, n_samples, cfg.stride)
        schedule = self.refit_schedule(eval_idx, regime_mode)

        model: Optional[nn.Module] = None
        A_t: Optional[torch.Tensor] = None
        preds = np.empty((len(eval_idx), self.y.shape[1]), dtype=float)

        for k, (pos, kind) in enumerate(schedule):
            e = int(eval_idx[pos])
            tr = slice(e - N, e)                          # most recent N samples
            Xtr_raw, ytr = self.X[tr], self.y[tr]
            lo, hi = minmax_fit(Xtr_raw)
            Xtr = minmax_apply(Xtr_raw, lo, hi)

            if edge_kind is not None:
                # Edges come from the same N-day block, never from the future.
                win_end = self.end_idx[e - 1] + 1
                win = self.ret.values[max(0, win_end - N): win_end]
                A = EDGE_BUILDERS[edge_kind](win, cfg, seed=seed)
                if cfg.normalize_adjacency:
                    A = row_normalize(A)
                A_t = torch.tensor(A, dtype=torch.float32,
                                   device=torch.device(cfg.device))

            if kind == "full":
                model = train(build_model(model_name, cfg), A_t, Xtr, ytr, cfg)
            else:                                         # regime change: fine-tune
                model = train(model, A_t, Xtr, ytr, cfg,
                              epochs=cfg.finetune_epochs)

            # forecast every day up to the next refit with this fixed model
            stop = schedule[k + 1][0] if k + 1 < len(schedule) else len(eval_idx)
            for b in range(pos, stop, _PREDICT_CHUNK):
                sl = eval_idx[b: min(b + _PREDICT_CHUNK, stop)]
                preds[b: b + len(sl)] = predict(
                    model, A_t, minmax_apply(self.X[sl], lo, hi), cfg)

        return RunResult(
            model=model_name if edge_kind is None else f"{edge_kind}-GNN",
            regime=regime_mode,
            dates=self.target_dates[eval_idx],
            y_true=self.y[eval_idx],
            y_pred=preds,
            n_refits=len(schedule),
            seconds=time.time() - t0,
        )


# =============================================================================
# 7. Metrics  (Sec 4, Eq 19-23)
# =============================================================================

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """RMSE, MAE, MAPE, correlation and hit ratio, pooled over all ETFs.

    Hit ratio (Eq 23) compares the sign of the realised change y_t - y_{t-1}
    with the sign of the predicted change yhat_t - y_{t-1}: the direction the
    model called from the last *observed* level.  See DEVIATIONS[5].
    """
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = yp - yt

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err / (yt + _EPS))) * 100.0)
    corr = float(np.corrcoef(yt.ravel(), yp.ravel())[0, 1])

    d_true = np.sign(yt[1:] - yt[:-1])
    d_pred = np.sign(yp[1:] - yt[:-1])
    hit = float(np.mean(d_true == d_pred))

    return {"RMSE": rmse, "MAE": mae, "MAPE": mape,
            "Correlation": corr, "HitRatio": hit}


def summarize(runs: Sequence[RunResult]) -> pd.DataFrame:
    """Mean +/- sd over repetitions, in the layout of the paper's Table 6."""
    rows = []
    for r in runs:
        m = metrics(r.y_true, r.y_pred)
        m.update(Regime=r.regime.capitalize(), Model=r.model,
                 refits=r.n_refits, seconds=round(r.seconds, 1))
        rows.append(m)
    df = pd.DataFrame(rows)
    num = ["RMSE", "MAE", "MAPE", "Correlation", "HitRatio"]
    agg = df.groupby(["Regime", "Model"], sort=False)[num].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


# Table 6 of the paper, for side-by-side comparison with a replication.
PAPER_TABLE6 = pd.DataFrame([
    ("Periodic", "LSTM",        0.5136, 0.3489, 22.3469, 0.6913, 0.5882),
    ("Periodic", "GRU",         0.4598, 0.3112, 19.9726, 0.7545, 0.6137),
    ("Periodic", "Pearson-GNN", 0.2162, 0.1495, 10.2881, 0.9514, 0.7010),
    ("Periodic", "Granger-GNN", 0.1858, 0.1231,  8.2433, 0.9509, 0.7027),
    ("Periodic", "TE-GNN",      0.2142, 0.1455,  9.9158, 0.9552, 0.7082),
    ("Periodic", "ETE-GNN",     0.2001, 0.1301,  8.4759, 0.9565, 0.7040),
    ("Hurst",    "LSTM",        0.5008, 0.3381, 21.2038, 0.6993, 0.5995),
    ("Hurst",    "GRU",         0.4494, 0.3059, 19.6617, 0.7337, 0.6166),
    ("Hurst",    "Pearson-GNN", 0.2021, 0.1371,  9.1044, 0.9471, 0.6904),
    ("Hurst",    "Granger-GNN", 0.1833, 0.1118,  6.9656, 0.9595, 0.7091),
    ("Hurst",    "TE-GNN",      0.1973, 0.1309,  8.5549, 0.9535, 0.7213),
    ("Hurst",    "ETE-GNN",     0.1623, 0.1024,  6.6555, 0.9605, 0.7206),
], columns=["Regime", "Model", "RMSE", "MAE", "MAPE", "Correlation", "HitRatio"])


# =============================================================================
# 8. Driver
# =============================================================================

DEFAULT_MODELS = [
    ("ETE-GNN", "GNN", "ETE"),
    ("TE-GNN", "GNN", "TE"),
    ("Granger-GNN", "GNN", "Granger"),
    ("Pearson-GNN", "GNN", "Pearson"),
    ("LSTM", "LSTM", None),
    ("GRU", "GRU", None),
]


def prepare(cfg: Config):
    """Download, transform and derive the regime calendar."""
    prices, proxy_ret, proxy_src = download_prices(cfg)
    ret = log_returns(prices)
    rv = realized_volatility(ret, cfg.rv_window)
    hurst = rolling_hurst(proxy_ret, cfg.hurst_window, cfg.hurst_correction)
    labels = regime_labels(hurst, cfg.regime_lookback, cfg.regime_sensitivity,
                           cfg.hurst_threshold)
    changes = regime_change_dates(labels)
    return prices, ret, rv, proxy_ret, proxy_src, hurst, labels, changes


def run_experiment(cfg: Config, models: Optional[Sequence[str]] = None,
                   regimes: Sequence[str] = ("hurst", "periodic"),
                   verbose: bool = True):
    cfg.apply_runtime()
    prices, ret, rv, proxy_ret, proxy_src, hurst, labels, changes = prepare(cfg)
    if verbose:
        print(f"device      {cfg.device}"
              + (f"  torch threads {torch.get_num_threads()}"
                 if cfg.device == "cpu" else ""))
        print(f"prices      {prices.shape}  {prices.index[0].date()} -> "
              f"{prices.index[-1].date()}")
        print(f"proxy       {proxy_ret.index[0].date()} -> "
              f"{proxy_ret.index[-1].date()}  ({len(proxy_ret)} obs)")
        print("proxy split " + ", ".join(
            f"{k}={v}" for k, v in proxy_src.value_counts().items()))
        print(f"hurst       mean={hurst.mean():.4f}  "
              f"above 0.5: {(hurst > 0.5).mean():.1%}")
        print(f"regime      {len(changes)} changes detected "
              f"(paper reports 47)\n")

    wanted = set(models) if models else None
    wf = WalkForward(cfg, ret, rv, changes)
    runs: List[RunResult] = []
    for regime in regimes:
        for label, kind, edge in DEFAULT_MODELS:
            if wanted and label not in wanted:
                continue
            for rep in range(cfg.n_repeats):
                r = wf.run(kind, edge, regime, seed=cfg.seed + rep)
                runs.append(r)
                if verbose:
                    m = metrics(r.y_true, r.y_pred)
                    print(f"[{regime:8s}] {label:12s} rep{rep} "
                          f"RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} "
                          f"MAPE={m['MAPE']:.3f} corr={m['Correlation']:.4f} "
                          f"hit={m['HitRatio']:.4f} "
                          f"refits={r.n_refits} {r.seconds:.0f}s")
    return runs, summarize(runs), dict(hurst=hurst, labels=labels,
                                       changes=changes, rv=rv, ret=ret)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Replicate Lee & Cho (2025) H-ETE-GNN")
    p.add_argument("--models", nargs="*", default=None,
                   help="subset of: " + " ".join(m[0] for m in DEFAULT_MODELS))
    p.add_argument("--regimes", nargs="*", default=["hurst", "periodic"])
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--start", default=Config.start)
    p.add_argument("--end", default=Config.end)
    p.add_argument("--device", default=None)
    p.add_argument("--threads", type=int, default=Config.torch_threads,
                   help="torch CPU threads; the net is tiny, so more is slower")
    p.add_argument("--raw-adjacency", action="store_true",
                   help="use raw edge weights instead of row-normalising them. "
                        "This is the literal paper, but it makes the four edge "
                        "types a comparison of weight scale rather than "
                        "topology (see DEVIATIONS item 4)")
    p.add_argument("--hurst-correction", default=Config.hurst_correction,
                   choices=["none", "anis_lloyd"])
    p.add_argument("--out", default="replication_results.csv")
    p.add_argument("--quick", action="store_true",
                   help="short smoke run: 1 repeat, stride 5, 2010 onward")
    args = p.parse_args()

    cfg = Config(n_repeats=args.repeats, stride=args.stride,
                 start=args.start, end=args.end,
                 torch_threads=args.threads,
                 normalize_adjacency=not args.raw_adjacency,
                 hurst_correction=args.hurst_correction)
    if args.device:
        cfg.device = args.device
    if args.quick:
        cfg.n_repeats, cfg.stride, cfg.start = 1, 5, "2010-01-01"
        cfg.epochs, cfg.n_shuffles = 20, 10

    print(json.dumps(asdict(cfg), indent=2, default=str))
    runs, table, _ = run_experiment(cfg, args.models, args.regimes)
    print("\n" + table.to_string(index=False))
    table.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")


# =============================================================================
# Deviations from the published description
# =============================================================================

DEVIATIONS = """
1. World-ETF proxy.  The paper names the iShares MSCI World ETF as the regime
   proxy over 2003-02-07..2024-05-29, but URTH did not begin trading until
   2012.  We splice log-returns newest-first across URTH -> ACWI -> IOO so the
   Hurst series covers the whole sample.  The regime calendar - and therefore
   the count of 47 regime changes - is sensitive to this choice.  Set
   Config.proxy_chain=("URTH",) to see the effect of the literal reading.

2. Unreported hyper-parameters.  Table 1 gives search spaces for the learning
   rate, batch size, both hidden sizes and the shuffle count m, but Table 3
   reports only M=20, N=750, s=6.  Defaults here are mid-range placeholders,
   not the paper's optima; re-tune before drawing conclusions from levels.

3. Convolution readout.  Eq (8) concatenates the ReLU'd feature maps without
   naming a pooling step, so conv_readout="flatten" is the literal reading.
   "avgpool" is offered because it is the common implementation and yields a
   far smaller first GCN layer.

4. Adjacency scaling -- THE MOST CONSEQUENTIAL OPEN CHOICE.  The paper never
   says whether A is normalised, and the answer decides the whole experiment.
   Eq (9) adds a self term h_i W1 to a neighbourhood term sum_j A_ji h_j W2, so
   the graph only matters in proportion to the size of A's entries.  Measured on
   the first N=750 block (.build/check_adjacency_scale.py), the mean magnitude
   of the neighbourhood term as a fraction of the self term is:

       raw weights            row-normalised
       ETE       0.007        ETE       0.648
       TE        0.063        TE        0.910
       Granger   2.122        Granger   0.648
       Pearson   3.803        Pearson   0.910

   On raw weights ETE contributes 0.7% of the self term - the graph is
   effectively decorative and ETE-GNN degenerates into a per-node MLP - while
   Pearson contributes 380%.  The four "GNN" variants would then be comparing
   edge *magnitude* (nats vs a 0/1 indicator vs a correlation), not the edge
   *topology* the paper claims to be testing, and no amount of tuning would
   recover the reported ordering with ETE-GNN on top.  Row-normalising puts all
   four in 0.65-0.91, which is the only setting under which the paper's own
   narrative - that ETE wins because it is "sparser but more informative" -
   can even be tested.

   normalize_adjacency therefore DEFAULTS TO TRUE here - a deliberate departure
   from the literal text, taken because the literal reading does not produce a
   baseline the Renyi model can meaningfully be compared against.  Pass
   --raw-adjacency (or set the field False) to reproduce the literal reading.

5. Hit ratio.  Eq (23)'s second sign term is ambiguous in the typeset paper.
   We use sign(yhat_t - y_{t-1}), the direction called from the last observed
   level, rather than sign(yhat_t - yhat_{t-1}).

6. Regime hysteresis.  Sec 3.1 defines Above and Below but not the case where
   neither count reaches s.  We carry the previous label forward, so a regime
   persists until positively displaced.

7. Fine-tuning budget.  "the model was fine-tuned accordingly" is unquantified;
   we use Config.finetune_epochs=30 with the same early stopping.

7b. Early-stopping patience.  The paper says "100 epochs and early stopping" and
   never gives a patience.  It matters a lot: measured on the real panel, the
   longest run of non-improving epochs before the eventual best validation loss
   was 50 for LSTM, 19 for GRU and 17 for the GNN.  A patience of 10 therefore
   stops all three early, and does so unevenly - it leaves both RNNs worse than
   predicting the training mean on held-out data (0.374 and 0.378 against a
   0.355 baseline) while the GNN is essentially unaffected (0.0373 vs 0.0374).
   Since train() restores the best-validation weights either way, early stopping
   is only a compute saving here, so Config.patience defaults to None (run the
   full budget) rather than to a number that would quietly widen the paper's
   headline GNN-vs-RNN gap.  Set an integer to trade accuracy for time.

8. Min-max scaling is fit on each N-day training block and applied to the
   forecast window, to avoid look-ahead.  The paper does not say where the
   scaler is fit.

9. Periodic control.  Sec 3.1 says the no-Hurst models are "periodically
   retrained every N days"; we take that literally (full retrain, edges
   rebuilt) rather than fine-tuned.

10. R/S small-sample bias.  Plain R/S on a 250-point window is biased upward:
   on i.i.d. normal draws it averages H = 0.57 and exceeds 0.5 in ~84% of
   windows.  On the real proxy that shows up as mean H = 0.539 with 80.7% of
   days above 0.5, so the "Above" regime dominates 4220 days to 883.  It does
   NOT, however, suppress regime switching: the s=6-of-10 rule still flips 50
   times, against the paper's 47, so hurst_correction="none" (the literal
   paper) is the right default and is what these defaults use.  The Anis-Lloyd
   correction is available for a robustness check - it recentres H to 0.448 and
   inverts the label balance, giving 39 changes.  Numbers measured 2026-09-07
   on the URTH/ACWI/IOO splice; re-check them if the proxy chain changes.
"""

if __name__ == "__main__":
    main()
