"""
Statistical significance toolkit for the HR-ETE-GNN thesis.

Answers the panel question: "is your model *significantly* better than the
Shannon ETE-GNN it is inspired by, or is the RMSE gap noise?"

Dependencies: numpy, pandas, scipy only (works in Colab out of the box).

Contents
--------
Losses            : qlike, mse, mae, qlike_log   -> per-observation loss series
Variance          : newey_west_lrv               -> HAC long-run variance
Pairwise tests    : diebold_mariano              -> DM (1995) + Harvey-Leybourne-Newbold
                    giacomini_white              -> GW (2006) conditional predictive ability
Multiplicity      : benjamini_hochberg           -> FDR control across the 10 ETFs
                    model_confidence_set         -> Hansen-Lunde-Nason (2011) MCS
Bootstrap         : stationary_bootstrap_indices -> Politis-Romano
Baselines         : har_rv_forecast              -> Corsi (2009) HAR-RV
                    random_walk_forecast
Design            : min_detectable_effect        -> power / MDE for a given test length
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "qlike", "mse", "mae", "qlike_log",
    "newey_west_lrv",
    "diebold_mariano", "giacomini_white",
    "benjamini_hochberg", "model_confidence_set",
    "stationary_bootstrap_indices",
    "har_rv_forecast", "random_walk_forecast",
    "min_detectable_effect",
    "seed_summary",
]

_EPS = 1e-12


# --------------------------------------------------------------------------
# 1. Loss functions
# --------------------------------------------------------------------------
# All return a *per-observation* loss array with the same shape as the inputs.
# Keep them per-observation: every test below needs the loss *series*, not the
# aggregate, because the aggregate throws away the serial-correlation
# information that the HAC variance needs.

def mse(actual: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Squared error. Sensitive to the noise in the RV proxy; report, don't lead with it."""
    return (np.asarray(actual, float) - np.asarray(pred, float)) ** 2


def mae(actual: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Absolute error. NOT robust in Patton's (2011) sense for a noisy volatility proxy."""
    return np.abs(np.asarray(actual, float) - np.asarray(pred, float))


def qlike(actual: np.ndarray, pred: np.ndarray, floor: float = 0.05,
          warn: bool = True) -> np.ndarray:
    """QLIKE on the *variance* scale.

    L = sigma2_actual / sigma2_pred - log(sigma2_actual / sigma2_pred) - 1

    `actual` and `pred` are passed as volatilities (RV in %), squared internally.
    QLIKE is one of only two loss families (with MSE) that are robust to noise in
    the volatility proxy (Patton 2011, J. Econometrics), and unlike MSE it does
    not let a handful of crisis days dominate the average. Use it as the primary
    metric and pre-register that choice.

    IMPORTANT: QLIKE is only defined for strictly positive forecasts, and it
    diverges as pred -> 0. The pilot's model ends in a bare `nn.Linear`, so it
    can emit zero or negative volatility -- economically meaningless, and enough
    for one bad day to swamp the whole test-set average. `floor` (0.05 = 0.05% daily vol, well below any realistic RV) clips the
    forecast volatility from below and, with `warn=True`, tells you how often
    that bound bound. The real fix is in the architecture: give the head a
    softplus/exp activation, or forecast log RV and exponentiate.
    """
    a = np.asarray(actual, float) ** 2
    p = np.asarray(pred, float)
    n_bad = int(np.sum(p < floor))
    if warn and n_bad:
        import warnings
        warnings.warn(
            f"qlike: {n_bad}/{p.size} forecasts below floor={floor} were clipped "
            "(non-positive volatility forecasts). Constrain the model output to "
            "be positive rather than relying on this clip.", RuntimeWarning)
    p = np.maximum(p, floor) ** 2
    r = np.maximum(a, _EPS) / p
    return r - np.log(r) - 1.0


def qlike_log(actual: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Squared error in log-variance. Symmetric, scale-free, easy to defend."""
    a = np.log(np.maximum(np.asarray(actual, float) ** 2, _EPS))
    p = np.log(np.maximum(np.asarray(pred, float) ** 2, _EPS))
    return (a - p) ** 2


# --------------------------------------------------------------------------
# 2. HAC long-run variance
# --------------------------------------------------------------------------

def newey_west_lrv(x: np.ndarray, lag: int | None = None) -> float:
    """Newey-West long-run variance of a scalar series.

    `lag=None` uses the standard automatic rule floor(4*(n/100)^(2/9)).
    Loss differentials for volatility forecasts are strongly autocorrelated
    (volatility clusters), so a plain i.i.d. variance understates the standard
    error and manufactures significance. Always use HAC.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return np.nan
    if lag is None:
        lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lag = max(0, min(lag, n - 2))
    e = x - x.mean()
    gamma0 = float(e @ e) / n
    lrv = gamma0
    for j in range(1, lag + 1):
        gj = float(e[j:] @ e[:-j]) / n
        w = 1.0 - j / (lag + 1.0)          # Bartlett kernel -> guarantees lrv >= 0
        lrv += 2.0 * w * gj
    return max(lrv, _EPS)


# --------------------------------------------------------------------------
# 3. Diebold-Mariano
# --------------------------------------------------------------------------

def diebold_mariano(loss_base: np.ndarray,
                    loss_new: np.ndarray,
                    h: int = 1,
                    lag: int | None = None,
                    alternative: str = "greater",
                    hln: bool = True) -> dict:
    """Diebold-Mariano (1995) test with the Harvey-Leybourne-Newbold (1997) correction.

    d_t = loss_base_t - loss_new_t, so d > 0 means the NEW model is better.

    H0: E[d_t] = 0            (equal predictive accuracy)
    H1: E[d_t] > 0            (`alternative="greater"`: the new model is better)

    Use the one-sided form and say so in the thesis -- you have a directional
    hypothesis, and a one-sided test at 5% is the honest way to state it.

    Caveat to disclose: DM treats the forecasts as given. With *estimated*
    models (a neural net) the asymptotics are only exact under a fixed
    estimation window. Report `giacomini_white` alongside it, which is valid for
    estimated models.
    """
    d = np.asarray(loss_base, float).ravel() - np.asarray(loss_new, float).ravel()
    d = d[np.isfinite(d)]
    n = d.size
    if n < 10:
        raise ValueError(f"need >=10 paired observations, got {n}")

    if lag is None:
        lag = h - 1
    lrv = newey_west_lrv(d, lag=lag)
    dm = d.mean() / np.sqrt(lrv / n)

    stat, dist, df = dm, "normal", np.inf
    if hln:
        corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        stat = dm * corr
        dist, df = "t", n - 1

    if dist == "t":
        cdf = stats.t.cdf(stat, df)
    else:
        cdf = stats.norm.cdf(stat)

    if alternative == "greater":
        p = 1.0 - cdf
    elif alternative == "less":
        p = cdf
    else:
        p = 2.0 * min(cdf, 1.0 - cdf)

    return {
        "stat": float(stat), "p_value": float(p), "n": int(n),
        "mean_diff": float(d.mean()),
        "pct_improvement": float(100.0 * d.mean() / max(np.mean(loss_base), _EPS)),
        "se": float(np.sqrt(lrv / n)), "lag": int(lag),
        "dist": dist, "df": float(df), "alternative": alternative,
    }


# --------------------------------------------------------------------------
# 4. Giacomini-White conditional predictive ability
# --------------------------------------------------------------------------

def giacomini_white(loss_base: np.ndarray,
                    loss_new: np.ndarray,
                    instruments: np.ndarray | None = None,
                    lag: int | None = None) -> dict:
    """Giacomini-White (2006) test of conditional predictive ability.

    H0: E[d_{t+1} | F_t] = 0, i.e. NO information available at time t predicts
    which model will win tomorrow.

    `instruments` is an (n, q) matrix of variables known at time t. A constant
    column is added automatically. With `instruments=None` this reduces to the
    unconditional GW test (asymptotically the DM test, but valid for estimated
    models under a fixed/rolling estimation window).

    THIS IS THE TEST THAT MATCHES YOUR THESIS. Pass the lagged Hurst-regime
    dummy as the instrument and the test directly answers "does the crisis
    regime predict when Renyi beats Shannon?" -- which is exactly the claim in
    the title of the thesis. A significant regime coefficient is a far stronger
    result than a marginally lower RMSE.

    Returns the Wald statistic (chi2 with q+1 df) plus the regression
    coefficients, which are what you actually interpret.
    """
    d = np.asarray(loss_base, float).ravel() - np.asarray(loss_new, float).ravel()
    n = d.size
    if instruments is None:
        H = np.ones((n, 1))
        names = ["const"]
    else:
        Z = np.asarray(instruments, float)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        if Z.shape[0] != n:
            raise ValueError(f"instruments have {Z.shape[0]} rows, d has {n}")
        H = np.column_stack([np.ones(n), Z])
        names = ["const"] + [f"z{i+1}" for i in range(Z.shape[1])]

    ok = np.isfinite(d) & np.all(np.isfinite(H), axis=1)
    d, H = d[ok], H[ok]
    n, q = H.shape

    # moment conditions m_t = h_t * d_{t+1}; H0: E[m_t] = 0
    M = H * d[:, None]
    mbar = M.mean(axis=0)

    # HAC covariance of the moment vector
    if lag is None:
        lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    E = M - mbar
    Omega = (E.T @ E) / n
    for j in range(1, min(lag, n - 2) + 1):
        G = (E[j:].T @ E[:-j]) / n
        w = 1.0 - j / (lag + 1.0)
        Omega += w * (G + G.T)
    Omega += np.eye(q) * 1e-10

    stat = float(n * mbar @ np.linalg.solve(Omega, mbar))
    p = float(stats.chi2.sf(stat, q))

    # OLS of d on H -- the interpretable form of the same moment condition
    beta, *_ = np.linalg.lstsq(H, d, rcond=None)
    resid = d - H @ beta
    XtX_inv = np.linalg.pinv(H.T @ H)
    Sx = np.zeros((q, q))
    U = H * resid[:, None]
    Sx += U.T @ U
    for j in range(1, min(lag, n - 2) + 1):
        G = U[j:].T @ U[:-j]
        w = 1.0 - j / (lag + 1.0)
        Sx += w * (G + G.T)
    Vb = XtX_inv @ Sx @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(Vb), 0.0))
    tstat = beta / np.maximum(se, _EPS)

    return {
        "stat": stat, "p_value": p, "df": q, "n": int(n), "lag": int(lag),
        "coef": pd.Series(beta, index=names),
        "se": pd.Series(se, index=names),
        "t": pd.Series(tstat, index=names),
        "p_coef": pd.Series(2 * stats.norm.sf(np.abs(tstat)), index=names),
    }


# --------------------------------------------------------------------------
# 5. Multiplicity control
# --------------------------------------------------------------------------

def benjamini_hochberg(pvalues, q: float = 0.10) -> pd.DataFrame:
    """Benjamini-Hochberg FDR control.

    You will run one DM test per ETF (10 tests) and probably per alpha as well.
    Reporting "3 of 10 are significant at 5%" without correction is exactly the
    kind of thing a panel catches: under the null you expect 0.5 hits by chance.
    BH at q=0.10 is the standard, defensible correction for this many tests
    (Bonferroni is valid but needlessly conservative with correlated series).
    """
    s = pd.Series(pvalues, dtype=float)
    m = s.notna().sum()
    order = s.rank(method="first")
    thresh = q * order / m
    reject_raw = s <= thresh
    if reject_raw.any():
        cutoff = s[reject_raw].max()
        reject = s <= cutoff
    else:
        reject = pd.Series(False, index=s.index)
    adj = (s * m / order).sort_index()
    adj = adj.clip(upper=1.0)
    return pd.DataFrame({"p_value": s, "bh_threshold": thresh,
                         "p_adj": adj, "reject": reject})


def stationary_bootstrap_indices(n: int, B: int, block: float = 20.0,
                                 seed: int = 0) -> np.ndarray:
    """Politis-Romano (1994) stationary bootstrap index matrix, shape (B, n).

    `block` is the *expected* block length; geometric with p = 1/block. Use
    ~20 for daily volatility data (roughly one trading month) so the resample
    preserves volatility clustering.
    """
    rng = np.random.default_rng(seed)
    p = 1.0 / max(block, 1.0)
    idx = np.empty((B, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=B)
    newblock = rng.random((B, n)) < p
    starts = rng.integers(0, n, size=(B, n))
    for t in range(1, n):
        cont = (idx[:, t - 1] + 1) % n
        idx[:, t] = np.where(newblock[:, t], starts[:, t], cont)
    return idx


def model_confidence_set(losses: dict, alpha: float = 0.10, B: int = 1000,
                         block: float = 20.0, seed: int = 0) -> dict:
    """Hansen-Lunde-Nason (2011) Model Confidence Set via the range statistic T_R.

    `losses` maps model name -> per-observation loss array (all same length).
    Returns the set of models that cannot be distinguished from the best at
    confidence 1-alpha, plus each model's MCS p-value.

    Why the panel will like this: with a full alpha grid plus HAR-RV plus the
    Shannon baseline plus ablations, you are comparing ~10 models on one test
    set. Pairwise DM tests do not control that. The MCS does, and it produces
    the single cleanest sentence you can put in a defence:
    "the 90% MCS contains {HR-ETE-GNN(alpha*)} alone and excludes the Shannon
    ETE-GNN baseline."
    """
    names = list(losses)
    L = np.column_stack([np.asarray(losses[k], float).ravel() for k in names])
    ok = np.all(np.isfinite(L), axis=1)
    L = L[ok]
    n, M0 = L.shape
    if M0 < 2:
        raise ValueError("need at least 2 models")

    bidx = stationary_bootstrap_indices(n, B, block=block, seed=seed)
    Lb = L[bidx]                      # (B, n, M) bootstrap resamples
    mean_b = Lb.mean(axis=1)          # (B, M)

    alive = list(range(M0))
    pvals, elim_order = {}, []
    running_max = 0.0

    while len(alive) > 1:
        A = np.array(alive)
        m = L[:, A].mean(axis=0)                             # (k,)
        dbar = m[:, None] - m[None, :]                       # (k, k)
        mb = mean_b[:, A]                                    # (B, k)
        db = mb[:, :, None] - mb[:, None, :]                 # (B, k, k)
        var = np.maximum(((db - dbar) ** 2).mean(axis=0), _EPS)
        sd = np.sqrt(var)

        t_obs = dbar / sd
        t_boot = (db - dbar) / sd

        iu = np.triu_indices(len(A), k=1)
        TR = np.abs(t_obs[iu]).max()
        TR_boot = np.abs(t_boot[:, iu[0], iu[1]]).max(axis=1)
        p = float((TR_boot >= TR).mean())

        running_max = max(running_max, p)     # MCS p-values are monotone by construction
        worst_local = int(np.argmax(t_obs.max(axis=1)))   # largest loss vs. best rival
        worst = A[worst_local]
        pvals[names[worst]] = running_max
        elim_order.append(names[worst])
        alive.remove(worst)

    pvals[names[alive[0]]] = 1.0
    included = [k for k in names if pvals[k] > alpha]
    return {
        "included": included,
        "excluded": [k for k in names if k not in included],
        "p_values": pd.Series(pvals).reindex(names),
        "elimination_order": elim_order,
        "alpha": alpha, "B": B, "n": int(n),
    }


# --------------------------------------------------------------------------
# 6. Baselines you must beat
# --------------------------------------------------------------------------

def har_rv_forecast(rv: np.ndarray, split: int,
                    lags=(1, 5, 22), expanding: bool = True) -> np.ndarray:
    """Corsi (2009) HAR-RV: the referee benchmark in volatility forecasting.

    RV_{t+1} = b0 + b_d*RV_t + b_w*mean(RV_{t-4:t}) + b_m*mean(RV_{t-21:t}) + e

    `rv` is a 1-D series of realized volatility. `split` is the index of the
    first test observation *in the target array*. Fitted by OLS on the training
    part only; with `expanding=True` the coefficients are refit at every test
    date on all data up to that date (no look-ahead).

    Beat this, or the graph machinery has not earned its place in the thesis.
    """
    rv = np.asarray(rv, float).ravel()
    maxlag = max(lags)
    X, y = [], []
    for t in range(maxlag - 1, len(rv) - 1):
        X.append([1.0] + [rv[t - l + 1: t + 1].mean() for l in lags])
        y.append(rv[t + 1])
    X, y = np.array(X), np.array(y)

    preds = np.empty(len(y) - split)
    for i in range(split, len(y)):
        end = i if expanding else split
        beta, *_ = np.linalg.lstsq(X[:end], y[:end], rcond=None)
        preds[i - split] = X[i] @ beta
    return preds


def random_walk_forecast(rv: np.ndarray, split: int, maxlag: int = 22) -> np.ndarray:
    """RV_{t+1} = RV_t. Aligned with `har_rv_forecast`'s indexing."""
    rv = np.asarray(rv, float).ravel()
    X = np.array([rv[t] for t in range(maxlag - 1, len(rv) - 1)])
    return X[split:]


# --------------------------------------------------------------------------
# 7. Design: how big a difference can you even detect?
# --------------------------------------------------------------------------

def min_detectable_effect(n: int, alpha: float = 0.05, power: float = 0.80,
                          one_sided: bool = True) -> dict:
    """Minimum detectable standardized loss differential for a DM test.

    Returns the required mean(d)/sd(d) ratio. Run this BEFORE the experiment and
    put it in the methodology chapter -- it pre-empts the "your test set is too
    small to conclude anything" objection, or tells you honestly that it is.
    """
    za = stats.norm.ppf(1 - alpha) if one_sided else stats.norm.ppf(1 - alpha / 2)
    zb = stats.norm.ppf(power)
    return {
        "n": n,
        "mde_ratio_at_power": float((za + zb) / np.sqrt(n)),
        "mde_ratio_at_significance": float(za / np.sqrt(n)),
        "note": "multiply by sd(d) to get the MDE in loss units; "
                "HAC inflation makes the true requirement larger.",
    }


def seed_summary(losses_by_seed: dict, name: str = "model") -> pd.DataFrame:
    """Summarize mean loss across training seeds.

    `losses_by_seed` maps seed -> per-observation loss array.
    Report mean +/- sd across seeds. If the sd across seeds is of the same order
    as the gap between your model and the baseline, the gap is initialization
    noise and no DM test on a single run can rescue it.
    """
    rows = {s: float(np.nanmean(v)) for s, v in losses_by_seed.items()}
    s = pd.Series(rows)
    return pd.DataFrame({"model": name, "n_seeds": len(s), "mean": s.mean(),
                         "sd": s.std(ddof=1), "min": s.min(), "max": s.max()},
                        index=[name])
