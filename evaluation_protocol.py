"""
Evaluation protocol for HR-ETE-GNN vs. the Shannon ETE-GNN baseline.

Drop-in replacement for section 7-8 of HR_ETE_GNN_Pilot.ipynb. Assumes the
notebook has already defined: TICKERS, rv, log_ret, hurst_series, HRETEGNN,
build_erte_adjacency, LOOKBACK, and torch.

Design rules this file enforces (each one closes a hole a panellist can open):

  R1  Identical seeds across arms. The baseline and the treatment share the
      seed list, so any performance gap cannot be an initialisation artefact.
  R2  Multiple seeds. The unit of comparison is the seed-ensemble forecast,
      and the across-seed spread is reported next to the effect size.
  R3  No look-ahead in the graph. The ER-TE adjacency is estimated on the
      TRAINING window only, never on data that overlaps the test set.
  R4  A validation split for alpha. alpha is chosen on validation, never on
      test, otherwise every p-value on the test set is invalid.
  R5  Baselines. HAR-RV, random walk, and a no-graph ablation are evaluated on
      the exact same targets so the tests are like-for-like.
  R6  Pre-registered primary metric and test, declared before looking at
      results (see PRIMARY_* constants below).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from thesis_stats import (
    qlike, mse, mae,
    diebold_mariano, giacomini_white,
    benjamini_hochberg, model_confidence_set,
    har_rv_forecast, random_walk_forecast,
    min_detectable_effect,
)

# ---------------------------------------------------------------------------
# PRE-REGISTRATION BLOCK -- fill this in and freeze it BEFORE running anything.
# Put it verbatim in the methodology chapter. It is the single cheapest way to
# make the result defensible: it proves the test was not chosen after seeing
# which one gave the answer you wanted.
# ---------------------------------------------------------------------------
PRIMARY_METRIC = "QLIKE"                 # robust to noise in the RV proxy (Patton 2011)
PRIMARY_TEST = "Giacomini-White (2006), unconditional, HAC"
PRIMARY_ALTERNATIVE = "greater"          # one-sided: HR-ETE-GNN beats Shannon ETE-GNN
PRIMARY_ALPHA = 0.05                     # significance level
SECONDARY_FDR_Q = 0.10                   # BH across the 10 ETFs
MCS_CONFIDENCE = 0.90                    # Hansen-Lunde-Nason model confidence set
MIN_MEANINGFUL_IMPROVEMENT = 0.05        # 5% QLIKE reduction = smallest effect worth claiming
N_SEEDS = 20
LOSS_FN = qlike


# ---------------------------------------------------------------------------
# 1. Multi-seed training -- R1, R2
# ---------------------------------------------------------------------------

def train_multiseed(model_factory, A_np, X_train, y_train, X_val, X_test,
                    seeds=range(N_SEEDS), epochs=2000, lr=1e-2,
                    patience=100, verbose=False):
    """Train one architecture across `seeds` and return per-seed + ensemble forecasts.

    Note the epochs/lr defaults. The pilot used 50 full-batch steps at lr=1e-3,
    which is ~50 gradient updates in total: neither arm had converged, so the
    pilot compared two arbitrary points on two different optimisation
    trajectories. Early stopping on a validation split is what makes the
    comparison about the *graph* rather than about who got luckier with the
    learning-rate schedule.
    """
    import torch
    import torch.nn.functional as F

    A = torch.tensor(_row_normalize(A_np), dtype=torch.float32)
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    Xva = torch.tensor(X_val, dtype=torch.float32)
    Xte = torch.tensor(X_test, dtype=torch.float32)

    per_seed_test, per_seed_val, epochs_used = [], [], []
    for s in seeds:
        torch.manual_seed(s)            # R1: the seed is an explicit argument,
        np.random.seed(s)               #     not whatever the RNG happened to be at
        model = model_factory()
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        best, best_state, bad = np.inf, None, 0
        n_tr = len(Xtr)
        cut = int(0.85 * n_tr)          # inner split for early stopping
        Xa, ya, Xb, yb = Xtr[:cut], ytr[:cut], Xtr[cut:], ytr[cut:]

        for ep in range(epochs):
            model.train(); opt.zero_grad()
            loss = F.mse_loss(model(Xa, A), ya)
            loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                v = float(F.mse_loss(model(Xb, A), yb))
            if v < best - 1e-6:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            per_seed_test.append(model(Xte, A).numpy())
            per_seed_val.append(model(Xva, A).numpy())
        epochs_used.append(ep + 1)
        if verbose:
            print(f"  seed {s}: stopped at epoch {ep+1}, val MSE {best:.5f}")

    P = np.stack(per_seed_test)         # (S, n_test, n_nodes)
    return {
        "pred": P.mean(axis=0),         # R2: the ensemble is the object we test
        "pred_val": np.stack(per_seed_val).mean(axis=0),
        "per_seed": P,
        "seed_spread": float(P.mean(axis=(1, 2)).std(ddof=1)),
        "epochs_used": epochs_used,
    }


def _row_normalize(A):
    A = np.asarray(A, float).copy()
    s = A.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return A / s


# ---------------------------------------------------------------------------
# 2. Leak-free splits and adjacency -- R3, R4
# ---------------------------------------------------------------------------

def make_splits(rv_mat, lookback=20, train_frac=0.60, val_frac=0.20):
    """Chronological train / validation / test split of the supervised samples."""
    Xs, ys, ts = [], [], []
    for t in range(lookback, len(rv_mat) - 1):
        Xs.append(rv_mat[t - lookback:t].T)
        ys.append(rv_mat[t + 1])
        ts.append(t + 1)
    Xs, ys, ts = np.stack(Xs), np.stack(ys), np.array(ts)
    n = len(Xs)
    i_tr, i_va = int(train_frac * n), int((train_frac + val_frac) * n)
    return {
        "X_train": Xs[:i_tr], "y_train": ys[:i_tr],
        "X_val": Xs[i_tr:i_va], "y_val": ys[i_tr:i_va],
        "X_test": Xs[i_va:], "y_test": ys[i_va:],
        "t_train": ts[:i_tr], "t_val": ts[i_tr:i_va], "t_test": ts[i_va:],
        "i_train_end": i_tr, "i_val_end": i_va, "lookback": lookback,
    }


def adjacency_from_training_only(log_ret, tickers, rv_index, splits, alpha,
                                 build_fn, **kw):
    """Estimate the ER-TE graph on the training window ONLY -- R3.

    The pilot built the adjacency from `log_ret.iloc[-600:]`, which overlaps the
    314-day test set. That is look-ahead bias: the graph the model uses to
    forecast the test period was partly *estimated from* the test period. It
    is the first thing a quantitative panellist checks, and on its own it
    invalidates any significance claim.
    """
    last_train_date = rv_index[splits["t_train"][-1]]
    train_returns = log_ret.loc[:last_train_date, tickers]
    return build_fn(train_returns, tickers, alpha=alpha, **kw)


# ---------------------------------------------------------------------------
# 3. The comparison itself
# ---------------------------------------------------------------------------

def compare(actual, forecasts: dict, regime=None, loss_fn=LOSS_FN,
            baseline_key="Shannon ETE-GNN", pooled=True):
    """Run the full significance battery.

    actual    : (n_test, n_nodes) realised RV
    forecasts : {model_name: (n_test, n_nodes) forecast}
    regime    : optional (n_test,) Hurst-regime indicator, KNOWN AT TIME t
                (use hurst_series shifted by one day), 1 = anti-persistent/crisis

    Returns a dict of DataFrames ready to paste into the results chapter.
    """
    actual = np.asarray(actual, float)
    losses = {k: loss_fn(actual, np.asarray(v, float)) for k, v in forecasts.items()}
    pooled_losses = {k: v.mean(axis=1) for k, v in losses.items()}   # avg over ETFs, per day

    # --- headline table ------------------------------------------------
    head = pd.DataFrame({
        k: {"loss": float(np.nanmean(v)),
            "RMSE": float(np.sqrt(np.nanmean(mse(actual, forecasts[k])))),
            "MAE": float(np.nanmean(mae(actual, forecasts[k])))}
        for k, v in losses.items()}).T.sort_values("loss")

    out = {"summary": head}

    # --- pooled DM + GW vs. the baseline -------------------------------
    if baseline_key in losses:
        rows = {}
        for k in losses:
            if k == baseline_key:
                continue
            dm = diebold_mariano(pooled_losses[baseline_key], pooled_losses[k],
                                 alternative=PRIMARY_ALTERNATIVE)
            gw = giacomini_white(pooled_losses[baseline_key], pooled_losses[k])
            rows[k] = {"DM_stat": dm["stat"], "DM_p": dm["p_value"],
                       "GW_stat": gw["stat"], "GW_p": gw["p_value"],
                       "improvement_%": dm["pct_improvement"],
                       "meets_min_effect":
                           dm["pct_improvement"] / 100 >= MIN_MEANINGFUL_IMPROVEMENT}
        out["pooled_vs_baseline"] = pd.DataFrame(rows).T

        # --- per-ETF DM with BH-FDR ------------------------------------
        per = {}
        for k in losses:
            if k == baseline_key:
                continue
            ps = {}
            for j in range(actual.shape[1]):
                ps[j] = diebold_mariano(losses[baseline_key][:, j], losses[k][:, j],
                                        alternative=PRIMARY_ALTERNATIVE)["p_value"]
            bh = benjamini_hochberg(pd.Series(ps), q=SECONDARY_FDR_Q)
            per[k] = bh
        out["per_etf"] = per

        # --- regime-conditional GW: THE headline test for this thesis ---
        if regime is not None:
            reg = np.asarray(regime, float).ravel()
            rows = {}
            for k in losses:
                if k == baseline_key:
                    continue
                g = giacomini_white(pooled_losses[baseline_key], pooled_losses[k],
                                    instruments=reg)
                rows[k] = {"GW_cond_stat": g["stat"], "GW_cond_p": g["p_value"],
                           "beta_const": g["coef"]["const"], "t_const": g["t"]["const"],
                           "beta_regime": g["coef"]["z1"], "t_regime": g["t"]["z1"],
                           "p_regime": g["p_coef"]["z1"]}
            out["regime_conditional"] = pd.DataFrame(rows).T

            # regime-stratified losses, with the honest n in each bucket
            strat = {}
            for k in losses:
                for lbl, msk in [("calm", reg < 0.5), ("crisis", reg >= 0.5)]:
                    strat[(k, lbl)] = {"loss": float(np.nanmean(pooled_losses[k][msk])),
                                       "n_days": int(msk.sum())}
            out["regime_stratified"] = pd.DataFrame(strat).T

    # --- model confidence set ------------------------------------------
    if len(pooled_losses) >= 2:
        out["mcs"] = model_confidence_set(pooled_losses,
                                          alpha=1 - MCS_CONFIDENCE, B=2000)

    out["power"] = min_detectable_effect(actual.shape[0], alpha=PRIMARY_ALPHA)
    return out


def report(res: dict) -> None:
    """Print the battery in the order you should present it to the panel."""
    print("=" * 72)
    print(f"PRIMARY METRIC : {PRIMARY_METRIC}")
    print(f"PRIMARY TEST   : {PRIMARY_TEST}, one-sided at {PRIMARY_ALPHA}")
    print(f"MIN EFFECT     : {MIN_MEANINGFUL_IMPROVEMENT:.0%} loss reduction")
    print("=" * 72)
    print("\n[1] Forecast accuracy\n", res["summary"].round(5))
    if "pooled_vs_baseline" in res:
        print("\n[2] Pooled predictive-ability tests vs. baseline\n",
              res["pooled_vs_baseline"].round(4))
    if "regime_conditional" in res:
        print("\n[3] Regime-conditional GW  (beta_regime > 0 and significant =")
        print("    the Renyi graph helps SPECIFICALLY in crisis regimes ->")
        print("    this is the thesis hypothesis, tested directly)\n",
              res["regime_conditional"].round(4))
        print("\n[3b] Regime-stratified loss\n", res["regime_stratified"].round(5))
    if "per_etf" in res:
        print("\n[4] Per-ETF DM with BH-FDR control")
        for k, bh in res["per_etf"].items():
            print(f"  {k}: {int(bh['reject'].sum())}/{len(bh)} ETFs significant "
                  f"after FDR at q={SECONDARY_FDR_Q}")
    if "mcs" in res:
        m = res["mcs"]
        print(f"\n[5] {MCS_CONFIDENCE:.0%} Model Confidence Set")
        print("  included:", m["included"])
        print("  excluded:", m["excluded"])
        print("  MCS p-values:", m["p_values"].round(3).to_dict())
    p = res["power"]
    print(f"\n[6] Power: with n={p['n']} test days you can only detect a "
          f"standardized\n    loss differential of {p['mde_ratio_at_power']:.3f} "
          f"at 80% power. If the observed\n    mean(d)/sd(d) is below that, the "
          f"test set is too short to conclude.")
