"""
Shared code for the per-alpha notebooks in this folder.

Each run_alpha_*.ipynb runs the full 2006-2024 walk-forward for ONE Renyi order
with the same 10 seeds, and compares it with the Shannon baseline (alpha = 1,
run_alpha_1.0.ipynb).  Nothing about the experiment is defined here:
configuration, data, estimator, the walk-forward and deterministic mode all come
unchanged from ../hrete_full.py, and the checkpoints go to ../runs/full/.

The comparison code is section 6 of run_full.ipynb, moved into functions so the
per-alpha notebooks and compare_all_alphas.ipynb share one copy of it.

Only alpha* (= 0.7, chosen on validation in run_full.ipynb) carries the
confirmatory test.  Every other order is a sensitivity result (manuscript
Section 3.5: alpha = 0.3 and 0.5 sit outside the selection grid and are
appendix-only).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent            # final_results/alpha_runs
FINAL = HERE.parent                               # final_results
ROOT = FINAL.parent                               # repo root
for p in (str(ROOT), str(FINAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

import hetegnn_base as hb  # noqa: E402
import hrete_full as hf  # noqa: E402
import thesis_stats as ts  # noqa: E402

BASELINE = 1.0
ALPHA_STAR = float(json.load(open(FINAL / "tables" / "alpha_star.json"))["alpha_star"])
# Every order with (or planned to have) a full run, in display order.
ALL_ALPHAS: tuple = (0.5, 0.7, 0.9, 1.0, 1.5, 2.0)

# The settings run_full.ipynb picked (its section 3 printed: device=cuda
# n_jobs=6 torch threads per run=2).  Every alpha must use the same device:
# cpu and cuda kernels round differently, so the same seed gives a different
# model on each (val MSE 0.744 on cpu vs 1.311 on deterministic cuda for
# alpha = 1, seed 0).  Mixing devices across alphas would not be like for like.
DEVICE = "cuda"
N_JOBS = 6
TORCH_THREADS = 2


def run_overrides(device: str = DEVICE, torch_threads: int = TORCH_THREADS,
                  **extra) -> dict:
    return dict(extra, device=device, torch_threads=torch_threads)


def panel_sha256(path: Path = hf.DATA_FILE) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def specs(alpha: float, seeds: Sequence[int] = hf.SEEDS,
          root: Path = hf.RUNS_DIR) -> list:
    return [hf.RunSpec("full", float(alpha), s, str(root)) for s in seeds]


def status(alphas: Sequence[float] = ALL_ALPHAS, seeds: Sequence[int] = hf.SEEDS,
           root: Path = hf.RUNS_DIR) -> pd.DataFrame:
    """How many of the seeds each alpha has on disk."""
    rows = []
    for a in alphas:
        found = [s for s in specs(a, seeds, root) if s.path.exists()]
        done = [s.seed for s in found]
        det = sum(hf.load_run(s)["deterministic"] for s in found)
        rows.append(dict(alpha=a, model=hf.arm_label(a), seeds_done=len(done),
                         of=len(seeds), complete=len(done) == len(seeds),
                         deterministic=det, missing=[s for s in seeds if s not in done]))
    return pd.DataFrame(rows).set_index("alpha")


def load(alpha: float, seeds: Sequence[int] = hf.SEEDS,
         root: Path = hf.RUNS_DIR) -> Dict[int, dict]:
    runs = {s.seed: hf.load_run(s) for s in specs(alpha, seeds, root)}
    bad = {s: (r["deterministic"], r["device"]) for s, r in runs.items()
           if not r["deterministic"] or r["device"] != DEVICE}
    if bad:
        raise RuntimeError(
            f"alpha={alpha:g}: seeds {sorted(bad)} were not made in deterministic mode on "
            f"{DEVICE} ({bad}).  Delete or archive those files and re-run them.")
    return runs


def qlike_mean(y, p) -> float:
    return float(ts.qlike(y, p, warn=False).mean())


# =============================================================================
# Full-span replication metrics (run_full.ipynb section 6.1)
# =============================================================================

PAPER_COLS = ["RMSE", "MAE", "MAPE", "Correlation", "HitRatio"]


def per_seed_metrics(runs_by_alpha: Dict[float, Dict[int, dict]]) -> pd.DataFrame:
    rows = []
    for a, runs in runs_by_alpha.items():
        for s, r in runs.items():
            m = hb.metrics(r["y_true"], r["y_pred"])
            m.update(alpha=a, Model=hf.arm_label(a), seed=s,
                     QLIKE=qlike_mean(r["y_true"], r["y_pred"]),
                     refits=r["n_refits"], minutes=r["seconds"] / 60)
            rows.append(m)
    return pd.DataFrame(rows)


def paper_table(per_seed: pd.DataFrame, with_paper: bool = True) -> pd.DataFrame:
    """Mean +/- SD over seeds, laid out like the paper's Table 6."""
    cols = PAPER_COLS + ["QLIKE"]
    agg = per_seed.groupby(["alpha", "Model"])[cols].agg(["mean", "std"])
    agg = agg.loc[sorted(agg.index, key=lambda ix: (ix[0] != BASELINE, ix[0]))]  # Shannon first
    fmt = pd.DataFrame({c: agg[(c, "mean")].map("{:.4f}".format) + " ± "
                        + agg[(c, "std")].map("{:.4f}".format) for c in cols})
    fmt.index = agg.index.get_level_values("Model")
    if with_paper:
        paper = hb.PAPER_TABLE6.query("Regime == 'Hurst' and Model == 'ETE-GNN'")[PAPER_COLS]
        fmt.loc["Lee & Cho (2025), H-ETE-GNN"] = (
            paper.iloc[0].map("{:.4f}".format).tolist() + ["-"])
    return fmt


# =============================================================================
# Test-span comparison against Shannon (run_full.ipynb section 6.2)
# =============================================================================

def ensemble(runs: Dict[int, dict], seeds: Sequence[int] = hf.SEEDS):
    """Seed-mean forecast: removes initialisation noise."""
    rs = [runs[s] for s in seeds]
    return rs[0]["dates"], rs[0]["y_true"], np.mean([r["y_pred"] for r in rs], axis=0)


def test_losses(runs: Dict[int, dict], seeds: Sequence[int] = hf.SEEDS):
    """Per-day, per-ETF QLIKE and squared error of the ensemble on the test span."""
    dates, y, p = ensemble(runs, seeds)
    test = dates > pd.Timestamp(hf.VAL_END)
    yt, pt = y[test], p[test]
    return dict(dates=dates[test], y=yt, p=pt,
                qlike=ts.qlike(yt, pt, warn=False), se=(pt - yt) ** 2)


def compare_to_shannon(alpha: float, base_runs: Dict[int, dict],
                       new_runs: Dict[int, dict], data: dict,
                       tickers: Sequence[str],
                       seeds: Sequence[int] = hf.SEEDS) -> dict:
    """Section 6.2 of run_full.ipynb for an arbitrary alpha.

    d = loss(H-ETE-GNN) - loss(HR-ETE-GNN(alpha)); d > 0 means Renyi is better.
    """
    B, N = test_losses(base_runs, seeds), test_losses(new_runs, seeds)
    assert (B["dates"] == N["dates"]).all() and np.array_equal(B["y"], N["y"]), \
        "baseline and alpha runs are not on the same forecast days"
    dt = B["dates"]
    Lb, Ln = B["qlike"], N["qlike"]
    lb, ln = Lb.mean(axis=1), Ln.mean(axis=1)

    gw = ts.giacomini_white(lb, ln)
    dm = ts.diebold_mariano(lb, ln, alternative="greater")
    dm_mse = ts.diebold_mariano(B["se"].mean(axis=1), N["se"].mean(axis=1),
                                alternative="greater")
    lab = (data["labels"].map({"Above": 1.0, "Below": 0.0}).shift(1)
           .reindex(dt).ffill().bfill())
    gw_reg = ts.giacomini_white(lb, ln, instruments=lab.values)
    gw_p1 = gw["p_value"] / 2 if dm["mean_diff"] > 0 else 1 - gw["p_value"] / 2

    label = hf.arm_label(alpha)
    headline = pd.DataFrame([
        ("QLIKE, H-ETE-GNN", lb.mean()), (f"QLIKE, {label}", ln.mean()),
        ("QLIKE improvement, %", dm["pct_improvement"]),
        ("GW unconditional: Wald stat", gw["stat"]),
        ("GW unconditional: p (two-sided)", gw["p_value"]),
        ("GW unconditional: p (one-sided)", gw_p1),
        ("DM (HLN) on QLIKE: stat", dm["stat"]),
        ("DM (HLN) on QLIKE: p one-sided", dm["p_value"]),
        ("MSE improvement, %", dm_mse["pct_improvement"]),
        ("DM on MSE: p one-sided", dm_mse["p_value"]),
        ("GW with lagged regime: Wald stat", gw_reg["stat"]),
        ("GW with lagged regime: p", gw_reg["p_value"]),
        ("GW regime coefficient", gw_reg["coef"].iloc[1]),
        ("GW regime coefficient p", gw_reg["p_coef"].iloc[1]),
    ], columns=["quantity", "value"]).set_index("quantity")

    per_etf = []
    for k, t in enumerate(tickers):
        d = ts.diebold_mariano(Lb[:, k], Ln[:, k], alternative="greater")
        per_etf.append(dict(ETF=t, QLIKE_base=Lb[:, k].mean(), QLIKE_new=Ln[:, k].mean(),
                            improvement_pct=d["pct_improvement"], DM_stat=d["stat"],
                            p=d["p_value"]))
    per_etf = pd.DataFrame(per_etf)
    bh = ts.benjamini_hochberg(per_etf["p"].values, q=0.10)
    per_etf = pd.concat([per_etf, bh.drop(columns=["p_value"]).reset_index(drop=True)],
                        axis=1)

    test = base_runs[seeds[0]]["dates"] > pd.Timestamp(hf.VAL_END)
    seed_tab = pd.DataFrame([dict(
        seed=s,
        QLIKE_base=qlike_mean(base_runs[s]["y_true"][test], base_runs[s]["y_pred"][test]),
        QLIKE_new=qlike_mean(new_runs[s]["y_true"][test], new_runs[s]["y_pred"][test]))
        for s in seeds])
    seed_tab["new_wins"] = seed_tab.QLIKE_new < seed_tab.QLIKE_base

    summary = dict(alpha=alpha, Model=label,
                   test_QLIKE=ln.mean(), test_QLIKE_shannon=lb.mean(),
                   QLIKE_improvement_pct=dm["pct_improvement"],
                   MSE_improvement_pct=dm_mse["pct_improvement"],
                   GW_stat=gw["stat"], GW_p_two_sided=gw["p_value"], GW_p_one_sided=gw_p1,
                   DM_p_one_sided=dm["p_value"], GW_regime_p=gw_reg["p_value"],
                   seeds_won=int(seed_tab.new_wins.sum()), n_seeds=len(seeds),
                   ETFs_won_BH=int(per_etf["reject"].sum()))
    return dict(headline=headline, per_etf=per_etf, per_seed=seed_tab,
                summary=summary, dates=dt, daily_base=lb, daily_new=ln)


def plot_cumulative(res: dict, alpha: float, changes: pd.DatetimeIndex, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3.8))
    dt = res["dates"]
    ax.plot(dt, np.cumsum(res["daily_base"] - res["daily_new"]), color="#1a1a2e", lw=1)
    ax.axhline(0, color="k", lw=.6)
    for d in changes[changes > dt[0]]:
        ax.axvline(d, color="#f39c12", lw=.4, alpha=.6)
    ax.set_title(f"Cumulative QLIKE advantage of {hf.arm_label(alpha)} over H-ETE-GNN "
                 "(up = Rényi better)")
    return ax


def check_device(device: str = DEVICE) -> None:
    """Stop rather than silently fall back to a device the baseline did not use."""
    import torch
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available.  Every alpha must run on the same device, and "
            "the plan is cuda; a cpu run would not be like for like with the others.  "
            "Fix the GPU before going on.")
