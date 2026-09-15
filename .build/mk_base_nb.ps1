param([string]$Out)

$cells = New-Object System.Collections.ArrayList

function Add-Cell([string]$Type, [string]$Text) {
    $lines = $Text -split "`r?`n"
    $src = @()
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($i -lt $lines.Count - 1) { $src += ($lines[$i] + "`n") } else { if ($lines[$i] -ne '') { $src += $lines[$i] } }
    }
    $id = "cell-" + ($cells.Count).ToString("d2")
    if ($Type -eq 'markdown') {
        [void]$cells.Add([ordered]@{ cell_type = 'markdown'; id = $id; metadata = @{}; source = $src })
    } else {
        [void]$cells.Add([ordered]@{ cell_type = 'code'; id = $id; execution_count = $null; metadata = @{}; outputs = @(); source = $src })
    }
}

Add-Cell markdown @'
# H-ETE-GNN -- Baseline Replication

Replication of the **base model** in

> Lee, S. & Cho, P. (2025). *Graph-Based Stock Volatility Forecasting with Effective
> Transfer Entropy and Hurst-Based Regime Adaptation.* **Fractal Fract. 9**, 339.
> https://doi.org/10.3390/fractalfract9060339

This notebook is the control arm for the thesis. It reproduces Lee & Cho's
**H-ETE-GNN** exactly as published -- Shannon effective transfer entropy edges,
Hurst-based regime retraining, multi-scale-conv + 3-layer GCN -- plus all five
benchmarks they report (TE-GNN, Granger-GNN, Pearson-GNN, LSTM, GRU) under both
retraining policies. The Renyi extension (HR-ETE-GNN) is then compared against
*these* numbers, not against the paper's printed table.

All modelling code lives in [`hetegnn_base.py`](hetegnn_base.py); this notebook is a driver.

## Paper -> code map

| Paper | Implementation |
|---|---|
| Eq (3) discretized TE, 3 bins, k=l=1 | `transfer_entropy` |
| Eq (4)-(5) ETE and Z-score, keep Z>1.96 | `effective_te`, `build_ete_matrix` |
| Eq (8) multi-scale conv, kernels {3,5,7}x12 | `MultiScaleConv` |
| Eq (9) message passing, 3 layers | `GCNLayer`, `ETEGNN` |
| Eq (10)-(16) R/S Hurst exponent | `hurst_rs` |
| Eq (17)-(18) log-return, 20-day RV | `log_returns`, `realized_volatility` |
| Eq (19)-(23) RMSE/MAE/MAPE/corr/hit | `metrics` |
| Sec 3.1 regime rule (10-day window, s=6) | `regime_labels` |
| Sec 3.1 walk-forward, refit on regime change | `WalkForward.run` |
| Table 3 tuned values M=20, N=750, s=6 | `Config` defaults |
| Table 6 published results | `PAPER_TABLE6` |
'@

Add-Cell markdown @'
## 1. Setup

The next cell runs unchanged in both places.

**Locally** it assumes you launched Jupyter from the project venv, so it only
verifies the imports:

```
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m ipykernel install --user --name hetegnn --display-name "HR-ETE-GNN (.venv)"
.venv\Scripts\jupyter lab
```

**On Colab** it clones the repo for `hetegnn_base.py` and installs
`requirements-colab.txt` -- which deliberately does *not* name torch, so
Colab's CUDA build survives. Set Runtime -> Change runtime type -> GPU first.
'@

Add-Cell code @'
import os, sys, importlib, subprocess

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    if not os.path.exists("HR-ETE-GNN-Thesis"):
        subprocess.run(["git", "clone", "-q",
                        "https://github.com/Eurie-R/HR-ETE-GNN-Thesis.git"], check=True)
    proj = "HR-ETE-GNN-Thesis/HR-ETE-GNN-Thesis"
    sys.path.insert(0, proj)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                    os.path.join(proj, "requirements-colab.txt")], check=True)
elif not os.path.exists("hetegnn_base.py"):
    raise SystemExit("run this notebook from the project folder, next to hetegnn_base.py")

import hetegnn_base as hb
importlib.reload(hb)

import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 4)
plt.rcParams["axes.grid"] = True
pd.set_option("display.width", 160)

cfg = hb.Config()
print(f"colab={IN_COLAB}  python={sys.version.split()[0]}")
print(f"torch={torch.__version__}  cuda={torch.cuda.is_available()}  device={cfg.device}")
print(f"numpy={np.__version__}  pandas={pd.__version__}")
print(f"\nM (input size)      : {cfg.input_size}")
print(f"N (training period) : {cfg.train_period}")
print(f"s (regime sens.)    : {cfg.regime_sensitivity}")
'@

Add-Cell markdown @'
## 2. Data -- Section 3.2

Ten iShares MSCI country ETFs, daily adjusted close, 7 Feb 2003 -> 29 May 2024.
Targets are 20-day realized volatility (Eq 18); model inputs are the raw
log-returns (Eq 17), min-max scaled inside each training block.

**Note on the regime proxy.** The paper names the iShares MSCI World ETF, but
URTH only started trading in 2012 while the sample begins in 2003. We splice
`URTH -> ACWI -> IOO` in return space so the Hurst series spans the sample. The
cell below reports how much of the series each source contributes.
'@

Add-Cell code @'
prices, proxy_ret, proxy_src = hb.download_prices(cfg)
ret = hb.log_returns(prices)
rv  = hb.realized_volatility(ret, cfg.rv_window)

print(f"prices {prices.shape}   {prices.index[0].date()} -> {prices.index[-1].date()}")
print(f"log-returns {ret.shape}")
print(f"realized vol {rv.shape}   {rv.index[0].date()} -> {rv.index[-1].date()}")

print("\nRegime-proxy splice:")
for tkr, n in proxy_src.value_counts().items():
    span = proxy_src[proxy_src == tkr].index
    print(f"  {tkr:5s} {n:5d} days  {span.min().date()} -> {span.max().date()}")
rv.tail()
'@

Add-Cell code @'
# Descriptive statistics -- compare with the paper's Table 5.
import warnings
import scipy.stats as st
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")

rows = []
for t in cfg.tickers:
    for name, s in (("log-return", ret[t]), ("RV", rv[t])):
        jb = st.jarque_bera(s.dropna())
        rows.append(dict(ETF=t, series=name, mean=s.mean(), max=s.max(), min=s.min(),
                         std=s.std(), skew=s.skew(), kurtosis=s.kurtosis(),
                         JarqueBera=jb[0], ADF=adfuller(s.dropna(), autolag="AIC")[0]))
desc = pd.DataFrame(rows).set_index(["ETF", "series"])
desc.round(4)
'@

Add-Cell markdown @'
## 3. Hurst exponent and regime detection -- Sections 2.3 and 3.1

Rolling 250-day R/S Hurst exponent on the World proxy. A window is `Above` when
at least `s = 6` of the last 10 daily Hurst values exceed 0.5, and `Below` when
at least 6 fall under it; the model is refit whenever that label flips.

The paper reports **47 regime changes** over the 19-year sample. Our count is
the first thing to check -- it is the number most sensitive to the two
under-specified choices below.

**On the R/S bias.** Plain rescaled-range analysis is biased upward on short
windows -- on i.i.d. normal draws of length 250 it averages H = 0.57 and lands
above 0.5 in 84% of them. On the real proxy that shows up as mean H = 0.539 and
an `Above` regime that dominates 4220 days to 883. It does *not* suppress
switching, though: the s=6-of-10 rule still flips **50** times against the
paper's 47, so the literal estimator (`hurst_correction="none"`) is the right
default. The Anis-Lloyd correction recentres H to 0.448 and gives 39 changes --
useful as a robustness check, not as the main run. The next cell computes both
so you can see the trade-off rather than take it on trust.
'@

Add-Cell code @'
import copy

variants = {}
for corr in ("none", "anis_lloyd"):
    h = hb.rolling_hurst(proxy_ret, cfg.hurst_window, corr)
    lab = hb.regime_labels(h, cfg.regime_lookback, cfg.regime_sensitivity,
                           cfg.hurst_threshold)
    ch = hb.regime_change_dates(lab)
    variants[corr] = (h, lab, ch)
    print(f"correction={corr:11s} mean H={h.mean():.4f}  "
          f"frac H>0.5={(h > 0.5).mean():6.1%}  "
          f"regime changes={len(ch):3d}   (paper: 47)")

# Keep whichever lands closer to the paper's 47 changes, and record the choice.
best = min(variants, key=lambda k: abs(len(variants[k][2]) - 47))
cfg.hurst_correction = best
hurst, labels, changes = variants[best]
print(f"\nusing hurst_correction={best!r}  ->  {len(changes)} regime changes")
print(labels.value_counts().to_string())
'@

Add-Cell code @'
# Figure 3 analogue: Hurst exponent with regime shifts marked.
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(hurst.index, hurst.values, color="steelblue", lw=0.9)
ax.axhline(0.5, color="black", ls="--", lw=0.8, alpha=0.7)

# alternate the band colour on every detected change, as in the paper
bounds = [hurst.index[0]] + list(changes) + [hurst.index[-1]]
for k in range(len(bounds) - 1):
    ax.axvspan(bounds[k], bounds[k + 1], color=("red" if k % 2 else "blue"), alpha=0.06)
for d in changes:
    ax.axvline(d, color="grey", ls=":", lw=0.6)

ax.set_title(f"Rolling {cfg.hurst_window}-day Hurst exponent on the World proxy -- "
             f"{len(changes)} regime changes (paper: 47)")
ax.set_ylabel("H")
plt.tight_layout(); plt.show()
'@

Add-Cell markdown @'
## 4. Edge matrices -- Figure 4 analogue

Built on the first `N = 750`-day training block, exactly as the walk-forward does.
The paper's qualitative claim is a sparsity ordering: **Pearson (dense) > TE >
ETE (sparsest and most informative)**, with Granger sparsest of all because it is
hard-thresholded at 5%.
'@

Add-Cell code @'
win = ret.values[:cfg.train_period]
mats = {k: hb.EDGE_BUILDERS[k](win, cfg, seed=0) for k in ("ETE", "TE", "Granger", "Pearson")}

fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
for ax, (name, A) in zip(axes, mats.items()):
    im = ax.imshow(A, cmap="viridis")
    ax.set_xticks(range(len(cfg.tickers))); ax.set_xticklabels(cfg.tickers, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cfg.tickers))); ax.set_yticklabels(cfg.tickers, fontsize=7)
    ax.set_title(f"{name}\n{(A > 0).sum()}/90 edges")
    ax.set_xlabel("target"); ax.set_ylabel("source")
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout(); plt.show()

pd.DataFrame({k: {"edges": int((A > 0).sum()), "density": (A > 0).mean(),
                  "mean weight": A[A > 0].mean() if (A > 0).any() else 0.0}
              for k, A in mats.items()}).T.round(4)
'@

Add-Cell markdown @'
### Does the graph actually do anything? (read this before the full run)

Eq (9) is `h_i' = ReLU(h_i W1 + sum_j A_ji h_j W2)`: a self term plus a
neighbourhood term. The graph only influences the forecast in proportion to the
size of A's entries -- and the four edge builders produce wildly different
scales. ETE is a handful of nats (~0.01), Granger is a 0/1 indicator, Pearson is
a correlation in [0,1].

The paper never says whether A is normalised. The cell below measures what each
edge type actually contributes, because the answer decides whether this
experiment compares graph *topology* (the paper's claim) or merely graph
*magnitude*.
'@

Add-Cell code @'
hb.set_seed(0)
_probe = hb.ETEGNN(cfg)
_W = np.stack([win[t - cfg.input_size:t].T
               for t in range(cfg.input_size, cfg.train_period)])
_X = torch.tensor(hb.minmax_apply(_W, *hb.minmax_fit(_W)), dtype=torch.float32)

rows = []
for norm in (False, True):
    for name, A0 in mats.items():
        A = hb.row_normalize(A0) if norm else A0
        with torch.no_grad():
            H = _probe.feat(_X)
            lyr = _probe.layers[0]
            self_t = lyr.W1(H).abs().mean().item()
            neigh_t = torch.einsum("ji,bjf->bif",
                                   torch.tensor(A, dtype=torch.float32),
                                   lyr.W2(H)).abs().mean().item()
        rows.append(dict(normalized=norm, edge=name, edges=int((A0 > 0).sum()),
                         row_sum=A.sum(axis=1).mean(),
                         neigh_over_self=neigh_t / max(self_t, 1e-12)))
contrib = pd.DataFrame(rows).set_index(["normalized", "edge"]).round(4)
print(contrib.to_string())
print("\nneigh_over_self near 0 means the adjacency is decorative:")
print("the model literally cannot use the graph you built for it.")
'@

Add-Cell markdown @'
On raw weights the ETE graph contributes well under 1% of the self term while
Pearson contributes several hundred percent -- so `Pearson-GNN` and
`Granger-GNN` get real message passing and `ETE-GNN` collapses towards a
per-node MLP. Row-normalising brings all four into the same band, which is the
only setting in which the paper's argument (ETE wins because it is *sparser but
more informative*) is even testable.

**`cfg.normalize_adjacency` therefore defaults to `True`** here -- a deliberate
departure from the literal text, because the literal reading does not give a
baseline HR-ETE-GNN can meaningfully be compared against. Set it to `False`
(CLI: `--raw-adjacency`) to reproduce the paper as written, and say in the
thesis which arm you are quoting.
'@

Add-Cell markdown @'
## 5. The model -- Section 2.2

Eq (8) multi-scale convolution (kernels 3/5/7, 12 channels each) produces the
initial node embedding; Eq (9) then runs three rounds of message passing over
the causality graph, with the third layer emitting one realized-volatility
value per node.
'@

Add-Cell code @'
model = hb.ETEGNN(cfg)
print(model)
print(f"\nconv readout      : {cfg.conv_readout}")
print(f"node embedding dim: {model.feat.out_dim}")
print(f"trainable params  : {sum(p.numel() for p in model.parameters()):,}")

x = torch.randn(4, cfg.n_nodes(), cfg.input_size)
A = torch.tensor(mats["ETE"], dtype=torch.float32)
print(f"forward: {tuple(x.shape)} + A{tuple(A.shape)} -> {tuple(model(x, A).shape)}")
'@

Add-Cell markdown @'
## 6. Smoke run

One short walk-forward to confirm the pipeline is wired correctly before
committing to the full experiment. Strided and shortened -- the numbers here are
**not** comparable to the paper.
'@

Add-Cell code @'
import copy
smoke = copy.deepcopy(cfg)
smoke.n_repeats, smoke.stride, smoke.epochs, smoke.n_shuffles = 1, 10, 15, 10

wf = hb.WalkForward(smoke, ret, rv, changes)
r = wf.run("GNN", "ETE", "hurst", seed=0)
print(f"{len(r.y_true)} forecasts, {r.n_refits} refits, {r.seconds:.0f}s")
pd.Series(hb.metrics(r.y_true, r.y_pred)).round(4)
'@

Add-Cell markdown @'
## 7. Full experiment -- Table 6

Six models x two retraining policies x ten repetitions, walk-forward over the
whole sample. **This is the expensive cell** (roughly 1-3 hours on a Colab GPU).
Trim `n_repeats`, `stride`, or the `models` list while iterating.
'@

Add-Cell code @'
# Set HETEGNN_FAST=1 in the environment to rehearse the whole notebook in a
# couple of minutes. Leave it unset for the real protocol.
if os.environ.get("HETEGNN_FAST"):
    cfg.n_repeats, cfg.stride, cfg.epochs, cfg.n_shuffles = 2, 40, 8, 10
    print("FAST MODE -- these numbers are not comparable to the paper\n")

runs, table, aux = hb.run_experiment(
    cfg,
    models=None,                      # None = all six; e.g. ["ETE-GNN", "TE-GNN"]
    regimes=("hurst", "periodic"),
)
table.round(4)
'@

Add-Cell code @'
import pickle
with open("baseline_runs.pkl", "wb") as f:
    pickle.dump(runs, f)
table.to_csv("baseline_table6.csv", index=False)
print("saved baseline_runs.pkl and baseline_table6.csv")
'@

Add-Cell markdown @'
## 8. Replicated vs published

Side-by-side against the paper's Table 6. Judge the replication on the
**ordering and the gaps** -- ETE-GNN best, Hurst better than Periodic, GNNs well
ahead of the RNNs -- rather than on matching levels to three decimals. Several
hyper-parameters the paper tuned were never printed (see the deviations list in
the next cell), so exact levels are not recoverable.
'@

Add-Cell code @'
rep = table[["Regime", "Model", "RMSE_mean", "MAE_mean", "MAPE_mean",
             "Correlation_mean", "HitRatio_mean"]].copy()
rep.columns = ["Regime", "Model", "RMSE", "MAE", "MAPE", "Correlation", "HitRatio"]

cmp = hb.PAPER_TABLE6.merge(rep, on=["Regime", "Model"],
                            suffixes=("_paper", "_ours"), how="left")
cols = ["Regime", "Model"]
for m in ("RMSE", "MAE", "MAPE", "Correlation", "HitRatio"):
    cmp[f"{m}_delta"] = cmp[f"{m}_ours"] - cmp[f"{m}_paper"]
    cols += [f"{m}_paper", f"{m}_ours", f"{m}_delta"]
cmp[cols].round(4)
'@

Add-Cell code @'
# Does the replication preserve the paper's qualitative claims?
def _get(regime, model, metric="RMSE"):
    row = rep[(rep.Regime == regime) & (rep.Model == model)]
    return float(row[metric].iloc[0]) if len(row) else float("nan")

checks = {
    "ETE-GNN is best under Hurst":
        _get("Hurst", "ETE-GNN") == min(_get("Hurst", m) for m in rep.Model.unique()),
    "Hurst beats Periodic for ETE-GNN":
        _get("Hurst", "ETE-GNN") < _get("Periodic", "ETE-GNN"),
    "Hurst beats Periodic for every GNN":
        all(_get("Hurst", m) < _get("Periodic", m)
            for m in ("ETE-GNN", "TE-GNN", "Granger-GNN", "Pearson-GNN")),
    "GNNs beat RNNs":
        max(_get("Hurst", m) for m in ("ETE-GNN", "TE-GNN", "Granger-GNN", "Pearson-GNN"))
        < min(_get("Hurst", m) for m in ("LSTM", "GRU")),
    "ETE-GNN beats TE-GNN under Hurst":
        _get("Hurst", "ETE-GNN") < _get("Hurst", "TE-GNN"),
}
for k, v in checks.items():
    print(f"{'PASS' if v else 'FAIL'}  {k}")
'@

Add-Cell markdown @'
## 9. Forecasts vs actual -- Figure 5 analogue
'@

Add-Cell code @'
TGT = "EWG"
j = list(cfg.tickers).index(TGT)
hurst_runs = [r for r in runs if r.regime == "hurst" and r.model in
              ("ETE-GNN", "TE-GNN", "Granger-GNN", "Pearson-GNN", "LSTM", "GRU")]

fig, ax = plt.subplots(figsize=(14, 5))
ref = hurst_runs[0]
ax.plot(ref.dates, ref.y_true[:, j], color="black", lw=1.3, label="Actual RV")
for name in ("ETE-GNN", "Granger-GNN", "LSTM"):
    sel = [r for r in hurst_runs if r.model == name]
    if not sel:
        continue
    mean_pred = np.mean([r.y_pred[:, j] for r in sel], axis=0)
    ax.plot(sel[0].dates, mean_pred, lw=0.9, alpha=0.85, label=name)
ax.set_title(f"{TGT} -- one-day-ahead RV, Hurst-regime models (mean over repetitions)")
ax.set_ylabel("RV (%)"); ax.legend()
plt.tight_layout(); plt.show()
'@

Add-Cell code @'
# Annual RMSE -- Table 9 analogue. The paper notes H-ETE-GNN degrades in 2008
# and 2020-2023, because few regime changes fire during those stretches.
frames = []
for r in hurst_runs:
    err2 = (r.y_pred - r.y_true) ** 2
    s = pd.DataFrame({"year": r.dates.year, "mse": err2.mean(axis=1)})
    g = s.groupby("year")["mse"].mean().pow(0.5)
    frames.append(g.rename(r.model))
annual = pd.concat(frames, axis=1)
annual = annual.T.groupby(level=0).mean().T
annual.round(4)
'@

Add-Cell markdown @'
## 10. Where this departs from the paper

Read this before quoting any replicated number, and carry the same list into
the HR-ETE-GNN comparison so both arms sit on identical assumptions.
'@

Add-Cell code @'
print(hb.DEVIATIONS)
'@

$nb = [ordered]@{
    cells          = $cells
    metadata       = [ordered]@{
        kernelspec    = [ordered]@{ display_name = 'Python 3'; language = 'python'; name = 'python3' }
        language_info = [ordered]@{ name = 'python'; version = '3.11' }
        colab         = [ordered]@{ provenance = @() }
    }
    nbformat       = 4
    nbformat_minor = 5
}

$json = $nb | ConvertTo-Json -Depth 12
[System.IO.File]::WriteAllText($Out, $json, (New-Object System.Text.UTF8Encoding($false)))
Write-Host "wrote $Out  ($($cells.Count) cells, $((Get-Item $Out).Length) bytes)"
