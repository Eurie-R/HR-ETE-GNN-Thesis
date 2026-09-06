import nbformat as nbf, json
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): C.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# HR-ETE-GNN — Statistical Validation

### Is the Rényi model *significantly* better than the Shannon ETE-GNN it is built on?

---

This notebook exists to answer one question that the pilot notebook could not:

> The pilot reported RMSE **0.2384** for the Shannon baseline and **0.2053** for the Rényi model — a 13.9% improvement. **Is that difference real, or is it noise?**

The honest answer for the pilot is *we cannot tell*, and not because the effect is small. It is because of **how the two numbers were produced**. Five separate design choices made the comparison untestable. This notebook fixes them, then applies the formal test.

### What was wrong, and what is done here instead

| # | Problem in the pilot | Consequence | Fix in this notebook |
|---|---|---|---|
| 1 | `torch.manual_seed(42)` called once at import, then two models built in sequence | The two arms got **different initial weights**. The gap could be pure initialisation luck. | Seed set *inside* the training loop; both arms get the **same seed list** (§5) |
| 2 | Adjacency built from `log_ret.iloc[-600:]`, which overlaps the test set | **Look-ahead bias.** The graph used to forecast the test period was estimated *from* the test period. | Graph estimated on the **training window only** (§4) |
| 3 | 50 full-batch steps at `lr=1e-3` | ~50 gradient updates total. Neither arm converged; two arbitrary points on two trajectories were compared. | Early stopping on a **validation block** (§5) |
| 4 | No validation split; α implicitly chosen against test | Any α tuning invalidates every test-set p-value. | Three-way **chronological** split (§3) |
| 5 | Edges kept at `z > 1.96` with only 20 surrogates | With 20 surrogates the null is not normal, so 1.96 is not a 5% test. Result rested on 3–5 edges. | **Exact permutation p-values**, 200 surrogates, **BH-FDR** across all 90 pairs (§4) |

### Two further corrections to the estimator itself

| Issue | Why it matters | Fix |
|---|---|---|
| Surrogates used `rng.permutation(y)` | Destroys Y's *own* autocorrelation, so the null tested was "Y is i.i.d. noise", not "Y carries no information about X". Returns are serially dependent — the two nulls are not the same. | **Circular-shift** surrogates (§2) |
| Raw returns fed to a k-NN entropy estimator | Distances are dominated by whichever series is most volatile, and the dimensional bias in the 1-D/2-D/3-D terms stops cancelling. | **Copula (rank) transform** to uniform margins (§2) |

### Where this notebook lands

Read §4 and §8 first if you only read two sections. §4 contains a result that is stronger than anything in the pilot, and §8 contains the caveat that keeps it honest.
""")

md("""
## §0 — Configuration and pre-registration

**Everything below is frozen before any result is looked at.** This block is the cheapest possible defence against the charge of test-shopping: it fixes the primary metric, the primary test and the direction of the alternative hypothesis in advance, so no reviewer can suspect the test was chosen after seeing which one gave the desired answer.

Copy this block verbatim into the methodology chapter.
""")

code('''
%pip install yfinance numpy pandas scipy matplotlib torch --quiet
''')

code('''
import numpy as np, pandas as pd, warnings, time
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.special import gamma as gamma_fn, digamma
from scipy import stats
import torch, torch.nn as nn, torch.nn.functional as F

plt.rcParams.update({'figure.figsize': (11, 4.2), 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'figure.facecolor': 'white'})
INK, ACCENT, CALM, WARN = '#1a1a2e', '#e94560', '#16a085', '#f39c12'
pd.set_option('display.precision', 4); pd.set_option('display.width', 130)

# ===========================================================================
#  PRE-REGISTRATION  —  frozen before any result is inspected
# ===========================================================================
PRIMARY_METRIC        = "QLIKE"        # robust to noise in the RV proxy (Patton 2011)
PRIMARY_TEST          = "Giacomini-White (2006), HAC"
PRIMARY_ALTERNATIVE   = "greater"      # one-sided: Renyi beats Shannon
PRIMARY_LEVEL         = 0.05
SECONDARY_FDR_Q       = 0.10           # Benjamini-Hochberg across the 10 ETFs
MCS_CONFIDENCE        = 0.90           # Hansen-Lunde-Nason model confidence set
MIN_MEANINGFUL_GAIN   = 0.05           # 5% loss reduction = smallest claim worth making

# ---- experiment configuration ---------------------------------------------
TICKERS      = ['EWA','EWC','EWG','EWJ','EWT','EWU','EWW','EWY','EWZ','EZA']
COUNTRY      = dict(zip(TICKERS, ['Australia','Canada','Germany','Japan','Taiwan',
                                  'UK','Mexico','South Korea','Brazil','South Africa']))
REGIME_PROXY = 'URTH'
START, END   = '2018-01-01', '2024-05-29'
LOOKBACK     = 20
RV_WINDOW    = 20
ALPHA_GRID   = [0.5, 1.0, 1.5]         # 1.0 == Shannon == the baseline
M_SURROGATES = 200                     # >= 200 needed for a usable permutation p-value
KNN_K        = 4
FDR_Q        = 0.10
MATCHED_K    = 20                      # edges per graph in the matched-density comparison
SEEDS        = list(range(8))          # raise to 20+ for the final thesis run
EPOCHS, LR, PATIENCE = 3000, 1e-2, 150
TRAIN_FRAC, VAL_FRAC = 0.60, 0.20

torch.set_num_threads(4)
print("Pre-registration frozen.")
print(f"  primary metric : {PRIMARY_METRIC}")
print(f"  primary test   : {PRIMARY_TEST}, one-sided at {PRIMARY_LEVEL}")
print(f"  seeds per arm  : {len(SEEDS)}   surrogates per pair : {M_SURROGATES}")
''')

md("""
---
## §1 — Data

Identical to the pilot, so nothing in the comparison is confounded by a different sample.
""")

code('''
import yfinance as yf
raw = yf.download(TICKERS + [REGIME_PROXY], start=START, end=END,
                  auto_adjust=True, progress=False)['Close'].dropna()
log_ret = np.log(raw / raw.shift(1)).dropna()
rv = (100*log_ret).pow(2).rolling(RV_WINDOW).mean().pow(0.5).dropna()

print(f"Prices     : {raw.shape[0]:,} days x {raw.shape[1]} series "
      f"({raw.index.min().date()} to {raw.index.max().date()})")
print(f"Log-returns: {log_ret.shape}")
print(f"Realized vol: {rv.shape}")
display(rv[TICKERS].describe().T[['mean','std','min','max']].round(3))
''')

md(r"""
---
## §2 — The ER-TE estimator, corrected

Three changes from the pilot, each with a specific justification.

### 2.1 Copula (rank) transform

The k-NN entropy estimator works by measuring Euclidean distances between points. Raw returns for different ETFs have different scales and different tail thickness, so distances in the joint space $(X_{t+1}, X_t, Y_t)$ get dominated by whichever series happens to be most volatile — and the dimensional bias in the 1-D, 2-D and 3-D entropy terms no longer cancels in the transfer-entropy difference.

Rank-transforming every margin to Uniform(0,1) removes scale entirely, so the estimator depends only on the **dependence structure** (the copula). That is exactly what transfer entropy is meant to measure.

### 2.2 Circular-shift surrogates

The pilot's null was generated by `rng.permutation(y)`, which scrambles Y completely — destroying both the Y→X link *and* Y's own autocorrelation. So the hypothesis actually being tested was *"Y is i.i.d. noise"*, which is trivially false for financial returns and therefore too easy to reject.

The intended null is *"Y carries no information about X's future beyond what X's own past already contains"*. A **circular shift** — rotating Y by a random offset — destroys the cross-dependence while leaving Y's autocorrelation intact. That is the correct null.

### 2.3 Exact permutation p-values instead of `z > 1.96`

With 20 surrogates, the sampling distribution of the z-score is nowhere near normal, so comparing it to 1.96 is not a 5% test. A permutation p-value

$$p = \frac{1 + \#\{\text{surrogates} \ge \text{observed}\}}{m + 1}$$

is exact for any $m$, requires no distributional assumption, and its resolution is bounded by $1/(m+1)$ — which is why $m = 200$ rather than 20.

### A caveat that must appear in the thesis

For $\alpha \neq 1$ the Rényi chain rule $H(A|B) = H(A,B) - H(B)$ **does not hold**. The implementation below (like the pilot's) uses the *difference-based* definition of Rényi transfer entropy, not the escort-distribution definition of Jizba, Kleinert & Shefaat (2012). It can take negative values in the population — which is why the pilot's sanity check printed `ERTE = -0.0095`. Name this explicitly in the methodology chapter and cite it; do not let a panellist find it first.
""")

code('''
# ---------------------------------------------------------------------------
# Renyi entropy: Leonenko-Pronzato-Savani k-NN estimator
# ---------------------------------------------------------------------------
def rank_uniform(x):
    """Copula transform: map to uniform margins via ranks (argsort, ~100x faster
    than scipy.stats.rankdata -- this is called on every surrogate)."""
    x = np.asarray(x, float)
    if x.ndim == 1:
        n = x.size; r = np.empty(n, float)
        r[np.argsort(x, kind='stable')] = np.arange(1, n+1)
        return r / (n + 1.0)
    return np.column_stack([rank_uniform(x[:, j]) for j in range(x.shape[1])])


def lps_renyi_entropy(X, alpha, k=KNN_K):
    """H_alpha via k-NN distances. alpha==1 falls back to Kozachenko-Leonenko."""
    X = np.asarray(X, float)
    if X.ndim == 1: X = X.reshape(-1, 1)
    N, d = X.shape
    if N <= k + 1: return np.nan
    dists, _ = cKDTree(X).query(X, k=k+1)
    rho = np.maximum(dists[:, k], 1e-12)
    B_d = np.pi**(d/2) / gamma_fn(d/2 + 1)              # volume of the unit d-ball
    if abs(alpha - 1.0) < 1e-8:
        return -digamma(k) + digamma(N) + np.log(B_d) + d*np.mean(np.log(rho))
    if k + 1 - alpha <= 0: return np.nan
    C_k = (gamma_fn(k) / gamma_fn(k + 1 - alpha))**(1.0/(1.0 - alpha))
    inner = (C_k**(1-alpha)) * ((N-1)**(1-alpha)) * (B_d**(1-alpha)) \\
            * np.mean(rho**(d*(1-alpha)))
    return (1.0/(1.0 - alpha)) * np.log(max(inner, 1e-300))


# ---------------------------------------------------------------------------
# Transfer entropy
# ---------------------------------------------------------------------------
def _te_blocks(y, x, lag=1):
    """Align the three series TE needs: X_{t+1}, X_t, Y_t."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    Xtp1 = x[lag+1:]; n = len(Xtp1)
    return Xtp1, x[lag:-1][:n], y[lag:-1][:n]


def _te_from_blocks(Xtp1, Xt, Yt, alpha, k, cache=None):
    """RTE = [H(X_{t+1},X_t) - H(X_t)] - [H(X_{t+1},X_t,Y_t) - H(X_t,Y_t)].

    NOTE: difference-based definition. The Renyi chain rule does not hold for
    alpha != 1, so this is NOT the escort-distribution RTE of Jizba et al. (2012).
    """
    if cache is None:
        H_x      = lps_renyi_entropy(Xt, alpha, k)
        H_xtp1_x = lps_renyi_entropy(np.column_stack([Xtp1, Xt]), alpha, k)
    else:
        H_x, H_xtp1_x = cache
    H_xy   = lps_renyi_entropy(np.column_stack([Xt, Yt]), alpha, k)
    H_full = lps_renyi_entropy(np.column_stack([Xtp1, Xt, Yt]), alpha, k)
    return (H_xtp1_x - H_x) - (H_full - H_xy)


def effective_rte(y, x, alpha, k=KNN_K, m_surrogates=M_SURROGATES, seed=0,
                  copula=True, surrogate='circular', lag=1):
    """ER-TE with an exact permutation p-value. Returns (erte, z, p).

    Speed: H(X_t) and H(X_{t+1},X_t) do not involve Y, so they are computed once
    and reused across all surrogates. The copula transform is likewise applied
    once -- rank-transforming a shifted series equals shifting the ranks, because
    ranks are equivariant under permutation.
    """
    rng = np.random.default_rng(seed)
    Xtp1, Xt, Yt = _te_blocks(y, x, lag)
    if copula:
        Xtp1, Xt, Yt = rank_uniform(Xtp1), rank_uniform(Xt), rank_uniform(Yt)
    cache = (lps_renyi_entropy(Xt, alpha, k),
             lps_renyi_entropy(np.column_stack([Xtp1, Xt]), alpha, k))
    obs = _te_from_blocks(Xtp1, Xt, Yt, alpha, k, cache=cache)

    n = len(Yt); surr = np.empty(m_surrogates)
    for i in range(m_surrogates):
        Ys = np.roll(Yt, int(rng.integers(1, n))) if surrogate == 'circular' \\
             else rng.permutation(Yt)
        surr[i] = _te_from_blocks(Xtp1, Xt, Ys, alpha, k, cache=cache)

    bias, sd = np.nanmean(surr), np.nanstd(surr, ddof=1)
    erte = obs - bias
    return erte, erte/(sd + 1e-12), (1.0 + np.sum(surr >= obs))/(m_surrogates + 1.0)

print("ER-TE estimator defined.")
''')

md("""
### Validating the estimator before trusting it

An estimator should be tested on data whose answer is known. We build a system where **Y genuinely drives X** with a one-day lag and nothing flows the other way, then check that the estimator recovers exactly that.

Three things must hold: Y→X is detected, X→Y is *not* falsely detected, and two independent series produce nothing.
""")

code('''
rng = np.random.default_rng(0); n = 800
y_s, x_s = np.zeros(n), np.zeros(n)
for t in range(1, n):                       # Y drives X at lag 1; X does not drive Y
    y_s[t] = 0.5*y_s[t-1] + rng.normal()
    x_s[t] = 0.3*x_s[t-1] + 0.6*y_s[t-1] + rng.normal()
u_s, v_s = rng.normal(size=n), rng.normal(size=n)     # independent control

rows = []
for a in ALPHA_GRID:
    e1, z1, p1 = effective_rte(y_s, x_s, a)          # true direction
    e2, z2, p2 = effective_rte(x_s, y_s, a)          # reverse direction
    _,  _,  p3 = effective_rte(u_s, v_s, a)          # independent
    rows.append({'alpha': a, 'ER-TE (Y->X)': e1, 'p (Y->X)': p1,
                 'ER-TE (X->Y)': e2, 'p (X->Y)': p2, 'p (independent)': p3})

val = pd.DataFrame(rows).set_index('alpha')
display(val.round(4))

ok = (val['p (Y->X)'] < 0.05).all() and (val['p (X->Y)'] > 0.05).all() and (val['p (independent)'] > 0.05).all()
print(f"\\nTrue direction detected at every alpha : {(val['p (Y->X)'] < 0.05).all()}")
print(f"Reverse direction correctly rejected   : {(val['p (X->Y)'] > 0.05).all()}")
print(f"Independent pair correctly rejected    : {(val['p (independent)'] > 0.05).all()}")
print(f"\\n{'ESTIMATOR VALIDATED' if ok else 'VALIDATION FAILED - do not proceed'}")
''')

md("""
---
## §3 — Leak-free chronological splits

The pilot used an 80/20 train/test split with no validation block, and built the graph from the last 600 days — which overlap the test set.

Here the sample is cut **chronologically** into three blocks:

- **Train (60%)** — fits the network weights, and is the *only* data the graph may see
- **Validation (20%)** — chooses the stopping epoch and α
- **Test (20%)** — touched exactly once, at the end

The cell below prints the actual dates so the separation can be verified by eye rather than taken on trust.
""")

code('''
def make_splits(rv_mat, lookback=LOOKBACK, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    Xs, ys, ts = [], [], []
    for t in range(lookback, len(rv_mat) - 1):
        Xs.append(rv_mat[t-lookback:t].T); ys.append(rv_mat[t+1]); ts.append(t+1)
    Xs, ys, ts = np.stack(Xs), np.stack(ys), np.array(ts)
    n = len(Xs); i_tr, i_va = int(train_frac*n), int((train_frac+val_frac)*n)
    return dict(X_train=Xs[:i_tr], y_train=ys[:i_tr], t_train=ts[:i_tr],
                X_val=Xs[i_tr:i_va], y_val=ys[i_tr:i_va], t_val=ts[i_tr:i_va],
                X_test=Xs[i_va:],  y_test=ys[i_va:],  t_test=ts[i_va:],
                lookback=lookback, n_samples=n)

sp = make_splits(rv[TICKERS].values)
LAST_TRAIN_DATE = rv.index[sp['t_train'][-1]]

display(pd.DataFrame([
    {'block': 'TRAIN', 'samples': len(sp['X_train']),
     'from': rv.index[sp['t_train'][0]].date(),  'to': rv.index[sp['t_train'][-1]].date()},
    {'block': 'VALIDATION', 'samples': len(sp['X_val']),
     'from': rv.index[sp['t_val'][0]].date(),    'to': rv.index[sp['t_val'][-1]].date()},
    {'block': 'TEST', 'samples': len(sp['X_test']),
     'from': rv.index[sp['t_test'][0]].date(),   'to': rv.index[sp['t_test'][-1]].date()},
]).set_index('block'))

fig, ax = plt.subplots(figsize=(12, 3.6))
avg = rv[TICKERS].mean(axis=1)
ax.plot(avg.index, avg.values, color=INK, linewidth=1)
for blk, c, lab in [('t_train', CALM, 'TRAIN — graph + weights'),
                    ('t_val', WARN, 'VALIDATION — stopping + alpha'),
                    ('t_test', ACCENT, 'TEST — touched once')]:
    ax.axvspan(rv.index[sp[blk][0]], rv.index[sp[blk][-1]], color=c, alpha=.18, label=lab)
ax.set_title('Chronological split — no block may see a later block')
ax.set_ylabel('Average RV, %'); ax.legend(loc='upper right', fontsize=9)
plt.tight_layout(); plt.show()
''')

md("""
### Demonstrating the leak the pilot had

This is worth making concrete rather than asserting. The pilot's graph window was `log_ret.iloc[-600:]`. The cell below counts how many days of that window fall *inside* the test period.
""")

code('''
pilot_window   = log_ret.iloc[-600:]
test_start     = rv.index[sp['t_test'][0]]
overlap_days   = int((pilot_window.index >= test_start).sum())

print(f"Pilot's graph window : {pilot_window.index[0].date()} to {pilot_window.index[-1].date()}  ({len(pilot_window)} days)")
print(f"Test period starts   : {test_start.date()}")
print(f"\\n>>> {overlap_days} of the {len(pilot_window)} days used to build the pilot's graph "
      f"fall INSIDE the test period ({100*overlap_days/len(pilot_window):.0f}%).")
print("\\nThis notebook instead estimates the graph on training data only:")
print(f"    graph window : up to {LAST_TRAIN_DATE.date()}   "
      f"(ends {(test_start - LAST_TRAIN_DATE).days} days before the test period opens)")

graph_returns = log_ret.loc[:LAST_TRAIN_DATE, TICKERS]
print(f"    shape        : {graph_returns.shape}")
''')
json.dump([c for c in C], open('/home/uriel/repositories/HR-ETE-GNN-THESIS_FINAL/.build/p1.json','w'))
print(f"part1: {len(C)} cells")
