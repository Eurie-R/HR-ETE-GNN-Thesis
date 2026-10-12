"""Reproduce the pilot's ER-TE matrices with the repo's own estimator.

Mirrors .build/mk_prelim.py exactly: same window, same k, same surrogate count,
same per-pair seeds, so the top-K out-degrees must match T5_h2b_transmitters.csv.
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/uriel/repositories/HR-ETE-GNN-THESIS_FINAL/.build")
sys.path.insert(0, "/home/uriel/repositories/HR-ETE-GNN-THESIS_FINAL")
import core as hr

ROOT = "/home/uriel/repositories/HR-ETE-GNN-THESIS_FINAL"
OUT = os.path.dirname(os.path.abspath(__file__))

TICKERS = ["EWA", "EWC", "EWG", "EWJ", "EWT", "EWU", "EWW", "EWY", "EWZ", "EZA"]
LOOKBACK = RV_WINDOW = 20
TRAIN_FRAC, VAL_FRAC = 0.60, 0.20
KNN_K = 4
M_SURROGATES = 100
FDR_Q = 0.10
MATCHED_K = 20
ALPHAS = [0.5, 1.0, 1.5]

px = pd.read_csv(f"{ROOT}/preliminary_results/prices_cache.csv", index_col=0, parse_dates=True)
log_ret = np.log(px).diff().dropna()
rv = (100 * log_ret).pow(2).rolling(RV_WINDOW).mean().pow(0.5).dropna()

sp = hr.make_splits(rv[TICKERS].values, lookback=LOOKBACK, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC)
last_train_date = rv.index[sp["t_train"][-1]]
train_ret = log_ret.loc[:last_train_date, TICKERS]
print(f"graph window: {train_ret.index.min().date()} -> {train_ret.index.max().date()} "
      f"({len(train_ret):,} days)")


def erte_graph(rdf, alpha, m=M_SURROGATES):
    n = len(TICKERS)
    E = np.zeros((n, n)); Z = np.zeros((n, n)); P = np.ones((n, n))
    t0 = time.time()
    for i, ti in enumerate(TICKERS):
        for j, tj in enumerate(TICKERS):
            if i == j:
                continue
            e, z, p = hr.effective_rte(rdf[ti].values, rdf[tj].values, alpha=alpha,
                                       k=KNN_K, m_surrogates=m, seed=1000 * i + j,
                                       copula=True, surrogate="circular")
            E[i, j], Z[i, j], P[i, j] = e, z, p
    off = ~np.eye(n, dtype=bool)
    keep = hr.benjamini_hochberg_mask(P[off], q=FDR_Q)
    A = np.zeros((n, n))
    A[off] = np.where(keep & np.isfinite(E[off]), np.maximum(E[off], 0.0), 0.0)
    print(f"  alpha={alpha}: {int((A > 0).sum())}/90 edges survive BH  ({time.time()-t0:.0f}s)")
    return dict(E=E, Z=Z, P=P, A_fdr=A)


def top_k_graph(E, k=MATCHED_K):
    n = E.shape[0]
    M = np.where(np.eye(n, dtype=bool), -np.inf, E)
    flat = M.ravel()
    idx = np.argsort(flat)[::-1][:k]
    out = np.zeros_like(E)
    out.ravel()[idx] = flat[idx]
    return out


G = {a: erte_graph(train_ret, a) for a in ALPHAS}
M = {a: top_k_graph(G[a]["E"]) for a in ALPHAS}

outdeg = pd.DataFrame({f"out-degree α={a}": (M[a] > 0).sum(axis=1) for a in ALPHAS}, index=TICKERS)
print("\nreproduced top-K out-degrees:")
print(outdeg)

t5 = pd.read_csv(f"{ROOT}/preliminary_results/T5_h2b_transmitters.csv", index_col=0)
print("\npublished T5 out-degrees:")
print(t5[[c for c in t5.columns if c.startswith("out-degree")]])

match = all(
    (outdeg[f"out-degree α={a}"].values == t5[f"out-degree α={a}"].values).all()
    for a in ALPHAS
)
print(f"\nMATCH T5 exactly: {match}")

np.savez(f"{OUT}/graphs.npz",
         **{f"E_{a}": G[a]["E"] for a in ALPHAS},
         **{f"A_{a}": G[a]["A_fdr"] for a in ALPHAS},
         **{f"P_{a}": G[a]["P"] for a in ALPHAS},
         **{f"M_{a}": M[a] for a in ALPHAS},
         tickers=np.array(TICKERS))
print(f"saved {OUT}/graphs.npz")
