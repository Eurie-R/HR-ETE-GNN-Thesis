"""Core routines for HR_ETE_GNN_stats.ipynb - tested standalone before packaging."""
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.special import gamma as gamma_fn, digamma
from scipy import stats

_EPS = 1e-12

# ---------------- entropy ----------------
def rank_uniform(x):
    """Copula transform: map a series to uniform margins via its ranks.

    Why: the k-NN entropy estimator measures Euclidean distances. Raw returns
    for different ETFs have different scales and fat tails, so distances in the
    joint space are dominated by whichever series happens to be most volatile,
    and the dimensional bias in H(1-D), H(2-D) and H(3-D) no longer cancels in
    the transfer-entropy difference. Rank-transforming every margin to Uniform(0,1)
    makes the estimator depend only on the *dependence structure* (the copula),
    which is exactly what transfer entropy is supposed to measure.

    Uses argsort rather than scipy.stats.rankdata: ~100x faster, and we call this
    on every surrogate.
    """
    x = np.asarray(x, float)
    if x.ndim == 1:
        n = x.size
        r = np.empty(n, float)
        r[np.argsort(x, kind="stable")] = np.arange(1, n + 1)
        return r / (n + 1.0)
    return np.column_stack([rank_uniform(x[:, j]) for j in range(x.shape[1])])


def lps_renyi_entropy(X, alpha, k=4):
    X = np.asarray(X, float)
    if X.ndim == 1: X = X.reshape(-1, 1)
    N, d = X.shape
    if N <= k + 1: return np.nan
    tree = cKDTree(X)
    dists, _ = tree.query(X, k=k + 1)
    rho = np.maximum(dists[:, k], 1e-12)
    B_d = np.pi ** (d / 2) / gamma_fn(d / 2 + 1)
    if abs(alpha - 1.0) < 1e-8:
        return -digamma(k) + digamma(N) + np.log(B_d) + d * np.mean(np.log(rho))
    if k + 1 - alpha <= 0: return np.nan
    C_k = (gamma_fn(k) / gamma_fn(k + 1 - alpha)) ** (1.0 / (1.0 - alpha))
    inner = (C_k ** (1 - alpha)) * ((N - 1) ** (1 - alpha)) * (B_d ** (1 - alpha)) \
            * np.mean(rho ** (d * (1 - alpha)))
    return (1.0 / (1.0 - alpha)) * np.log(max(inner, 1e-300))

# ---------------- transfer entropy ----------------
def _te_blocks(y, x, lag=1):
    """Align the three series TE needs: X_{t+1}, X_t, Y_t."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    Xtp1 = x[lag + 1:]; n = len(Xtp1)
    return Xtp1, x[lag:-1][:n], y[lag:-1][:n]


def _te_from_blocks(Xtp1, Xt, Yt, alpha, k, cache=None):
    """RTE = [H(X_{t+1},X_t) - H(X_t)] - [H(X_{t+1},X_t,Y_t) - H(X_t,Y_t)].

    CAVEAT, and you must state this in the thesis: for alpha != 1 the Renyi
    chain rule H(A|B) = H(A,B) - H(B) does NOT hold. This is the
    *difference-based* definition of Renyi transfer entropy, not the
    escort-distribution definition of Jizba, Kleinert & Shefaat (2012). It can
    take negative values in the population. Name it explicitly and cite it.
    """
    if cache is None:
        H_x      = lps_renyi_entropy(Xt, alpha, k)
        H_xtp1_x = lps_renyi_entropy(np.column_stack([Xtp1, Xt]), alpha, k)
    else:
        H_x, H_xtp1_x = cache
    H_xy   = lps_renyi_entropy(np.column_stack([Xt, Yt]), alpha, k)
    H_full = lps_renyi_entropy(np.column_stack([Xtp1, Xt, Yt]), alpha, k)
    return (H_xtp1_x - H_x) - (H_full - H_xy)


def renyi_te(y, x, alpha, k=4, lag=1, copula=True):
    """Renyi transfer entropy from Y to X, order alpha, history length 1."""
    Xtp1, Xt, Yt = _te_blocks(y, x, lag)
    if copula:
        Xtp1, Xt, Yt = rank_uniform(Xtp1), rank_uniform(Xt), rank_uniform(Yt)
    return _te_from_blocks(Xtp1, Xt, Yt, alpha, k)


def effective_rte(y, x, alpha, k=4, m_surrogates=200, seed=0, copula=True,
                  surrogate="circular", lag=1):
    """Effective Renyi TE with an exact permutation p-value.

    ER-TE = RTE(Y->X) - mean(RTE(Y_surrogate->X)), which removes the finite-sample
    bias of the k-NN estimator.

    surrogate="circular" (default) rotates Y by a random offset. This destroys
    the Y->X cross-dependence while PRESERVING Y's own autocorrelation. The pilot
    used a full random permutation, which destroys both -- so its null hypothesis
    was "Y is i.i.d. noise" rather than the intended "Y carries no information
    about X beyond X's own past". Financial returns are serially dependent, so
    the two nulls are not interchangeable.

    Returns (erte, z, p_permutation). Prefer the permutation p-value: with a
    finite surrogate count the z-score's null is not normal, so comparing z to
    1.96 is not a 5% test.

    Speed: H(X_t) and H(X_{t+1},X_t) do not involve Y at all, so they are
    computed once and reused across all surrogates (~2x). The copula transform
    is also applied once -- rank-transforming a circular shift gives the same
    answer as circularly shifting the ranks, because ranks are equivariant under
    permutation.
    """
    rng = np.random.default_rng(seed)
    Xtp1, Xt, Yt = _te_blocks(y, x, lag)
    if copula:
        Xtp1, Xt, Yt = rank_uniform(Xtp1), rank_uniform(Xt), rank_uniform(Yt)

    cache = (lps_renyi_entropy(Xt, alpha, k),
             lps_renyi_entropy(np.column_stack([Xtp1, Xt]), alpha, k))
    obs = _te_from_blocks(Xtp1, Xt, Yt, alpha, k, cache=cache)

    n = len(Yt)
    surr = np.empty(m_surrogates)
    for i in range(m_surrogates):
        if surrogate == "circular":
            Ys = np.roll(Yt, int(rng.integers(1, n)))
        else:
            Ys = rng.permutation(Yt)
        surr[i] = _te_from_blocks(Xtp1, Xt, Ys, alpha, k, cache=cache)

    bias = np.nanmean(surr)
    sd = np.nanstd(surr, ddof=1)
    erte = obs - bias
    z = erte / (sd + 1e-12)
    p = (1.0 + np.sum(surr >= obs)) / (m_surrogates + 1.0)
    return erte, z, p


def benjamini_hochberg_mask(p, q=0.10):
    p = np.asarray(p, float); ok = np.isfinite(p); m = ok.sum()
    if m == 0: return np.zeros_like(p, bool)
    idx = np.argsort(np.where(ok, p, np.inf))
    ranks = np.arange(1, len(p) + 1)
    thr = q * ranks / m
    srt = p[idx]
    passed = srt <= thr
    mask = np.zeros_like(p, bool)
    if passed.any():
        kmax = np.max(np.where(passed)[0])
        mask[idx[:kmax + 1]] = True
    return mask & ok

def build_erte_adjacency(returns_df, tickers, alpha, m_surrogates=200, k=4,
                         fdr_q=0.10, copula=True, verbose=False):
    n = len(tickers)
    E = np.zeros((n, n)); Z = np.zeros((n, n)); P = np.ones((n, n))
    pairs = []
    for i, ti in enumerate(tickers):
        for j, tj in enumerate(tickers):
            if i == j: continue
            e, z, p = effective_rte(returns_df[ti].values, returns_df[tj].values,
                                    alpha=alpha, k=k, m_surrogates=m_surrogates,
                                    seed=1000 * i + j, copula=copula)
            E[i, j], Z[i, j], P[i, j] = e, z, p
            pairs.append((i, j))
    pv = np.array([P[i, j] for i, j in pairs])
    keep = benjamini_hochberg_mask(pv, q=fdr_q)
    A = np.zeros((n, n))
    for (i, j), kp in zip(pairs, keep):
        if kp and np.isfinite(E[i, j]):
            A[i, j] = max(E[i, j], 0.0)
    if verbose:
        print(f"  alpha={alpha}: {int((A>0).sum())}/{n*(n-1)} edges survive BH-FDR q={fdr_q}")
    return A, Z, P


# ============================================================================
# Leak-free splits
# ============================================================================
def make_splits(rv_mat, lookback=20, train_frac=0.60, val_frac=0.20):
    """Chronological train/val/test split of the supervised samples.

    Three-way, not two-way. The validation block is what lets you choose alpha,
    the learning rate and the stopping epoch WITHOUT touching the test set. If
    any of those are tuned on test, every p-value computed on test is void.
    """
    Xs, ys, ts = [], [], []
    for t in range(lookback, len(rv_mat) - 1):
        Xs.append(rv_mat[t - lookback:t].T)
        ys.append(rv_mat[t + 1])
        ts.append(t + 1)
    Xs, ys, ts = np.stack(Xs), np.stack(ys), np.array(ts)
    n = len(Xs)
    i_tr, i_va = int(train_frac * n), int((train_frac + val_frac) * n)
    return dict(X_train=Xs[:i_tr], y_train=ys[:i_tr], t_train=ts[:i_tr],
                X_val=Xs[i_tr:i_va], y_val=ys[i_tr:i_va], t_val=ts[i_tr:i_va],
                X_test=Xs[i_va:], y_test=ys[i_va:], t_test=ts[i_va:],
                i_train_end=i_tr, i_val_end=i_va, lookback=lookback, n_samples=n)


def row_normalize(A):
    A = np.asarray(A, float).copy()
    s = A.sum(axis=1, keepdims=True); s[s == 0] = 1.0
    return A / s


# ============================================================================
# Model - identical to the pilot except for a positivity-constrained head
# ============================================================================
def build_model_classes():
    import torch, torch.nn as nn, torch.nn.functional as F

    class MultiScaleConv(nn.Module):
        def __init__(self, in_len, channels=12, kernels=(3, 5, 7)):
            super().__init__()
            self.convs = nn.ModuleList(
                [nn.Conv1d(1, channels, kernel_size=k, padding=k // 2) for k in kernels])
            self.out_dim = channels * len(kernels)
        def forward(self, x):
            B, N, L = x.shape
            x = x.reshape(B * N, 1, L)
            feats = [F.relu(c(x)).mean(dim=2) for c in self.convs]
            return torch.cat(feats, dim=1).view(B, N, -1)

    class GCNLayer(nn.Module):
        def __init__(self, i, o):
            super().__init__(); self.W1 = nn.Linear(i, o); self.W2 = nn.Linear(i, o)
        def forward(self, H, A):
            return F.relu(self.W1(H) + torch.einsum('ij,bjf->bif', A, self.W2(H)))

    class HRETEGNN(nn.Module):
        """Positivity-constrained head: softplus guarantees RV_hat > 0.

        The pilot ended in a bare nn.Linear, which can emit negative volatility --
        economically meaningless, and it makes the QLIKE loss diverge.
        """
        def __init__(self, n_nodes, lookback, hidden=32):
            super().__init__()
            self.feat = MultiScaleConv(lookback)
            d = self.feat.out_dim
            self.g1, self.g2, self.g3 = GCNLayer(d, hidden), GCNLayer(hidden, hidden), GCNLayer(hidden, hidden)
            self.head = nn.Linear(hidden, 1)
        def forward(self, x, A):
            h = self.feat(x)
            h = self.g3(self.g2(self.g1(h, A), A), A)
            return F.softplus(self.head(h).squeeze(-1)) + 1e-4

    return MultiScaleConv, GCNLayer, HRETEGNN


# ============================================================================
# Multi-seed training protocol
# ============================================================================
def train_multiseed(A_np, splits, n_nodes, seeds=range(20), hidden=32,
                    epochs=3000, lr=1e-2, patience=150, verbose=False):
    """Train one architecture across a FIXED seed list; return the seed ensemble.

    Two rules make this comparison fair, and both were violated in the pilot:

      1. torch.manual_seed(s) is called inside the loop, immediately before the
         model is constructed. In the pilot the seed was set once at import, so
         the two arms drew different initial weights and the RMSE gap could not
         be separated from initialisation luck.
      2. Every arm receives the SAME `seeds` list, so seed s of the baseline and
         seed s of the treatment start from identical weights. The only thing
         that differs between arms is the adjacency matrix.

    Early stopping uses the validation block, so each arm is compared at its own
    best point rather than after an arbitrary 50 gradient steps.
    """
    import torch, torch.nn.functional as F
    _, _, HRETEGNN = build_model_classes()

    A = torch.tensor(row_normalize(A_np), dtype=torch.float32)
    Xtr = torch.tensor(splits['X_train'], dtype=torch.float32)
    ytr = torch.tensor(splits['y_train'], dtype=torch.float32)
    Xva = torch.tensor(splits['X_val'],   dtype=torch.float32)
    yva = torch.tensor(splits['y_val'],   dtype=torch.float32)
    Xte = torch.tensor(splits['X_test'],  dtype=torch.float32)

    test_preds, val_preds, stops = [], [], []
    for s in seeds:
        torch.manual_seed(int(s)); np.random.seed(int(s))
        model = HRETEGNN(n_nodes=n_nodes, lookback=splits['lookback'], hidden=hidden)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        best, best_state, bad, ep = np.inf, None, 0, 0
        for ep in range(epochs):
            model.train(); opt.zero_grad()
            F.mse_loss(model(Xtr, A), ytr).backward(); opt.step()
            model.eval()
            with torch.no_grad():
                v = float(F.mse_loss(model(Xva, A), yva))
            if v < best - 1e-7:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience: break
        if best_state is not None: model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_preds.append(model(Xte, A).numpy()); val_preds.append(model(Xva, A).numpy())
        stops.append(ep + 1)
        if verbose: print(f"    seed {s:>2}: stop@{ep+1:<5} val_mse={best:.5f}")

    P = np.stack(test_preds)
    return dict(pred=P.mean(axis=0), pred_val=np.stack(val_preds).mean(axis=0),
                per_seed=P, per_seed_rmse=None, stops=stops,
                seed_spread=float(P.mean(axis=(1, 2)).std(ddof=1)))
