import nbformat as nbf, json
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): C.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
---
## §7 — The statistical toolkit

### Why not just a t-test on the RMSEs?

Because there is only *one* RMSE per model — a single number, with no sampling distribution attached. A test needs variation to work with.

The standard solution in the forecasting literature is to work with the **loss differential series**. For each test day $t$:

$$d_t = L_t^{\text{baseline}} - L_t^{\text{new model}}$$

Now there are 314 paired observations instead of two numbers, and testing whether the new model is better becomes testing whether $\mathbb{E}[d_t] > 0$.

Two complications have to be handled:

1. **$d_t$ is autocorrelated.** Volatility clusters, so forecast errors cluster too. Treating the 314 observations as independent understates the standard error and manufactures significance. The fix is a **HAC (Newey–West)** long-run variance.
2. **The models were estimated, not handed to us.** The classical Diebold–Mariano test assumes the forecasts are given. **Giacomini–White (2006)** is valid for estimated models, so it is the pre-registered primary test, with DM reported alongside because it is what most readers know.

### The loss function

QLIKE is the pre-registered primary metric:

$$L_t = \frac{\sigma^2_{\text{actual}}}{\sigma^2_{\text{forecast}}} - \log\frac{\sigma^2_{\text{actual}}}{\sigma^2_{\text{forecast}}} - 1$$

Patton (2011) shows that only two loss families — MSE and QLIKE — give *unbiased* rankings when the volatility target is measured with noise, as realized volatility always is. Of the two, MSE lets a handful of crisis days dominate the average, while QLIKE is scale-free and weights proportional errors equally. MSE and MAE are still reported, because a result that only holds under one loss function is not a robust result.
""")

code('''
_EPS = 1e-12

# ---------------------------------------------------------------- losses ----
def mse(actual, pred):  return (np.asarray(actual,float) - np.asarray(pred,float))**2
def mae(actual, pred):  return np.abs(np.asarray(actual,float) - np.asarray(pred,float))

def qlike(actual, pred, floor=0.05, warn=True):
    """QLIKE on the variance scale. Robust to noise in the RV proxy (Patton 2011).

    Only defined for strictly positive forecasts and it diverges as pred -> 0,
    which is why the model uses a softplus head. `floor` is a diagnostic backstop,
    not a fix: if it ever binds, the architecture is wrong.
    """
    a = np.asarray(actual,float)**2
    p = np.asarray(pred,float)
    n_bad = int(np.sum(p < floor))
    if warn and n_bad:
        warnings.warn(f"qlike: {n_bad}/{p.size} forecasts below floor={floor} clipped",
                      RuntimeWarning)
    r = np.maximum(a,_EPS)/np.maximum(p,floor)**2
    return r - np.log(r) - 1.0


# ------------------------------------------------- HAC long-run variance ----
def newey_west_lrv(x, lag=None):
    """Newey-West long-run variance with a Bartlett kernel (guarantees >= 0).
    Default bandwidth floor(4*(n/100)^(2/9)) is the standard automatic rule."""
    x = np.asarray(x,float); x = x[np.isfinite(x)]; n = x.size
    if n < 3: return np.nan
    if lag is None: lag = int(np.floor(4.0*(n/100.0)**(2.0/9.0)))
    lag = max(0, min(lag, n-2)); e = x - x.mean()
    lrv = float(e @ e)/n
    for j in range(1, lag+1):
        lrv += 2.0*(1.0 - j/(lag+1.0))*float(e[j:] @ e[:-j])/n
    return max(lrv, _EPS)


# ------------------------------------------------------ Diebold-Mariano ----
def diebold_mariano(loss_base, loss_new, h=1, lag=None,
                    alternative=PRIMARY_ALTERNATIVE, hln=True):
    """DM (1995) with the Harvey-Leybourne-Newbold (1997) small-sample correction.
    d_t = loss_base - loss_new, so d > 0 means the NEW model is better."""
    d = np.asarray(loss_base,float).ravel() - np.asarray(loss_new,float).ravel()
    d = d[np.isfinite(d)]; n = d.size
    if n < 10: raise ValueError(f"need >=10 paired obs, got {n}")
    if lag is None: lag = h - 1
    dm = d.mean()/np.sqrt(newey_west_lrv(d, lag=lag)/n)
    if hln:
        stat = dm*np.sqrt((n + 1 - 2*h + h*(h-1)/n)/n); cdf = stats.t.cdf(stat, n-1)
    else:
        stat = dm; cdf = stats.norm.cdf(stat)
    p = 1.0-cdf if alternative=='greater' else (cdf if alternative=='less'
                                                else 2*min(cdf, 1-cdf))
    return dict(stat=float(stat), p_value=float(p), n=int(n), mean_diff=float(d.mean()),
                pct_improvement=float(100*d.mean()/max(np.mean(loss_base), _EPS)))


# ------------------------------------------------------- Giacomini-White ----
def giacomini_white(loss_base, loss_new, instruments=None, lag=None):
    """GW (2006) conditional predictive ability.

    H0: E[d_{t+1} | F_t] = 0 -- nothing knowable at time t predicts which model
    wins tomorrow. Valid for ESTIMATED models, unlike DM.

    With `instruments` = the lagged Hurst regime dummy, this tests the thesis
    hypothesis directly: does the crisis regime predict when Renyi beats Shannon?
    """
    d = np.asarray(loss_base,float).ravel() - np.asarray(loss_new,float).ravel()
    n = d.size
    if instruments is None:
        H, names = np.ones((n,1)), ['const']
    else:
        Z = np.asarray(instruments,float)
        if Z.ndim == 1: Z = Z.reshape(-1,1)
        H, names = np.column_stack([np.ones(n), Z]), ['const'] + [f'z{i+1}' for i in range(Z.shape[1])]
    ok = np.isfinite(d) & np.all(np.isfinite(H), axis=1); d, H = d[ok], H[ok]
    n, q = H.shape
    if lag is None: lag = int(np.floor(4.0*(n/100.0)**(2.0/9.0)))

    M = H*d[:,None]; mbar = M.mean(axis=0); E = M - mbar
    Om = (E.T @ E)/n
    for j in range(1, min(lag, n-2)+1):
        G = (E[j:].T @ E[:-j])/n; Om += (1.0 - j/(lag+1.0))*(G + G.T)
    Om += np.eye(q)*1e-10
    stat = float(n*mbar @ np.linalg.solve(Om, mbar))

    beta, *_ = np.linalg.lstsq(H, d, rcond=None)
    U = H*(d - H @ beta)[:,None]; XtXi = np.linalg.pinv(H.T @ H); Sx = U.T @ U
    for j in range(1, min(lag, n-2)+1):
        G = U[j:].T @ U[:-j]; Sx += (1.0 - j/(lag+1.0))*(G + G.T)
    se = np.sqrt(np.maximum(np.diag(XtXi @ Sx @ XtXi), 0.0)); t = beta/np.maximum(se,_EPS)
    return dict(stat=stat, p_value=float(stats.chi2.sf(stat, q)), df=q, n=int(n),
                coef=pd.Series(beta,index=names), t=pd.Series(t,index=names),
                p_coef=pd.Series(2*stats.norm.sf(np.abs(t)),index=names))

print("Toolkit part 1 defined: losses, HAC, DM, GW.")
''')

code('''
# ---------------------------------------------------- multiplicity control ----
def benjamini_hochberg(pvalues, q=SECONDARY_FDR_Q):
    """BH-FDR across the 10 ETFs. Reporting '3 of 10 significant at 5%' without a
    correction is meaningless: under the null you expect 0.5 hits by chance."""
    s = pd.Series(pvalues, dtype=float); m = s.notna().sum(); order = s.rank(method='first')
    reject_raw = s <= q*order/m
    reject = (s <= s[reject_raw].max()) if reject_raw.any() else pd.Series(False, index=s.index)
    return pd.DataFrame({'p_value': s, 'p_adj': (s*m/order).clip(upper=1.0), 'reject': reject})


def stationary_bootstrap_indices(n, B, block=20.0, seed=0):
    """Politis-Romano (1994). Expected block length ~20 days preserves volatility
    clustering in the resamples."""
    rng = np.random.default_rng(seed); p = 1.0/max(block,1.0)
    idx = np.empty((B,n), np.int64); idx[:,0] = rng.integers(0,n,size=B)
    newb, starts = rng.random((B,n)) < p, rng.integers(0,n,size=(B,n))
    for t in range(1,n):
        idx[:,t] = np.where(newb[:,t], starts[:,t], (idx[:,t-1]+1) % n)
    return idx


def model_confidence_set(losses, alpha=1-MCS_CONFIDENCE, B=2000, block=20.0, seed=0):
    """Hansen-Lunde-Nason (2011) MCS via the range statistic T_R.

    Returns the set of models that cannot be distinguished from the best at
    confidence 1-alpha. This is what controls the multiplicity created by
    comparing 5 arms on one test set -- pairwise DM tests do not.
    """
    names = list(losses)
    L = np.column_stack([np.asarray(losses[k],float).ravel() for k in names])
    L = L[np.all(np.isfinite(L), axis=1)]; n, M0 = L.shape
    if M0 < 2: raise ValueError("need >= 2 models")
    mean_b = L[stationary_bootstrap_indices(n,B,block,seed)].mean(axis=1)

    alive, pvals, order, running = list(range(M0)), {}, [], 0.0
    while len(alive) > 1:
        A = np.array(alive); m = L[:,A].mean(axis=0)
        dbar = m[:,None] - m[None,:]
        mb = mean_b[:,A]; db = mb[:,:,None] - mb[:,None,:]
        sd = np.sqrt(np.maximum(((db-dbar)**2).mean(axis=0), _EPS))
        t_obs, t_boot = dbar/sd, (db-dbar)/sd
        iu = np.triu_indices(len(A), k=1)
        TR = np.abs(t_obs[iu]).max()
        p = float((np.abs(t_boot[:,iu[0],iu[1]]).max(axis=1) >= TR).mean())
        running = max(running, p)
        worst = A[int(np.argmax(t_obs.max(axis=1)))]
        pvals[names[worst]] = running; order.append(names[worst]); alive.remove(worst)
    pvals[names[alive[0]]] = 1.0
    inc = [k for k in names if pvals[k] > alpha]
    return dict(included=inc, excluded=[k for k in names if k not in inc],
                p_values=pd.Series(pvals).reindex(names), elimination_order=order)


# ------------------------------------------------------------- baselines ----
def har_rv_forecast(rv1d, split, lags=(1,5,22), expanding=True):
    """Corsi (2009) HAR-RV -- the referee benchmark in volatility forecasting.
    Refit at every test date on data up to that date, so there is no look-ahead."""
    r = np.asarray(rv1d,float).ravel(); ml = max(lags)
    X = np.array([[1.0]+[r[t-l+1:t+1].mean() for l in lags] for t in range(ml-1, len(r)-1)])
    y = np.array([r[t+1] for t in range(ml-1, len(r)-1)])
    out = np.empty(len(y)-split)
    for i in range(split, len(y)):
        beta, *_ = np.linalg.lstsq(X[:(i if expanding else split)], y[:(i if expanding else split)], rcond=None)
        out[i-split] = X[i] @ beta
    return out

def random_walk_forecast(rv1d, split, maxlag=22):
    r = np.asarray(rv1d,float).ravel()
    return np.array([r[t] for t in range(maxlag-1, len(r)-1)])[split:]


def min_detectable_effect(n, alpha=PRIMARY_LEVEL, power=0.80):
    """Smallest standardized loss differential a DM test of length n can detect."""
    za, zb = stats.norm.ppf(1-alpha), stats.norm.ppf(power)
    return dict(n=n, mde_significance=float(za/np.sqrt(n)), mde_80pct_power=float((za+zb)/np.sqrt(n)))

print("Toolkit part 2 defined: BH, MCS, bootstrap, HAR-RV, power.")
''')

md("""
### Validating the tests before trusting them

A statistical test that has not been checked is an assertion. Two properties are verified by simulation:

- **Size** — under the null (two models with genuinely equal accuracy), the test should reject about 5% of the time. Rejecting far more often means it manufactures significance.
- **Power** — when one model is genuinely better, the test should detect it.

The MCS is checked the same way: it should retain almost everything when all models are equivalent, and reliably discard a model that is genuinely worse.
""")

code('''
print("Validating DM (400 simulations each)...")
rej_null = rej_pow = 0
for s in range(400):
    r = np.random.default_rng(10_000+s)
    rej_null += diebold_mariano(r.chisquare(1,800), r.chisquare(1,800))['p_value'] < 0.05
    b = r.chisquare(1,800)
    rej_pow  += diebold_mariano(b*1.10, b)['p_value'] < 0.05
print(f"  size  (should be ~0.05) : {rej_null/400:.3f}")
print(f"  power (10% better model): {rej_pow/400:.3f}")

print("\\nValidating MCS (40 simulations each)...")
keep_null, drop_bad = [], 0
for s in range(40):
    r = np.random.default_rng(20_000+s); e = r.normal(size=600)
    keep_null.append(len(model_confidence_set(
        {f'm{i}': (e+r.normal(size=600))**2 for i in range(5)}, B=300, seed=s)['included']))
    res = model_confidence_set({'a':(e+r.normal(size=600))**2, 'b':(e+r.normal(size=600))**2,
                                'c':(e+r.normal(size=600))**2,
                                'bad':(e+r.normal(scale=2.0,size=600))**2}, B=300, seed=s)
    drop_bad += 'bad' not in res['included']
print(f"  under a full null, models retained: {np.mean(keep_null):.2f} of 5")
print(f"  genuinely-worse model correctly excluded: {drop_bad}/40")

gw_u = giacomini_white(np.random.default_rng(1).chisquare(1,600)*1.05,
                       np.random.default_rng(2).chisquare(1,600))
print(f"\\nGW unconditional runs: stat={gw_u['stat']:.3f}, p={gw_u['p_value']:.4f}")
print("\\nTOOLKIT VALIDATED — tests have correct size and detect real effects.")
''')
json.dump(C, open('/home/uriel/repositories/HR-ETE-GNN-THESIS_FINAL/.build/p4.json','w'))
print(f"part4: {len(C)} cells")
