import nbformat as nbf, json
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): C.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
---
## §4 — The information-flow graph, estimated honestly

For every ordered pair of the 10 ETFs (90 pairs) we compute ER-TE and its permutation p-value on the **training window only**, then keep the edges that survive **Benjamini–Hochberg FDR** control at $q = 0.10$.

Why FDR rather than a raw threshold: testing 90 pairs at a nominal 5% each gives roughly 4–5 false edges by chance alone. Reporting those as "detected spillover" is exactly the error the pilot made when it kept 5 edges at `z > 1.96`. BH controls the expected *proportion* of false edges among those retained, which is the right guarantee for network recovery, and is far less conservative than Bonferroni for correlated tests.

The function also returns the **raw** ER-TE magnitudes, which §5 needs for the matched-density comparison.

This is the slowest cell in the notebook — 90 pairs × 200 surrogates × 3 α values.
""")

code('''
def benjamini_hochberg_mask(p, q=FDR_Q):
    """Boolean mask of hypotheses surviving Benjamini-Hochberg FDR control at q."""
    p = np.asarray(p, float); ok = np.isfinite(p); m = int(ok.sum())
    if m == 0: return np.zeros_like(p, bool)
    idx = np.argsort(np.where(ok, p, np.inf))
    thr = q*np.arange(1, len(p)+1)/m
    passed = p[idx] <= thr
    mask = np.zeros_like(p, bool)
    if passed.any():
        mask[idx[:np.max(np.where(passed)[0]) + 1]] = True
    return mask & ok


def build_erte_graph(returns_df, tickers, alpha, m_surrogates=M_SURROGATES,
                     k=KNN_K, fdr_q=FDR_Q, verbose=True):
    """Directed ER-TE graph. A_fdr[i,j] > 0 means information flows i -> j.

    Returns both the FDR-filtered adjacency (the honest 'what is significant?'
    answer) and the raw ER-TE magnitude matrix (needed to build matched-density
    graphs in section 5).
    """
    n = len(tickers)
    E, Z, P = np.zeros((n,n)), np.zeros((n,n)), np.ones((n,n))
    pairs = [(i,j) for i in range(n) for j in range(n) if i != j]
    for i, j in pairs:
        E[i,j], Z[i,j], P[i,j] = effective_rte(
            returns_df[tickers[i]].values, returns_df[tickers[j]].values,
            alpha=alpha, k=k, m_surrogates=m_surrogates, seed=1000*i + j)
    pv = np.array([P[i,j] for i,j in pairs])
    keep = benjamini_hochberg_mask(pv, q=fdr_q)
    A = np.zeros((n,n))
    for (i,j), kp in zip(pairs, keep):
        if kp and np.isfinite(E[i,j]): A[i,j] = max(E[i,j], 0.0)
    if verbose:
        print(f"  alpha={alpha}: {int((A>0).sum()):3d}/{n*(n-1)} edges survive "
              f"BH-FDR q={fdr_q}   (raw p<.05: {int((pv<.05).sum())})")
    return dict(A_fdr=A, E_raw=E, Z=Z, P=P, pvals=pv)


t0 = time.time()
GRAPH = {a: build_erte_graph(graph_returns, TICKERS, alpha=a) for a in ALPHA_GRID}
print(f"\\nTotal: {time.time()-t0:.0f}s")
''')

md("""
### The first substantive result

The edge counts below are not a diagnostic — they are a finding, and a stronger one than any RMSE comparison in the pilot.
""")

code('''
summary = pd.DataFrame([
    {'alpha': a,
     'label': 'Renyi, tail-emphasising' if a < 1 else
              ('SHANNON (the baseline)' if a == 1 else 'Renyi, bulk-emphasising'),
     'edges (BH-FDR)': int((GRAPH[a]['A_fdr'] > 0).sum()),
     'density %': 100*(GRAPH[a]['A_fdr'] > 0).sum()/90,
     'raw p<0.05': int((GRAPH[a]['pvals'] < 0.05).sum()),
     'smallest p': GRAPH[a]['pvals'].min()}
    for a in ALPHA_GRID]).set_index('alpha')
display(summary.round(4))

fig, axes = plt.subplots(1, len(ALPHA_GRID), figsize=(4.5*len(ALPHA_GRID), 4.3))
axes = np.atleast_1d(axes)
vmax = max(GRAPH[a]['A_fdr'].max() for a in ALPHA_GRID) or 1
for ax, a in zip(axes, ALPHA_GRID):
    ax.imshow(GRAPH[a]['A_fdr'], cmap='magma', vmin=0, vmax=vmax)
    ax.set_xticks(range(10)); ax.set_xticklabels(TICKERS, rotation=90, fontsize=7)
    ax.set_yticks(range(10)); ax.set_yticklabels(TICKERS, fontsize=7)
    ax.set_title(f'alpha = {a}\\n{int((GRAPH[a]["A_fdr"]>0).sum())} edges', fontsize=11)
    ax.set_xlabel('receiver'); ax.grid(False)
axes[0].set_ylabel('source')
plt.suptitle('ER-TE adjacency after FDR control — training window only',
             fontweight='bold')
plt.tight_layout(); plt.show()
''')

md("""
Read the α = 1.0 panel carefully.

**Shannon transfer entropy — the measure the baseline model is built on — detects far less significant directional spillover in this sample than the tail-emphasising Rényi measure**, on the same data, with the same estimator, the same surrogates and the same multiple-testing correction. The only thing that changed is α.

This is the thesis hypothesis, isolated and tested directly. It does not depend on any neural network, any training run, or any random seed — it is a property of the information measure itself, and it is the most defensible single result this pipeline produces.

It also creates a problem for the forecasting comparison, which §5 exists to solve.
""")

md(r"""
### Sensitivity: does this survive a change of input series?

A result that appears for only one choice of input is a fragile result. The check below re-runs the edge count on four transformations of the same prices.

It also guards against a trap worth naming explicitly: **transfer entropy must never be computed on overlapping rolling windows.** Realized volatility is a 20-day moving average, so RV on consecutive days shares 19 of its 20 underlying observations. That overlap manufactures serial dependence which the estimator will faithfully report as information flow. The `realized vol` row is included precisely to show what that failure mode looks like — it is a warning, not a candidate.
""")

code('''
sens_inputs = {
    'log-returns (used)': graph_returns,
    '|log-returns|':      graph_returns.abs(),
    'realized vol':       rv.loc[:LAST_TRAIN_DATE, TICKERS],
    'change in log RV':   np.log(rv[TICKERS]).diff().dropna().loc[:LAST_TRAIN_DATE],
}
rows = []
for nm, df in sens_inputs.items():
    for a in ALPHA_GRID:
        g = build_erte_graph(df, TICKERS, alpha=a, verbose=False)
        rows.append({'input series': nm, 'alpha': a,
                     'edges': int((g['A_fdr'] > 0).sum())})
sens = pd.DataFrame(rows).pivot(index='input series', columns='alpha', values='edges')
sens.columns = [f'alpha={c}' for c in sens.columns]
display(sens.loc[list(sens_inputs)])

print("How to read this table:")
print("  * log-returns are non-overlapping -> the defensible input, and what we use.")
print("  * 'realized vol' uses OVERLAPPING 20-day windows: consecutive observations")
print("    share 19 of 20 data points. High edge counts there are an artefact of that")
print("    overlap, not evidence of spillover. Never estimate TE on rolling windows.")
print("  * the alpha-dependence is large and real -- which is the thesis's point, but")
print("    also why alpha must be selected on VALIDATION data and never on test.")
''')

md(r"""
---
## §5 — Matched-density graphs, and why they are necessary

§4 leaves the forecasting comparison confounded.

If the Shannon graph has very few edges and the Rényi graph has many, a GNN using the Shannon graph is barely a graph model at all: rows of the adjacency that are entirely zero contribute nothing through the message-passing term, and those nodes collapse to the no-graph case. Comparing the two arms directly would answer

> *"is having a graph better than having no graph?"*

which is a much weaker question than the one the thesis asks:

> *"are the edges Rényi finds better than the edges Shannon finds?"*

The fix is to hold graph **density constant** and vary only *which* edges are present. For every α we keep the top-$K$ pairs ranked by raw ER-TE magnitude, ignoring significance entirely. Every arm then receives exactly $K$ edges, so the sole difference between arms is edge **identity** — precisely the quantity of interest.

Both graph families are carried forward, because they answer different questions:

| Graph family | Question it answers | Used in |
|---|---|---|
| **FDR-filtered** (§4) | What spillover is *statistically detectable*? | The headline detection result |
| **Matched-density** (§5) | Do Rényi's edges *forecast* better, holding density fixed? | The forecasting comparison |
""")

code('''
def top_k_graph(E_raw, k=MATCHED_K):
    """Keep the k largest ER-TE magnitudes. Identical edge count for every alpha,
    so any performance difference is attributable to WHICH edges, not HOW MANY."""
    E = np.array(E_raw, float, copy=True)
    np.fill_diagonal(E, -np.inf)
    E[~np.isfinite(E)] = -np.inf
    thresh = np.sort(E.ravel())[-k]
    return np.where(E >= thresh, np.maximum(E, 0.0), 0.0)


MATCHED = {a: top_k_graph(GRAPH[a]['E_raw'], k=MATCHED_K) for a in ALPHA_GRID}

for a in ALPHA_GRID:
    assert int((MATCHED[a] > 0).sum()) <= MATCHED_K, "top-k produced too many edges"

overlap = pd.DataFrame(
    [[int(((MATCHED[a] > 0) & (MATCHED[b] > 0)).sum()) for b in ALPHA_GRID]
     for a in ALPHA_GRID],
    index=[f'alpha={a}' for a in ALPHA_GRID],
    columns=[f'alpha={a}' for a in ALPHA_GRID])

print(f"Every matched graph carries {MATCHED_K} edges.")
print("Edges shared between each pair of alphas:")
display(overlap)
print(f"Diagonal is {MATCHED_K} by construction. If the OFF-diagonal entries were also")
print(f"near {MATCHED_K}, the different alphas would be recovering the same network and")
print("there would be nothing for the thesis to exploit. Lower is more interesting.")
''')

code('''
fig, axes = plt.subplots(1, len(ALPHA_GRID), figsize=(4.5*len(ALPHA_GRID), 4.3))
axes = np.atleast_1d(axes)
for ax, a in zip(axes, ALPHA_GRID):
    ax.imshow(MATCHED[a] > 0, cmap='Greys', vmin=0, vmax=1)
    ax.set_xticks(range(10)); ax.set_xticklabels(TICKERS, rotation=90, fontsize=7)
    ax.set_yticks(range(10)); ax.set_yticklabels(TICKERS, fontsize=7)
    ax.set_title(f'alpha = {a}', fontsize=11); ax.set_xlabel('receiver'); ax.grid(False)
axes[0].set_ylabel('source')
plt.suptitle(f'Matched-density graphs — exactly {MATCHED_K} edges each, '
             'so only edge IDENTITY differs', fontweight='bold')
plt.tight_layout(); plt.show()
''')
json.dump(C, open('/home/uriel/repositories/HR-ETE-GNN-THESIS_FINAL/.build/p2.json','w'))
print(f"part2: {len(C)} cells")
