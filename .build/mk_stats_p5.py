import nbformat as nbf, json
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): C.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
---
## §8 — The Hurst regime, and an honest power check

The pilot computed the Hurst exponent, plotted it, and never used it again. Since the "HR" in HR-ETE-GNN stands for *Hurst-Regime adaptive*, this is the first thing a panellist will ask about.

Here the regime enters in the way that matters most: as the **conditioning variable** in the Giacomini–White test. That turns the vague claim *"our model is better"* into the specific, falsifiable claim the thesis actually makes:

> **The Rényi graph should help *specifically when markets are turbulent*.**

Two details make the test legitimate:

- The regime indicator is **lagged by one day**, so it is genuinely known at the time the forecast is made. Conditioning on same-day information would be look-ahead bias smuggled in through the back door.
- The Hurst exponent is computed on the MSCI World proxy over a trailing 250-day window, so it uses no information from the ETFs being forecast.
""")

code('''
def hurst_rs(series):
    """Rescaled-range estimator of the Hurst exponent."""
    s = np.asarray(series, float); N = len(s)
    if N < 20: return np.nan
    out = []
    for lag in np.unique(np.logspace(1, np.log10(N//2), 10).astype(int)):
        vals = []
        for j in range(N//lag):
            seg = s[j*lag:(j+1)*lag]; sd = seg.std(ddof=1)
            if sd > 0:
                dev = np.cumsum(seg - seg.mean()); vals.append((dev.max()-dev.min())/sd)
        if vals: out.append((lag, np.mean(vals)))
    if len(out) < 2: return np.nan
    L, R = zip(*out); return np.polyfit(np.log(L), np.log(R), 1)[0]

hurst = log_ret[REGIME_PROXY].rolling(250).apply(hurst_rs, raw=True).dropna()

# lag by one day so the indicator is genuinely known when the forecast is made
regime_full = (hurst < 0.5).astype(float).shift(1).dropna()
test_dates  = rv.index[sp['t_test']]
regime_test = regime_full.reindex(test_dates).fillna(0.0).values

n_crisis = int(regime_test.sum())
print(f"Whole sample : {int((hurst<0.5).sum()):,} turbulent days of {len(hurst):,} "
      f"({100*(hurst<0.5).mean():.1f}%)")
print(f"TEST period  : {n_crisis} turbulent days of {len(regime_test)} "
      f"({100*regime_test.mean():.1f}%)")

fig, ax = plt.subplots(figsize=(12, 3.8))
ax.plot(hurst.index, hurst.values, color=INK, linewidth=1)
ax.axhline(0.5, color=ACCENT, ls='--')
ax.fill_between(hurst.index, hurst.values, 0.5, where=(hurst.values<0.5),
                color=ACCENT, alpha=.3, label='turbulent (H<0.5)')
ax.axvspan(test_dates[0], test_dates[-1], color='grey', alpha=.15, label='TEST period')
ax.set_title('Hurst regime — and how much crisis data the test period actually contains')
ax.set_ylabel('H'); ax.legend(fontsize=9)
plt.tight_layout(); plt.show()
''')

md("""
### Read the crisis-day count above before reading any result below

This is the most important caveat in the notebook, and it is better stated up front than discovered by a panellist.

The thesis hypothesis is about **crisis regimes**. If the test period contains few or no turbulent days, then the regime-conditional test has almost no data to work with, and a null result there means *"this sample cannot answer the question"* — **not** *"the hypothesis is false"*. Those are very different conclusions and must not be conflated.

The `min_detectable_effect` output in §9 quantifies the same problem for the unconditional test. Both belong in the methodology chapter as a stated limitation, and both are the strongest possible argument for extending the sample back through 2008.
""")

md("""
---
## §9 — Results

Everything now comes together. The battery runs in the pre-registered order.
""")

code('''
y_test = sp['y_test']
LOSSES_2D, FORECASTS = {}, {}

for name, r in RESULTS.items():
    FORECASTS[name] = r['pred']

# --- baselines that are not neural networks -------------------------------
split_idx = sp['t_test'][0] - 21
har  = np.column_stack([har_rv_forecast(rv[t].values, split_idx)[:len(y_test)] for t in TICKERS])
rw   = np.column_stack([random_walk_forecast(rv[t].values, split_idx)[:len(y_test)] for t in TICKERS])
FORECASTS['HAR-RV (Corsi 2009)'] = har
FORECASTS['Random walk']         = rw

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    LOSSES_2D = {k: qlike(y_test, v, warn=False) for k, v in FORECASTS.items()}
POOLED = {k: v.mean(axis=1) for k, v in LOSSES_2D.items()}   # average over ETFs, per day

acc = pd.DataFrame({
    k: {'QLIKE': float(np.nanmean(LOSSES_2D[k])),
        'RMSE':  float(np.sqrt(np.nanmean(mse(y_test, v)))),
        'MAE':   float(np.nanmean(mae(y_test, v)))}
    for k, v in FORECASTS.items()}).T.sort_values('QLIKE')
print("[1] FORECAST ACCURACY  (lower is better; QLIKE is the pre-registered metric)")
display(acc.round(5))
''')

code('''
print(f"[2] PREDICTIVE-ABILITY TESTS vs. {BASELINE}")
print(f"    one-sided, alternative = the listed model beats the baseline\\n")
rows = {}
for k in FORECASTS:
    if k == BASELINE: continue
    dm = diebold_mariano(POOLED[BASELINE], POOLED[k])
    gw = giacomini_white(POOLED[BASELINE], POOLED[k])
    rows[k] = {'DM stat': dm['stat'], 'DM p': dm['p_value'],
               'GW stat': gw['stat'], 'GW p': gw['p_value'],
               'QLIKE improvement %': dm['pct_improvement'],
               'sig at 5%': dm['p_value'] < PRIMARY_LEVEL,
               'meets 5% min effect': dm['pct_improvement']/100 >= MIN_MEANINGFUL_GAIN}
tests = pd.DataFrame(rows).T
display(tests)
''')

code('''
print("[3] REGIME-CONDITIONAL GIACOMINI-WHITE  <-- the thesis hypothesis, tested directly")
print("    beta_regime > 0 and significant  =>  the model helps SPECIFICALLY in crises\\n")
if n_crisis < 10:
    print(f"    *** Only {n_crisis} turbulent days in the test period. This test is")
    print("        UNDERPOWERED and its result must not be interpreted as evidence")
    print("        either way. Reported for completeness only. ***\\n")
rows = {}
for k in FORECASTS:
    if k == BASELINE: continue
    g = giacomini_white(POOLED[BASELINE], POOLED[k], instruments=regime_test)
    rows[k] = {'GW cond stat': g['stat'], 'GW cond p': g['p_value'],
               'beta const': g['coef']['const'], 't const': g['t']['const'],
               'beta regime': g['coef']['z1'], 't regime': g['t']['z1'],
               'p regime': g['p_coef']['z1']}
display(pd.DataFrame(rows).T.round(4))

if n_crisis > 0:
    strat = pd.DataFrame({
        k: {'calm QLIKE': float(np.nanmean(POOLED[k][regime_test < .5])),
            'crisis QLIKE': float(np.nanmean(POOLED[k][regime_test >= .5]))}
        for k in FORECASTS}).T
    strat['calm days'] = int((regime_test < .5).sum())
    strat['crisis days'] = n_crisis
    print("\\n[3b] REGIME-STRATIFIED LOSS")
    display(strat.round(5))
''')

code('''
print("[4] PER-ETF DIEBOLD-MARIANO with Benjamini-Hochberg FDR control")
print(f"    (uncorrected, ~0.5 of 10 ETFs would look significant by chance alone)\\n")
per_etf = {}
for k in FORECASTS:
    if k == BASELINE: continue
    ps = {COUNTRY[t]: diebold_mariano(LOSSES_2D[BASELINE][:,j], LOSSES_2D[k][:,j])['p_value']
          for j, t in enumerate(TICKERS)}
    bh = benjamini_hochberg(pd.Series(ps))
    per_etf[k] = bh
    print(f"  {k:<28} raw p<0.05: {int((bh['p_value']<0.05).sum()):>2}/10   "
          f"after FDR q={SECONDARY_FDR_Q}: {int(bh['reject'].sum()):>2}/10")

print(f"\\n  Detail for the primary treatment arm:")
main_arm = next((k for k in FORECASTS if k.startswith('HR-ETE-GNN (a=0.5')), None)
if main_arm: display(per_etf[main_arm].round(4))
''')

code('''
print(f"[5] {MCS_CONFIDENCE:.0%} MODEL CONFIDENCE SET (Hansen-Lunde-Nason 2011)")
print("    the set of models that cannot be statistically distinguished from the best\\n")
mcs = model_confidence_set(POOLED, alpha=1-MCS_CONFIDENCE, B=2000)
print("  INCLUDED (indistinguishable from best):")
for m in mcs['included']: print(f"     * {m}")
print("  EXCLUDED (significantly worse):")
for m in mcs['excluded']: print(f"     - {m}")
print()
display(mcs['p_values'].sort_values(ascending=False).round(4).to_frame('MCS p-value'))
''')

code('''
print("[6] POWER — can this test set detect anything at all?\\n")
mde = min_detectable_effect(len(y_test))
d_main = POOLED[BASELINE] - POOLED[main_arm] if main_arm else None
print(f"  test-set length n = {mde['n']}")
print(f"  smallest detectable mean(d)/sd(d) at 5%      : {mde['mde_significance']:.4f}")
print(f"  smallest detectable mean(d)/sd(d) at 80% power: {mde['mde_80pct_power']:.4f}")
if d_main is not None:
    obs = d_main.mean()/d_main.std(ddof=1)
    print(f"\\n  OBSERVED mean(d)/sd(d) for {main_arm}: {obs:.4f}")
    if abs(obs) < mde['mde_80pct_power']:
        print("  -> below the 80%-power threshold: this test set is too SHORT to")
        print("     reliably detect an effect of this size. A null result here is")
        print("     inconclusive, not negative. Extend the out-of-sample window.")
    else:
        print("  -> above the 80%-power threshold: the test set is adequately powered.")
''')

code('''
fig, (a1, a2) = plt.subplots(2, 1, figsize=(12.5, 8), height_ratios=[2, 1])
j = TICKERS.index('EWG')
a1.plot(test_dates, y_test[:, j], color='black', linewidth=1.6, label='Actual RV', zorder=5)
for k in ['HAR-RV (Corsi 2009)', BASELINE] + ([main_arm] if main_arm else []):
    a1.plot(test_dates, FORECASTS[k][:, j], linewidth=1.2, alpha=.85,
            label=f"{k}  (QLIKE={acc.loc[k,'QLIKE']:.4f})")
a1.set_title(f'One-day-ahead RV forecast — {COUNTRY["EWG"]} (EWG), test set')
a1.set_ylabel('RV, %'); a1.legend(fontsize=9)

if main_arm:
    cum = np.cumsum(POOLED[BASELINE] - POOLED[main_arm])
    a2.plot(test_dates, cum, color=ACCENT, linewidth=1.5)
    a2.axhline(0, color='black', linewidth=.9)
    a2.fill_between(test_dates, cum, 0, where=(cum > 0), color=CALM, alpha=.25)
    a2.fill_between(test_dates, cum, 0, where=(cum <= 0), color=ACCENT, alpha=.25)
    a2.set_title('Cumulative QLIKE advantage of the Renyi arm over the Shannon baseline\\n'
                 '(rising = Renyi winning; a single jump = one lucky day, not a real edge)',
                 fontsize=11)
    a2.set_ylabel('cumulative loss\\ndifferential')
plt.tight_layout(); plt.show()
''')

md("""
The lower panel is worth dwelling on. A genuine forecasting advantage accumulates **steadily** — the line drifts upward across the whole test period. An advantage that arrives as a **single step** came from one or two days, will not survive into a new sample, and no p-value should persuade you otherwise. Always look at this plot before believing a DM statistic.
""")

md(r"""
---
## §10 — What can and cannot be claimed

### Reporting template

Fill this in from the tables above and put it in the results chapter. It states the effect size, the test, the correction and the limitation in one paragraph, which is what a rigorous panel wants to see.

> Using QLIKE as the pre-registered loss function on a 314-day out-of-sample period, the HR-ETE-GNN with α = 0.5 achieved a mean loss of **____** against **____** for the Shannon ETE-GNN baseline, an improvement of **____%**. A one-sided Giacomini–White test of conditional predictive ability, using a HAC covariance estimator, gives a statistic of **____** (p = **____**). Across the ten individual ETFs, **____** of 10 remained significant after Benjamini–Hochberg FDR control at q = 0.10. The 90% Model Confidence Set contained **____**. Both arms were trained with identical seed lists over **____** seeds and evaluated as seed ensembles; the adjacency matrices were estimated on training data only.

### The distinction that matters most

| What the evidence supports | What it does **not** support |
|---|---|
| Rényi ER-TE detects directional spillover that Shannon ER-TE does not, at matched estimator settings and under FDR control | That this is universal — it is one sample, one asset class, one frequency |
| The two α values recover genuinely different networks | That the difference is *economically* meaningful without a VaR or utility test |
| The forecasting comparison is now free of look-ahead, seed and convergence confounds | That a non-significant regime interaction disproves the hypothesis — the test period may simply contain too few crisis days |

### Remaining work, in priority order

1. **Extend the sample through 2008.** Both the power calculation in §9 and the crisis-day count in §8 point at the same limitation, and it is the binding constraint on every claim in this notebook.
2. **Make the "HR" real.** The regime currently enters only as a *test* conditioner. The thesis claims regime-*adaptive* α — that means $A_t = A(\alpha^*(\text{regime}_t))$, with α re-selected on validation data within each regime.
3. **Walk-forward evaluation.** One fixed split gives 314 test points. A rolling-origin scheme over the full sample gives thousands, which is the cheapest available route to real statistical power.
4. **Raise `SEEDS` to 20+** and `M_SURROGATES` to 500+ for the final run.
5. **Economic significance.** Kupiec and Christoffersen VaR backtests, and a Fleming–Kirby–Ostdiek volatility-timing utility gain, convert "lower QLIKE" into basis points a panel can weigh.
6. **Resolve the Rényi chain-rule issue** (§2) — either adopt the Jizba et al. escort-distribution RTE, or name and cite the difference-based variant explicitly.

### The three questions to rehearse before the defence

**"How do you know the improvement isn't just a lucky random seed?"**
→ §6. Both arms use identical seed lists, all results are seed ensembles, and the within-arm spread is reported so the reader can compare it against the between-arm gap directly.

**"Did your graph see the test data?"**
→ §3. The graph window ends before the test period opens, and the overlap in the pilot's approach is quantified explicitly for contrast.

**"Why should α < 1 help at all?"**
→ §4. Not a story — a measurement. Same data, same estimator, same correction, only α changes, and the number of detectable edges changes with it.
""")
json.dump(C, open('/home/uriel/repositories/HR-ETE-GNN-THESIS_FINAL/.build/p5.json','w'))
print(f"part5: {len(C)} cells")
