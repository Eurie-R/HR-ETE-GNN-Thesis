# Hypotheses for HR-ETE-GNN — Readable Statements, Plain-Language Meaning, and Expected Test Outputs

Prepared for: L.U.L. Ramoy, *HR-ETE-GNN: Robust Volatility Forecasting via Hurst-Regime-Adaptive Rényi Effective Transfer Entropy Graph Neural Networks*

This document has three parts:

1. **Manuscript-ready hypothesis statements** — the same hypotheses, written so a panelist can read them without decoding notation.
2. **What each hypothesis actually claims** — plain-language explanation.
3. **What numbers each RQ1 test produces** — what you will literally put in a results table.

A fourth section flags problems you should fix before the defense.

---

## Part 1 — Manuscript-Ready Hypothesis Statements

### Notation used throughout

| Symbol | Meaning |
|---|---|
| `RV_hat` | one-day-ahead realized volatility forecast |
| `L_t(M)` | forecast loss of model `M` on test day `t` |
| `d_t` | loss differential: `L_t(H-ETE-GNN) − L_t(HR-ETE-GNN)`, averaged across the ten ETFs on day `t` |
| `α` | Rényi order; `α = 1` recovers Shannon, `α < 1` up-weights rare/extreme events |
| `T` | number of test days |

**Reading rule for `d_t`:** a *positive* `d_t` means the Rényi model lost less than the Shannon model on day `t` — the Rényi model won that day. So every "improvement" hypothesis below is a one-sided test that the average of `d_t` is greater than zero.

---

### Research Question 1

> Does replacing Shannon-based Effective Transfer Entropy with Effective Rényi Transfer Entropy as the graph-construction metric produce statistically significant forecasting gains over H-ETE-GNN and the secondary baselines?

---

**Hypothesis 1a — Headline test (pre-registered primary).**

- **H₀:** On average across the test period, HR-ETE-GNN and H-ETE-GNN produce forecasts of equal expected accuracy. The average loss differential is zero, and any observed difference is sampling noise.
- **H₁:** HR-ETE-GNN produces forecasts of strictly lower expected loss than H-ETE-GNN. The average loss differential is positive.

*Test:* Giacomini–White conditional predictive ability test, HAC-robust standard errors, one-sided (greater). *Reported alongside:* Diebold–Mariano with the Harvey–Leybourne–Newbold small-sample correction, and the same machinery applied to RMSE and MAE differentials as secondary loss functions.

---

**Hypothesis 1b — Regime-conditional test (the actual "Hurst-Regime-Adaptive" claim).**

- **H₀:** Whatever accuracy advantage HR-ETE-GNN has, it is the same in turbulent and calm regimes. The lagged Hurst-based turbulence indicator carries no information about when the Rényi model wins; the regime coefficient in the Giacomini–White instrument regression is zero.
- **H₁:** The advantage is concentrated in turbulent regimes. The loss differential is systematically larger on days following an anti-persistent (Hurst < 0.5) regime signal; the regime coefficient is positive.

*Test:* Giacomini–White with the lagged regime indicator as instrument; one-sided on the regime coefficient, HAC standard errors.

---

**Hypothesis 1c — Per-ETF robustness.**

- **H₀:** For each individual ETF *i* (i = 1, …, 10), the average loss differential for that ETF is zero.
- **H₁:** For at least one ETF, the average loss differential is positive — the gain is not driven by a single market.

*Test:* Diebold–Mariano per ETF, with Benjamini–Hochberg false-discovery-rate control at q = 0.10 across the ten tests.

---

**Hypothesis 1d — Multi-model comparison.**

- **H₀:** All candidate models — HR-ETE-GNN at each α, H-ETE-GNN, GNN-ETE, GNN-TE, GNN-Granger, GNN-Pearson, GRU, LSTM, HAR-RV — have equal expected loss.
- **H₁:** At least one model has strictly higher expected loss and can be eliminated with 90% confidence.

*Test:* Hansen–Lunde–Nason Model Confidence Set at 90% confidence, stationary-bootstrap resampling. *Decision rule for RQ1:* the claim is supported if the surviving set contains HR-ETE-GNN and excludes H-ETE-GNN.

---

### Research Question 2

> When the directed weighted adjacency matrix is reconstructed from ER-TE versus Shannon ETE on the same crisis and non-crisis windows, does the resulting topology differ — specifically, do the top volatility-information transmitters among the ten country ETFs change identity under tail-sensitive α < 1 measurement during crisis periods?

---

**Hypothesis 2a — Edge existence.**

For each ordered pair of ETFs (i → j), i ≠ j, at a given α and window:

- **H₀:** ETF *i*'s past returns carry no information about ETF *j*'s future returns beyond what *j*'s own past already provides. The measured ER-TE is indistinguishable from what the circular-shift surrogate null produces.
- **H₁:** A genuine directional information flow exists from *i* to *j*; measured ER-TE exceeds the surrogate null.

*Test:* Permutation test against circular-shift surrogates, with Benjamini–Hochberg FDR control at q = 0.10 across all 90 ordered pairs. Estimated separately for α = 1 and α < 1, and separately for crisis and non-crisis windows.

---

**Hypothesis 2b — Transmitter identity.**

- **H₀:** During crisis windows, the set of dominant volatility-information transmitters — the top-ranked source ETFs by significant out-degree — is the same under Shannon (α = 1) and tail-sensitive Rényi (α < 1) measurement. The two FDR-significant edge sets are statistically indistinguishable.
- **H₁:** At least one ETF's status as a dominant transmitter changes between the Shannon and Rényi crisis networks; the significant edge sets and out-degree rankings diverge under tail-sensitive measurement.

*Status note for the methodology chapter:* as currently implemented this comparison is **descriptive** — edge-overlap tables and adjacency heatmaps, with no attached p-value. To give H2b the same evidentiary weight as H2a, formalize it as a permutation test on the top-K overlap count, or a stationary-bootstrap confidence interval on the Kendall's τ rank concordance between the two out-degree rankings.

---

## Part 2 — What Each Hypothesis Actually Claims

**H1a is the "does it work at all" question.** You take every test day, compute how badly each of the two models missed, subtract, and ask whether the average of that difference is reliably above zero rather than a lucky streak. It answers: *is HR-ETE-GNN better on average across the whole out-of-sample period?* Nothing about when or why.

**H1b is your actual thesis.** H1a can pass with a model that is uniformly 1% better everywhere — which would be a fine result but would not support your Significance of Study argument at all. Your argument is that α < 1 up-weights the tails of the return distribution, and tails only matter when returns are non-Gaussian, which is what crisis regimes are. H1b tests exactly that: not "is the gain positive" but "is the gain *bigger when the market is turbulent*." If H1a passes and H1b fails, you have a model that works but not for the reason you claimed, and the honest write-up says so.

**H1c guards against a lucky market.** A pooled average across ten ETFs can be driven by one country — EWZ or EWY, say — where the model happens to do well. H1c re-runs the test separately per ETF. The Benjamini–Hochberg correction is not optional: with ten independent tests at α = 0.05 you expect roughly half a false positive by chance, so an uncorrected "3 of 10 significant" claim means very little. BH controls the *expected proportion of your significant findings that are false* at 10%.

**H1d places you on the leaderboard.** Pairwise tests only compare two models at a time. The Model Confidence Set takes all candidates at once, repeatedly eliminates the worst-performing model until it can no longer reject equal predictive ability among the survivors, and returns a set that contains the true best model with 90% probability. It is the multiple-comparison-safe way to say "our model is among the best," and the interesting result for you is whether the Shannon baseline gets eliminated while your model survives.

**H2a asks whether an edge is real.** ER-TE is always ≥ 0 by construction and a finite-sample estimate is essentially never exactly zero, so a raw ER-TE value tells you nothing on its own. The circular-shift surrogate destroys the cross-series timing relationship while preserving each series' own autocorrelation structure, giving you a null distribution of "what ER-TE looks like when there is no real flow." The permutation p-value is the fraction of surrogates that beat the observed value.

**H2b asks whether the two measures see a different world.** This is the structural, non-forecasting contribution: if Shannon says Japan is the dominant crisis transmitter and Rényi says Brazil is, that is a finding about measurement, independent of whether either model forecasts better.

---

## Part 3 — What Values the RQ1 Tests Produce

This is what goes in your results table. Numbers below are **illustrative placeholders showing the format**, not results.

### H1a — Giacomini–White and Diebold–Mariano

Both tests are built on the same quantity: the mean loss differential and its standard error.

**Step by step:**

1. `d̄ = (1/T) Σ d_t` — mean loss differential, in loss units.
2. `σ̂_HAC` — Newey–West HAC long-run standard deviation of `d_t`. Report the bandwidth you used (e.g. `⌊4(T/100)^{2/9}⌋`).
3. `SE(d̄) = σ̂_HAC / √T`
4. `t = d̄ / SE(d̄)`
5. One-sided p-value = `1 − Φ(t)` for GW; for HLN-corrected DM, multiply `t` by `√((T + 1 − 2h + h(h−1)/T) / T)` with h = 1, and compare to a t-distribution with T − 1 degrees of freedom.

**Output table format:**

| Quantity | Symbol | Example value | How to read it |
|---|---|---|---|
| Test days | T | 1,043 | Sample size for every test below |
| Mean loss differential | d̄ | 0.0134 | Positive → Rényi wins on average |
| Relative improvement | d̄ / mean baseline loss | 3.1% | The economically meaningful number |
| HAC standard error | SE(d̄) | 0.0058 | Newey–West, bandwidth 6 |
| GW statistic | t | 2.31 | d̄ divided by its standard error |
| One-sided p-value | p | 0.010 | P(seeing this if models are equal) |
| 95% one-sided lower bound | — | 0.0038 | Report this; a p-value alone is not an effect size |
| Decision at 5% | — | Reject H₀ | |
| DM (HLN-corrected) | DM* | 2.28 | Robustness check, same direction |
| DM one-sided p | p | 0.011 | |

**Interpretation guide:**

- `t > 1.645` → reject at 5% one-sided; `t > 2.326` → reject at 1%.
- `d̄ < 0` with a large |t| means the **Shannon baseline wins**. Report it as such — do not switch to a two-sided test after the fact.
- A significant p-value with `d̄ / baseline loss` under ~1% is statistically detectable but practically uninteresting; say so rather than letting the panel say it for you.

**Note on the conditional form.** With q instruments the Giacomini–White statistic is `T · Z̄' Ω̂⁻¹ Z̄`, distributed χ²(q), where `Z_t = h_{t−1} · d_t`. That form is an omnibus, two-sided test — it detects "the models differ" but not "ours is better." For a directional claim, use the single-instrument t-form above. Use the χ² form only for H1b's joint test.

---

### H1b — Regime-conditional coefficient

Estimate, with HAC standard errors:

```
d_t = β₀ + β₁ · Turbulent_{t−1} + ε_t
```

where `Turbulent_{t−1} = 1` when the lagged Hurst estimate is below 0.5.

**Output table format:**

| Quantity | Example value | How to read it |
|---|---|---|
| β̂₀ (calm-regime mean differential) | 0.0061 | Advantage when markets are persistent |
| β̂₁ (extra advantage in turbulence) | 0.0180 | The number your whole thesis rests on |
| HAC SE(β̂₁) | 0.0079 | |
| t-statistic for β̂₁ | 2.28 | One-sided |
| One-sided p-value | 0.011 | |
| Turbulent-regime mean differential (β̂₀ + β̂₁) | 0.0241 | |
| Days in turbulent regime | 287 of 1,043 | **Report this** — it drives your power |
| Joint GW χ²(2) | 7.94 | Test that β₀ = β₁ = 0 |
| χ² p-value | 0.019 | |

**Interpretation guide:**

- `β̂₁ > 0`, significant → the tail-sensitivity story holds; the gain concentrates in turbulence.
- `β̂₀ > 0` but `β̂₁ ≈ 0` → the model is better, but uniformly. Your mechanism claim is unsupported and the manuscript must say so.
- `β̂₀ ≈ 0` and `β̂₁ > 0` → the cleanest possible result for you: no cost in calm markets, real gain in crises. This is exactly the "improving robustness under extreme regimes while preserving calm-period performance" claim in your Research Objectives.
- If the turbulent-regime day count is small (say under 150), a null result is uninformative rather than evidence of no effect. Report the count and the minimum detectable effect.

---

### H1c — Per-ETF DM with Benjamini–Hochberg

**Output table format** (one row per ETF, sorted by raw p-value):

| Rank i | ETF | d̄ᵢ | DM* | Raw p | BH threshold (i/10)·0.10 | BH-adjusted p | Significant? |
|---|---|---|---|---|---|---|---|
| 1 | EWZ | 0.0281 | 3.12 | 0.001 | 0.010 | 0.010 | Yes |
| 2 | EWY | 0.0203 | 2.44 | 0.008 | 0.020 | 0.040 | Yes |
| 3 | EWW | 0.0155 | 2.01 | 0.023 | 0.030 | 0.077 | Yes |
| 4 | EWT | 0.0102 | 1.51 | 0.066 | 0.040 | 0.165 | No |
| … | … | … | … | … | … | … | … |
| 10 | EWU | −0.0041 | −0.62 | 0.732 | 0.100 | 0.732 | No |

**Procedure:** sort the ten raw p-values ascending; find the largest rank *k* where `p₍ₖ₎ ≤ (k/10) · 0.10`; declare ranks 1 through *k* significant. In the example, k = 3.

**Reportable summary:** "3 of 10 ETFs show significant improvement after BH-FDR control at q = 0.10; the mean differential is positive for 8 of 10, and no ETF shows significant deterioration." The last clause matters — it is what rules out the model being harmful somewhere.

---

### H1d — Model Confidence Set

**Output table format:**

| Elimination step | Model eliminated | Range statistic T_R | Bootstrap p | MCS p-value | In 90% set? |
|---|---|---|---|---|---|
| 1 | Random walk | 5.81 | 0.000 | 0.000 | No |
| 2 | HAR-RV | 4.22 | 0.001 | 0.001 | No |
| 3 | LSTM | 3.44 | 0.004 | 0.004 | No |
| 4 | GNN-Pearson | 2.91 | 0.021 | 0.021 | No |
| 5 | GNN-Granger | 2.40 | 0.058 | 0.058 | No |
| 6 | H-ETE-GNN | 2.02 | 0.087 | 0.087 | No |
| — | HR-ETE-GNN (α = 0.7) | — | — | 1.000 | **Yes** |
| — | HR-ETE-GNN (α = 0.5) | — | — | 0.412 | Yes |
| — | GNN-ETE | — | — | 0.183 | Yes |

**How to read it:** the MCS p-value for a model is the significance level at which it would be eliminated. Models with MCS p ≥ 0.10 form the 90% confidence set. The best-performing surviving model always has MCS p = 1.000 by construction — that is not evidence of anything on its own; what matters is *which* models were forced out.

**Note on how you state H1d.** The MCS is a set-construction procedure with a coverage guarantee, not a single hypothesis test. At each step it tests equal predictive ability *among the models still in the set* and eliminates the worst if it rejects. Phrase the null as "all models currently in the set have equal expected loss" rather than "all candidate models have equal expected loss," and report the coverage guarantee rather than a single p-value.

---

## Part 4 — Issues to Resolve Before the Defense

These are ordered by how much damage they do if a panelist raises them first.

**1. Your primary metric is inconsistent between the manuscript and the pre-registration.** Section 3.8 of the manuscript defines RMSE, MAE, MAPE, correlation, and hit ratio. RQ1 on page 4 promises "statistically significant reductions in RMSE, MAE, MAPE, and improvements in correlation and hit-ratio." But the hypotheses above make QLIKE the pre-registered primary metric, and QLIKE appears nowhere in the manuscript. Pick one and make the document internally consistent. QLIKE is the better choice — it is robust to noise in the realized-volatility proxy (Patton, 2011) in a way MAPE and correlation are not — but then you must add it to §3.8 with its formula and justify it there. Leaving the mismatch is the single most likely thing to get caught.

**2. Diebold–Mariano is not valid here as a co-primary test, and you should say why you are still reporting it.** Your α-grid includes 1.0, at which ER-TE reduces to Shannon ETE. That makes H-ETE-GNN a limiting special case of HR-ETE-GNN — the models are *nested*. Standard DM asymptotics assume non-nested models and vanishing parameter-estimation error; under nesting the DM statistic is not asymptotically standard normal and the test is undersized. Giacomini–White is specifically designed to survive this, but only under a **fixed-length rolling estimation window**. So: (a) confirm your regime-adaptive retraining protocol uses a fixed-length window, not an expanding one, and state it explicitly in §3.6; (b) demote DM to "descriptive robustness, reported because it is the familiar benchmark, with the nesting caveat noted"; (c) if a panelist pushes, Clark–West is the standard nested-model alternative.

**3. QLIKE requires strictly positive forecasts.** `QLIKE = RV/RV_hat − log(RV/RV_hat) − 1` is undefined for `RV_hat ≤ 0`, and a GCN trained under MSE loss can and will emit negative volatility forecasts. Decide now whether you clip, apply a softplus output, or forecast log-RV — and report how many forecasts required intervention. If the answer is "many," QLIKE is not usable and you should say so before someone finds it.

**4. Training loss and evaluation loss are different.** You train under MSE and evaluate primarily under QLIKE. This is defensible — QLIKE is proxy-robust while MSE is what the architecture inherits from Lee and Cho — but it is a deliberate choice and needs one sentence of justification in §3.8, not silence.

**5. "Hurst < 0.5 = turbulent" is an assumption, not a definition.** Hurst below 0.5 means anti-persistent, i.e. mean-reverting. That correlates with crisis periods but is not the same thing, and a panelist who knows fractal time series will ask. Either justify the mapping empirically (show that your crisis sub-samples — 2008, 2020, 2022 — actually coincide with H < 0.5 on the MSCI World proxy) or run H1b a second way, using the explicit crisis-window dummy from §3.2 as the instrument. Agreement between the two is a strong result; disagreement is something you need to know before the defense, not during it.

**6. α selection must be out-of-sample, and stated.** If α is chosen by test-set performance, every p-value in Part 3 is invalid. Confirm that α is selected on a validation split and state it in §3.5. If you instead report all seven α values as separate models, the MCS in H1d handles the multiplicity — but then the headline H1a must fix a single pre-specified α, not the best one.

**7. 200 surrogates is thin for H2a.** With m = 200 the smallest achievable permutation p-value is 1/201 ≈ 0.005, and you are applying BH across 90 hypotheses at q = 0.10. That floor limits how many edges can survive correction regardless of how strong the true signal is. Raise it to at least 1,000; the estimator cost is linear and you have the time.

**8. Your α = 0.3 results are outside the estimator's validity bound.** You already flag this in Limitations #2 — that α = 0.3 falls below the α > 0.6 threshold for k = 1. If α = 0.3 turns out to be your best-performing configuration, that limitation becomes the headline objection. Either raise k for the low-α runs, or drop α = 0.3 from the headline and report it in an appendix sensitivity table only.

**9. Confirm `d_t` is a daily cross-sectional average, not a pooled panel.** Your definition says "on test day t," which implies you average the ten ETFs' losses first and then run the test on T daily values. That is the right choice — pooling 10 × T observations would induce strong cross-sectional correlation on each date that a time-series HAC estimator does not correct for, inflating your t-statistics. Make it explicit in the methodology so no one has to guess.

**10. The secondary-metric extensions in your scope note need the right tests.** If you extend beyond QLIKE/RMSE/MAE: hit ratio should use the Pesaran–Timmermann directional accuracy test, not a plain two-proportion test; and comparing two correlations computed against the *same* actuals makes them dependent and overlapping, which requires the Steiger or Williams test, not a plain Fisher-z. Get these right or leave them explicitly out of scope — the second option is entirely defensible for an undergraduate thesis.
