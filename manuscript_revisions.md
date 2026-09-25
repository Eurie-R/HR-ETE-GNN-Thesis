# HR-ETE-GNN — Drop-in manuscript revisions
### RQ1 (H1a) confirmatory · RQ2 descriptive

Every block below is paste-ready prose. Citation numbers [21]–[29] are new and
are defined in section 17 at the end.

**Scope of this revision.** Six hypotheses become one plus a descriptive
analysis. H1a carries Research Question 1 and all inferential weight; Research
Question 2 is a descriptive network comparison reported as secondary findings.
H1c (per-ETF), H1d (model confidence set) and H2a (edge existence) are dropped.
H1b (regime-conditional accuracy) is **held back**, not abandoned — its text is
preserved in section 18 and can be reinstated without disturbing anything else.

**Consequence of holding back H1b.** H1b was the only test of the
regime-conditional mechanism, which is the claim the thesis title names and
which Section 1.5 advances. The Hurst regime adaptation remains in the
architecture, in both arms, exactly as Lee and Cho [9] specify it; what is no
longer evaluated is whether the Rényi advantage concentrates in anti-persistent
regimes. Sections 1.5 and 1.4 are revised below so that the mechanism is
presented as the motivation for the study rather than as something it
establishes. This is the honest framing and it is also the one that makes
reinstating H1b later straightforward.

**Single loss function.** QLIKE is the only criterion under which any hypothesis
is tested. RMSE, MAE, MAPE, correlation and hit ratio survive in one place only —
Table 4.0, the replication audit against Lee and Cho's Table 6 — where they carry
no test and exist solely to show that the baseline reproduces the published
result. Because QLIKE is now a single point of failure, its positivity
contingency is pre-specified in Section 3.8 rather than resolved on inspection.

**Controlled-comparison principle.** The two arms of this study — H-ETE-GNN at
α = 1 and HR-ETE-GNN at α\* — must differ in the Rényi order and in nothing
else. Fidelity to Lee and Cho [9] is a separate and subordinate goal: where the
two conflict, control wins; where they do not, the published specification is
followed exactly. Under the revisions below the forecasting pipeline departs
from Lee and Cho in exactly one respect, the row normalisation of Step 6, which
is itself required for control. The remaining changes — QLIKE as primary metric,
and hyperparameters frozen across arms — sit downstream of the models or tighten
the control further. No new estimation machinery is introduced anywhere.

---

## 1. ABSTRACT (p. ii) — replace the final sentence

**Replace:**
> Using daily price data of 10 major country-level ETFs over 19 years (2003-2024), Forecasting performance is assessed via RMSE, MAE, MAPE, Hit-Ratio and correlation, stratified by Hurst regime and crisis sub-samples (2008 GFC, 2020 COVID-19, 2022 Russia-Ukraine).

**With:**

Using daily closing prices of ten country-level iShares MSCI ETFs from February 2003 to May 2024, forecasting performance is evaluated under QLIKE, adopted as the single pre-registered loss function on the grounds that it is robust to noise in the realized-volatility proxy. The confirmatory claim tested is that ER-TE graph construction lowers expected out-of-sample loss relative to H-ETE-GNN, assessed with the Giacomini–White test of conditional predictive ability under heteroskedasticity- and autocorrelation-consistent standard errors, with the two models differing in the Rényi order and in no other respect. As a secondary and descriptive contribution, the ER-TE and Shannon ETE adjacency matrices are compared as networks, characterising whether the countries that emerge as dominant volatility-information transmitters differ under tail-sensitive measurement during crisis windows.

---

## 2. SECTION 1.2, Objective 2 (p. 3) — replace

**Replace:**
> 2. To evaluate whether ER-TE based graph construction yields statistically significant forecasting gains over H-ETE-GNN and competing benchmarks (ETE-GNN, TE-GNN, Granger-GNN, Pearson-GNN, GRU, LSTM).

**With:**

2. To evaluate whether ER-TE based graph construction yields a statistically significant reduction in expected forecast loss relative to H-ETE-GNN over the out-of-sample period. Secondary benchmarks (GNN-ETE, GNN-TE, GNN-Granger, GNN-Pearson, GRU, LSTM, HAR-RV) are reported for descriptive context only.

**Also replace Objective 3:**
> 3. To compare ER-TE and Shannon ETE network topologies across crisis and non-crisis windows, identifying whether tail-sensitive information flow changes which countries are dominant transmitters of volatility information.

**With:**

3. To characterise, descriptively, how ER-TE and Shannon ETE network topologies differ across crisis and non-crisis windows, and whether the countries that emerge as dominant transmitters of volatility information change under tail-sensitive measurement. This objective is secondary to Objective 2 and no statistical claim is attached to it.

---

## 3. SECTION 1.3, Research Question 1 (pp. 3–4) — replace

**Replace:**
> 1 Does replacing Shannon-based Effective Transfer Entropy with Effective Rényi Transfer Entropy (ER-TE) as the graph construction metric in the Graph Neural Network of the H-ETE-GNN framework produce statistically significant reductions in RMSE, MAE, MAPE, and improvements in correlation and hit-ratio for one-day ahead realized-volatility forecasts relative to the H-ETE-GNN and other secondary baseline models.

**With:**

1. Does replacing Shannon-based Effective Transfer Entropy with Effective Rényi Transfer Entropy (ER-TE) as the graph-construction metric in the H-ETE-GNN framework produce a statistically significant reduction in expected QLIKE loss for one-day-ahead realized-volatility forecasts relative to H-ETE-GNN?

---

## 4. SECTION 1.3, Research Question 2 (p. 4) — replace

**With:**

2. When the directed weighted adjacency matrix is reconstructed from ER-TE versus Shannon ETE on the same crisis and non-crisis windows, does the resulting topology differ in a measurable way — specifically, do the top volatility-information transmitters among the ten country ETFs change identity under tail-sensitive α < 1 measurement during crisis periods relative to Shannon measurement?

**Add immediately after the two research questions:**

Research Question 1 is the confirmatory component of this study and carries its inferential claims. Research Question 2 is addressed descriptively: it characterises the structure of the estimated networks, and its findings are reported as secondary.

---

## 5. SECTION 1.4, Scope (pp. 4–5) — replace, then add a paragraph

**Replace:**
> In advancing the H-ETE-GNN hybrid model, only the entropy basis used for graph construction is changed. The multi-scale 1D convolution, three GCN layers, MSE loss, Adam optimizer, and early-stopping criteria are held identical to Lee and Cho [9]. ER-TE is estimated with k = 1, an α-grid of {0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2}, and m-shuffled surrogates for bias correction. Comparisons are drawn against ETE-GNN, TE-GNN, Granger-GNN, Pearson-GNN, GRU, LSTM, and H-ETE-GNN as the primary baseline.

**With:**

In advancing the H-ETE-GNN hybrid model, only the entropy basis used for graph construction is changed. The multi-scale 1D convolution, three GCN layers, MSE training loss, Adam optimizer, and the Z-score edge-retention screen are held identical to Lee and Cho [9]. Two matters they do not settle are resolved explicitly and applied identically to every model compared: the early-stopping patience, which they leave unspecified (Section 3.5), and the scaling of the adjacency matrix, which is the single deliberate departure from their specification and is made in the service of experimental control (Section 3.3, Step 6). ER-TE is estimated with k = l = 1 over the candidate α-grid {0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0}, with m = 100 shuffled surrogates for bias correction, fixed rather than tuned. All remaining hyperparameters are tuned once on the validation split at the baseline configuration α = 1.0 and then frozen, so that the Rényi order is the only quantity differing between the two arms (Section 3.5). A single Rényi order α\* is selected on the validation split, before any test-set forecast is computed, and that value alone carries the confirmatory hypotheses; the remaining orders are reported in an appendix sensitivity table. Selection is restricted to α ≥ 0.6 so that the selected configuration lies inside the α-Gaussian validity region of the k = 1 estimator (Limitation 2). Comparisons are drawn against H-ETE-GNN as the primary and only inferential baseline; ETE-GNN, TE-GNN, Granger-GNN, Pearson-GNN, GRU, LSTM, and HAR-RV are reported descriptively.

**Add as a new paragraph immediately after:**

Three forms of inference are deliberately excluded. First, the study makes no per-asset claim: all hypothesis tests are conducted on the daily cross-sectional average loss differential across the ten ETFs, and no test is performed on any individual ETF. Second, no multiplicity-controlled ranking of the full model set is attempted; a model-confidence-set procedure would be the appropriate instrument for such a claim and is not undertaken here. Third, no edge-level significance testing is performed: the adjacency matrices are constructed by the retention rule of Lee and Cho [9], and the network comparison of Research Question 2 is conducted on the matrices that rule produces, without an accompanying test of whether any individual edge differs from a surrogate null. Accordingly, no statement is made about which markets drive an observed gain, about the standing of HR-ETE-GNN relative to the secondary benchmarks beyond descriptive comparison, or about the statistical significance of any structural difference between the two networks.

---

## 6. SECTION 1.4, Limitations (p. 5) — add items 4 and 5

4. Because all inference is conducted on the cross-sectional average loss differential, a positive result cannot be attributed to any particular market. The gain may be concentrated in a subset of the ten ETFs, and this study does not identify which.

5. Comparisons against the secondary benchmarks are descriptive. No multiplicity-controlled ranking of the model set is claimed, so the results do not establish that HR-ETE-GNN is superior to any model other than H-ETE-GNN.

6. The network comparison of Research Question 2 is descriptive. Edges are retained by the Z-score screen of Lee and Cho [9] rather than by a test controlling the false-discovery rate across the ninety ordered pairs, so the structural differences reported may in part reflect estimation noise. No significance is claimed for them, and the study's inferential conclusions rest on Research Question 1 alone.

7. The regime-conditional mechanism is not tested. The theoretical case set out in Section 2.1.5 and Section 1.5 is that tail-sensitive measurement should matter when returns deviate from Gaussianity and should be immaterial when they do not, which implies that any advantage of ER-TE ought to concentrate in anti-persistent regimes. This study establishes only whether an advantage exists on average across the out-of-sample period; it does not establish whether the advantage arises for the reason proposed. A formal test would condition the loss differential on the lagged regime indicator, and it is identified in Chapter V as the principal extension of this work.

---

## 6b. SECTION 1.5, Significance of Study (pp. 5–6) — replace the first paragraph

**Replace:**
> The study directly operationalizes the stated recommendations of Lee and Cho [9] that non-Shannon entropies are to be explored within the H-ETE-GNN framework by providing a calibrated α-grid and a regime-stratified evaluation protocol. In bridging non-Shannon entropies, we implement the information-theoretic framework of Jizba, Lavička, and Tabachová [11] on Rényi Transfer Entropy with the H-ETE-GNN model. By grounding graph construction in a tunable-order entropy, the work supplies a principled rationale for why tail-sensitive α < 1 should matter precisely when returns deviate from Gaussianity, and why it should be immaterial when they do not.

**With:**

The study directly operationalizes the stated recommendation of Lee and Cho [9] that non-Shannon entropies be explored within the H-ETE-GNN framework, by providing a calibrated α-grid and a controlled comparison in which the entropy basis is the only quantity that varies between the two models. In bridging non-Shannon entropies, we implement the information-theoretic framework of Jizba, Lavička, and Tabachová [11] on Rényi Transfer Entropy within the H-ETE-GNN model.

Grounding graph construction in a tunable-order entropy supplies a principled rationale for why tail-sensitive α < 1 should matter precisely when returns deviate from Gaussianity, and why it should be immaterial when they do not: by Theorem 3.1 the Rényi and Shannon measures coincide exactly for Gaussian processes, and by Theorem 3.2 they separate as returns become heavy-tailed. That rationale is the motivation for this study and is what makes the substitution worth making. It is not, however, what this study tests. The confirmatory question here is whether the substitution improves forecasts at all, averaged across the out-of-sample period; whether any improvement arises through the tail-sensitivity channel specifically is a distinct question, requiring the loss differential to be conditioned on the prevailing regime, and it is left to the extension identified in Chapter V. Distinguishing the two is deliberate: a mechanism cannot usefully be attributed to an effect that has not first been shown to exist.

---

## 7. SECTION 3.1, Data (p. 19) — add at the end of the section

The supervised samples are formed with a look-back window of 20 trading days and split chronologically into training (60%), validation (20%), and test (20%) blocks, with no shuffling at any point. All graph estimation, hyperparameter selection, and Rényi-order selection use the training and validation blocks only; the ER-TE adjacency matrix is estimated on training-block returns alone, so that no test-period observation informs the graph the model uses to forecast it. On the full February 2003 to May 2024 sample this places the start of the test block in 2020, so that the COVID-19 episode falls inside the evaluation period.

The composition of the test block is reported rather than assumed. Chapter IV states the number of test days classified as anti-persistent by the procedure of Section 3.6 and the number falling inside the crisis windows of Section 3.2, together with mean losses stratified by regime. These figures are descriptive under the present scope; they are reported because a forecasting comparison conducted almost entirely within calm markets supports a narrower claim than one spanning both regimes, and a reader is entitled to know which was the case. The networks compared under Research Question 2 are estimated on the crisis and non-crisis windows of Section 3.2 directly, independently of this forecasting split.

---

## 8. SECTION 3.2, Crisis sub-samples (p. 20) — replace the second paragraph

**Replace:**
> These periods are selected based on the Hurst exponent dropping below 0.5, indicating anti-persistent and crisis-characteristic behavior. These time periods are also well-documented episodes of extreme market stress.

**With:**

These windows are defined ex ante by reference to the documented episodes of extreme market stress themselves, and not by the behaviour of the Hurst exponent. Defining them by the Hurst exponent and then offering the Hurst exponent as evidence of their crisis character would be circular.

The relationship between the two is instead treated as an empirical question. A Hurst exponent below 0.5 indicates anti-persistence, that is, mean reversion, which is a property associated with but not identical to market crisis. Chapter IV therefore reports the degree of overlap between the three calendar windows above and the days classified as anti-persistent by the regime-detection procedure of Section 3.6, so that the correspondence between the two notions is established on this sample rather than assumed from the literature.

These windows serve two purposes under the present scope. They define the crisis and non-crisis periods on which the two adjacency matrices are estimated and compared for Research Question 2, which is where they carry most of their weight. They also define the strata for the descriptive regime-stratified loss table of Chapter IV. They are not used to condition any hypothesis test; that use is identified in Chapter V as an extension.

---

## 9. SECTION 3.3, Steps 4–6 (p. 21) — replace, and add Step 7

**Replace Step 4's closing text with (keep the existing formula):**

**Step 4 — Effective correction.** For each pair, generate m = 100 shuffled surrogates of Y and compute [existing formula]. Shuffling destroys the temporal structure of Y while preserving its marginal distribution, so that the mean surrogate RTE estimates the finite-sample bias, following Marschinski and Kantz [10]. The number of surrogates is fixed at m = 100, the upper end of the range searched by Lee and Cho [9], and is not tuned. The ER-TE point estimate, and therefore which edges survive the screen of Step 5, is sensitive to m; allowing it to be selected separately for each arm would permit the graph itself to be chosen on outcome.

**Step 5 — retain Lee and Cho's text unchanged. Do not replace it.** The Z > 1.96 screen against the shuffled distribution stays exactly as specified. Append only the following sentence:

This retention rule is that of Lee and Cho [9] and is applied without modification, so that the adjacency-construction pipeline feeding the graph network is identical to theirs in every respect except the entropy basis. The formal edge-significance testing required by Hypothesis 2a is a separate analysis, conducted in Step 7, and does not alter the matrix used for forecasting.

**Replace Step 6 in full:**

**Step 6 — Adjacency matrix.** The resulting n × n sparse, directed, weighted matrix A_α serves as the GCN adjacency for a given training window and Rényi order α. Rows of A_α are normalised to sum to one before use.

Row normalisation is the single deliberate departure from the literal specification of Lee and Cho [9], and it is made in the service of experimental control rather than in spite of it. Under their node-update rule (Section 3.4) the neighbourhood term enters in proportion to the raw magnitude of the adjacency entries, so the influence of the graph on the forecast is a function of the scale of the edge weights and not only of their pattern. Rényi and Shannon transfer entropies do not share a common scale: the order α enters the Rényi entropy through the factor 1/(1 − α), so ER-TE magnitudes vary systematically with α even where the underlying network is unchanged. On raw weights, a comparison between the α = 1 and α = α\* arms would therefore confound a difference in edge topology, which is the object of this study, with a difference in edge magnitude, which is an artefact of the estimator. Row normalisation removes that confound by placing every adjacency on a common scale, and it is applied identically to both arms and to every secondary graph model.

A second consideration reinforces the choice. Measured on the first training block of this sample, the mean magnitude of the neighbourhood term under a raw ETE adjacency is 0.007 of the self term, against 0.648 after row normalisation. Under raw weights the graph is accordingly close to inert and the architecture degenerates toward a per-node multilayer perceptron, in which case neither arm is meaningfully a graph model and the comparison has little to detect. The baseline reported throughout is Lee and Cho's model with a row-normalised adjacency, and results under the literal raw-weight specification are reported as a robustness check in the appendix.

**Add Step 7:**

**Step 7 — Network comparison (descriptive).** For Research Question 2, Steps 1–6 are executed twice on the same window, once at α = 1 (Shannon) and once at the selected tail-sensitive order α\*, yielding two retained adjacency matrices. Because both are produced by the same pipeline that generates the forecasting graphs, this comparison requires no additional estimation and introduces no procedure not already used for Research Question 1.

For each matrix, each ETF's out-degree is computed as the number of retained outgoing edges and its weighted out-degree as the sum of their ER-TE values. The two networks are then compared on three quantities: (i) the identity and ordering of the top-K transmitters, K = 3; (ii) the number of directed edges common to both matrices, reported as an overlap table alongside adjacency heatmaps; and (iii) Kendall's rank correlation τ [22] between the two out-degree rankings, with a 90% confidence interval obtained by stationary bootstrap [21] with expected block length 20 trading days. All three are reported separately for crisis and non-crisis windows.

The confidence interval on τ is reported to convey how much of the observed rank divergence is compatible with sampling variation. It is a single pre-specified quantity rather than a battery of edge-level tests, and it is presented as a descriptive aid; the language used to discuss it in Chapters IV and V is that of compatibility with noise, not of statistical significance.

---

## 10. SECTION 3.5, Table 3.5.1 (pp. 22–23) — amend two rows

**Replace the "Shuffle Simulations (m)" row's Search Space cell:**
> [ 10 ~ 100 ]

**With:**
> Fixed at 100 (not tuned; see Section 3.3, Step 4)

**Replace the "Rényi Order (α)" row's Search Space cell:**
> {0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0}

**With:**
> Candidate set {0.7, 0.9, 1.0, 1.5, 2.0}; a single α\* selected on the validation split

---

## 11. SECTION 3.5 (p. 23) — replace the closing sentence, add two paragraphs

**Replace:**
> When α = 1.0, HR-ETE-GNN reduces exactly to the original H-ETE-GNN [9], ensuring backward compatibility.

**With:**

**Controlled comparison protocol.** The comparison between H-ETE-GNN and HR-ETE-GNN is designed so that the Rényi order α is the only quantity differing between the two arms. All remaining hyperparameters in Table 3.5.1 — learning rate, batch size, both hidden-state sizes, input size M, training period N, and regime sensitivity s — are tuned once by Bayesian optimisation on the validation split at the baseline configuration α = 1.0, and are then frozen and applied unchanged at every value of α. The number of surrogate series m is likewise fixed at 100 rather than tuned.

Tuning each arm separately would be the more conventional protocol, but it would defeat the purpose of the study: the two arms would then differ in α and in seven further quantities simultaneously, and no observed difference in forecast accuracy could be attributed to the entropy basis. Freezing the configuration at the baseline optimum is additionally conservative with respect to the hypothesis under test, since HR-ETE-GNN is evaluated under hyperparameters selected for its competitor rather than for itself; a gain observed under this protocol cannot be an artefact of more favourable tuning. Per-arm tuning is reported in the appendix as a sensitivity analysis.

The Rényi order α is not tuned on test-set performance. A single α\* is selected by validation loss and that value alone is used for the confirmatory tests of Section 3.9; orders below 0.6 are excluded from selection for the reason given in Limitation 2, and α = 0.3 and α = 0.5 are reported in an appendix sensitivity table only.

When α = 1.0, HR-ETE-GNN reduces exactly to the original H-ETE-GNN [9], ensuring backward compatibility. This exact nesting has a consequence for inference which is addressed in Section 3.9.1.

**Early stopping.** Lee and Cho [9] specify a budget of 100 epochs with early stopping but do not report the patience parameter, so no choice here can be said to follow them. The full epoch budget is used without early termination, and best-validation weights are restored in every case. For the primary comparison this choice is close to immaterial, since both arms are the same architecture and are affected identically by it. It matters for the descriptive benchmark table of Section 3.7: measured on this panel, the longest run of non-improving epochs preceding the eventual best validation loss was 50 epochs for LSTM and 19 for GRU, against 17 for the graph model, so a short patience would truncate the recurrent baselines while leaving the graph models untouched and would overstate the graph-versus-recurrent gap.

---

## 12. SECTION 3.6 (pp. 23–24) — add as a closing paragraph

The estimation window is of fixed length N throughout and does not expand as the test period progresses. Each retraining event re-estimates the adjacency matrix and fine-tunes the network on the most recent N days only, discarding earlier observations. This is not merely an implementation convenience. The Giacomini–White test of conditional predictive ability, which is the primary inferential instrument of Section 3.9, is valid for forecasts from estimated models only under a fixed-length rolling estimation scheme, because it treats the finite-window forecasting method — model, estimation window, and all — as the object of the test rather than a limiting population model [26]. An expanding window would invalidate that asymptotic argument.

---

## 13. SECTION 3.7 (pp. 24–25) — amend the table, add a note

In Table 3.7.1, change the HAR-RV row label to: **HAR-RV (univariate) [23]**

**Add immediately below the table:**

All models other than H-ETE-GNN are reported for descriptive context only. No inferential comparison is made against them and no multiplicity correction across the model set is applied; see Section 1.4. All graph-based models use the row-normalised adjacency described in Section 3.3, Step 6, so that differences among them reflect edge topology rather than edge scale.

---

## 14. SECTION 3.8, Evaluation Metrics (p. 25) — replace the section in full

### 3.8 Evaluation Metrics

**Primary loss function.** The pre-registered primary loss function is QLIKE, evaluated on the variance scale:

  QLIKE_t,i = ( RV_t,i / R̂V_t,i ) − ln( RV_t,i / R̂V_t,i ) − 1

where RV_t,i and R̂V_t,i denote the realized and forecast variance of ETF i on day t. QLIKE is non-negative, attains zero only at R̂V = RV, and belongs to the family of loss functions that Patton [24] shows to be robust to noise in the volatility proxy. This property is essential here: realized volatility is itself an estimate of latent volatility rather than an observation of it, and a loss function lacking robustness in this sense can rank a worse forecast above a better one purely as an artefact of proxy noise. Of the metrics used by Lee and Cho [9], only RMSE has this property; MAPE and correlation do not. QLIKE is preferred to RMSE as the primary metric because it is additionally scale-free in the forecast ratio and does not permit a small number of extreme days to dominate the sample average. The latter property matters here: realized volatility in this sample spans two orders of magnitude between calm and crisis periods, so under a squared-error criterion a handful of crisis days would determine the comparison, and a difference in average accuracy would in practice be a difference in crisis-day accuracy without being reported as one.

**Positivity.** QLIKE is defined only for strictly positive forecasts and diverges as R̂V → 0. The network terminates in a linear layer and can therefore emit non-positive volatility forecasts, which are economically meaningless and, left uncorrected, sufficient for a single day to dominate the test-period average. Forecast volatility is accordingly floored at 0.05 in the units of RV — that is, 0.05% daily volatility, below any realized value in the sample — before squaring. The floor is applied identically to both arms, so it cannot advantage either, and the number and proportion of forecasts requiring it is reported in Chapter IV for every model.

Because QLIKE is the sole criterion under which any hypothesis is tested, this intervention is load-bearing in a way it would not be alongside a secondary loss, and the contingency is therefore specified in advance rather than decided on inspection. If more than 1% of either arm's forecasts fall below the floor, the QLIKE comparison is reported as unreliable and the confirmatory test is conducted instead on the log-variance criterion

  L_t,i = ( ln RV_t,i − ln R̂V_t,i )²

which is scale-free, is defined for any non-zero forecast, and is likewise robust to proxy noise. This substitution is declared here so that it cannot be made after the p-value is known; the reported results state which criterion was used and why. The preferable remedy is architectural — constraining the output head to be positive, or forecasting log-RV and exponentiating — and is noted in Chapter V as an extension, since altering the output layer would depart from the architecture held fixed in Section 3.4.

**A single loss function.** QLIKE is the only loss function under which any hypothesis is tested. No secondary loss carries an inferential claim, and the test of Section 3.9.1 is not repeated under an alternative criterion.

This is a restriction rather than an omission, and it is made for a reason. Testing the same hypothesis under several loss functions and reporting whichever attains significance is an uncontrolled multiple comparison: three loss functions afford three opportunities to reject a true null, and the nominal 5% level is no longer the actual one. Pre-committing to a single proxy-robust criterion, chosen on the grounds given above and fixed before the test block is used, is the stricter procedure. The consequence is stated plainly: this study establishes whether the Rényi substitution improves forecasts *under QLIKE*, and does not establish that the result would survive under a different criterion.

**Training versus evaluation loss.** The network is trained under mean squared error and evaluated primarily under QLIKE. This is a deliberate choice rather than an inconsistency. MSE is inherited unchanged from Lee and Cho [9] and is retained so that the training objective is not altered alongside the entropy basis, which would confound the attribution that this study is designed to make; QLIKE is adopted for evaluation because comparing forecasts against a noisy proxy requires a proxy-robust loss in the sense of Patton [24]. Both models in the primary comparison are trained under the identical objective, so the choice cannot advantage either.

**Metrics retained for replication only.** RMSE, MAE, MAPE, correlation, and hit ratio, as defined by Lee and Cho [9], are computed and reported once, in Chapter IV, in the format of their Table 6. Their sole purpose is to establish the fidelity of the baseline replication against the published figures: a reader cannot judge whether the H-ETE-GNN implemented here reproduces the H-ETE-GNN of the source paper unless it is reported in the source paper's own units. Replication is assessed on the ordering of models and the gaps between them rather than on exact agreement of levels, since Lee and Cho do not report the tuned learning rate, batch size, hidden sizes, or shuffle count that produced their table.

  [retain the existing RMSE, MAE, MAPE, correlation, and hit-ratio formulas here]

None of these five quantities carries a hypothesis test, and none is used to compare HR-ETE-GNN against H-ETE-GNN. Beyond the multiplicity argument above, three of them could not support such a test in any case: MAPE and correlation are not robust to volatility-proxy noise in Patton's [24] sense, a valid comparison of hit ratios requires a test of directional accuracy, and a valid comparison of two correlations computed against the same realized series requires a procedure for dependent, overlapping correlations rather than an independent-samples test. The latter two instruments lie outside the scope defined in Section 1.4.

**The loss differential.** All tests in Section 3.9 operate on a single daily series. For each test day t, the loss differential is

  d_t = (1/n) Σ_i [ L_t,i(H-ETE-GNN) − L_t,i(HR-ETE-GNN) ]

that is, the cross-sectional average across the n = 10 ETFs of the per-asset loss difference, taken first, so that each test is applied to T daily observations rather than to a pooled panel of n × T. A positive d_t means HR-ETE-GNN incurred the smaller loss on day t, and every hypothesis below is therefore one-sided in the direction d > 0. Pooling the panel would be inappropriate: the ten ETFs' losses are strongly correlated within a date, a time-series HAC estimator does not correct for that cross-sectional dependence, and a pooled test would consequently understate the standard error and overstate significance.

**Standard errors.** All standard errors are heteroskedasticity- and autocorrelation-consistent, computed with the Newey–West estimator [25] under a Bartlett kernel with bandwidth ⌊4(T/100)^(2/9)⌋, where T is the number of test days; the bandwidth actually used is reported. HAC correction is necessary rather than precautionary here, since volatility clusters and loss differentials are in consequence strongly serially correlated, so that an i.i.d. variance estimate would manufacture significance.

---

## 15. NEW SECTION 3.9 — insert after 3.8

### 3.9 Hypotheses and Test Procedures

Research Question 1 is operationalized as a single confirmatory hypothesis, stated in advance of any test-set evaluation: the Rényi order α\* and all hyperparameters are fixed on the validation split before the test block is used. Research Question 2 is addressed descriptively and is set out in Section 3.9.2.

#### 3.9.1 Hypothesis 1 — Average predictive accuracy

**H₀:** E[d_t] = 0. Over the test period, HR-ETE-GNN and H-ETE-GNN have equal expected loss, and any observed difference is sampling variation.

**H₁:** E[d_t] > 0. HR-ETE-GNN has strictly lower expected loss than H-ETE-GNN.

**Test.** The Giacomini–White test of conditional predictive ability [26] in its unconditional form, one-sided, with the HAC standard error of Section 3.8. The statistic is t = d̄ / SE(d̄) and the one-sided p-value is 1 − Φ(t). Reported alongside, on the same QLIKE differential: the Diebold–Mariano statistic [27] with the Harvey–Leybourne–Newbold small-sample correction [28].

**Decision rule.** H₀ is rejected at the 5% level for t > 1.645. In addition to the p-value, the mean differential d̄, its expression as a percentage of the baseline's mean loss, and a one-sided 95% lower confidence bound are reported. A statistically significant improvement smaller than 5% of baseline loss is reported as statistically detectable but economically immaterial. A negative d̄ with a large statistic is reported as evidence that the Shannon baseline is superior; the test is not converted to a two-sided form after the fact.

**Nesting caveat.** At α = 1 the Effective Rényi Transfer Entropy reduces exactly to Shannon Effective Transfer Entropy, so H-ETE-GNN is a limiting special case of HR-ETE-GNN and the two models are nested. Under nesting the Diebold–Mariano statistic is not asymptotically standard normal and the test is undersized [29]. Diebold–Mariano is therefore reported as descriptive corroboration, because it is the familiar benchmark in this literature, and not as evidence. Giacomini–White is the test of record: its asymptotics are derived for forecasts produced by models estimated on a fixed-length rolling window [26], which is the scheme specified in Section 3.6, and do not require the models to be non-nested. Clark and West [29] provide the standard nested-model alternative should a more specialised instrument be required.

#### 3.9.2 Research Question 2 — Descriptive network comparison

Research Question 2 concerns the structure of the estimated networks rather than the accuracy of the forecasts they support, and it is addressed descriptively. No hypothesis test is attached to it, and no claim of statistical significance is made about any structural difference reported under it. Its findings are secondary to those of Section 3.9.1.

**Expectation.** During crisis windows, the set of dominant volatility-information transmitters — the top-ranked source ETFs by out-degree in the retained adjacency matrix — is expected to differ between Shannon (α = 1) and tail-sensitive (α\*) measurement, and to coincide outside them. The basis for this expectation is the theory of Section 2.1.5 and is directional: by Theorem 3.1 the two measures agree exactly when returns are Gaussian, which is the calm-period case, and by Theorem 3.2 they diverge as returns become heavy-tailed.

**Evidence reported.** The three quantities of Section 3.3, Step 7 — top-K transmitter identity, edge overlap counts with adjacency heatmaps, and Kendall's τ between the two out-degree rankings with a stationary-bootstrap confidence interval — computed separately for crisis and non-crisis windows.

**Why this is reported descriptively.** A formal test of the null that the two transmitter rankings are identical would require a resampling procedure over the edge set together with correction for the ninety ordered pairs screened simultaneously. That machinery lies outside the scope defined in Section 1.4. The consequence is stated plainly rather than left to the reader to infer: the structural differences reported in Chapter IV may in part reflect estimation noise, and no significance is claimed for them. They are offered as a characterisation of what the two measures see, and as context for the forecasting results, which are where the inferential weight of this study rests.

**Relation to the forecasting hypotheses.** The structural comparison is a claim about measurement, not about prediction, and the two are independent by construction. The two entropy bases may describe visibly different networks without that difference translating into forecast accuracy under the architecture of Section 3.4, and conversely. The two sets of results are therefore reported without conditioning either on the other, and in particular a divergence in network structure is not offered as evidence for, nor its absence as evidence against, the hypothesis of Section 3.9.1.

---

## 16. CHAPTER IV — table shells

**Table 4.0** Baseline replication, in the format of Lee and Cho's Table 6
Columns: Model | RMSE | MAE | MAPE | Correlation | Hit ratio
Rows: the full benchmark set of Table 3.7.1. Descriptive, and the only place these five metrics appear. Establishes that the replicated H-ETE-GNN reproduces the published ordering and gaps before any comparison against HR-ETE-GNN is made. Judge it on ordering and gaps, not on exact levels.

**Table 4.1** Hypothesis 1a — average predictive accuracy (primary loss: QLIKE)
Columns: Quantity | Symbol | Value | Note
Rows: Test days (T); Turbulent days in test block; Mean loss differential (d̄); Relative improvement (d̄ / mean baseline loss); HAC standard error (bandwidth); GW statistic; GW one-sided p; 95% one-sided lower bound; DM\* (HLN-corrected); DM one-sided p; Decision at 5%
There is no equivalent table under a second loss function; QLIKE is the sole inferential criterion (Section 3.8).

**Table 4.4** Regime-stratified losses, descriptive
Columns: Regime | Test days | Share of test block | Mean QLIKE, H-ETE-GNN | Mean QLIKE, HR-ETE-GNN | Difference
Rows: persistent (H ≥ 0.5); anti-persistent (H < 0.5); inside a Section 3.2 crisis window; outside one.
Descriptive only — no test is attached. Reported so that the composition of the evaluation period is visible and so that a reader can see whether the average result of Table 4.1 is drawn from one regime or from both.

**Table 4.4b** Robustness — primary test under the literal raw-weight adjacency
Columns: Loss | d̄ | GW t | one-sided p | DM\* | one-sided p
One row, QLIKE. Supports the departure recorded in Section 3.3, Step 6.

**Table 4.5** Network structure, descriptive (Research Question 2)
Columns: Window | α | Retained edges | Density | Mean edge weight | Top-3 transmitters by out-degree
Rows: crisis × {1.0, α\*}, non-crisis × {1.0, α\*}

**Table 4.6** Network divergence, descriptive (Research Question 2)
Columns: Window | Top-3 transmitters, Shannon | Top-3 transmitters, α\* | Common to both | Edges in both matrices | Edges in exactly one | Kendall τ | 90% bootstrap CI
Accompany with paired adjacency heatmaps, Shannon against α\*, one pair per window.

**Table 4.7** Diagnostics, reported for every model
Columns: Model | Forecasts below QLIKE floor (n, %) | Epochs used | Across-seed sd of test loss

**Table 4.8** Regime validation (supports Section 3.2)
Columns: Crisis window | Trading days | Days with H < 0.5 | Share | Days flagged anti-persistent by the Section 3.6 rule | Share

---

## 17. BIBLIOGRAPHY — new entries

[21] Politis, D.N., and Romano, J.P. 1994. The stationary bootstrap. *Journal of the American Statistical Association* 89, 428 (1994), 1303–1313.

[22] Kendall, M.G. 1938. A new measure of rank correlation. *Biometrika* 30, 1–2 (1938), 81–93.

[23] Corsi, F. 2009. A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics* 7, 2 (2009), 174–196.

[24] Patton, A.J. 2011. Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics* 160, 1 (2011), 246–256.

[25] Newey, W.K., and West, K.D. 1987. A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica* 55, 3 (1987), 703–708.

[26] Giacomini, R., and White, H. 2006. Tests of conditional predictive ability. *Econometrica* 74, 6 (2006), 1545–1578.

[27] Diebold, F.X., and Mariano, R.S. 1995. Comparing predictive accuracy. *Journal of Business & Economic Statistics* 13, 3 (1995), 253–263.

[28] Harvey, D., Leybourne, S., and Newbold, P. 1997. Testing the equality of prediction mean squared errors. *International Journal of Forecasting* 13, 2 (1997), 281–291.

[29] Clark, T.E., and West, K.D. 2007. Approximately normal tests for equal predictive accuracy in nested models. *Journal of Econometrics* 138, 1 (2007), 291–311.

**Housekeeping:** [12] Kupiec (1995) and [13] Christoffersen (1998) appear in the bibliography but are not cited anywhere in the text. They are VaR backtests and have no role under the revised scope. Delete them and renumber, or cite them.

---

## 18. HELD BACK — Hypothesis 1b, for later reinstatement

Nothing in sections 1–17 depends on this block. To reinstate H1b: paste it back as Section 3.9.2, renumber the current 3.9.2 to 3.9.3, restore clause (b) of RQ1 and the regime clause of Objective 2, revert Limitation 7 and the second paragraph of the Section 1.5 replacement, and restore Table 4.4 to the regression format given at the end of this section. The regime instrument itself is already produced by the pipeline (Section 3.6), so no new estimation is required.

### 3.9.2 Hypothesis 1b — Regime-conditional accuracy

The regression of interest, estimated with HAC standard errors, is

  d_t = β₀ + β₁ · Turbulent_{t−1} + ε_t

where Turbulent_{t−1} = 1 if the regime label of Section 3.6 is anti-persistent on day t − 1, and 0 otherwise. The indicator is lagged by one day so that it is measurable with respect to the information set available when the forecast is made.

**H₀:** β₁ = 0. Whatever accuracy difference exists between the two models is the same in anti-persistent and persistent regimes; the lagged regime signal carries no information about which model will win.

**H₁:** β₁ > 0. The accuracy advantage of HR-ETE-GNN is larger on days following an anti-persistent regime signal.

**Test.** Giacomini–White [26] with the lagged regime indicator as the conditioning instrument, one-sided on β₁, HAC standard errors. The joint Wald statistic for β₀ = β₁ = 0, distributed χ² with 2 degrees of freedom, is reported alongside as an omnibus test; note that the joint form is two-sided and detects difference rather than superiority, so the directional claim rests on the coefficient t-statistic and not on the χ² statistic. As specified in Section 3.2, the regression is estimated a second time with a lagged calendar crisis dummy in place of the Hurst indicator.

**Reported quantities.** β̂₀ (the calm-regime mean differential), β̂₁ (the incremental turbulent-regime differential), β̂₀ + β̂₁ (the turbulent-regime mean differential), HAC standard errors and one-sided p-values for each, the joint χ² statistic, and the number of turbulent days in the test block.

**Interpretation, stated in advance.** Four outcomes are distinguished, and each is reported as such rather than reinterpreted after the fact.

(i) β̂₁ > 0 and significant: the tail-sensitivity mechanism proposed in Section 1.5 is supported, and the gain concentrates where the theory of Section 2.1.5 predicts it should.

(ii) β̂₀ > 0 significant but β̂₁ indistinguishable from zero: the model forecasts better, but uniformly. The improvement is reported, and the manuscript states that the mechanism advanced to explain it is not corroborated by its own test.

(iii) β̂₀ indistinguishable from zero and β̂₁ > 0: the outcome most consistent with the research objective of Section 1.2, namely improved robustness under extreme regimes at no cost to calm-period performance.

(iv) Neither coefficient significant: no evidence of improvement.

Because the power of this test is governed by the number of turbulent days in the test block, a null result accompanied by a small turbulent-day count is reported as uninformative rather than as evidence of no effect, together with the minimum effect detectable at that sample size.

### Table 4.4, regression format

Columns: Quantity | Hurst instrument | Crisis-dummy instrument
Rows: β̂₀; HAC SE(β̂₀); t(β̂₀); one-sided p; β̂₁; HAC SE(β̂₁); t(β̂₁); one-sided p; β̂₀+β̂₁; turbulent days / total days; joint χ²(2); χ² p; minimum detectable effect

### Also restore, if reinstating

**Section 3.1**, in place of the second paragraph of the replacement text: the test block must contain enough anti-persistent days for the test to have power, which on the full sample it does, the split placing the test block start in 2020 so that COVID-19 falls inside it. Reduced-scope configurations whose test block begins later contain too few such days and are not used for confirmatory inference.

**Section 3.2**, in place of the "two purposes" paragraph: Hypothesis 1b is estimated twice, once with the lagged Hurst-regime indicator and once with a lagged calendar crisis dummy; agreement between the two specifications strengthens the regime interpretation, disagreement is reported as such.
