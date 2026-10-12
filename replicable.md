# Replication Configuration: Lee & Cho (2025) H-ETE-GNN

Source: Lee, M.; Cho, P. "Graph-Based Stock Volatility Forecasting with Effective-Transfer
Entropy and Hurst-Based Regime Adaptation." *Fractal Fract.* 2025, 9, 339.

This file lists every configuration item needed to replicate the paper's experiment. It is the
reference for the controlled comparison: both arms (H-ETE-GNN at α = 1 and HR-ETE-GNN at α\*)
must use exactly these settings, with the Rényi order as the only difference.

**Status legend**
- **Specified**: stated in the paper; use as given.
- **Tuned, not reported**: a search space is given but the selected value is not.
- **Not reported**: the paper is silent; the thesis must fix a value.
- **Departure**: the thesis deliberately differs from the paper.

---

## 1. Data

| Item | Value | Status | Paper source |
|---|---|---|---|
| Target assets | 10 iShares MSCI country ETFs: EWA (Australia), EWC (Canada), EWG (Germany), EWJ (Japan), EWT (Taiwan), EWU (UK), EWW (Mexico), EWY (South Korea), EWZ (Brazil), EZA (South Africa) | Specified | §3.2, Table 4 |
| Regime proxy | iShares MSCI World ETF | Specified | §3.2, Table 4 |
| Data source | Yahoo Finance, daily close prices | Specified | Abstract, §1 |
| Sample period | 7 Feb 2003 – 29 May 2024 | Specified | §3.2 |
| Reported evaluation period | 2006 – 2024 (after the first training block) | Specified | Abstract, Table 9 |
| Model input | Log returns, `r_t = ln(p_t / p_{t-1})` | Specified | Eq. 17 |
| Input scaling | Min-max to [0, 1] | Specified | §3.1 |
| Data the scaler is fitted on | — | Not reported | — |
| Prediction target | One-day-ahead realized volatility | Specified | §3.1 |
| RV formula | `RV_t = sqrt( (1/n) Σ_{i=1..n} (100 · r_{t-i})² )` | Specified | Eq. 18 |
| RV window n | 20 days | Specified | §3.2 |
| World ETF history | URTH lists only from 2012; the paper does not say how 2003–2012 is covered | Not reported | — |

## 2. Transfer Entropy and ETE (graph construction)

| Item | Value | Status | Paper source |
|---|---|---|---|
| TE lags | k = l = 1 | Specified | §2.1, Eq. 3 |
| TE estimator | Discretized TE (histogram probability mass function) | Specified | §2.1 |
| Number of bins | 3 (gain / neutral / loss) | Specified | §2.1 |
| Bin edges | Equal-width intervals between the window's min and max | Specified | §2.1 |
| Bin edges for surrogates | — | Not reported | — |
| Estimation window | The most recent N days (same as the training block) | Specified | §3.1 |
| ETE definition | `ETE = TE_{Y→X} − (1/m) Σ_i TE_{Y(i)→X}` | Specified | Eq. 4 |
| Surrogate method | Random shuffle of the source series Y (destroys temporal structure, keeps marginal distribution) | Specified | §2.1 |
| Number of surrogates m | Search space [10, 100] | Tuned, not reported | Table 1 |
| Edge significance | `Z = (TE − μ_shuffled) / σ_shuffled` | Specified | Eq. 5 |
| Edge retention rule | Keep the edge only if Z > 1.96 | Specified | §2.1 |
| Edge weight | ETE value of the retained edge | Specified | §2.2 |
| Direction | Directed: `A_ji` = edge from node j to node i | Specified | §2.2 |
| Adjacency normalisation | None (raw weights) | Specified | Eq. 9 |
| Self-loops | None; the self term is handled by `W1` | Specified | Eq. 9 |

## 3. Model architecture

| Item | Value | Status | Paper source |
|---|---|---|---|
| Feature extractor | Multi-scale 1D convolution, outputs concatenated | Specified | Eq. 8 |
| Kernel sizes | 3, 5, 7 | Specified | §3.1 |
| Channels per kernel | 12 | Specified | §3.1 |
| Convolution activation | ReLU | Specified | Eq. 8 |
| Message-passing layers | 3 | Specified | §3.1 |
| Node update rule | `h_i^(l+1) = ReLU( h_i W1 + Σ_{j∈N(i)} A_ji h_j W2 )` | Specified | Eq. 9 |
| Hidden size, layer 1 | Search space [10, 100] | Tuned, not reported | Table 1 |
| Hidden size, layer 2 | Search space [10, 100] | Tuned, not reported | Table 1 |
| Activation | ReLU | Specified | §3.1 |

## 4. Training

| Item | Value | Status | Paper source |
|---|---|---|---|
| Loss | Mean squared error (MSE) | Specified | §3.1 |
| Optimizer | Adam | Specified | §3.1 |
| Learning rate | Search space [1e-5, 1e-2] | Tuned, not reported | Table 1 |
| Batch size | Search space {4, 8, 16} | Tuned, not reported | Table 1 |
| Max epochs | 100 | Specified | §3.1 |
| Early stopping | Used | Specified | §3.1 |
| Early-stopping patience | — | Not reported | — |
| Train / validation split | 7 : 3 within the N-day block | Specified | §3.1 |
| Input size M | Searched {20, 40, 60}; selected **20** | Specified | Tables 1, 3 |
| Training period N | Searched {250, 500, 750}; selected **750** | Specified | Tables 1, 3 |
| Forecasting scheme | Sliding window: the last M days predict the next day's RV | Specified | §3.1 |
| Hyperparameter tuning | Bayesian optimization | Specified | §3.1 |
| Tuning objective | — | Not reported | — |
| Repetitions | 10 runs per model; mean ± SD reported | Specified | §4.1 |
| Random seeds | — | Not reported | — |
| Software | Python 3.8.20, PyTorch 2.0.0 | Specified | §3.1 |
| Hardware | AMD Ryzen 9 4.5 GHz, 32 GB RAM, NVIDIA GTX 4090 24 GB | Specified | §3.1 |

## 5. Hurst regime detection

| Item | Value | Status | Paper source |
|---|---|---|---|
| Estimator | Rescaled range (R/S); H = OLS slope of log(R/S)_w on log w | Specified | Eqs. 10–16 |
| R/S scales w inside each window | — | Not reported | — |
| Series | Log returns of the MSCI World ETF | Specified | §3.2 |
| Rolling window | 250 days (500 and 750 gave too few changes: 29 and 14) | Specified | §3.1 |
| Regime threshold | H = 0.5, two regimes ("Above" / "Below") | Specified | §2.3 |
| Lookback for the switching rule | Last 10 days | Specified | §3.1 |
| Sensitivity s | Searched {6, 7, 8}; selected **6** (switch when ≥ 6 of 10 days fall on one side) | Specified | Tables 1, 3 |
| Action on regime change | Rebuild the edge matrix on the last N days, then **fine-tune** (not full retrain) | Specified | §3.1 |
| Fine-tuning epochs / learning rate | — | Not reported | — |
| Action with no regime change | Reuse the current model without training | Specified | §3.1 |
| Regime changes found | 47 | Specified | §3.3, §4.2 |
| Periodic arm (no Hurst) | Retrain every N days | Specified | §3.1 |

## 6. Benchmark models

| Model | Edge matrix | Notes | Paper source |
|---|---|---|---|
| ETE-GNN / H-ETE-GNN | ETE, Z > 1.96 screen | Proposed model | §2.1–2.2 |
| TE-GNN | Raw transfer entropy | Same estimator, no surrogate correction | §3.1 |
| Granger-GNN | Binary: 1 if p < 0.05, else 0 | Lag order p not reported | §3.1, Eq. 6 |
| Pearson-GNN | Absolute correlation \|ρ\| | Dense and symmetric | §3.1, Eq. 7 |
| LSTM | No graph; past N-day log returns | Uses Hurst retraining | §3.1 |
| GRU | No graph; past N-day log returns | Uses Hurst retraining | §3.1 |

**RNN tuning grid (Table 2)**

| Hyperparameter | Search space |
|---|---|
| Learning rate | [1e-5, 1e-2] |
| Batch size | {4, 8, 16} |
| Hidden size per layer | [10, 100] |
| Number of layers | {1, 2, 3, 4, 5} |
| Dropout | [0, 0.5] |
| Input size M | {20, 40, 60} |
| Training period N | {250, 500, 750} |
| Sensitivity s | {6, 7, 8} |

## 7. Evaluation metrics

| Metric | Formula | Paper source |
|---|---|---|
| RMSE | `sqrt( (1/n) Σ (ŷ_i − y_i)² )` | Eq. 19 |
| MAE | `(1/n) Σ \|ŷ_i − y_i\|` | Eq. 20 |
| MAPE | `(1/n) Σ \|(ŷ_i − y_i) / y_i\| × 100` | Eq. 21 |
| Correlation | Pearson correlation of y and ŷ | Eq. 22 |
| Hit ratio | `(1/(n−1)) Σ_{i=2..n} I[ sign(ŷ_i − ŷ_{i−1}) = sign(y_i − y_{i−1}) ]` | Eq. 23 |
| Significance | One-tailed t-tests between model pairs | Table 7 |

---

## 8. Gaps the thesis must fix

Each value below must be chosen once and applied identically to both arms and every benchmark.

| # | Unreported item | Thesis choice | Where it is justified |
|---|---|---|---|
| 1 | Learning rate, batch size, hidden sizes | Tuned once on validation at α = 1, then frozen for both arms | `manuscript_revisions.md` §3.5 |
| 2 | Number of surrogates m | Fixed at m = 100 (top of the paper's range), not tuned | `manuscript_revisions.md` Step 4 |
| 3 | Early-stopping patience | None: full 100 epochs, best-validation weights restored | `README_replication.md` |
| 4 | World ETF before 2012 | Splice URTH → ACWI → IOO | `README_replication.md` |
| 5 | Fine-tuning epochs / learning rate after a regime change | **To decide** | — |
| 6 | R/S scales w | **To decide** (literal R/S gives 50 changes vs the paper's 47) | `README_replication.md` |
| 7 | Granger lag order p | **To decide** | — |
| 8 | Scaler fit data | **To decide** (fit on the training block only, to avoid look-ahead) | — |
| 9 | Surrogate bin edges | **To decide** | — |
| 10 | Bayesian-optimization objective | **To decide** | — |
| 11 | Random seeds | **To decide**; use the same seeds in both arms | — |

## 9. Deliberate departures from the paper

| Item | Paper | Thesis | Reason |
|---|---|---|---|
| Adjacency scaling | Raw ETE weights | Row-normalised (`D⁻¹A`), all arms and benchmarks | Rényi and Shannon TE have different scales; raw ETE makes the neighbour term ≈ 0.7% of the self term. Literal raw-weight results go in the appendix |
| Primary metric | RMSE, MAE, MAPE, correlation, hit ratio | QLIKE (paper metrics reported once, for the replication audit) | QLIKE is robust to proxy noise (Patton) |

## 10. Open consistency issue

| Item | Paper | `manuscript_revisions.md` | Code (`README_replication.md:239`) |
|---|---|---|---|
| Surrogate type | Random shuffle | Random shuffle | Thesis arm `effective_rte` defaults to **circular shift**; baseline `hetegnn_base.effective_te` uses shuffle |

If the arms use different surrogate types, two things differ between them, not only α. Set
`surrogate="shuffle"` explicitly in both arms (or circular in both) and state the choice in §3.3.
