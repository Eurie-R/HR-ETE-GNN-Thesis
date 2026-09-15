# H-ETE-GNN baseline replication

Control arm for the thesis: a faithful reimplementation of

> Lee, S. & Cho, P. (2025). *Graph-Based Stock Volatility Forecasting with Effective
> Transfer Entropy and Hurst-Based Regime Adaptation.* Fractal Fract. **9**, 339.
> <https://doi.org/10.3390/fractalfract9060339>

The Renyi extension (HR-ETE-GNN) is compared against the numbers this code
produces, not against the paper's printed table.

| File | What it is |
|---|---|
| `hetegnn_base.py` | The whole model, all four edge builders, both retraining policies, metrics. Importable and runnable as a CLI. |
| `HETE_GNN_Base_Replication.ipynb` | Driver notebook. Runs identically locally and on Colab. |
| `requirements.txt` | Pinned local environment. |
| `requirements-colab.txt` | Colab additions only (no torch, so the CUDA build survives). |
| `smoke_test.py` | **Run this first.** ~15 s offline gate: everything builds, every model's loss falls, walk-forward completes. |
| `preliminary_results.ipynb` | The *thesis-side* experiment (HR-ETE-GNN vs this baseline) end to end at reduced scope, ~10 min. See below. |
| `.build/selftest.py` | Offline test suite: no network, synthetic data, ~28 assertions (~4 min). |
| `.build/check_rnn_earlystop.py` | Checks that early stopping is not truncating the LSTM/GRU benchmarks. |
| `.build/check_data.py` | Downloads the real panel and reports the regime calendar. |
| `.build/check_adjacency_scale.py` | Measures how much each edge type contributes through Eq (9). |
| `.build/check_equiv.py` | Proves the batched walk-forward equals the one-day-at-a-time loop. |
| `.build/bench_threads.py` | Picks a torch thread count for this machine. |

## Running it locally

```bash
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python smoke_test.py
```

`smoke_test.py` takes ~15 seconds and answers one question: does the model work?
It builds all four edge matrices, trains all six models and checks the loss
actually falls and stays finite, verifies gradients reach the graph term, and
runs a walk-forward under both retraining policies. Add `--network` to also
check the yfinance download. **Run it before starting anything long** — it is
the cheap way to find out that something is broken.

`.build/selftest.py` is the thorough version (~4 minutes, ~28 assertions,
including the RNN benchmarks at the paper's real training size). Then either
drive things from the notebook:

```bash
.venv\Scripts\python -m ipykernel install --user --name hetegnn --display-name "HR-ETE-GNN (.venv)"
.venv\Scripts\jupyter lab
```

or from the command line:

```bash
.venv\Scripts\python hetegnn_base.py --quick
.venv\Scripts\python hetegnn_base.py --models ETE-GNN TE-GNN --repeats 3
.venv\Scripts\python hetegnn_base.py --out replication_results.csv
```

`--quick` is a smoke run (1 repeat, stride 5, 2010 onward, 20 epochs). The
no-flag form is the full paper protocol: 6 models x 2 regimes x 10 repeats.

## Running it on Colab

Open `HETE_GNN_Base_Replication.ipynb`, set **Runtime -> Change runtime type ->
GPU**, and run the setup cell. It detects Colab, clones this repo, and installs
`requirements-colab.txt`.

Do **not** `pip install -r requirements.txt` on Colab: it pins a CPU torch wheel
and would silently take away the accelerator.

## What the paper fixes, and what it leaves open

Reported and implemented as-is: ten country ETFs over 2003-02-07..2024-05-29;
20-day realized volatility as the target; min-max scaled log-returns as input;
3-bin discretized transfer entropy with k=l=1; ETE as TE minus the mean of m
shuffled surrogates, keeping only Z > 1.96; a 250-day rolling R/S Hurst exponent
on a World-ETF proxy; regimes from a 6-of-10-day rule around H = 0.5; multi-scale
convolution with kernels {3,5,7} x 12 channels; three message-passing layers;
MSE, Adam, 100 epochs, early stopping, 7:3 train/validation. Table 3's tuned
values M=20, N=750, s=6 are the defaults.

Left open by the paper, and therefore a judgement call here: the learning rate,
batch size, both hidden sizes and the shuffle count m (Table 1 gives search
spaces, Table 3 never reports the optima); whether Eq (8) pools over time;
whether the adjacency is normalised; the fine-tuning budget; and which R/S
variant produced Figure 3. Every one of these is a field on `Config`, and all of
them are enumerated with reasoning in `hetegnn_base.DEVIATIONS`:

```python
import hetegnn_base as hb
print(hb.DEVIATIONS)
```

## The one decision that changes everything

Eq (9) adds a self term `h_i W1` to a neighbourhood term `sum_j A_ji h_j W2`, so
the graph matters only in proportion to how big `A`'s entries are. The four edge
builders are on completely different scales: ETE is a few hundredths of a nat,
Granger is a 0/1 indicator, Pearson is a correlation. The paper never says
whether `A` is normalised.

Measured on the first N=750 block (`.build/check_adjacency_scale.py`), the mean
size of the neighbourhood term as a fraction of the self term:

| edge | raw | row-normalised |
|---|---|---|
| ETE | 0.007 | 0.648 |
| TE | 0.063 | 0.910 |
| Granger | 2.122 | 0.648 |
| Pearson | 3.803 | 0.910 |

With raw weights the ETE graph contributes **0.7%** of the self term -- it is
decorative, and `ETE-GNN` degenerates into a per-node MLP -- while Pearson
contributes 380%. The four variants would then be comparing edge *magnitude*,
not the edge *topology* the paper claims to test, and no amount of tuning would
put ETE-GNN on top.

**`normalize_adjacency` therefore defaults to `True`** -- a deliberate departure
from the literal text, taken because the literal reading does not yield a
baseline HR-ETE-GNN can meaningfully be compared against. To reproduce the
literal paper instead:

```bash
.venv\Scripts\python hetegnn_base.py --raw-adjacency --out results_literal.csv
```

Say so explicitly in the thesis: the baseline is Lee & Cho's model with a
row-normalised adjacency, and that choice is forced by Eq (9), not cosmetic.

## Two more things worth knowing before you read results

**The World-ETF proxy does not exist for the whole sample.** The paper names the
iShares MSCI World ETF, but URTH only listed in 2012 while the sample starts in
2003. We splice log-returns newest-first across `URTH -> ACWI -> IOO`
(3112 / 957 / 1292 days respectively) and report the split at load time. The
regime calendar depends on this; `Config.proxy_chain = ("URTH",)` shows the
literal reading.

**The regime count comes out close.** With the literal R/S estimator the 6-of-10
rule fires **50** regime changes against the paper's 47 -- close enough to treat
the regime machinery as replicated. The Anis-Lloyd small-sample correction
(`Config.hurst_correction = "anis_lloyd"`) gives 39 and is available as a
robustness check.

Judge the replication on ordering and gaps -- ETE-GNN best, Hurst better than
Periodic, GNNs well ahead of the RNNs -- rather than on matching Table 6 to
three decimals. The unreported hyper-parameters make exact levels unrecoverable.

## Early stopping is not a free compute saving

The paper says "100 epochs and early stopping" and never gives a patience. It
turns out to matter a great deal. Measured on the real panel, the longest run of
non-improving epochs *before* the eventual best validation loss was:

| model | longest plateau | best epoch |
|---|---|---|
| LSTM | 50 epochs | 92 / 100 |
| GRU | 19 epochs | 40 / 100 |
| GNN | 17 epochs | 69 / 100 |

A patience of 10 stops all three early — and does so unevenly. It leaves both
RNNs *worse than predicting the training mean* on held-out data (0.374 and 0.378
against a 0.355 baseline) while the GNN is untouched (0.0373 vs 0.0374), which
would have inflated the paper's headline GNN-vs-RNN gap by roughly 2x for
reasons that have nothing to do with the models.

Since `train()` restores the best-validation weights regardless, early stopping
here only saves time and never guards against overfitting. So `Config.patience`
defaults to `None` (run the full budget). Set an integer to trade accuracy for
time, and re-run `.build/check_rnn_earlystop.py` if you do.

Related: the LSTM's best epoch is 92 of 100, so the paper's own budget is
marginal for it. Its poor showing in Table 6 may be partly a training-budget
artefact.

## Performance notes

Two things dominate runtime, both already handled:

- **Torch threads.** The net is tiny (576 -> 64 -> 32 -> 1) and batched at 8, so
  torch's default of one thread per core loses more to synchronisation than it
  gains. On this 12-core box a full fit takes 34 s at 12 threads and 9.6 s at 2.
  `Config.torch_threads` defaults to 4; `--threads N` overrides it.
- **Batched forecasting.** Between two refits the model and adjacency are fixed,
  so the days in that stretch are forecast in one pass rather than one at a time.
  `.build/check_equiv.py` proves this is bit-for-bit the same as the naive loop
  (max prediction difference ~3e-7, float32 noise).

`stride > 1` subsamples the *evaluation* only. A regime change landing on a day
that is not being forecast still triggers its refit, so the Hurst arm keeps the
retraining schedule the paper specifies. This matters more than it sounds: with
a naive exact-date match, `stride=10` fired 1 refit instead of 14, which would
have made any strided Hurst-vs-Periodic comparison meaningless.

Set `HETEGNN_FAST=1` to rehearse the notebook end to end in a couple of minutes
before committing to the real run.

## Preliminary results: the whole experiment, scaled down

`preliminary_results.ipynb` runs the *thesis* experiment — HR-ETE-GNN against the
Shannon baseline this repo replicates — from data through to the reporting
paragraph, in about ten minutes instead of days. Build it with
`.venv\Scripts\python .build\mk_prelim.py`, then run it like any notebook.

It exists to settle three things before the real run starts: that the pipeline
executes end to end on real data, that the results tables have their final shape,
and that the binding constraints are known early. It is **not** a preview of the
final numbers, and §0 of the notebook states exactly which five knobs were turned
down and what each one costs.

One switch controls everything:

```python
MODE = "preliminary"     # -> "thesis" for the full run; nothing else changes
```

Two constraints it surfaced immediately, both scope artefacts rather than model
problems:

- **The test block contains no crisis.** Starting the sample in 2018 puts the
  20% test block in 2023–24. The lagged Hurst indicator flags 4 of 314 days as
  turbulent, and a high-volatility dummy fixed on the training block flags 0 —
  the two definitions agree. H1b, the hypothesis the thesis rests on, therefore
  cannot be estimated at this sample length, and no amount of extra compute on
  this window fixes it. The thesis run starts in 2003 and spans 2008 and 2020.
- **The surrogate budget caps the edge count.** With *m* surrogates no
  permutation *p* can fall below 1/(*m*+1), and BH across 90 ordered pairs needs
  many pairs sitting at that floor before any edge survives. At *m* = 100 that
  threshold is 10 pairs; at *m* = 1000 it is 1. A low edge count at preliminary
  scope is a statement about the surrogate budget, not about the market.

Tables and figures are written to `preliminary_results/` as CSV and PNG, one file
per numbered table, ready to paste into the manuscript.
