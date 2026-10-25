"""
Full controlled experiment: H-ETE-GNN (alpha = 1) vs HR-ETE-GNN (alpha*).

Everything not listed here is taken unchanged from ../hetegnn_base.py, which
replicates Lee & Cho (2025).  The configuration follows ../replicable.md.  The
two arms share every setting, the seed list and the surrogate permutations, and
differ in one thing only: the Renyi order alpha used to build the edge matrix.

The Renyi arm uses the same estimator as the paper's ETE: 3-bin equal-width
discretisation, k = l = 1, m shuffled surrogates, keep the edge only if Z > 1.96.
Shannon entropy is replaced by the plug-in Renyi entropy of order alpha:

    H_a(p) = log(sum p^a) / (1 - a)
    RTE_a  = [H_a(X+, X) - H_a(X)] - [H_a(X+, X, Y) - H_a(X, Y)]

This is the difference-based Renyi TE (the chain rule does not hold for
alpha != 1, so it is not the Jizba et al. escort form).  At alpha = 1 the
builder calls hetegnn_base's own ETE code, so HR-ETE-GNN(1) and H-ETE-GNN are
the same model, bit for bit.

Only the Hurst retraining policy is run: both arms are "H-" models, and the
periodic control and the four benchmark models are dropped to save compute.
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Deterministic GPU kernels.  cuBLAS reads this when it first initialises, so it
# must be set before torch touches the GPU; importing this module sets it in the
# notebook kernel and in every spawned worker.  See set_deterministic().
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hetegnn_base as hb  # noqa: E402

_EPS = hb._EPS

# =============================================================================
# Frozen configuration  (replicable.md, sections 1-5 and 8)
# =============================================================================

# Candidate Renyi orders.  alpha* is chosen from these on the validation span;
# orders below 0.6 are excluded from selection (manuscript_revisions.md, Lim. 2).
ALPHA_GRID: tuple = (0.7, 0.9, 1.0, 1.5, 2.0)

# Validation span: forecast days up to and including this date pick alpha*.
# The confirmatory test uses only forecast days after it.  The walk-forward is
# causal, so a forecast for day t never depends on anything after t.
VAL_END = "2009-12-31"

SEEDS: tuple = tuple(range(10))          # paper: 10 repetitions; same seeds both arms
VAL_SEEDS: tuple = (0, 1, 2)             # seeds for the alpha-selection runs


# On cuda, cuDNN's default convolution kernels add in a run-dependent order, so
# the same seed gave val MSE 0.912 / 1.369 / 1.410 on three reruns (alpha = 1,
# seed 0).  With these flags reruns are bit-identical, at no measured cost.
# Runs made before this was switched on are kept in archive_nondeterministic_2026-10-12/ next to this file.
DETERMINISTIC = True


def set_deterministic(on: bool = DETERMINISTIC) -> None:
    """Make every torch op use a fixed-order kernel, or raise if it has none."""
    import torch
    torch.backends.cudnn.deterministic = on
    if on:
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(on)


def make_config(**overrides) -> hb.Config:
    """Lee & Cho's configuration, with replicable.md section 8 gaps filled in."""
    cfg = hb.Config(
        # Section 1: data
        start="2003-02-07", end="2024-05-30", rv_window=20,
        # Table 3: tuned values the paper reports
        input_size=20, train_period=750, regime_sensitivity=6,
        # Section 2: TE / ETE.  Gap 2: m fixed at 100, top of the paper's range
        n_bins=3, te_lag=1, z_threshold=1.96, n_shuffles=100,
        # Sections 3-4: model and training
        conv_kernels=(3, 5, 7), conv_channels=12, epochs=100, val_fraction=0.3,
        # Gap 1: lr / batch / hidden sizes are frozen at hetegnn_base defaults
        # (mid-range of Table 1) and are identical in both arms
        lr=1e-3, batch_size=8, hidden1=64, hidden2=32,
        # Gap 3: no early-stopping patience; best-validation weights restored
        patience=None,
        # Gap 5: fine-tune for 30 epochs at the same lr after a regime change
        finetune_epochs=30,
        # Section 5: Hurst.  Gap 6: literal R/S, no small-sample correction
        hurst_window=250, regime_lookback=10, hurst_threshold=0.5,
        hurst_correction="none",
        # Section 9: row-normalised adjacency in both arms
        normalize_adjacency=True,
        n_repeats=len(SEEDS), seed=0, stride=1,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# =============================================================================
# Discretised Renyi transfer entropy
# =============================================================================

def renyi_entropy(p: np.ndarray, alpha: float) -> float:
    """Plug-in Renyi entropy of a probability mass function, in nats."""
    p = p[p > 0]
    if abs(alpha - 1.0) < 1e-12:
        return float(-np.sum(p * np.log(p)))
    return float(np.log(np.sum(p ** alpha)) / (1.0 - alpha))


def _rte_from_symbols(sy: np.ndarray, sx: np.ndarray, n_bins: int,
                      lag: int, alpha: float) -> float:
    """Renyi TE_{Y->X} of order alpha from pre-symbolised series.

    Uses the same sample alignment as hetegnn_base._te_from_symbols.
    """
    if abs(alpha - 1.0) < 1e-12:
        return hb._te_from_symbols(sy, sx, n_bins, lag)
    xf = sx[lag + 1:]
    n = xf.size
    if n < 10:
        return np.nan
    xp = sx[lag:-1][:n]
    yp = sy[lag:-1][:n]

    b = n_bins
    counts = np.bincount(xf * b * b + xp * b + yp, minlength=b ** 3)
    p = counts.reshape(b, b, b).astype(float) / n          # p[x_{t+1}, x_t, y_t]

    h_full = renyi_entropy(p.ravel(), alpha)
    h_xf_xp = renyi_entropy(p.sum(axis=2).ravel(), alpha)
    h_xp_yp = renyi_entropy(p.sum(axis=0).ravel(), alpha)
    h_xp = renyi_entropy(p.sum(axis=(0, 2)), alpha)
    return (h_xf_xp - h_xp) - (h_full - h_xp_yp)


def effective_rte(y: np.ndarray, x: np.ndarray, alpha: float, n_bins: int = 3,
                  lag: int = 1, m_shuffles: int = 100,
                  rng: Optional[np.random.Generator] = None):
    """Eq (4)-(5) with Renyi TE.  Returns (ERTE, Z).

    Mirrors hetegnn_base.effective_te line for line, including how the rng is
    consumed, so both arms see the same surrogate permutations.
    """
    if abs(alpha - 1.0) < 1e-12:
        return hb.effective_te(y, x, n_bins, lag, m_shuffles, rng)
    rng = rng or np.random.default_rng(0)
    sy = hb.discretize(y, n_bins)
    sx = hb.discretize(x, n_bins)

    te = _rte_from_symbols(sy, sx, n_bins, lag, alpha)
    if not np.isfinite(te):
        return np.nan, np.nan

    surr = np.array([_rte_from_symbols(rng.permutation(sy), sx, n_bins, lag, alpha)
                     for _ in range(m_shuffles)])
    mu, sd = np.nanmean(surr), np.nanstd(surr)
    return te - mu, (te - mu) / (sd + _EPS)


def build_erte_matrix(returns: np.ndarray, cfg: hb.Config, seed: int = 0,
                      alpha: float = 1.0) -> np.ndarray:
    """ER-TE adjacency: keep ERTE where Z > 1.96, zero elsewhere.

    Same loop order and rng as hetegnn_base.build_ete_matrix.
    """
    if abs(alpha - 1.0) < 1e-12:
        return hb.build_ete_matrix(returns, cfg, seed)
    n = returns.shape[1]
    A = np.zeros((n, n))
    rng = np.random.default_rng(seed)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ete, z = effective_rte(returns[:, i], returns[:, j], alpha,
                                   cfg.n_bins, cfg.te_lag, cfg.n_shuffles, rng)
            if np.isfinite(z) and z > cfg.z_threshold:
                A[i, j] = max(ete, 0.0)
    return A


def edge_kind(alpha: float) -> str:
    """Register the alpha builder with hetegnn_base and return its key."""
    if abs(alpha - 1.0) < 1e-12:
        return "ETE"
    key = f"RETE{alpha:g}"
    if key not in hb.EDGE_BUILDERS:
        def _builder(returns, cfg, seed=0, _a=alpha):
            return build_erte_matrix(returns, cfg, seed, _a)
        hb.EDGE_BUILDERS[key] = _builder
    return key


def arm_label(alpha: float) -> str:
    return "H-ETE-GNN" if abs(alpha - 1.0) < 1e-12 else f"HR-ETE-GNN(a={alpha:g})"


# =============================================================================
# Data
# =============================================================================

DATA_FILE = HERE / "data" / "panel.pkl"


def prepare_data(cfg: hb.Config, path: Path = DATA_FILE,
                 refresh: bool = False) -> dict:
    """Download and transform once, then cache, so every worker and every rerun
    uses the identical panel even if Yahoo revises its history later."""
    if path.exists() and not refresh:
        with open(path, "rb") as f:
            return pickle.load(f)
    prices, ret, rv, proxy_ret, proxy_src, hurst, labels, changes = hb.prepare(cfg)
    data = dict(prices=prices, ret=ret, rv=rv, proxy_ret=proxy_ret,
                proxy_src=proxy_src, hurst=hurst, labels=labels,
                changes=changes, config=asdict(cfg))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return data


# =============================================================================
# Runs, checkpointed one file per (span, alpha, seed)
# =============================================================================

RUNS_DIR = HERE / "runs"


@dataclass(frozen=True)
class RunSpec:
    span: str          # "val" (forecast days <= VAL_END) or "full"
    alpha: float
    seed: int
    root: str = str(RUNS_DIR)   # carried in the spec so spawned workers see it

    @property
    def path(self) -> Path:
        return Path(self.root) / self.span / f"alpha{self.alpha:g}_seed{self.seed}.npz"


def _walk_forward(cfg: hb.Config, data: dict, span: str) -> hb.WalkForward:
    ret, rv = data["ret"], data["rv"]
    if span == "val":
        # Keep one extra day so the last target date is VAL_END itself.
        cut = rv.index[rv.index > pd.Timestamp(VAL_END)]
        last = cut[0] if len(cut) else rv.index[-1]
        ret, rv = ret.loc[:last], rv.loc[:last]
    return hb.WalkForward(cfg, ret, rv, data["changes"])


def run_one(spec: RunSpec, cfg_overrides: Optional[dict] = None,
            data_path: str = str(DATA_FILE)) -> dict:
    """One walk-forward run.  Skips work if its checkpoint already exists."""
    if spec.path.exists():
        return {"spec": spec, "skipped": True, "seconds": 0.0}
    set_deterministic()
    cfg = make_config(**(cfg_overrides or {})).apply_runtime()
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    wf = _walk_forward(cfg, data, spec.span)
    r = wf.run("GNN", edge_kind(spec.alpha), "hurst", seed=spec.seed)
    if spec.span == "val":
        keep = r.dates <= pd.Timestamp(VAL_END)
        r.dates, r.y_true, r.y_pred = r.dates[keep], r.y_true[keep], r.y_pred[keep]
    spec.path.parent.mkdir(parents=True, exist_ok=True)
    tmp = spec.path.with_name(spec.path.stem + ".tmp.npz")
    np.savez_compressed(tmp, dates=r.dates.values.astype("datetime64[ns]"),
                        y_true=r.y_true, y_pred=r.y_pred,
                        n_refits=r.n_refits, seconds=r.seconds,
                        alpha=spec.alpha, seed=spec.seed,
                        deterministic=DETERMINISTIC, device=cfg.device)
    os.replace(tmp, spec.path)                     # atomic: no half-written runs
    return {"spec": spec, "skipped": False, "seconds": r.seconds,
            "n_refits": r.n_refits}


def run_many(specs: Sequence[RunSpec], n_jobs: int = 1,
             cfg_overrides: Optional[dict] = None) -> List[dict]:
    """Run specs, n_jobs at a time, printing progress and an ETA."""
    todo = [s for s in specs if not s.path.exists()]
    print(f"{len(specs) - len(todo)} of {len(specs)} runs already done; "
          f"{len(todo)} to go with n_jobs={n_jobs}")
    out: List[dict] = []
    t0 = time.time()

    def _report(res: dict, k: int) -> None:
        el = time.time() - t0
        eta = el / k * (len(todo) - k)
        s = res["spec"]
        print(f"[{k:3d}/{len(todo)}] {s.span:4s} alpha={s.alpha:<4g} seed={s.seed}  "
              f"{res['seconds'] / 60:5.1f} min  refits={res.get('n_refits', '-')}  "
              f"elapsed {el / 3600:5.2f} h  ETA {eta / 3600:5.2f} h", flush=True)

    if n_jobs <= 1:
        for k, s in enumerate(todo, 1):
            res = run_one(s, cfg_overrides)
            out.append(res)
            _report(res, k)
        return out

    # spawn: required for CUDA in workers and the only option on Windows
    with ProcessPoolExecutor(n_jobs, mp_context=get_context("spawn")) as ex:
        futs = [ex.submit(run_one, s, cfg_overrides) for s in todo]
        for k, f in enumerate(as_completed(futs), 1):
            res = f.result()
            out.append(res)
            _report(res, k)
    return out


def load_run(spec: RunSpec) -> dict:
    z = np.load(spec.path)
    return dict(dates=pd.DatetimeIndex(z["dates"]), y_true=z["y_true"],
                y_pred=z["y_pred"], n_refits=int(z["n_refits"]),
                seconds=float(z["seconds"]), alpha=float(z["alpha"]),
                seed=int(z["seed"]),
                deterministic=bool(z["deterministic"]) if "deterministic" in z.files else False,
                device=str(z["device"]) if "device" in z.files else "unknown")


# =============================================================================
# Timing, for the run-time estimate
# =============================================================================

def benchmark(cfg: hb.Config, data: dict, alphas: Sequence[float] = (1.0, 0.7),
              repeats: int = 1) -> dict:
    """Time one graph build per alpha and one full 100-epoch fit on cfg.device."""
    import torch
    set_deterministic()
    cfg.apply_runtime()
    wf = hb.WalkForward(cfg, data["ret"], data["rv"], data["changes"])
    N = cfg.train_period
    win = wf.ret.values[:N]
    graph = {}
    for a in alphas:
        key = edge_kind(a)
        t = time.time()
        for _ in range(repeats):
            A = hb.EDGE_BUILDERS[key](win, cfg, seed=0)
        graph[a] = (time.time() - t) / repeats
    A = hb.row_normalize(A)
    A_t = torch.tensor(A, dtype=torch.float32, device=torch.device(cfg.device))
    lo, hi = hb.minmax_fit(wf.X[:N])
    X = hb.minmax_apply(wf.X[:N], lo, hi)
    hb.set_seed(0)
    t = time.time()
    hb.train(hb.build_model("GNN", cfg), A_t, X, wf.y[:N], cfg)
    if cfg.device.startswith("cuda"):
        torch.cuda.synchronize()
    fit = time.time() - t
    return dict(device=cfg.device, graph_seconds=graph, fit_seconds=fit)


def project_runtime(bench: dict, data: dict, cfg: hb.Config,
                    alpha_grid: Sequence[float] = ALPHA_GRID,
                    val_seeds: Sequence[int] = VAL_SEEDS,
                    seeds: Sequence[int] = SEEDS, n_jobs: int = 1) -> pd.DataFrame:
    """Serial cost of each stage from the benchmark, divided by n_jobs."""
    wf = hb.WalkForward(cfg, data["ret"], data["rv"], data["changes"])
    eval_dates = wf.target_dates[cfg.train_period:]
    changes = pd.DatetimeIndex(data["changes"])
    in_eval = changes[(changes > eval_dates[0]) & (changes <= eval_dates[-1])]
    n_ch_full = len(in_eval)
    n_ch_val = int((in_eval <= pd.Timestamp(VAL_END)).sum())

    fit = bench["fit_seconds"]
    ft = fit * cfg.finetune_epochs / cfg.epochs

    def run_cost(alpha: float, n_ch: int) -> float:
        g = bench["graph_seconds"].get(alpha, bench["graph_seconds"].get(
            0.7 if alpha != 1.0 else 1.0))
        return fit + n_ch * ft + (n_ch + 1) * g

    stage1 = sum(run_cost(a, n_ch_val) for a in alpha_grid) * len(val_seeds)
    stage2 = sum(run_cost(a, n_ch_full) for a in (1.0, 0.7)) * len(seeds)
    rows = [("Stage 1: alpha selection (val span)",
             len(alpha_grid) * len(val_seeds), n_ch_val, stage1),
            ("Stage 2: H-ETE-GNN vs HR-ETE-GNN (full span)",
             2 * len(seeds), n_ch_full, stage2)]
    df = pd.DataFrame(rows, columns=["stage", "runs", "refits per run",
                                     "serial hours"])
    df["serial hours"] = df["serial hours"] / 3600
    df[f"hours at n_jobs={n_jobs}"] = df["serial hours"] / n_jobs
    total = df.sum(numeric_only=True)
    total["refits per run"] = np.nan
    df.loc[len(df)] = ["TOTAL", *total.values]
    return df
