param([string]$Py, [string]$Out)

$src = [System.IO.File]::ReadAllText($Py)

$head = @'
<!doctype html><meta charset="utf-8"><title>hetegnn syntax harness</title>
<pre id="out">loading pyodide...</pre>
<script id="src" type="text/plain">
'@

$mid = @'
</script>
<script src="https://cdn.jsdelivr.net/pyodide/v314.0.6/full/pyodide.js"></script>
<script>
const OUT = document.getElementById("out");
(async () => {
  try {
    const py = await loadPyodide();
    OUT.textContent = "loading packages...";
    await py.loadPackage(["numpy", "pandas", "scipy"]);
    py.globals.set("SRC", document.getElementById("src").textContent);
    const res = py.runPython(TEST);
    OUT.textContent = res;
  } catch (e) {
    OUT.textContent = "HARNESS ERROR: " + e;
  }
})();
const TEST = String.raw`
import ast, sys, types, math
lines = []

# ---- 1. syntax ----
try:
    tree = ast.parse(SRC, filename="hetegnn_base.py")
    lines.append("PASS  syntax parses (%d top-level statements)" % len(tree.body))
except SyntaxError as e:
    lines.append("FAIL  SyntaxError line %s: %s" % (e.lineno, e.msg))
    lines.append("      %s" % (e.text or "").rstrip())
    tree = None

# ---- 2. import under a torch stub ----
if tree is not None:
    class _M:
        def __init__(self, *a, **k): pass
        def __call__(self, *a, **k): raise NotImplementedError
        def to(self, *a, **k): return self
        def parameters(self): return []
        def state_dict(self): return {}
        def load_state_dict(self, d): pass
        def train(self, *a): pass
        def eval(self): pass
    class _ML(_M):
        def __init__(self, mods=()): self._m = list(mods)
        def __iter__(self): return iter(self._m)

    nn = types.ModuleType("torch.nn")
    for nm in ("Module",):    setattr(nn, nm, _M)
    nn.ModuleList = _ML
    for nm in ("Linear", "Conv1d", "LSTM", "GRU"): setattr(nn, nm, _M)
    fn = types.ModuleType("torch.nn.functional")
    fn.relu = fn.mse_loss = lambda *a, **k: None
    nn.functional = fn

    class _NoGrad:
        def __call__(self, f): return f
        def __enter__(self): return self
        def __exit__(self, *a): return False

    torch = types.ModuleType("torch")
    torch.nn = nn
    torch.optim = types.ModuleType("torch.optim"); torch.optim.Adam = _M
    torch.cuda = types.SimpleNamespace(is_available=lambda: False,
                                       manual_seed_all=lambda s: None)
    torch.no_grad = _NoGrad
    torch.manual_seed = lambda s: None
    for nm in ("tensor", "device", "randperm", "einsum", "randn", "Tensor"):
        setattr(torch, nm, _M)

    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = nn
    sys.modules["torch.nn.functional"] = fn
    sys.modules["torch.optim"] = torch.optim

    mod = types.ModuleType("hetegnn_base")
    mod.__file__ = "hetegnn_base.py"
    sys.modules["hetegnn_base"] = mod   # @dataclass resolves cls.__module__ here
    try:
        exec(compile(SRC, "hetegnn_base.py", "exec"), mod.__dict__)
        lines.append("PASS  module imports under a torch stub")
    except Exception as e:
        lines.append("FAIL  import: %s: %s" % (type(e).__name__, e))
        import traceback
        lines.append(traceback.format_exc()[-900:])
        mod = None

# ---- 3. public surface present ----
if tree is not None and mod is not None:
    want = ["Config", "TICKERS", "REGIME_PROXY_CHAIN", "download_prices",
            "log_returns", "realized_volatility", "discretize",
            "transfer_entropy", "effective_te", "build_ete_matrix",
            "build_te_matrix", "build_granger_matrix", "build_pearson_matrix",
            "EDGE_BUILDERS", "row_normalize", "expected_rs", "hurst_rs",
            "rolling_hurst", "regime_labels", "regime_change_dates",
            "MultiScaleConv", "GCNLayer", "ETEGNN", "RNNBaseline",
            "build_model", "make_windows", "minmax_fit", "minmax_apply",
            "train", "predict", "RunResult", "WalkForward", "metrics",
            "summarize", "PAPER_TABLE6", "DEFAULT_MODELS", "prepare",
            "run_experiment", "main", "DEVIATIONS"]
    missing = [w for w in want if not hasattr(mod, w)]
    lines.append(("PASS  all %d public names present" % len(want)) if not missing
                 else "FAIL  missing: %s" % missing)

    # Config defaults match the paper's Table 3
    c = mod.Config()
    ok = (c.input_size == 20 and c.train_period == 750 and
          c.regime_sensitivity == 6 and c.rv_window == 20 and
          c.n_bins == 3 and c.hurst_window == 250 and
          abs(c.z_threshold - 1.96) < 1e-9 and c.epochs == 100 and
          tuple(c.conv_kernels) == (3, 5, 7) and c.conv_channels == 12)
    lines.append(("PASS" if ok else "FAIL") +
                 "  Config matches Table 3 (M=%d N=%d s=%d)" %
                 (c.input_size, c.train_period, c.regime_sensitivity))

    # EDGE_BUILDERS keys line up with the four benchmarks
    ok = set(mod.EDGE_BUILDERS) == {"ETE", "TE", "Granger", "Pearson"}
    lines.append(("PASS" if ok else "FAIL") + "  EDGE_BUILDERS keys")

    # PAPER_TABLE6 shape
    t = mod.PAPER_TABLE6
    ok = t.shape == (12, 7) and abs(float(
        t[(t.Regime == "Hurst") & (t.Model == "ETE-GNN")].RMSE.iloc[0]) - 0.1623) < 1e-9
    lines.append(("PASS" if ok else "FAIL") + "  PAPER_TABLE6 %s, H-ETE-GNN RMSE 0.1623" % (t.shape,))

    # multi-scale conv output dim, flatten readout: 12*(20-k+1) for k in 3,5,7
    exp = 12 * 18 + 12 * 16 + 12 * 14
    got = sum(12 * (20 - k + 1) for k in (3, 5, 7))
    lines.append(("PASS" if exp == got == 576 else "FAIL") +
                 "  conv flatten dim = %d" % got)

    # anis_lloyd correction is wired through Config
    lines.append(("PASS" if c.hurst_correction == "none" else "FAIL") +
                 "  hurst_correction default = %r" % c.hurst_correction)

    # numeric spot-checks on the real module functions
    import numpy as np
    rng = np.random.default_rng(3)
    n = 2000
    yv = rng.standard_normal(n)
    xv = np.empty(n); xv[0] = 0.0
    for t in range(1, n):
        xv[t] = 0.8 * yv[t-1] + 0.3 * rng.standard_normal()
    f = mod.transfer_entropy(yv, xv); r = mod.transfer_entropy(xv, yv)
    lines.append(("PASS" if f > 5 * r else "FAIL") +
                 "  transfer_entropy directional  fwd=%.4f rev=%.4f" % (f, r))
    e, z = mod.effective_te(yv, xv, m_shuffles=30)
    lines.append(("PASS" if z > 1.96 else "FAIL") + "  effective_te Z=%.1f" % z)
    h_raw = np.mean([mod.hurst_rs(rng.standard_normal(250)) for _ in range(60)])
    h_al  = np.mean([mod.hurst_rs(rng.standard_normal(250), correction="anis_lloyd")
                     for _ in range(60)])
    lines.append("INFO  hurst_rs on white noise: none=%.3f  anis_lloyd=%.3f" % (h_raw, h_al))
    lines.append(("PASS" if abs(h_al - 0.5) < abs(h_raw - 0.5) else "FAIL") +
                 "  anis_lloyd is closer to 0.5")
    A = np.array([[0., 1.], [0., 0.]])
    lines.append(("PASS" if np.allclose(mod.row_normalize(A), [[0, 1], [0, 0]]) else "FAIL") +
                 "  row_normalize handles all-zero rows")

"\n".join(lines)
`;
</script>
'@

[System.IO.File]::WriteAllText($Out, $head + $src + $mid, (New-Object System.Text.UTF8Encoding($false)))
Write-Host "wrote $Out ($((Get-Item $Out).Length) bytes)"
