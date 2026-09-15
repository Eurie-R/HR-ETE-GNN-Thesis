"""Does early stopping truncate the LSTM/GRU benchmarks before they learn?

The paper trains everything with "100 epochs and early stopping" but never says
the patience.  RNNs sit on a plateau near the training mean for a while before
they escape it; if patience is short they get stopped there, which would make
the GNN-vs-RNN gap an artefact of the stopping rule rather than a property of
the models.  This runs both arms on the real panel and reports the difference.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import hetegnn_base as hb

cfg = hb.Config(device="cpu").apply_runtime()
prices, proxy, _ = hb.download_prices(cfg)
ret = hb.log_returns(prices)
rv = hb.realized_volatility(ret, cfg.rv_window)

common = ret.index.intersection(rv.index)
X, y, _ = hb.make_windows(ret.loc[common].values, rv.loc[common].values,
                          cfg.input_size)
N = cfg.train_period
lo, hi = hb.minmax_fit(X[:N])
Xtr, ytr = hb.minmax_apply(X[:N], lo, hi), y[:N]

# held-out block straight after the training window, as the walk-forward does
Xte, yte = hb.minmax_apply(X[N:N + 250], lo, hi), y[N:N + 250]
mean_pred = ytr.mean(axis=0)
base_tr = float(np.mean((ytr - mean_pred) ** 2))
base_te = float(np.mean((yte - mean_pred) ** 2))

print(f"real panel: {X.shape}, training on the first {N} samples")
print(f"predict-the-mean MSE   train={base_tr:.4f}  test={base_te:.4f}\n")
print(f"{'model':6s} {'patience':>9s} {'epochs run':>11s} "
      f"{'train MSE':>10s} {'test MSE':>9s}  verdict")
print("-" * 64)

for kind in ("LSTM", "GRU", "GNN"):
    for patience in (10, 100):
        c = hb.Config(device="cpu", epochs=100, patience=patience,
                      torch_threads=cfg.torch_threads)
        A = None
        if kind == "GNN":
            Am = hb.build_ete_matrix(ret.values[:N], c, seed=0)
            import torch
            A = torch.tensor(hb.row_normalize(Am), dtype=torch.float32)

        hb.set_seed(0)
        t0 = time.time()
        m = hb.train(hb.build_model(kind, c), A, Xtr, ytr, c)
        tr = float(np.mean((hb.predict(m, A, Xtr, c) - ytr) ** 2))
        te = float(np.mean((hb.predict(m, A, Xte, c) - yte) ** 2))
        verdict = "beats mean" if te < base_te else "STUCK AT THE MEAN"
        print(f"{kind:6s} {patience:9d} {'~' + str(int(time.time()-t0)) + 's':>11s} "
              f"{tr:10.4f} {te:9.4f}  {verdict}")
    print()

print("If patience=10 is stuck but patience=100 is not, the RNN benchmarks are")
print("being truncated and Config.patience should be raised before the real run.")
