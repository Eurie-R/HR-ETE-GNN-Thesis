"""How much signal does each edge type actually push through Eq (9)?

Eq (9) is  h_i' = ReLU(h_i W1 + sum_j A_ji h_j W2).  The two terms are only
comparable if A is on a sensible scale.  ETE weights are ~0.01 nats while
Granger's are exactly 1.0, so on raw weights the ETE graph contributes ~100x
less than the Granger graph and ETE-GNN degenerates towards a per-node MLP.
This quantifies the gap, and what row-normalising does to it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import hetegnn_base as hb

cfg = hb.Config().apply_runtime()
prices, proxy_ret, _ = hb.download_prices(cfg)
ret = hb.log_returns(prices)
win = ret.values[:cfg.train_period]

mats = {k: hb.EDGE_BUILDERS[k](win, cfg, seed=0)
        for k in ("ETE", "TE", "Granger", "Pearson")}

hb.set_seed(0)
model = hb.ETEGNN(cfg)
X = torch.tensor(
    hb.minmax_apply(*(lambda W: (W, *hb.minmax_fit(W)))(
        np.stack([win[t - cfg.input_size:t].T
                  for t in range(cfg.input_size, cfg.train_period)]))),
    dtype=torch.float32)

print(f"{'edge':9s} {'edges':>6s} {'row sum':>9s} {'mean w':>9s}   "
      f"{'|self|':>9s} {'|neigh|':>9s} {'neigh/self':>11s}")
print("-" * 72)
for norm in (False, True):
    print(f"[normalize_adjacency = {norm}]")
    for name, A0 in mats.items():
        A = hb.row_normalize(A0) if norm else A0
        At = torch.tensor(A, dtype=torch.float32)
        with torch.no_grad():
            H = model.feat(X)
            layer = model.layers[0]
            self_term = layer.W1(H)
            neigh_term = torch.einsum("ji,bjf->bif", At, layer.W2(H))
            s = float(self_term.abs().mean())
            g = float(neigh_term.abs().mean())
        print(f"  {name:7s} {int((A0 > 0).sum()):6d} {A.sum(axis=1).mean():9.4f} "
              f"{(A[A > 0].mean() if (A > 0).any() else 0):9.4f}   "
              f"{s:9.4f} {g:9.4f} {g / max(s, 1e-12):11.3f}")
    print()

print("neigh/self is the ratio the graph actually contributes in layer 1.")
print("Near zero means the adjacency is decorative: the model cannot use it.")
