"""How long does one real training call take, and does thread count help?

The model is tiny (576->64->32->1) and batched at 8, so intra-op parallelism can
cost more in synchronisation than it saves.  This measures one full-size fit at
several thread counts to pick a default before committing to the long run.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import hetegnn_base as hb

cfg = hb.Config(device="cpu")
rng = np.random.default_rng(0)
N, n = cfg.train_period, cfg.n_nodes()
X = rng.standard_normal((N, n, cfg.input_size)).astype(np.float32)
y = np.abs(rng.standard_normal((N, n))).astype(np.float32) + 1.0
A = torch.tensor(rng.random((n, n)) * 0.01, dtype=torch.float32)

print(f"torch {torch.__version__}  logical cores {os.cpu_count()}")
print(f"one fit = {cfg.epochs} epochs x {int(N*0.7)//cfg.batch_size} batches "
      f"(early stopping patience {cfg.patience})\n")

for nt in (1, 2, 4, os.cpu_count()):
    torch.set_num_threads(nt)
    hb.set_seed(0)
    t0 = time.time()
    hb.train(hb.ETEGNN(cfg), A, X, y, cfg)
    dt = time.time() - t0
    print(f"  threads={nt:<3d} {dt:6.1f}s per full fit")

# What the whole protocol costs, given a per-fit cost.
print("\nprojection for the full protocol (6 models x 2 regimes x 10 repeats):")
print("  hurst   : 1 full fit + ~49 fine-tunes (30 epochs) per run")
print("  periodic: ~6 full fits per run")
