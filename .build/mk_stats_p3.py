import nbformat as nbf, json
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): C.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
---
## §6 — The model, and the multi-seed training protocol

### One change to the architecture

The network is identical to the pilot's — multi-scale 1-D convolution, three GCN layers, linear head — with a single modification: the head is wrapped in a **softplus** so the forecast is guaranteed positive.

The pilot ended in a bare `nn.Linear`, which can emit a *negative* volatility forecast. That is economically meaningless (a market cannot swing by −0.3% on a typical day), and it makes the QLIKE loss diverge, so a single bad day can swamp the entire test-set average.

### Why multiple seeds, restated precisely

This is the fix for the pilot's most serious problem. In the pilot:

```python
np.random.seed(42); torch.manual_seed(42)      # cell 3, executed once
...
results['Shannon'] = train_model(A_shannon)     # builds a model -> consumes RNG
results['Renyi']   = train_model(A_renyi05)     # builds a model -> DIFFERENT weights
```

The second model was constructed from a different point in the random stream, so **the two arms did not start from the same initial weights**. The reported gap therefore mixes together two effects that cannot be separated after the fact: the effect of the graph, and the effect of initialisation luck.

The protocol below fixes both halves of that:

1. `torch.manual_seed(s)` is called **inside** the loop, immediately before the model is constructed.
2. Every arm receives the **same `SEEDS` list**, so seed *s* of the baseline and seed *s* of the treatment begin from byte-identical weights.

The forecast that goes into the statistical tests is the **average across seeds**. Averaging removes the initialisation noise that is not a property of the model, leaving the graph as the only systematic difference between arms.
""")

code('''
class MultiScaleConv(nn.Module):
    def __init__(self, in_len, channels=12, kernels=(3,5,7)):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(1, channels, kernel_size=k, padding=k//2) for k in kernels])
        self.out_dim = channels*len(kernels)
    def forward(self, x):
        B, N, L = x.shape
        x = x.reshape(B*N, 1, L)
        return torch.cat([F.relu(c(x)).mean(dim=2) for c in self.convs], dim=1).view(B, N, -1)


class GCNLayer(nn.Module):
    def __init__(self, i, o):
        super().__init__(); self.W1, self.W2 = nn.Linear(i, o), nn.Linear(i, o)
    def forward(self, H, A):
        return F.relu(self.W1(H) + torch.einsum('ij,bjf->bif', A, self.W2(H)))


class HRETEGNN(nn.Module):
    """Pilot architecture + a positivity-constrained head (softplus)."""
    def __init__(self, n_nodes, lookback, hidden=32):
        super().__init__()
        self.feat = MultiScaleConv(lookback); d = self.feat.out_dim
        self.g1, self.g2, self.g3 = GCNLayer(d, hidden), GCNLayer(hidden, hidden), GCNLayer(hidden, hidden)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x, A):
        h = self.feat(x)
        h = self.g3(self.g2(self.g1(h, A), A), A)
        return F.softplus(self.head(h).squeeze(-1)) + 1e-4


def row_normalize(A):
    A = np.asarray(A, float).copy()
    s = A.sum(axis=1, keepdims=True); s[s == 0] = 1.0
    return A/s


def train_multiseed(A_np, splits, seeds=SEEDS, hidden=32, epochs=EPOCHS,
                    lr=LR, patience=PATIENCE, label='', verbose=False):
    """Train one architecture across a FIXED seed list; return the seed ensemble."""
    A   = torch.tensor(row_normalize(A_np), dtype=torch.float32)
    Xtr = torch.tensor(splits['X_train'], dtype=torch.float32)
    ytr = torch.tensor(splits['y_train'], dtype=torch.float32)
    Xva = torch.tensor(splits['X_val'],   dtype=torch.float32)
    yva = torch.tensor(splits['y_val'],   dtype=torch.float32)
    Xte = torch.tensor(splits['X_test'],  dtype=torch.float32)

    te, va, stops, vmses = [], [], [], []
    for s in seeds:
        torch.manual_seed(int(s)); np.random.seed(int(s))    # <-- INSIDE the loop
        model = HRETEGNN(len(TICKERS), splits['lookback'], hidden)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        best, best_state, bad, ep = np.inf, None, 0, 0
        for ep in range(epochs):
            model.train(); opt.zero_grad()
            F.mse_loss(model(Xtr, A), ytr).backward(); opt.step()
            model.eval()
            with torch.no_grad(): v = float(F.mse_loss(model(Xva, A), yva))
            if v < best - 1e-7:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience: break
        if best_state is not None: model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            te.append(model(Xte, A).numpy()); va.append(model(Xva, A).numpy())
        stops.append(ep+1); vmses.append(best)
        if verbose: print(f"    seed {s:>2}: stop@{ep+1:<5} val_mse={best:.5f}")
    P = np.stack(te)
    return dict(label=label, pred=P.mean(axis=0), pred_val=np.stack(va).mean(axis=0),
                per_seed=P, stops=stops, val_mses=vmses)

print("Model and training protocol defined.")
''')

md("""
### The arms

Five models are trained. Three of them exist purely to make the comparison interpretable — without them, a win for the Rényi arm could be explained away in at least two different ways.

| Arm | Graph | What it rules out |
|---|---|---|
| **Shannon ETE-GNN** | matched-density, α = 1.0 | *the baseline — this is what we must beat* |
| **HR-ETE-GNN α = 0.5** | matched-density, α = 0.5 | the treatment |
| **HR-ETE-GNN α = 1.5** | matched-density, α = 1.5 | that any α ≠ 1 would do just as well |
| **No-graph ablation** | identity matrix | that the graph contributes anything at all |
| **Random graph** | random, same density | that *any* graph helps, not specifically a TE graph |

The random-graph control is the one most often missing from papers of this kind, and it is the one a sharp panellist will ask for: it separates "transfer entropy found the right structure" from "message-passing between correlated series helps regardless of structure".
""")

code('''
rng_g = np.random.default_rng(12345)
A_random = np.zeros((len(TICKERS), len(TICKERS)))
off = [(i,j) for i in range(len(TICKERS)) for j in range(len(TICKERS)) if i != j]
for idx in rng_g.choice(len(off), size=MATCHED_K, replace=False):
    i, j = off[idx]; A_random[i, j] = 1.0

ARMS = {f'HR-ETE-GNN (a={a})' if a != 1.0 else 'Shannon ETE-GNN (a=1.0)': MATCHED[a]
        for a in ALPHA_GRID}
ARMS['No-graph ablation'] = np.eye(len(TICKERS))
ARMS['Random graph']      = A_random
BASELINE = 'Shannon ETE-GNN (a=1.0)'

t0 = time.time(); RESULTS = {}
for name, A in ARMS.items():
    print(f"Training {name}  ({len(SEEDS)} seeds, {int((A>0).sum())} edges) ...")
    RESULTS[name] = train_multiseed(A, sp, label=name)
print(f"\\nAll arms trained in {time.time()-t0:.0f}s")
''')

md("""
### How much of the pilot's gap could have been seed luck?

Before any forecast is scored, look at the spread of validation MSE across seeds *within a single arm*. Every one of those runs used the same data and the same graph; the only thing that differed was the initial weights.

If that spread is comparable to the gap the pilot reported between arms, then the pilot's result was not measuring the graph at all.
""")

code('''
spread = pd.DataFrame([
    {'arm': n, 'best seed': min(r['val_mses']), 'worst seed': max(r['val_mses']),
     'mean': np.mean(r['val_mses']), 'sd across seeds': np.std(r['val_mses'], ddof=1),
     'worst/best': max(r['val_mses'])/min(r['val_mses']),
     'median stop epoch': int(np.median(r['stops']))}
    for n, r in RESULTS.items()]).set_index('arm')
display(spread.round(5))

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.boxplot([RESULTS[n]['val_mses'] for n in RESULTS], labels=list(RESULTS),
           patch_artist=True,
           boxprops=dict(facecolor='#dfe6ee', edgecolor=INK),
           medianprops=dict(color=ACCENT, linewidth=2))
for i, n in enumerate(RESULTS):
    ax.scatter([i+1]*len(SEEDS), RESULTS[n]['val_mses'], color=INK, s=16, zorder=3, alpha=.7)
ax.set_ylabel('Validation MSE'); ax.set_xticklabels(list(RESULTS), rotation=18, ha='right', fontsize=9)
ax.set_title('Spread across random seeds WITHIN each arm\\n'
             '(same data, same graph — only the initial weights differ)')
plt.tight_layout(); plt.show()

worst_ratio = spread['worst/best'].max()
print(f"Within a single arm, the worst seed is up to {worst_ratio:.2f}x the best seed's")
print("validation MSE. The pilot trained ONE seed per arm and compared the results")
print("directly. Any gap smaller than this spread is indistinguishable from luck --")
print("which is exactly why the tests below run on the seed ENSEMBLE, not a single run.")
''')
json.dump(C, open('/home/uriel/repositories/HR-ETE-GNN-THESIS_FINAL/.build/p3.json','w'))
print(f"part3: {len(C)} cells")
