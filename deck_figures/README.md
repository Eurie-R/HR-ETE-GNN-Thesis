# Deck Figures — where each picture comes from and which slide it belongs to

Two sources feed the deck:

- **`preliminary_results/F*.png`** — produced by `preliminary_results.ipynb`. These are the
  *results* figures and are already embedded in `Ramoy_Midterm_Defense.pptx`.
- **`deck_figures/S*.png`** — the *explanatory* figures. Most were extracted from the
  rendered outputs of `explanation.ipynb`; three (`S14`, `S15`, `S28`) were generated for
  the deck by `deck_build/make_figs.py`. **All are embedded in the .pptx.**

Rebuild anything with:

```bash
.venv/bin/python deck_build/repro_graphs.py   # re-measures the ER-TE graphs (~45 s)
.venv/bin/python deck_build/make_figs.py      # redraws S14 / S15 / S28
.venv/bin/python deck_build/build_deck.py     # rebuilds Ramoy_Midterm_Defense.pptx
```

`deck_build/graphs.npz` holds the reproduced ER-TE matrices. `repro_graphs.py` verifies them
against `T4_h2a_edge_existence.csv` and `T5_h2b_transmitters.csv` — it prints `MATCH T5
exactly: True`, so the diagrams and the tables cannot drift apart.

To regenerate any `S*` figure at a different size or style, re-run the corresponding cell in
`explanation.ipynb` and save with `fig.savefig(..., dpi=200, bbox_inches='tight')`.

## Slide-by-slide

| Slide | Visual called for | File | Status |
|---|---|---|---|
| 2 | Weather-map metaphor | — | **Does not exist.** Decorative only; safe to drop. |
| 4 | Price panel + returns panel | `S04a_ewg_price.png`, `S04b_ewg_returns.png` | ready |
| 5 | Two paths, same endpoints | `S05_two_investments.png` | ready |
| 6 | Four scenario shapes | `S06_S07_four_scenarios.png` | ready (shows paths **and** RV) |
| 7 | Four scenarios with RV underneath | `S06_S07_four_scenarios.png` | ready (same figure as slide 6) |
| 8 | *(no Visual line)* | `S08_rolling_window.png` | optional — the 20-day window sliding, incl. the COVID window |
| 9 | RV over time, all ten countries | `preliminary_results/F1_sample_and_regimes.png` | **already in the .pptx** |
| 9 | alternative | `S09_rv_over_time.png` | Germany only, cleaner for a first look |
| 10 | Direction vs. volatility autocorrelation bars | `S10_direction_vs_size.png` | ready |
| 11 | [past] → [model] → [10 numbers] | `S11_forecasting_task.png` | ready |
| 12 | March 2020, all markets collapsing together | `S12a_covid_all_countries.png` | ready |
| 12 | supporting: correlation is symmetric | `S12b_correlation_heatmap.png` | optional |
| 13 | *(no Visual line)* | `S13_transfer_entropy_idea.png` | embedded — Error A vs Error B, drawn |
| 14 | 10-node directed graph | `S14_gnn_architecture.png` | embedded — **generated**: the full GNN, with the real α=0.5 graph inside it |
| 14 | alternative | `S14_information_network.png` | the plain 10-node graph, if the architecture panel is too much |
| 15 | Camera exposure metaphor | `S15_alpha_dial.png` | embedded — **generated**: what α actually does, measured on your returns |
| 16 | Hurst bands vs. elevated volatility | `S16_hurst_regime_detector.png` | embedded (replaces F1 here) |
| 22 | *(no Visual line)* | `S22_pipeline.png` | optional — full pipeline, end to end |
| 28 | *(no Visual line)* | `S28_topology_alpha_contrast.png` | embedded — **generated**: the RQ2 topology contrast, Shannon vs Rényi |
| 30 | Adjacency heatmaps, transmitters | `preliminary_results/F2*.png`, `F3*.png` | embedded, but small — S28 makes the same point far better |

## The three generated figures

**`S15_alpha_dial.png`** — replaces the camera-exposure metaphor with the measurement itself.
Left: the real distribution of daily moves across all ten ETFs over the graph-estimation window
(>3% days are 5.8% of the sample). Right: the Rényi escort weight pᵅ/Σpᵅ relative to p, per
move-size bin. **A crash day carries 3.5× the weight at α = 0.5 that it carries under Shannon** —
that number is the whole mechanism claim, computed rather than asserted.

**`S14_gnn_architecture.png`** — the model end to end: 20-day window per country → multi-scale
1-D convolution (kernels 3/5/7, 12 channels → 576 features) → three graph layers (576→64→32→1)
→ one RV forecast per country. Dimensions are read off `hetegnn_base.Config` and `ETEGNN`, and
the graph in the middle panel is the real measured α=0.5 topology, not a sketch.

**`S28_topology_alpha_contrast.png`** — the RQ2 figure. Both networks at matched density
(top-20 edges), same circular layout, circle size = out-degree, red = a country whose rank moves
3+ places (Australia, Canada, South Africa). Out-degrees match `T5` exactly.

**Slide 2 (weather map)** is still the one visual with no asset. It is decorative; the words carry
the analogy fine on their own.

## Known limitation

`S14` and `S28` are dense, and on a slide that also carries body text and a table they land about
2.8 inches tall — readable as a gist while you narrate, not readable label-by-label from the back
of the room. Giving each its own slide is the only real fix.

## Unused results figures

`preliminary_results/F4_forecasts.png` (forecast vs. actual traces) and
`F5_dashboard.png` (summary dashboard) are not referenced by any slide. F4 would strengthen
slide 23; F5 suits a backup slide.
