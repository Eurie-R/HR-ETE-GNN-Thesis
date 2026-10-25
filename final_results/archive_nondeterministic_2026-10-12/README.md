# Archived: non-deterministic full runs (2026-10-12)

These are the stage-2 runs (alpha = 0.7 and 1.0, 10 seeds each) and the results tables that
`final_results/run_full.ipynb` produced on 2026-10-12, on cuda **without** deterministic mode.

They were archived on 2026-10-21. A rerun check showed that, in that mode, the same seed does
not reproduce: alpha = 1, seed 0 on the validation span gave val MSE 0.912, 1.369 and 1.410 on
three runs. `final_results/hrete_full.py` now enables deterministic mode (`DETERMINISTIC = True`),
and every alpha is re-run that way from `final_results/alpha_runs/`.

Kept in place in `final_results/`: `runs/val/`, `tables/stage1_alpha_selection.csv`, and
`tables/alpha_star.json`. These record how alpha* = 0.7 was chosen, and alpha* stays frozen.
They also came from the non-deterministic mode.
