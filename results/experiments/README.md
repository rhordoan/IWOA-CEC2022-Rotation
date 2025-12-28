## Experiment results layout

This directory contains **generated experiment outputs** produced by `benchmarks/experiments.py`.

### Folder structure

- `stage1_removal/`: **Removal ablations** (IWOA full + leave-one-out variants + WOA baseline)
  - `10D/`, `20D/`, ...: dimension subfolders
- `stage2_addition/`: **Addition ladder** (WOA -> WOAPlus ladder -> IWOA full reference)
- `stage3_interactions/`: **Core interactions** (factorial {restart,crossover,nelder_mead} on harder functions)

Each run folder is timestamped, e.g. `20251227T215137Z_cec2022_20D/`.

### Expected files in a run folder

- `summary.csv`: aggregated metrics per (method, function)
- `summary.tex`: LaTeX table version (if generated)
- `meta.json`: config + environment metadata for reproducibility
- `results.csv` / `results.jsonl`: per-run outputs (often large; usually ignored by git)

### Notes

- Some older/legacy runs may be missing `meta.json` or `summary.tex` if they were generated before the latest runner.


