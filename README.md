# Hybrid IWOA: A Rotation-Invariant Whale Optimization Algorithm

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Benchmark: CEC 2022](https://img.shields.io/badge/Benchmark-CEC%202022-red)](https://github.com/P-N-Suganthan/CEC2022)

**Author:** Roberto Sergiu Hordoan  
**Status:** Pre-print / Research Code

## 🚀 Overview

This repository contains the implementation of **Hybrid IWOA**, a variant of the Whale Optimization Algorithm (WOA) tailored for **rotated, ill-conditioned optimization problems** where standard approaches often degrade due to strong variable covariance.

The standard WOA relies on largely coordinate-independent updates, which makes it effective on simple or separable functions but less reliable on rotated, non-separable problems. Hybrid IWOA addresses this by combining three mechanisms:

- **Boosted Nelder-Mead (BNM):** A covariance-guided local search operator for deep valley exploitation.
- **Adaptive DE Crossover:** An adaptive recombination step that implicitly learns variable linkage during the mid-search phase.
- **Dynamic Diversity Restoration:** A restart mechanism based on population variance to escape local basins and restore exploration pressure.

In combination, these components make WOA more robust on rotated landscapes and other problems with strong variable coupling.

## 🏆 Key Results (IEEE CEC 2022 - 20 Dimensions)

Hybrid IWOA was evaluated on the **IEEE CEC 2022** benchmark suite (20D, 200k FEs) and compared against the competition winner (**EA4eig**) and runner-up (**NL-SHADE-RSP**).

### The "Discriminator" Test: F11

Composition Function 3 (**F11**) has a pronounced rotated valley structure and is commonly used as a difficult test case.

- **NL-SHADE-RSP (Runner-Up):** Failed (mean error: 150.0)  
- **Hybrid IWOA (Ours):** Solved (mean error: 0.99)

### Comparative Summary

| Function | Landscape Type        | Hybrid IWOA (Ours) | NL-SHADE-RSP | Verdict               |
| :------- | :-------------------- | :----------------- | :----------- | :-------------------- |
| F1 (Zakharov)   | Unimodal / Rotated  | 7.35E-07           | 1.00E-08     | ✅ Tier 1 (Solved)     |
| F2 (Rosenbrock) | Valley / Ill-Cond. | **5.02 (Best)**    | 8.93 (Mean)  | ⚡ Superior Peak Perf. |
| F3 (Schaffer)   | Basic               | 0.00               | 1.00E-08     | ✅ Solved              |
| F11 (Comp 3)    | Rotated / Complex   | **0.99**           | 150.0        | 🏆 State-of-the-Art    |

> **Note:** Hybrid IWOA behaves as a **rotation-oriented variant**. It trades some performance on simple separable grids (e.g., F4) in order to achieve higher accuracy on complex rotated manifolds (e.g., F11), which are closer to many real-world engineering scenarios.

## 🛠️ Installation

Clone the repository and install the Python dependencies:

```bash
# Clone the repository
git clone https://github.com/rhordoan/IWOA-CEC2022-Rotation.git
cd IWOA-CEC2022-Rotation

# Install dependencies
pip install -r requirements.txt
```

## 📊 Reproduction

### 1. Run the Full Benchmark

To execute the full validation suite (12 functions, 30 independent runs per function) and generate **CSV + LaTeX-ready tables**:

```bash
python benchmarks/experiments.py --dims 20 --runs 30 --functions F1-12
```

This run uses parallel processing (`joblib`) and may take **30–40 minutes**, depending on your CPU core count.

#### Methods and ablations

The experiment runner supports:
- Baseline: `woa`
- Proposed: `iwoa` (expand into component ablations via `--iwoa_variants`)

Example (baseline + full IWOA + ablations):

```bash
python benchmarks/experiments.py \
  --dims 20 --runs 30 --functions F1-12 \
  --methods iwoa,woa \
  --iwoa_variants full,no_restart,no_crossover,no_chaos,no_levy,no_nelder_mead
```

Outputs are written into a timestamped folder under `results/experiments/` and include:
- `results.csv` / `results.jsonl` (per-run)
- `summary.csv` / `summary.tex` (aggregated)
- `meta.json` (config + environment metadata)

### Ablation Study (solid: removal + addition + interactions)

This repo includes **preset ablation suites** so you don’t accidentally omit variants:
- `ablation_removal`: IWOA full + leave-one-out variants (+ WOA baseline)
- `ablation_addition`: WOA → WOAPlus ladder (adds operators progressively) → IWOA(full) reference
- `ablation_interactions`: factorial interactions for {restart, crossover, nelder_mead} (+ WOA baseline)

#### Stage 1 (breadth): removal ablations on F1–F12 for 10D/20D/30D

- PowerShell example:

```powershell
foreach ($d in 10,20,30) {
  python benchmarks/experiments.py --preset ablation_removal --dims $d --runs 30 --functions F1-12
}
```

- Bash example:

```bash
for d in 10 20 30; do
  python benchmarks/experiments.py --preset ablation_removal --dims "$d" --runs 30 --functions F1-12
done
```

#### Stage 2 (depth): addition ladder on hard functions (F7–F12) at 20D

```bash
python benchmarks/experiments.py --preset ablation_addition --dims 20 --runs 30 --functions F7-12
```

#### Stage 3 (synergy): interactions of {restart,crossover,nelder_mead} on F7–F12 at 20D

```bash
python benchmarks/experiments.py --preset ablation_interactions --dims 20 --runs 30 --functions F7-12
```

#### Post-processing: ablation contribution tables (Δ vs WOA / Δ vs full / success rates)

Run this inside an experiment output folder (replace `<RUN_FOLDER>`):

```bash
python benchmarks/ablation_analysis.py --input_dir results/experiments/<RUN_FOLDER>
```

This creates:
- `ablation_delta_vs_woa.csv/.tex`
- `ablation_delta_vs_full.csv/.tex`
- `success_rates.csv/.tex`
- `ablation_plots/*.png` (heatmaps, default 20D)

### 2. Statistical tables (ranks + Wilcoxon + Holm)

After running experiments, compute ranking and statistical tests:

```bash
python benchmarks/stats.py --input results/experiments/<RUN_FOLDER>/summary.csv --baseline WOA
```

This writes `avg_ranks.*`, `ranks_per_function.*`, and `wilcoxon.*` into the same folder (CSV + LaTeX).

### 2. Generate Convergence Plots

To visualize **median + IQR** log-scale convergence behavior across multiple runs:

```bash
python benchmarks/plot_convergence.py \
  --dims 20 --runs 30 --max_fes 200000 --log_every 500 \
  --functions F1,F11 \
  --methods iwoa,woa \
  --iwoa_variants full
```

Convergence outputs are saved under `results/convergence/<RUN_FOLDER>/`:
- `convergence.csv` (per-run curves)
- `convergence_agg.csv` (median + quantiles)
- `plots/*.png`

### 3. Run unit tests (budget + reproducibility)

```bash
python -m unittest discover -s tests -v
```

## 🐳 Docker (recommended for artifact reproduction)

Build:

```bash
docker build -t iwoa-cec2022 .
```

Run experiments and write outputs to your local `results/` folder:

```bash
docker run --rm -v ${PWD}/results:/app/results iwoa-cec2022 \
  python benchmarks/experiments.py --dims 20 --runs 30 --functions F1-12
```

## 📂 Repository Structure

```text
IWOA-CEC2022-Rotation/
├── src/
│   ├── iwoa.py             # Core algorithm implementation (Gold Standard)
│   ├── woa.py              # Baseline WOA (for comparisons)
│   ├── woa_plus.py         # WOA + optional operators (for addition ablations)
│   └── evaluator.py        # CEC-compliant evaluation wrapper
├── benchmarks/
│   ├── experiments.py      # Canonical deterministic experiment runner
│   ├── reporting.py        # Output helpers (CSV/JSONL/LaTeX)
│   ├── stats.py            # Ranks + Wilcoxon(+Holm) tables
│   ├── ablation_presets.py # Preset method suites for ablation studies
│   ├── ablation_analysis.py# Δ tables + success rates + heatmaps from an experiment folder
│   ├── run_cec2022.py      # Backward-compatible wrapper (calls experiments.py)
│   └── plot_convergence.py # Median+IQR convergence plots (multi-run)
├── tests/
│   ├── test_budget.py
│   └── test_reproducibility.py
├── results/
│   └── plots/              # Output directory for convergence figures
├── Dockerfile
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 📜 Citation

If you use this code or the accompanying results in your research, please cite:

```bibtex
@misc{hordoan2025iwoa,
  author       = {Hordoan, Roberto Sergiu},
  title        = {An Enhanced Whale Optimization Algorithm with Covariance-Guided Local Search},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/rhordoan/IWOA-CEC2022-Rotation}}
}
```

## 📝 License

This project is licensed under the **MIT License**. See the `LICENSE` file for full details.


