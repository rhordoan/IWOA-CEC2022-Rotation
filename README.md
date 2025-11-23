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

To execute the full validation suite (12 functions, 30 independent runs per function) and generate LaTeX-ready tables:

```bash
python benchmarks/run_cec2022.py --dims 20 --runs 30
```

This run uses parallel processing (`joblib`) and may take **30–40 minutes**, depending on your CPU core count.

### 2. Generate Convergence Plots

To visualize the log-scale convergence behavior on the critical functions (F1 and F11):

```bash
python benchmarks/plot_convergence.py
```

Convergence images will be saved under `results/plots/`.

## 📂 Repository Structure

```text
IWOA-CEC2022-Rotation/
├── src/
│   ├── iwoa.py             # Core algorithm implementation (Gold Standard)
│   └── evaluator.py        # CEC-compliant evaluation wrapper
├── benchmarks/
│   ├── run_cec2022.py      # Main validation harness
│   └── plot_convergence.py # Plotting script
├── results/
│   └── plots/              # Output directory for convergence figures
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


