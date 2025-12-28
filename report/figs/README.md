## Convergence figures

This directory is expected by `report/technical.tex` for including convergence figures.

### Expected filenames

- `F22022_convergence.png`
- `F62022_convergence.png`
- `F112022_convergence.png`
- `F122022_convergence.png`

### How to generate

Run from the repository root:

```bash
python benchmarks/plot_convergence.py --dims 20 --runs 30 --max_fes 200000 --log_every 500 --functions F2,F6,F11,F12 --methods iwoa,woa --iwoa_variants full
```

Then copy the generated `plots/*.png` into this folder (or change the paths in `technical.tex`).


