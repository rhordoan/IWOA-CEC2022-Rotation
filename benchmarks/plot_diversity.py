import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluator import CEC_Evaluator
from src.iwoa import IWOA_Strict

try:
    import opfunu.cec_based.cec2022 as cec2022
except ImportError as e:
    raise SystemExit("Missing dependency: opfunu. Install with: pip install -r requirements.txt") from e


@dataclass(frozen=True)
class DiversityConfig:
    dims: int
    runs: int
    max_fes: int
    log_every: int
    seed: int
    seed_strategy: str
    function: str
    pop_size: int
    min_pop_size: int
    n_jobs: int


def _stable_hash_int(*parts: Any, mod: int = 2**32 - 1) -> int:
    import hashlib

    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest(), "little") % mod


def _set_global_seeds(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)


def _make_run_seed(cfg: DiversityConfig, *, run_id: int) -> int:
    if cfg.seed_strategy == "base_plus_run":
        return int(cfg.seed + 1000 * run_id + _stable_hash_int(cfg.function) % 1000)
    if cfg.seed_strategy == "hash":
        return int(_stable_hash_int(cfg.seed, cfg.dims, cfg.max_fes, cfg.function, run_id))
    raise ValueError(f"Unknown seed_strategy: {cfg.seed_strategy}")


def _normalize_func_name(func: str) -> str:
    f = func.strip().upper()
    if f in {"ALL", "F1-12", "F1..12"}:
        raise ValueError("plot_diversity.py expects a single function, e.g. F11 or F112022.")
    if not f.startswith("F"):
        f = "F" + f
    if not f.endswith("2022"):
        f = f + "2022"
    return f


def _make_checkpoints(max_fes: int, log_every: int) -> List[int]:
    cps = [1]
    log_every = int(log_every) if log_every and int(log_every) > 0 else 500
    cps.extend(list(range(log_every, max_fes + 1, log_every)))
    if cps[-1] != max_fes:
        cps.append(max_fes)
    return sorted(set(int(c) for c in cps if 1 <= c <= max_fes))


class DiversityTrace:
    """
    Records diversity at fixed FE checkpoints using a "last known value" fill.
    The optimizer calls us at irregular times (per-generation, restarts, etc.).
    """

    def __init__(self, checkpoints: Sequence[int]) -> None:
        self.checkpoints = list(checkpoints)
        self._i = 0
        self.history_fes: List[int] = []
        self.history_diversity: List[float] = []
        self.history_restarted: List[int] = []

        self._last_div = float("nan")
        self._last_restarted = 0

    def __call__(self, fes: int, diversity: float, restarted: bool) -> None:
        fes_i = int(fes)
        self._last_div = float(diversity)
        self._last_restarted = 1 if restarted else 0

        while self._i < len(self.checkpoints) and fes_i >= self.checkpoints[self._i]:
            self.history_fes.append(int(self.checkpoints[self._i]))
            self.history_diversity.append(float(self._last_div))
            self.history_restarted.append(int(self._last_restarted))
            self._i += 1


def _run_single(cfg: DiversityConfig, run_id: int) -> pd.DataFrame:
    import warnings

    warnings.filterwarnings("ignore")

    seed = _make_run_seed(cfg, run_id=run_id)
    _set_global_seeds(seed)

    func_name = _normalize_func_name(cfg.function)
    problem = getattr(cec2022, func_name)(ndim=cfg.dims)
    bounds = np.array([problem.lb, problem.ub])

    evaluator = CEC_Evaluator(problem.evaluate, cfg.max_fes, problem.lb, problem.ub)
    checkpoints = _make_checkpoints(cfg.max_fes, cfg.log_every)
    trace = DiversityTrace(checkpoints)

    opt = IWOA_Strict(
        evaluator=evaluator,
        dim=cfg.dims,
        bounds=bounds,
        pop_size=cfg.pop_size,
        min_pop_size=cfg.min_pop_size,
    )
    opt.run(trace_hook=trace)

    df = pd.DataFrame(
        {
            "function": func_name,
            "dims": cfg.dims,
            "run_id": run_id,
            "seed": seed,
            "fes": trace.history_fes,
            "diversity": trace.history_diversity,
            "restarted": trace.history_restarted,
        }
    )
    return df


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["function", "dims", "fes"], as_index=False)["diversity"]
    out = g.agg(
        median_div="median",
        q25_div=lambda x: float(np.nanpercentile(x, 25)),
        q75_div=lambda x: float(np.nanpercentile(x, 75)),
    )
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Generate a diversity (population spread) plot for Hybrid IWOA.")
    p.add_argument("--dims", type=int, default=20)
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--max_fes", type=int, default=200000)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--function", type=str, default="F11")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--seed_strategy", type=str, default="base_plus_run", choices=["base_plus_run", "hash"])
    p.add_argument("--pop_size", type=int, default=100)
    p.add_argument("--min_pop_size", type=int, default=20)
    p.add_argument("--n_jobs", type=int, default=1)
    p.add_argument("--out_png", type=str, default=os.path.join("report", "diversity_plot.png"))
    p.add_argument("--out_dir", type=str, default=os.path.join("results", "convergence", "diversity"))
    args = p.parse_args()

    cfg = DiversityConfig(
        dims=int(args.dims),
        runs=int(args.runs),
        max_fes=int(args.max_fes),
        log_every=int(args.log_every),
        seed=int(args.seed),
        seed_strategy=str(args.seed_strategy),
        function=str(args.function),
        pop_size=int(args.pop_size),
        min_pop_size=int(args.min_pop_size),
        n_jobs=int(args.n_jobs),
    )

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)

    dfs = Parallel(n_jobs=cfg.n_jobs)(
        delayed(_run_single)(cfg, run_id=r) for r in range(cfg.runs)
    )
    df = pd.concat(dfs, ignore_index=True)
    agg = _aggregate(df)

    # Save data for traceability
    raw_csv = os.path.join(args.out_dir, f"diversity_{_normalize_func_name(cfg.function)}_{cfg.dims}D_raw.csv")
    agg_csv = os.path.join(args.out_dir, f"diversity_{_normalize_func_name(cfg.function)}_{cfg.dims}D_agg.csv")
    df.to_csv(raw_csv, index=False)
    agg.to_csv(agg_csv, index=False)

    # Plot
    plt.figure(figsize=(7.2, 4.0))
    x = agg["fes"].to_numpy()
    med = agg["median_div"].to_numpy()
    q25 = agg["q25_div"].to_numpy()
    q75 = agg["q75_div"].to_numpy()

    plt.plot(x, med, label="Median diversity", linewidth=2.0)
    plt.fill_between(x, q25, q75, alpha=0.25, label="IQR (25–75%)")

    plt.xlabel("Function evaluations (FEs)")
    plt.ylabel("Mean std(pop) across dimensions")
    plt.title(f"Population diversity vs FEs (Hybrid IWOA) — {_normalize_func_name(cfg.function)}, D={cfg.dims}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    plt.savefig(args.out_png, dpi=200)
    plt.close()

    print(f"Wrote: {args.out_png}")
    print(f"Wrote: {raw_csv}")
    print(f"Wrote: {agg_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


