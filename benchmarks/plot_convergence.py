import argparse
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluator import CEC_Evaluator
from src.iwoa import IWOA_Strict
from src.woa import WOA

try:
    import opfunu.cec_based.cec2022 as cec2022
except ImportError as e:
    raise SystemExit("Missing dependency: opfunu. Install with: pip install -r requirements.txt") from e


@dataclass(frozen=True)
class ConvergenceConfig:
    dims: int
    runs: int
    max_fes: int
    log_every: int
    seed: int
    seed_strategy: str
    functions: Tuple[str, ...]
    methods: Tuple[str, ...]
    pop_size: int
    min_pop_size: int
    n_jobs: int


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def _make_run_seed(cfg: ConvergenceConfig, *, method: str, func_name: str, run_id: int) -> int:
    if cfg.seed_strategy == "base_plus_run":
        return int(cfg.seed + 1000 * run_id + _stable_hash_int(method, func_name) % 1000)
    if cfg.seed_strategy == "hash":
        return int(_stable_hash_int(cfg.seed, cfg.dims, cfg.max_fes, method, func_name, run_id))
    raise ValueError(f"Unknown seed_strategy: {cfg.seed_strategy}")


def _parse_functions(spec: str) -> Tuple[str, ...]:
    s = spec.strip()
    if s.upper() in {"ALL", "F1-12", "F1..12"}:
        return tuple(f"F{i}2022" for i in range(1, 13))
    parts = [p.strip() for p in s.split(",") if p.strip()]
    funcs: List[str] = []
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            a = a.strip().upper().replace("F", "")
            b = b.strip().upper().replace("F", "")
            lo, hi = int(a), int(b)
            funcs.extend([f"F{i}2022" for i in range(lo, hi + 1)])
        else:
            q = p.upper()
            if not q.startswith("F"):
                q = "F" + q
            if not q.endswith("2022"):
                q = q + "2022"
            funcs.append(q)
    seen = set()
    out = []
    for f in funcs:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return tuple(out)


def _parse_csv(spec: str) -> Tuple[str, ...]:
    return tuple(p.strip() for p in spec.split(",") if p.strip())


def _iwoa_flags_from_variant(variant: str) -> Dict[str, bool]:
    v = variant.strip().lower()
    if v in {"full", "default"}:
        return dict(
            enable_restart=True,
            enable_crossover=True,
            enable_chaos=True,
            enable_levy=True,
            enable_nelder_mead=True,
        )
    if v == "no_restart":
        d = _iwoa_flags_from_variant("full")
        d["enable_restart"] = False
        return d
    if v == "no_crossover":
        d = _iwoa_flags_from_variant("full")
        d["enable_crossover"] = False
        return d
    if v == "no_chaos":
        d = _iwoa_flags_from_variant("full")
        d["enable_chaos"] = False
        return d
    if v == "no_levy":
        d = _iwoa_flags_from_variant("full")
        d["enable_levy"] = False
        return d
    if v == "no_nelder_mead":
        d = _iwoa_flags_from_variant("full")
        d["enable_nelder_mead"] = False
        return d
    raise ValueError(
        f"Unknown IWOA variant '{variant}'. Expected one of: full,no_restart,no_crossover,no_chaos,no_levy,no_nelder_mead"
    )


def _method_label(method: str) -> str:
    if method.startswith("iwoa:"):
        return f"IWOA_Strict[{method.split(':', 1)[1]}]"
    if method == "iwoa":
        return "IWOA_Strict[full]"
    if method == "woa":
        return "WOA"
    return method


class EvaluatorWithHistory:
    """
    Tracks best-so-far error at fixed FE checkpoints.
    """

    def __init__(self, func, max_fes: int, lb, ub, optimal: float, checkpoints: Sequence[int]) -> None:
        self.func = func
        self.max_fes = int(max_fes)
        self.calls = 0
        self.lb = lb
        self.ub = ub
        self.optimal = float(optimal)
        self.stop_flag = False

        self._checkpoints_set = set(int(c) for c in checkpoints)
        self.history_fes: List[int] = []
        self.history_error: List[float] = []

        self._best_error = float("inf")

    def evaluate(self, x):
        if self.calls >= self.max_fes:
            self.stop_flag = True
            return 1e15

        val = self.func(np.clip(x, self.lb, self.ub))
        self.calls += 1

        err = abs(float(val) - self.optimal)
        if err == 0.0:
            err = 1e-15
        if err < self._best_error:
            self._best_error = err

        if self.calls in self._checkpoints_set:
            self.history_fes.append(int(self.calls))
            self.history_error.append(float(self._best_error))

        return val

    def __call__(self, x):
        return self.evaluate(x)


def _make_checkpoints(max_fes: int, log_every: int) -> List[int]:
    cps = [1]
    if log_every <= 0:
        log_every = 500
    cps.extend(list(range(log_every, max_fes + 1, log_every)))
    if cps[-1] != max_fes:
        cps.append(max_fes)
    # unique + sorted
    return sorted(set(int(c) for c in cps if 1 <= c <= max_fes))


def _run_single(cfg: ConvergenceConfig, method: str, func_name: str, run_id: int) -> pd.DataFrame:
    import warnings

    warnings.filterwarnings("ignore")
    import opfunu.cec_based.cec2022 as cec2022_local

    seed = _make_run_seed(cfg, method=method, func_name=func_name, run_id=run_id)
    _set_global_seeds(seed)

    problem = getattr(cec2022_local, func_name)(ndim=cfg.dims)
    bounds = np.array([problem.lb, problem.ub])
    checkpoints = _make_checkpoints(cfg.max_fes, cfg.log_every)

    eval_hist = EvaluatorWithHistory(
        problem.evaluate, cfg.max_fes, problem.lb, problem.ub, problem.f_global, checkpoints=checkpoints
    )

    if method.startswith("iwoa:"):
        flags = _iwoa_flags_from_variant(method.split(":", 1)[1])
        optimizer = IWOA_Strict(
            evaluator=eval_hist, dim=cfg.dims, bounds=bounds, pop_size=cfg.pop_size, min_pop_size=cfg.min_pop_size, **flags
        )
        label = _method_label(method)
    elif method == "iwoa":
        optimizer = IWOA_Strict(
            evaluator=eval_hist, dim=cfg.dims, bounds=bounds, pop_size=cfg.pop_size, min_pop_size=cfg.min_pop_size
        )
        label = _method_label(method)
    elif method == "woa":
        optimizer = WOA(evaluator=eval_hist, dim=cfg.dims, bounds=bounds, pop_size=cfg.pop_size)
        label = _method_label(method)
    else:
        raise ValueError(f"Unknown method spec: {method}")

    optimizer.run()

    df = pd.DataFrame(
        {
            "method": label,
            "method_raw": method,
            "function": func_name,
            "dims": cfg.dims,
            "run_id": run_id,
            "seed": seed,
            "fes": eval_hist.history_fes,
            "error": eval_hist.history_error,
        }
    )
    return df


def _aggregate_convergence(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["function", "dims", "method", "fes"], as_index=False)["error"]
    out = grp.agg(
        median_error="median",
        q25_error=lambda x: float(np.nanpercentile(x, 25)),
        q75_error=lambda x: float(np.nanpercentile(x, 75)),
        mean_error="mean",
    )
    return out


def _plot_function(out_dir: str, func_name: str, agg: pd.DataFrame) -> str:
    fig_dir = os.path.join(out_dir, "plots")
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sub = agg[agg["function"] == func_name]
    for method in sorted(sub["method"].unique()):
        mdf = sub[sub["method"] == method].sort_values("fes")
        x = mdf["fes"].to_numpy()
        y = mdf["median_error"].to_numpy()
        lo = mdf["q25_error"].to_numpy()
        hi = mdf["q75_error"].to_numpy()
        plt.semilogy(x, y, linewidth=2.0, label=f"{method} (median)")
        plt.fill_between(x, lo, hi, alpha=0.15)

    plt.title(f"Convergence: {func_name} ({int(sub['dims'].iloc[0])}D)", fontsize=14, fontweight="bold")
    plt.xlabel("Function Evaluations (FEs)", fontsize=12)
    plt.ylabel("Error (log scale)", fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend(fontsize=10)

    out_file = os.path.join(fig_dir, f"{func_name}_convergence.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    return out_file


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-run convergence plotting (median + IQR).")
    p.add_argument("--dims", type=int, default=20)
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--max_fes", type=int, default=200000)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--functions", type=str, default="F1,F11")
    p.add_argument("--methods", type=str, default="iwoa,woa")
    p.add_argument(
        "--iwoa_variants",
        type=str,
        default="full",
        help="Comma list: full,no_restart,no_crossover,no_chaos,no_levy,no_nelder_mead",
    )
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--seed_strategy", type=str, default="base_plus_run", choices=["base_plus_run", "hash"])
    p.add_argument("--pop_size", type=int, default=100)
    p.add_argument("--min_pop_size", type=int, default=20)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory. Default: results/convergence/<utc_timestamp>_cec2022_<dims>D",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    functions = _parse_functions(args.functions)
    methods_raw = _parse_csv(args.methods)
    variants = _parse_csv(args.iwoa_variants)

    expanded_methods: List[str] = []
    for m in methods_raw:
        ml = m.strip().lower()
        if ml == "iwoa":
            for v in variants:
                vv = v.strip().lower()
                if vv in {"full", "default"}:
                    expanded_methods.append("iwoa")
                else:
                    expanded_methods.append(f"iwoa:{vv}")
        else:
            expanded_methods.append(ml)
    seen = set()
    methods: List[str] = []
    for m in expanded_methods:
        if m not in seen:
            seen.add(m)
            methods.append(m)

    cfg = ConvergenceConfig(
        dims=int(args.dims),
        runs=int(args.runs),
        max_fes=int(args.max_fes),
        log_every=int(args.log_every),
        seed=int(args.seed),
        seed_strategy=str(args.seed_strategy),
        functions=tuple(functions),
        methods=tuple(methods),
        pop_size=int(args.pop_size),
        min_pop_size=int(args.min_pop_size),
        n_jobs=int(args.n_jobs),
    )

    out_dir = args.out
    if out_dir is None:
        out_dir = os.path.join("results", "convergence", f"{_utc_now_compact()}_cec2022_{cfg.dims}D")
    os.makedirs(out_dir, exist_ok=True)

    tasks = [(m, f, r) for m in cfg.methods for f in cfg.functions for r in range(cfg.runs)]
    if cfg.n_jobs == 1:
        dfs = [_run_single(cfg, m, f, r) for (m, f, r) in tasks]
    else:
        dfs = Parallel(n_jobs=cfg.n_jobs, verbose=10)(delayed(_run_single)(cfg, m, f, r) for (m, f, r) in tasks)

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(out_dir, "convergence.csv"), index=False)

    agg = _aggregate_convergence(df)
    agg.to_csv(os.path.join(out_dir, "convergence_agg.csv"), index=False)

    for fn in cfg.functions:
        out_file = _plot_function(out_dir, fn, agg)
        print(f"Saved plot: {out_file}")

    meta = {
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "config": {
            "dims": cfg.dims,
            "runs": cfg.runs,
            "max_fes": cfg.max_fes,
            "log_every": cfg.log_every,
            "seed": cfg.seed,
            "seed_strategy": cfg.seed_strategy,
            "functions": list(cfg.functions),
            "methods": list(cfg.methods),
            "pop_size": cfg.pop_size,
            "min_pop_size": cfg.min_pop_size,
        },
    }
    import json

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print(f"\nSaved convergence outputs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())