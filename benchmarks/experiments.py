import argparse
import os
import platform
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from joblib import Parallel, delayed

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluator import CEC_Evaluator
from src.iwoa import IWOA_Strict
from src.woa import WOA
from src.woa_plus import WOAPlus
from benchmarks.reporting import write_experiment_outputs

try:
    import opfunu.cec_based.cec2022 as cec2022
except ImportError as e:
    raise SystemExit(
        "Missing dependency: opfunu. Install with: pip install -r requirements.txt"
    ) from e


@dataclass(frozen=True)
class ExperimentConfig:
    dims: int
    runs: int
    max_fes: int
    pop_size: int
    min_pop_size: int
    seed: int
    seed_strategy: str  # "base_plus_run" | "hash"
    functions: Tuple[str, ...]
    n_jobs: int
    methods: Tuple[str, ...]
    iwoa_variants: Tuple[str, ...]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _stable_hash_int(*parts: Any, mod: int = 2**32 - 1) -> int:
    # Stable across processes and Python versions.
    # Avoid Python's built-in hash() which is salted per process.
    import hashlib

    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest(), "little") % mod


def _make_run_seed(cfg: ExperimentConfig, *, method: str, func_name: str, run_id: int) -> int:
    if cfg.seed_strategy == "base_plus_run":
        # Keep it simple and auditable.
        return int(cfg.seed + 1000 * run_id + _stable_hash_int(method, func_name) % 1000)
    if cfg.seed_strategy == "hash":
        return int(_stable_hash_int(cfg.seed, cfg.dims, cfg.max_fes, method, func_name, run_id))
    raise ValueError(f"Unknown seed_strategy: {cfg.seed_strategy}")


def _set_global_seeds(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)


def _get_env_metadata() -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    try:
        import importlib.metadata as md

        pkgs = [
            "numpy",
            "pandas",
            "scipy",
            "matplotlib",
            "joblib",
            "opfunu",
            "tabulate",
        ]
        meta["packages"] = {p: md.version(p) for p in pkgs if _safe_has_dist(md, p)}
    except Exception:
        meta["packages"] = {}
    return meta


def _safe_has_dist(md_module, name: str) -> bool:
    try:
        md_module.version(name)
        return True
    except Exception:
        return False


def _iwoa_flags_from_variant(variant: str) -> Dict[str, bool]:
    v = variant.strip().lower()
    if v in {"full", "default"}:
        return dict(
            enable_restart=True,
            enable_crossover=True,
            enable_chaos=True,
            enable_levy=True,
            enable_nelder_mead=True,
            enable_init_chaos_opposition=True,
            enable_quasi_reflection=True,
            enable_lpsr=True,
        )

    # Factorial interaction encoding: r{0|1}_c{0|1}_nm{0|1}
    if v.startswith("r") and "_c" in v and "_nm" in v:
        try:
            parts = v.split("_")
            r = int(parts[0][1:])
            c = int(parts[1][1:])
            nm = int(parts[2][2:])
            d = _iwoa_flags_from_variant("full")
            d["enable_restart"] = bool(r)
            d["enable_crossover"] = bool(c)
            d["enable_nelder_mead"] = bool(nm)
            return d
        except Exception as e:
            raise ValueError(f"Bad interaction variant format: {variant}") from e

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
    if v == "no_init":
        d = _iwoa_flags_from_variant("full")
        d["enable_init_chaos_opposition"] = False
        return d
    if v == "no_qr":
        d = _iwoa_flags_from_variant("full")
        d["enable_quasi_reflection"] = False
        return d
    if v == "no_lpsr":
        d = _iwoa_flags_from_variant("full")
        d["enable_lpsr"] = False
        return d
    raise ValueError(
        "Unknown IWOA variant "
        f"'{variant}'. Expected one of: full,no_restart,no_crossover,no_chaos,no_levy,no_nelder_mead,no_init,no_qr,no_lpsr,"
        " or interaction encoding r{0|1}_c{0|1}_nm{0|1}"
    )


def _woa_plus_flags_from_variant(variant: str) -> Dict[str, bool]:
    v = variant.strip().lower()
    # Ladder variants (cumulative)
    if v == "init":
        return dict(
            enable_init_chaos_opposition=True,
            enable_quasi_reflection=False,
            enable_crossover=False,
            enable_restart=False,
            enable_perturb=False,
            enable_chaos_local=False,
            enable_nelder_mead=False,
            enable_lpsr=False,
        )
    if v == "init_qr":
        d = _woa_plus_flags_from_variant("init")
        d["enable_quasi_reflection"] = True
        return d
    if v == "init_qr_crossover":
        d = _woa_plus_flags_from_variant("init_qr")
        d["enable_crossover"] = True
        return d
    if v == "init_qr_crossover_restart":
        d = _woa_plus_flags_from_variant("init_qr_crossover")
        d["enable_restart"] = True
        return d
    if v == "init_qr_crossover_restart_perturb":
        d = _woa_plus_flags_from_variant("init_qr_crossover_restart")
        d["enable_perturb"] = True
        return d
    if v == "init_qr_crossover_restart_perturb_chaos":
        d = _woa_plus_flags_from_variant("init_qr_crossover_restart_perturb")
        d["enable_chaos_local"] = True
        return d
    if v == "init_qr_crossover_restart_perturb_chaos_nm":
        d = _woa_plus_flags_from_variant("init_qr_crossover_restart_perturb_chaos")
        d["enable_nelder_mead"] = True
        return d
    if v in {"all", "full"}:
        # All operators on in the ladder (including LPSR).
        d = _woa_plus_flags_from_variant("init_qr_crossover_restart_perturb_chaos_nm")
        d["enable_lpsr"] = True
        return d
    raise ValueError(
        f"Unknown WOAPlus variant '{variant}'. Expected one of: init,init_qr,init_qr_crossover,init_qr_crossover_restart,"
        "init_qr_crossover_restart_perturb,init_qr_crossover_restart_perturb_chaos,init_qr_crossover_restart_perturb_chaos_nm,all"
    )


def _single_run(cfg: ExperimentConfig, method: str, func_name: str, run_id: int) -> Dict[str, Any]:
    # Re-import inside worker (joblib best practice).
    import warnings

    warnings.filterwarnings("ignore")
    import opfunu.cec_based.cec2022 as cec2022_local

    seed = _make_run_seed(cfg, method=method, func_name=func_name, run_id=run_id)
    _set_global_seeds(seed)

    problem = getattr(cec2022_local, func_name)(ndim=cfg.dims)
    evaluator = CEC_Evaluator(problem.evaluate, cfg.max_fes, problem.lb, problem.ub)
    bounds = np.array([problem.lb, problem.ub])

    if method.startswith("iwoa:"):
        variant = method.split(":", 1)[1]
        flags = _iwoa_flags_from_variant(variant)
        optimizer = IWOA_Strict(
            evaluator=evaluator,
            dim=cfg.dims,
            bounds=bounds,
            pop_size=cfg.pop_size,
            min_pop_size=cfg.min_pop_size,
            **flags,
        )
        method_label = f"IWOA_Strict[{variant}]"
    elif method == "iwoa":
        optimizer = IWOA_Strict(
            evaluator=evaluator,
            dim=cfg.dims,
            bounds=bounds,
            pop_size=cfg.pop_size,
            min_pop_size=cfg.min_pop_size,
        )
        method_label = "IWOA_Strict[full]"
    elif method == "woa":
        optimizer = WOA(
            evaluator=evaluator,
            dim=cfg.dims,
            bounds=bounds,
            pop_size=cfg.pop_size,
        )
        method_label = "WOA"
    elif method.startswith("woa_plus:"):
        variant = method.split(":", 1)[1]
        flags = _woa_plus_flags_from_variant(variant)
        optimizer = WOAPlus(
            evaluator=evaluator,
            dim=cfg.dims,
            bounds=bounds,
            pop_size=cfg.pop_size,
            min_pop_size=cfg.min_pop_size,
            **flags,
        )
        method_label = f"WOAPlus[{variant}]"
    else:
        raise ValueError(f"Unknown method spec: {method}")

    t0 = time.time()
    _, best_fit = optimizer.run()
    t1 = time.time()

    error = float(abs(best_fit - problem.f_global))
    if error < 1e-8:
        error = 0.0

    return {
        "method": method_label,
        "function": func_name,
        "dims": cfg.dims,
        "run_id": int(run_id),
        "seed": int(seed),
        "max_fes": int(cfg.max_fes),
        "fes_used": int(evaluator.calls),
        "best_fit": float(best_fit),
        "f_global": float(problem.f_global),
        "error": float(error),
        "wall_time_s": float(t1 - t0),
    }

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deterministic CEC2022 experiment runner (opfunu).")
    p.add_argument("--dims", type=int, default=20)
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--max_fes", type=int, default=None, help="Defaults to 10000*dims.")
    p.add_argument("--functions", type=str, default="F1-12", help='Examples: "F1-12", "F1,F2,F11"')
    p.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Optional ablation preset: ablation_removal, ablation_addition, ablation_interactions.",
    )
    p.add_argument(
        "--methods",
        type=str,
        default="iwoa,woa",
        help='Comma list. Supported: "iwoa", "woa", "iwoa:<variant>", "woa_plus:<variant>".',
    )
    p.add_argument(
        "--iwoa_variants",
        type=str,
        default="full",
        help=(
            "Comma list for IWOA ablations: "
            "full,no_restart,no_crossover,no_chaos,no_levy,no_nelder_mead,no_init,no_qr,no_lpsr "
            "and interaction encoding r{0|1}_c{0|1}_nm{0|1}"
        ),
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
        help="Output directory. Default: results/experiments/<utc_timestamp>_cec2022_<dims>D",
    )
    return p.parse_args(argv)


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
    # Deduplicate while preserving order
    seen = set()
    out = []
    for f in funcs:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return tuple(out)

def _parse_csv(spec: str) -> Tuple[str, ...]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    return tuple(parts)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    functions = _parse_functions(args.functions)
    iwoa_variants = _parse_csv(args.iwoa_variants)

    if args.preset is not None:
        from benchmarks.ablation_presets import get_preset

        preset = get_preset(args.preset)
        print(f"Using preset: {preset.name} ({preset.description})")
        expanded_methods = [m.strip().lower() for m in preset.methods if m.strip()]
    else:
        methods_raw = _parse_csv(args.methods)
        # Expand "iwoa" into requested variants if provided.
        expanded_methods = []
        for m in methods_raw:
            ml = m.strip().lower()
            if ml == "iwoa":
                for v in iwoa_variants:
                    vv = v.strip().lower()
                    if vv in {"full", "default"}:
                        expanded_methods.append("iwoa")
                    else:
                        expanded_methods.append(f"iwoa:{vv}")
            else:
                expanded_methods.append(ml)

    # Deduplicate while preserving order
    seen = set()
    methods: List[str] = []
    for m in expanded_methods:
        if m not in seen:
            seen.add(m)
            methods.append(m)

    max_fes = int(args.max_fes) if args.max_fes is not None else int(10000 * args.dims)

    cfg = ExperimentConfig(
        dims=int(args.dims),
        runs=int(args.runs),
        max_fes=max_fes,
        pop_size=int(args.pop_size),
        min_pop_size=int(args.min_pop_size),
        seed=int(args.seed),
        seed_strategy=str(args.seed_strategy),
        functions=tuple(functions),
        n_jobs=int(args.n_jobs),
        methods=tuple(methods),
        iwoa_variants=tuple(iwoa_variants),
    )

    out_dir = args.out
    if out_dir is None:
        out_dir = os.path.join("results", "experiments", f"{_utc_now_compact()}_cec2022_{cfg.dims}D")
    _ensure_dir(out_dir)

    # Build tasks
    tasks = [(method, fn, run_id) for method in cfg.methods for fn in cfg.functions for run_id in range(cfg.runs)]

    results: List[Dict[str, Any]] = []
    if cfg.n_jobs == 1:
        for method, fn, run_id in tasks:
            results.append(_single_run(cfg, method, fn, run_id))
    else:
        results = Parallel(n_jobs=cfg.n_jobs, verbose=10)(
            delayed(_single_run)(cfg, method, fn, run_id) for method, fn, run_id in tasks
        )

    write_experiment_outputs(
        out_dir,
        cfg=cfg,
        meta=_get_env_metadata(),
        results=results,
        latex_float_fmt="%.2E",
    )
    print(f"\nSaved results to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


