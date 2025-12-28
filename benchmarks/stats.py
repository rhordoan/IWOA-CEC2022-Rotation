import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except Exception as e:
    raise SystemExit("Missing dependency: scipy. Install with: pip install -r requirements.txt") from e


def _holm_adjust(pvals: List[float]) -> List[float]:
    """
    Holm-Bonferroni adjusted p-values.
    Returns adjusted p-values in the original order.
    """
    m = len(pvals)
    order = np.argsort(pvals)
    adj = [0.0] * m
    prev = 0.0
    for k, idx in enumerate(order):
        factor = m - k
        val = min(1.0, float(pvals[idx]) * factor)
        # ensure monotonicity
        val = max(val, prev)
        prev = val
        adj[idx] = val
    return adj


def _rank_row(values: pd.Series) -> pd.Series:
    # Smaller is better -> rank 1 best.
    return values.rank(method="average", ascending=True)


def compute_ranks(summary: pd.DataFrame, metric: str = "mean_error") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - ranks_per_function: rows=function, cols=method, values=rank
      - avg_ranks: rows=method with avg_rank
    """
    pivot = summary.pivot_table(index="function", columns="method", values=metric, aggfunc="first")
    ranks = pivot.apply(_rank_row, axis=1)
    avg = ranks.mean(axis=0).sort_values()
    avg_df = avg.reset_index()
    avg_df.columns = ["method", "avg_rank"]
    return ranks.reset_index(), avg_df


def compute_wilcoxon_vs_baseline(
    summary: pd.DataFrame,
    *,
    baseline: str,
    metric: str = "mean_error",
    alternative: str = "two-sided",
    holm: bool = True,
) -> pd.DataFrame:
    pivot = summary.pivot_table(index="function", columns="method", values=metric, aggfunc="first")
    if baseline not in pivot.columns:
        raise ValueError(f"Baseline method '{baseline}' not found. Available: {list(pivot.columns)}")

    base = pivot[baseline]
    rows: List[Dict[str, object]] = []
    pvals: List[float] = []

    for m in pivot.columns:
        if m == baseline:
            continue
        x = pivot[m]
        # Wilcoxon requires paired samples; drop functions missing either value
        df_pair = pd.concat([base, x], axis=1, keys=["base", "x"]).dropna()
        if len(df_pair) < 3:
            p = float("nan")
            stat = float("nan")
        else:
            # Wilcoxon on differences; smaller metric means better.
            stat, p = wilcoxon(
                df_pair["x"].to_numpy(),
                df_pair["base"].to_numpy(),
                alternative=alternative,
                zero_method="wilcox",
            )
            stat = float(stat)
            p = float(p)
        pvals.append(p)
        rows.append(
            {
                "baseline": baseline,
                "method": m,
                "n_functions": int(len(df_pair)),
                "wilcoxon_stat": stat,
                "p_value": p,
            }
        )

    out = pd.DataFrame(rows)
    if holm and len(out) > 0:
        # Only adjust finite p-values; keep NaNs as NaN.
        finite_mask = np.isfinite(out["p_value"].to_numpy())
        finite_pvals = out.loc[finite_mask, "p_value"].tolist()
        adj = _holm_adjust(finite_pvals) if finite_pvals else []
        out["p_holm"] = np.nan
        if adj:
            out.loc[finite_mask, "p_holm"] = adj
    return out.sort_values("p_value", na_position="last").reset_index(drop=True)


def write_tables(out_dir: str, *, ranks_per_function: pd.DataFrame, avg_ranks: pd.DataFrame, wilcox: pd.DataFrame) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ranks_per_function.to_csv(os.path.join(out_dir, "ranks_per_function.csv"), index=False)
    avg_ranks.to_csv(os.path.join(out_dir, "avg_ranks.csv"), index=False)
    wilcox.to_csv(os.path.join(out_dir, "wilcoxon.csv"), index=False)

    ranks_per_function.to_latex(os.path.join(out_dir, "ranks_per_function.tex"), index=False, float_format="%.3f")
    avg_ranks.to_latex(os.path.join(out_dir, "avg_ranks.tex"), index=False, float_format="%.3f")
    wilcox.to_latex(os.path.join(out_dir, "wilcoxon.tex"), index=False, float_format="%.3E")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute ranks and Wilcoxon tests from summary.csv.")
    p.add_argument("--input", type=str, required=True, help="Path to summary.csv from an experiment run.")
    p.add_argument("--out", type=str, default=None, help="Output directory (default: same dir as input).")
    p.add_argument("--metric", type=str, default="mean_error", help="Metric column in summary.csv to compare.")
    p.add_argument("--baseline", type=str, default="WOA", help="Baseline method name as it appears in summary.csv.")
    p.add_argument("--alternative", type=str, default="two-sided", choices=["two-sided", "less", "greater"])
    p.add_argument("--no_holm", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    summary = pd.read_csv(args.input)

    ranks_pf, avg_ranks = compute_ranks(summary, metric=args.metric)
    wilcox = compute_wilcoxon_vs_baseline(
        summary,
        baseline=args.baseline,
        metric=args.metric,
        alternative=args.alternative,
        holm=not args.no_holm,
    )

    out_dir = args.out if args.out is not None else os.path.dirname(os.path.abspath(args.input))
    write_tables(out_dir, ranks_per_function=ranks_pf, avg_ranks=avg_ranks, wilcox=wilcox)
    print(f"Saved stats tables to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



