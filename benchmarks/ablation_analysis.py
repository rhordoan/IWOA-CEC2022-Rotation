import argparse
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_required(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def _latex(df: pd.DataFrame, path: str, float_fmt: str = "%.3E") -> None:
    df.to_latex(path, index=False, float_format=float_fmt)


def _compute_deltas(
    summary: pd.DataFrame,
    *,
    baseline_method: str,
    metric: str = "mean_error",
    eps: float = 1e-15,
) -> pd.DataFrame:
    base = summary[summary["method"] == baseline_method][["function", "dims", "max_fes", metric]].rename(
        columns={metric: f"{metric}_baseline"}
    )
    merged = summary.merge(base, on=["function", "dims", "max_fes"], how="left")
    merged["delta"] = merged[metric] - merged[f"{metric}_baseline"]
    merged["ratio"] = (merged[metric] + eps) / (merged[f"{metric}_baseline"] + eps)
    merged["log10_ratio"] = np.log10(merged["ratio"])
    return merged


def _success_rates(
    results: pd.DataFrame,
    thresholds: Tuple[float, ...],
) -> pd.DataFrame:
    grp_cols = ["method", "function", "dims", "max_fes"]
    out = results.groupby(grp_cols, as_index=False).agg(runs=("error", "count"))
    for thr in thresholds:
        col = f"success_le_{thr:.0e}"
        s = results.assign(_ok=results["error"] <= thr).groupby(grp_cols, as_index=False)["_ok"].mean()
        out = out.merge(s, on=grp_cols, how="left")
        out = out.rename(columns={"_ok": col})
    return out


def _heatmap(
    df: pd.DataFrame,
    *,
    out_file: str,
    title: str,
    value_col: str,
    dim: int,
    baseline_label: str,
) -> None:
    sub = df[df["dims"] == dim].copy()
    pivot = sub.pivot_table(index="function", columns="method", values=value_col, aggfunc="first")

    # Drop baseline column (it's always 0/1 depending on metric); keep others.
    if baseline_label in pivot.columns:
        pivot = pivot.drop(columns=[baseline_label])

    functions = list(pivot.index)
    methods = list(pivot.columns)
    mat = pivot.to_numpy()

    plt.figure(figsize=(max(10, 0.6 * len(methods)), max(6, 0.4 * len(functions))))
    im = plt.imshow(mat, aspect="auto", interpolation="nearest")
    plt.colorbar(im, fraction=0.02, pad=0.02)
    plt.xticks(range(len(methods)), methods, rotation=45, ha="right")
    plt.yticks(range(len(functions)), functions)
    plt.title(title, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation analysis: deltas + success rates + optional heatmaps.")
    p.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Experiment output folder containing summary.csv and results.csv.",
    )
    p.add_argument("--out", type=str, default=None, help="Output folder (default: input_dir).")
    p.add_argument("--metric", type=str, default="mean_error", help="Metric column in summary.csv.")
    p.add_argument("--baseline_woa", type=str, default="WOA", help="Baseline label for WOA in summary/results.")
    p.add_argument(
        "--baseline_full",
        type=str,
        default="IWOA_Strict[full]",
        help="Baseline label for full method in summary/results.",
    )
    p.add_argument(
        "--thresholds",
        type=str,
        default="1e-8,1e-6",
        help="Comma list of success thresholds for error (e.g., 1e-8,1e-6).",
    )
    p.add_argument("--heatmap_dim", type=int, default=20, help="Which dimension to use for heatmap outputs.")
    p.add_argument("--no_heatmaps", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    in_dir = args.input_dir
    out_dir = args.out if args.out is not None else in_dir
    os.makedirs(out_dir, exist_ok=True)

    summary = _read_required(os.path.join(in_dir, "summary.csv"))
    results = _read_required(os.path.join(in_dir, "results.csv"))

    thresholds = tuple(float(x) for x in args.thresholds.split(",") if x.strip())

    # Δ vs WOA
    d_woa = _compute_deltas(summary, baseline_method=args.baseline_woa, metric=args.metric)
    d_woa_out = d_woa.sort_values(["dims", "function", "method"]).reset_index(drop=True)
    d_woa_csv = os.path.join(out_dir, "ablation_delta_vs_woa.csv")
    d_woa_out.to_csv(d_woa_csv, index=False)
    _latex(d_woa_out, os.path.join(out_dir, "ablation_delta_vs_woa.tex"))

    # Δ vs full
    d_full = _compute_deltas(summary, baseline_method=args.baseline_full, metric=args.metric)
    d_full_out = d_full.sort_values(["dims", "function", "method"]).reset_index(drop=True)
    d_full_csv = os.path.join(out_dir, "ablation_delta_vs_full.csv")
    d_full_out.to_csv(d_full_csv, index=False)
    _latex(d_full_out, os.path.join(out_dir, "ablation_delta_vs_full.tex"))

    # Success rates
    sr = _success_rates(results, thresholds)
    sr_out = sr.sort_values(["dims", "function", "method"]).reset_index(drop=True)
    sr_out.to_csv(os.path.join(out_dir, "success_rates.csv"), index=False)
    _latex(sr_out, os.path.join(out_dir, "success_rates.tex"), float_fmt="%.3f")

    # Optional heatmaps
    if not args.no_heatmaps:
        plots_dir = os.path.join(out_dir, "ablation_plots")
        _heatmap(
            d_woa_out,
            out_file=os.path.join(plots_dir, f"heatmap_log10ratio_vs_woa_{args.heatmap_dim}D.png"),
            title=f"log10 ratio vs {args.baseline_woa} ({args.heatmap_dim}D)",
            value_col="log10_ratio",
            dim=int(args.heatmap_dim),
            baseline_label=args.baseline_woa,
        )
        _heatmap(
            d_full_out,
            out_file=os.path.join(plots_dir, f"heatmap_log10ratio_vs_full_{args.heatmap_dim}D.png"),
            title=f"log10 ratio vs {args.baseline_full} ({args.heatmap_dim}D)",
            value_col="log10_ratio",
            dim=int(args.heatmap_dim),
            baseline_label=args.baseline_full,
        )

    print(f"Saved ablation analysis outputs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



