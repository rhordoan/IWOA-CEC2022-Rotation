from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def results_to_dataframe(results: List[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(list(results))


def summarize_errors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-run results into per-(method,function,dims,budget) summaries.
    Expects an `error` column.
    """
    grp_cols = ["method", "function", "dims", "max_fes"]
    grp = df.groupby(grp_cols, as_index=False)

    def iqr(x: pd.Series) -> float:
        q75 = np.nanpercentile(x, 75)
        q25 = np.nanpercentile(x, 25)
        return float(q75 - q25)

    summary = grp["error"].agg(
        runs="count",
        mean_error="mean",
        std_error="std",
        median_error="median",
        iqr_error=iqr,
        best_error="min",
        worst_error="max",
    )
    return summary.reset_index(drop=True)


def write_jsonl(path: str, rows: Iterable[Mapping[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(dict(r), sort_keys=True) + "\n")


def write_experiment_outputs(
    out_dir: str,
    *,
    cfg: Optional[Any] = None,
    meta: Optional[Dict[str, Any]] = None,
    results: List[Mapping[str, Any]],
    latex_float_fmt: str = "%.2E",
) -> Dict[str, str]:
    """
    Writes:
      - results.csv (per-run)
      - results.jsonl (per-run)
      - summary.csv (aggregated)
      - summary.tex (aggregated, LaTeX)
      - meta.json (config + environment metadata)
    """
    ensure_dir(out_dir)

    df = results_to_dataframe(results)
    results_csv = os.path.join(out_dir, "results.csv")
    df.to_csv(results_csv, index=False)

    results_jsonl = os.path.join(out_dir, "results.jsonl")
    write_jsonl(results_jsonl, results)

    summary_df = summarize_errors(df)
    summary_csv = os.path.join(out_dir, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    summary_tex = os.path.join(out_dir, "summary.tex")
    summary_df.to_latex(summary_tex, index=False, float_format=latex_float_fmt)

    meta_payload: Dict[str, Any] = {}
    if cfg is not None:
        try:
            meta_payload["config"] = asdict(cfg)
        except Exception:
            meta_payload["config"] = cfg  # best-effort
    if meta is not None:
        meta_payload["env"] = meta
    meta_json = os.path.join(out_dir, "meta.json")
    if meta_payload:
        with open(meta_json, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2, sort_keys=True)

    return {
        "results_csv": results_csv,
        "results_jsonl": results_jsonl,
        "summary_csv": summary_csv,
        "summary_tex": summary_tex,
        "meta_json": meta_json,
    }



