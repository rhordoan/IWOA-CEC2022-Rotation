import sys
import os
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluator import CEC_Evaluator
from src.iwoa import IWOA_Strict

# Check for CEC 2022 library
try:
    import opfunu.cec_based.cec2022 as cec2022
    from tabulate import tabulate
    HAS_OPFUNU = True
except ImportError:
    HAS_OPFUNU = False
    print("Error: opfunu not installed. pip install opfunu tabulate")

def single_run_task(func_name, dims, run_id):
    import warnings
    warnings.filterwarnings("ignore")
    
    # Re-import inside worker
    import opfunu.cec_based.cec2022 as cec2022 
    
    problem = getattr(cec2022, func_name)(ndim=dims)
    MAX_FES = 10000 * dims
    
    evaluator = CEC_Evaluator(problem.evaluate, MAX_FES, problem.lb, problem.ub)
    bounds = np.array([problem.lb, problem.ub])
    
    optimizer = IWOA_Strict(
        evaluator=evaluator,
        dim=dims,
        bounds=bounds,
        pop_size=100,       
        min_pop_size=20,
    )
    
    _, best_fit = optimizer.run()
    
    error = abs(best_fit - problem.f_global)
    if error < 1e-8: error = 0.0
    
    return error

def run_cec2022_final_validation(dims=20, runs=30):
    if not HAS_OPFUNU: return

    max_fes = 10000 * dims
    print(f"\n{'='*75}")
    print(f"🚀 STRICT PAPER VALIDATION RUN - GOLD STANDARD (Dims={dims}, Runs={runs})")
    print(f"{'='*75}")
    
    final_stats = []
    
    for i in range(1, 13):
        func_name = f"F{i}2022"
        print(f"Processing {func_name:<10} ...", end=" ", flush=True)
        
        start_t = time.time()
        
        errors = Parallel(n_jobs=-1, verbose=0)(
            delayed(single_run_task)(func_name, dims, r) for r in range(runs)
        )
        
        duration = time.time() - start_t
        
        avg_err = np.mean(errors)
        best_err = np.min(errors)
        worst_err = np.max(errors)
        std_err = np.std(errors)
        
        print(f"Done ({duration:.1f}s). Mean: {avg_err:.2E} | Best: {best_err:.2E}")
        
        final_stats.append({
            "Function": func_name,
            "Best": best_err,
            "Worst": worst_err,
            "Mean Error": avg_err,
            "Std Dev": std_err,
            "Avg Time (s)": duration / runs
        })

    print("\n" + tabulate(pd.DataFrame(final_stats), headers="keys", tablefmt="latex", floatfmt=".2E"))
    print("\n[INFO] Table formatted for LaTeX. Copy-paste into your paper.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, default=20)
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()
    
    run_cec2022_final_validation(dims=args.dims, runs=args.runs)