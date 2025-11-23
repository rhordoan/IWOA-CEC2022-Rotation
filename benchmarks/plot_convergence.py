import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import opfunu.cec_based.cec2022 as cec2022

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.iwoa import IWOA_Strict

class EvaluatorWithHistory:
    def __init__(self, func, max_fes, lb, ub, optimal):
        self.func = func
        self.max_fes = int(max_fes)
        self.calls = 0
        self.lb = lb
        self.ub = ub
        self.optimal = optimal
        self.history_fes = []
        self.history_error = []
        self.stop_flag = False

    def evaluate(self, x):
        if self.calls >= self.max_fes:
            self.stop_flag = True
            return 1e15
        
        val = self.func(np.clip(x, self.lb, self.ub))
        self.calls += 1
        
        # Log every 500 FEs
        if self.calls % 500 == 0 or self.calls == 1:
            error = abs(val - self.optimal)
            if error == 0: error = 1e-15 
            self.history_fes.append(self.calls)
            self.history_error.append(error)
            
        return val

    def __call__(self, x):
        return self.evaluate(x)

class StandardWOA:
    def __init__(self, evaluator, dim, bounds, pop_size=100):
        self.eval = evaluator
        self.dim = dim
        self.lb, self.ub = bounds[0], bounds[1]
        self.pop_size = pop_size

    def run(self):
        pop = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fits = np.array([self.eval(x) for x in pop])
        best_idx = np.argmin(fits)
        best_X = pop[best_idx].copy()
        
        while not self.eval.stop_flag:
            a = 2.0 - 2.0 * (self.eval.calls / self.eval.max_fes)
            for i in range(self.pop_size):
                if self.eval.stop_flag: break
                r1, r2 = random.random(), random.random()
                A, C = 2*a*r1 - a, 2*r2
                l = random.uniform(-1, 1)
                p = random.random()
                
                if p < 0.5:
                    if abs(A) < 1:
                        D = abs(C * best_X - pop[i])
                        pop[i] = best_X - A * D
                    else:
                        rand_idx = random.randint(0, self.pop_size-1)
                        D = abs(C * pop[rand_idx] - pop[i])
                        pop[i] = pop[rand_idx] - A * D
                else:
                    D = abs(best_X - pop[i])
                    pop[i] = D * math.exp(1*l) * math.cos(2*math.pi*l) + best_X
                
                pop[i] = np.clip(pop[i], self.lb, self.ub)
                fits[i] = self.eval(pop[i])
                
                if fits[i] < fits[best_idx]:
                    best_idx = i
                    best_X = pop[i].copy()

def generate_comparison_plot(func_name, dims=20):
    print(f"Generating plot for {func_name}...")
    
    problem = getattr(cec2022, func_name)(ndim=dims)
    max_fes = 200000
    bounds = np.array([problem.lb, problem.ub])

    # --- Run Baseline ---
    eval_base = EvaluatorWithHistory(problem.evaluate, max_fes, problem.lb, problem.ub, problem.f_global)
    woa = StandardWOA(eval_base, dims, bounds)
    try: woa.run()
    except: pass 

    # --- Run Proposed IWOA ---
    eval_prop = EvaluatorWithHistory(problem.evaluate, max_fes, problem.lb, problem.ub, problem.f_global)
    iwoa = IWOA_Strict(eval_prop, dims, bounds)
    try: iwoa.run()
    except: pass

    base_x, base_y = eval_base.history_fes, eval_base.history_error
    prop_x, prop_y = eval_prop.history_fes, eval_prop.history_error

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))
    plt.semilogy(base_x, base_y, 'r--', linewidth=1.5, label='Standard WOA')
    plt.semilogy(prop_x, prop_y, 'b-', linewidth=2.0, label='Proposed IWOA')

    plt.title(f'Convergence Analysis: {func_name} (20D)', fontsize=14, fontweight='bold')
    plt.xlabel('Function Evaluations (FEs)', fontsize=12)
    plt.ylabel('Error (Log Scale)', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend(fontsize=12)
    
    if func_name == "F112022":
        plt.axhline(y=150.0, color='gray', linestyle=':', label='SOTA Runner-Up (150.0)')
    
    output_dir = os.path.join(os.path.dirname(__file__), '../results/plots')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{func_name}_convergence.png")
    
    plt.savefig(output_file, dpi=300)
    print(f"Saved {output_file}")
    plt.close()

if __name__ == "__main__":
    generate_comparison_plot("F12022")  
    generate_comparison_plot("F112022")