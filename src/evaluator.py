import numpy as np

class CEC_Evaluator:
    """
    Wraps the objective function to strictly count every invocation.
    Prevents 'hidden' evaluations during restarts or local search.
    """
    def __init__(self, func, max_fes, lb, ub):
        self.func = func
        self.max_fes = int(max_fes)
        self.calls = 0
        self.lb = lb
        self.ub = ub
        self.stop_flag = False

    def evaluate(self, x):
        if self.calls >= self.max_fes:
            self.stop_flag = True
            return 1e15 # Penalty to stop optimizers
        
        # Strict Bound Enforcement
        x_clipped = np.clip(x, self.lb, self.ub)
        
        val = self.func(x_clipped)
        self.calls += 1
        return val

    def __call__(self, x):
        return self.evaluate(x)