import math
import random
from typing import Tuple

import numpy as np

from src.evaluator import CEC_Evaluator


class WOA:
    """
    A minimal baseline Whale Optimization Algorithm (WOA).

    This is intentionally simple and deterministic given the RNG seed,
    suitable for baseline comparisons and replication-style experiments.
    """

    def __init__(
        self,
        evaluator: CEC_Evaluator,
        dim: int,
        bounds: np.ndarray,
        *,
        pop_size: int = 100,
        spiral_shape_const: float = 1.0,
    ) -> None:
        self.eval = evaluator
        self.dim = int(dim)
        self.lb, self.ub = bounds[0], bounds[1]
        self.pop_size = int(pop_size)
        self.spiral_shape_const = float(spiral_shape_const)

    def run(self) -> Tuple[np.ndarray, float]:
        pop = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fits = np.array([self.eval(x) for x in pop])

        idx_best = int(np.argmin(fits))
        best_x = pop[idx_best].copy()
        best_fit = float(fits[idx_best])

        while not self.eval.stop_flag:
            progress = self.eval.calls / self.eval.max_fes
            a = 2.0 - 2.0 * progress

            for i in range(self.pop_size):
                if self.eval.stop_flag:
                    break

                r1, r2 = random.random(), random.random()
                A, C = 2.0 * a * r1 - a, 2.0 * r2
                l = random.uniform(-1, 1)
                p = random.random()

                x = pop[i].copy()
                if p < 0.5:
                    if abs(A) < 1:
                        D = np.abs(C * best_x - x)
                        x_new = best_x - A * D
                    else:
                        rand_idx = random.randint(0, self.pop_size - 1)
                        D = np.abs(C * pop[rand_idx] - x)
                        x_new = pop[rand_idx] - A * D
                else:
                    D = np.abs(best_x - x)
                    b = self.spiral_shape_const
                    x_new = D * math.exp(b * l) * math.cos(2 * math.pi * l) + best_x

                x_new = np.clip(x_new, self.lb, self.ub)
                f_new = self.eval(x_new)

                pop[i] = x_new
                fits[i] = f_new

                if f_new < best_fit:
                    best_fit = float(f_new)
                    best_x = x_new.copy()

        return best_x, best_fit



