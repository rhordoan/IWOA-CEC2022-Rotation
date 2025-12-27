import math
import random
from typing import Tuple

import numpy as np
from scipy.optimize import minimize

from src.evaluator import CEC_Evaluator


class WOAPlus:
    """
    WOA dynamics + optional extra operators (used for \"addition\" ablations).

    This is NOT intended to be \"the proposed method\"; it's a controlled ladder
    to answer: which operators added to WOA drive improvements under the same FE budget?
    """

    def __init__(
        self,
        evaluator: CEC_Evaluator,
        dim: int,
        bounds: np.ndarray,
        *,
        pop_size: int = 100,
        min_pop_size: int = 20,
        spiral_shape_const: float = 1.0,
        enable_init_chaos_opposition: bool = True,
        enable_quasi_reflection: bool = True,
        enable_crossover: bool = True,
        enable_restart: bool = True,
        enable_perturb: bool = True,
        enable_chaos_local: bool = True,
        enable_nelder_mead: bool = True,
        enable_lpsr: bool = True,
    ) -> None:
        self.eval = evaluator
        self.dim = int(dim)
        self.lb, self.ub = bounds[0], bounds[1]
        self.pop_size = int(pop_size)
        self.initial_pop_size = int(pop_size)
        self.min_pop_size = int(min_pop_size)
        self.spiral_shape_const = float(spiral_shape_const)

        self.enable_init_chaos_opposition = bool(enable_init_chaos_opposition)
        self.enable_quasi_reflection = bool(enable_quasi_reflection)
        self.enable_crossover = bool(enable_crossover)
        self.enable_restart = bool(enable_restart)
        self.enable_perturb = bool(enable_perturb)
        self.enable_chaos_local = bool(enable_chaos_local)
        self.enable_nelder_mead = bool(enable_nelder_mead)
        self.enable_lpsr = bool(enable_lpsr)

    # ---- Init helpers (mirrors IWOA init for controlled ablations) ----
    def gauss_map(self, x):
        x = np.where(x == 0, 1e-9, x)
        return (1.0 / x) % 1.0

    def logistic_map(self, x):
        return 4 * x * (1 - x)

    def _initial_candidates(self, size=None):
        n = size if size is not None else self.pop_size
        if not self.enable_init_chaos_opposition:
            X = np.random.rand(n, self.dim)
            X = X * (self.ub - self.lb) + self.lb
            return X, self.lb + self.ub - X

        half = n // 2
        if n < 2:
            half = 0

        X_gauss = np.random.rand(half, self.dim)
        X_logis = np.random.rand(n - half, self.dim)

        for _ in range(10):
            X_gauss = self.gauss_map(X_gauss)
            X_logis = self.logistic_map(X_logis)

        X = np.vstack((X_gauss, X_logis)) if half > 0 else X_logis
        X = X * (self.ub - self.lb) + self.lb
        return X, self.lb + self.ub - X

    def initialize_population(self):
        X, X_opp = self._initial_candidates()
        f_X = np.array([self.eval(x) for x in X])

        if not self.enable_init_chaos_opposition:
            return X, f_X

        f_Xopp = []
        for x in X_opp:
            if self.eval.calls < self.eval.max_fes:
                f_Xopp.append(self.eval(x))
            else:
                f_Xopp.append(1e15)
        f_Xopp = np.array(f_Xopp)

        take_opp = f_Xopp < f_X
        pop = np.where(take_opp[:, None], X_opp, X)
        fits = np.where(take_opp, f_Xopp, f_X)
        return pop, fits

    # ---- Operators ----
    def get_quasi_reflection(self, x, curr_min, curr_max):
        center = (curr_min + curr_max) / 2.0
        r = np.random.rand(*x.shape)
        return center + (x - center) * r

    def levy_flight(self, beta: float = 1.5) -> np.ndarray:
        sigma_u = (
            math.gamma(1 + beta)
            * math.sin(math.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = np.random.randn(self.dim) * sigma_u
        v = np.random.randn(self.dim)
        return u / (np.abs(v) ** (1 / beta) + 1e-9)

    def apply_crossover(self, x, pop, fits, idx):
        if pop.shape[0] < 4:
            return x

        # Choose a \"good\" reference from top 20% by fitness (closer to IWOA behavior).
        top_k = max(1, int(pop.shape[0] * 0.2))
        top_idx = np.argsort(fits)[:top_k]
        r_best = int(np.random.choice(top_idx))

        candidates = [i for i in range(pop.shape[0]) if i != idx and i != r_best]
        if len(candidates) < 2:
            return x

        r1, r2 = np.random.choice(candidates, 2, replace=False)
        F = 0.5 + 0.4 * random.random()
        mutant = x + F * (pop[r_best] - x) + F * (pop[r1] - pop[r2])
        mask = np.random.rand(self.dim) < 0.9
        return np.where(mask, mutant, x)

    def check_and_restart(self, pop, fits, progress):
        if progress > 0.90:
            return pop, fits, False
        if self.eval.stop_flag:
            return pop, fits, False

        std_pos = np.std(pop, axis=0)
        avg_diversity = np.mean(std_pos)

        domain_range = np.mean(self.ub - self.lb)
        threshold = 1e-4 * domain_range

        if avg_diversity < threshold:
            num_keep = max(1, int(pop.shape[0] * 0.1))
            sorted_idx = np.argsort(fits)

            num_replace = pop.shape[0] - num_keep
            if num_replace < 1:
                return pop, fits, False

            X_new, _ = self._initial_candidates(size=num_replace)
            replace_indices = sorted_idx[num_keep:]

            for i, idx in enumerate(replace_indices):
                if self.eval.stop_flag:
                    break
                pop[idx] = X_new[i]
                fits[idx] = self.eval(pop[idx])

            return pop, fits, True

        return pop, fits, False

    # ---- Main run ----
    def run(self) -> Tuple[np.ndarray, float]:
        pop, fits = self.initialize_population()

        best_idx = int(np.argmin(fits))
        best_x = pop[best_idx].copy()
        best_fit = float(fits[best_idx])

        stagnation_counter = 0
        gen = 0

        while self.eval.calls < self.eval.max_fes and not self.eval.stop_flag:
            gen += 1
            progress = self.eval.calls / self.eval.max_fes

            if self.enable_lpsr:
                plan_pop = int(
                    np.round((self.min_pop_size - self.initial_pop_size) * progress + self.initial_pop_size)
                )
                if pop.shape[0] > plan_pop:
                    sorted_idx = np.argsort(fits)
                    pop = pop[sorted_idx[:plan_pop]]
                    fits = fits[sorted_idx[:plan_pop]]

            if self.enable_restart:
                pop, fits, restarted = self.check_and_restart(pop, fits, progress)
                if restarted:
                    best_idx = int(np.argmin(fits))
                    if fits[best_idx] < best_fit:
                        best_fit = float(fits[best_idx])
                        best_x = pop[best_idx].copy()
                    stagnation_counter = 0
                    continue

            a = 2.0 - 2.0 * progress
            curr_min, curr_max = np.min(pop, axis=0), np.max(pop, axis=0)

            updated_any = False
            for i in range(pop.shape[0]):
                if self.eval.stop_flag:
                    break

                x = pop[i].copy()
                r1, r2 = random.random(), random.random()
                A, C = 2.0 * a * r1 - a, 2.0 * r2
                l = random.uniform(-1, 1)
                p = random.random()

                if p < 0.5:
                    if abs(A) < 1:
                        D = np.abs(C * best_x - x)
                        x_new = best_x - A * D
                    else:
                        rand_idx = random.randint(0, pop.shape[0] - 1)
                        D = np.abs(C * pop[rand_idx] - x)
                        x_new = pop[rand_idx] - A * D
                else:
                    D = np.abs(best_x - x)
                    b = self.spiral_shape_const
                    x_new = D * math.exp(b * l) * math.cos(2 * math.pi * l) + best_x

                if self.enable_perturb and stagnation_counter > 10:
                    if random.random() < 0.5:
                        x_new += self.levy_flight() * (x_new - best_x) * 0.5
                    else:
                        x_new += np.random.normal(0, 1.0, self.dim) * (self.ub - self.lb) * 0.01

                if self.enable_crossover:
                    # Keep schedule simple: always on when enabled (this is an ablation ladder).
                    if random.random() < 0.9:
                        x_new = self.apply_crossover(x, pop, fits, i)

                x_new = np.clip(x_new, self.lb, self.ub)

                if self.enable_quasi_reflection:
                    x_qr = self.get_quasi_reflection(x_new, curr_min, curr_max)
                    x_qr = np.clip(x_qr, self.lb, self.ub)
                    f_new = self.eval(x_new)
                    f_qr = self.eval(x_qr)
                    if self.eval.stop_flag:
                        break
                    if f_qr < f_new:
                        x_new, f_new = x_qr, f_qr
                else:
                    f_new = self.eval(x_new)

                if self.eval.stop_flag:
                    break

                pop[i] = x_new
                fits[i] = f_new

                if f_new < best_fit:
                    best_fit = float(f_new)
                    best_x = x_new.copy()
                    updated_any = True

            if updated_any:
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if self.enable_chaos_local and progress > 0.9 and not updated_any and not self.eval.stop_flag:
                chaos_val = np.random.rand(self.dim)
                for _ in range(5):
                    chaos_val = 4.0 * chaos_val * (1 - chaos_val)
                epsilon = 1e-3 * (1 - progress)
                x_chaos = best_x + (chaos_val - 0.5) * epsilon * (self.ub - self.lb)
                x_chaos = np.clip(x_chaos, self.lb, self.ub)
                f_chaos = self.eval(x_chaos)
                if f_chaos < best_fit:
                    best_x, best_fit = x_chaos, float(f_chaos)
                    worst_idx = int(np.argmax(fits))
                    pop[worst_idx] = best_x
                    fits[worst_idx] = best_fit
                    stagnation_counter = 0

            if self.enable_nelder_mead and not self.eval.stop_flag:
                is_stuck = stagnation_counter > 15 and stagnation_counter % 5 == 0
                is_final = progress > 0.95 and gen % 10 == 0
                if is_stuck or is_final:
                    rem_budget = self.eval.max_fes - self.eval.calls
                    nm_budget = min(500, rem_budget)
                    if nm_budget > self.dim * 2:
                        res = minimize(
                            self.eval,
                            best_x,
                            method="Nelder-Mead",
                            bounds=list(zip(self.lb, self.ub)),
                            options={"maxfev": nm_budget, "xatol": 1e-8},
                        )
                        if res.fun < best_fit:
                            best_x, best_fit = res.x, float(res.fun)
                            worst_idx = int(np.argmax(fits))
                            pop[worst_idx] = best_x
                            fits[worst_idx] = best_fit
                            stagnation_counter = 0

        return best_x, best_fit


