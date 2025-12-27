import random
import unittest

import numpy as np

from src.evaluator import CEC_Evaluator
from src.iwoa import IWOA_Strict
from src.woa import WOA


def sphere(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(x * x))


class TestBudget(unittest.TestCase):
    def test_iwoa_budget_never_exceeds(self) -> None:
        random.seed(123)
        np.random.seed(123)

        dim = 3
        max_fes = 200
        lb = np.full(dim, -5.0)
        ub = np.full(dim, 5.0)
        bounds = np.array([lb, ub])

        evaluator = CEC_Evaluator(sphere, max_fes, lb, ub)
        opt = IWOA_Strict(
            evaluator=evaluator,
            dim=dim,
            bounds=bounds,
            pop_size=12,
            min_pop_size=4,
            enable_nelder_mead=False,  # keep tests fast + deterministic
        )

        _x, _f = opt.run()
        self.assertLessEqual(evaluator.calls, max_fes)
        self.assertTrue(evaluator.stop_flag or evaluator.calls == max_fes)

    def test_woa_budget_never_exceeds(self) -> None:
        random.seed(123)
        np.random.seed(123)

        dim = 3
        max_fes = 150
        lb = np.full(dim, -5.0)
        ub = np.full(dim, 5.0)
        bounds = np.array([lb, ub])

        evaluator = CEC_Evaluator(sphere, max_fes, lb, ub)
        opt = WOA(
            evaluator=evaluator,
            dim=dim,
            bounds=bounds,
            pop_size=10,
        )

        _x, _f = opt.run()
        self.assertLessEqual(evaluator.calls, max_fes)
        self.assertTrue(evaluator.stop_flag or evaluator.calls == max_fes)


if __name__ == "__main__":
    unittest.main()


