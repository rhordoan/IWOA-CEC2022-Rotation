import random
import unittest

import numpy as np

from src.evaluator import CEC_Evaluator
from src.iwoa import IWOA_Strict


def sphere(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(x * x))


class TestReproducibility(unittest.TestCase):
    def test_iwoa_same_seed_same_result(self) -> None:
        dim = 4
        max_fes = 250
        lb = np.full(dim, -5.0)
        ub = np.full(dim, 5.0)
        bounds = np.array([lb, ub])

        def run_once(seed: int):
            random.seed(seed)
            np.random.seed(seed)
            evaluator = CEC_Evaluator(sphere, max_fes, lb, ub)
            opt = IWOA_Strict(
                evaluator=evaluator,
                dim=dim,
                bounds=bounds,
                pop_size=14,
                min_pop_size=6,
                enable_nelder_mead=False,
            )
            x, f = opt.run()
            return x, f, evaluator.calls

        x1, f1, c1 = run_once(999)
        x2, f2, c2 = run_once(999)

        self.assertEqual(c1, c2)
        self.assertTrue(np.allclose(x1, x2))
        self.assertAlmostEqual(float(f1), float(f2), places=15)


if __name__ == "__main__":
    unittest.main()



