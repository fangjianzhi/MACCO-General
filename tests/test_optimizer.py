import unittest
import numpy as np

from macco import minimize,minimize_delayed_hybrid,minimize_hybrid,minimize_low_rank
from baselines import cbo, de, pso
from strong_baselines import cma_es, lshade
from classic23_benchmarks import make_classic23_suite


def sphere(x):
    return float(np.sum(x ** 2))


class TestMACCO(unittest.TestCase):
    def test_reproducible_exact_budget_and_monotone(self):
        kw = dict(dim=8, lb=-5, ub=5, population_size=16,
                  max_evaluations=1000, seed=7)
        a, b = minimize(sphere, **kw), minimize(sphere, **kw)
        self.assertEqual(a.best_f, b.best_f)
        np.testing.assert_array_equal(a.best_x, b.best_x)
        self.assertEqual(a.evaluations, 1000)
        self.assertTrue(np.all(np.diff(a.history) <= 1e-15))

    def test_vector_bounds(self):
        result = minimize(sphere, 3, [-1, -2, -3], [1, 2, 3],
                          population_size=10, max_evaluations=500, seed=2)
        self.assertTrue(np.all(result.best_x >= [-1, -2, -3]))
        self.assertTrue(np.all(result.best_x <= [1, 2, 3]))

    def test_equal_budget_baselines(self):
        for optimizer in (de, pso, cbo):
            result=optimizer(sphere,5,-5,5,population_size=10,max_evaluations=300,seed=3)
            self.assertEqual(result.evaluations,300)
            self.assertTrue(np.isfinite(result.best_f))

    def test_strong_baselines_exact_budget(self):
        for optimizer in (cma_es,lshade):
            result=optimizer(sphere,5,-5,5,population_size=10,max_evaluations=300,seed=4)
            self.assertEqual(result.evaluations,300)
            self.assertTrue(np.isfinite(result.best_f))

    def test_low_rank_exact_budget_and_reproducible(self):
        kw=dict(dim=8,lb=-5,ub=5,population_size=16,max_evaluations=1000,seed=9,rank=3)
        a,b=minimize_low_rank(sphere,**kw),minimize_low_rank(sphere,**kw)
        self.assertEqual(a.evaluations,1000); self.assertEqual(a.best_f,b.best_f)
        self.assertTrue(np.all(np.diff(a.history)<=1e-15))

    def test_hybrid_exact_budget_and_reproducible(self):
        kw=dict(dim=8,lb=-5,ub=5,population_size=16,
                max_evaluations=1000,seed=11,rank=3)
        a,b=minimize_hybrid(sphere,**kw),minimize_hybrid(sphere,**kw)
        self.assertEqual(a.evaluations,1000); self.assertEqual(a.best_f,b.best_f)
        np.testing.assert_array_equal(a.best_x,b.best_x)
        self.assertTrue(np.all(np.diff(a.history)<=1e-15))

    def test_delayed_hybrid_exact_budget_and_reproducible(self):
        kw=dict(dim=8,lb=-5,ub=5,population_size=16,
                max_evaluations=1000,seed=13,rank=3,credit_window=3)
        a,b=minimize_delayed_hybrid(sphere,**kw),minimize_delayed_hybrid(sphere,**kw)
        self.assertEqual(a.evaluations,1000); self.assertEqual(a.best_f,b.best_f)
        np.testing.assert_array_equal(a.best_x,b.best_x)
        self.assertTrue(np.all(np.diff(a.history)<=1e-15))

    def test_classic23_shapes_bounds_and_known_points(self):
        suite=make_classic23_suite()
        self.assertEqual(len(suite),23)
        for _,(_,lb,ub,dim) in suite.items():
            self.assertEqual(np.broadcast_to(lb,(dim,)).shape,(dim,))
            self.assertTrue(np.all(np.broadcast_to(ub,(dim,))>np.broadcast_to(lb,(dim,))))
        self.assertAlmostEqual(suite["F1"][0](np.zeros(30)),0.)
        self.assertAlmostEqual(suite["F5"][0](np.ones(30)),0.)
        self.assertAlmostEqual(suite["F18"][0](np.array([0.,-1.])),3.)


if __name__ == "__main__":
    unittest.main()
