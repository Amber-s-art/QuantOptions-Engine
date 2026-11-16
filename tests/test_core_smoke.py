"""Small deterministic checks for the core package."""

import unittest

import numpy as np

from core.pricing import black_scholes_price, calculate_greeks
from core.implied_vol import implied_volatility
from core.monte_carlo import simulate_gbm_paths, price_european_option_mc


class CoreSmokeTests(unittest.TestCase):
    def test_hull_standard_atm_example(self):
        # Standard Black-Scholes benchmark: S=K=100, r=5%, sigma=20%, T=1.
        call = black_scholes_price(100, 100, 1, 0.05, 0.20, "call")
        put = black_scholes_price(100, 100, 1, 0.05, 0.20, "put")
        self.assertAlmostEqual(call, 10.4505835722, places=7)
        self.assertAlmostEqual(put, 5.5735260223, places=7)

    def test_put_call_parity(self):
        call = black_scholes_price(100, 105, 0.75, 0.04, 0.25, "call")
        put = black_scholes_price(100, 105, 0.75, 0.04, 0.25, "put")
        expected = 100 - 105 * np.exp(-0.04 * 0.75)
        self.assertAlmostEqual(call - put, expected, places=10)

    def test_greeks(self):
        values = calculate_greeks(100, 100, 1, 0.05, 0.20, "call")
        self.assertAlmostEqual(values["delta"], 0.6368306512, places=7)
        self.assertAlmostEqual(values["gamma"], 0.0187620173, places=7)
        self.assertGreater(values["vega"], 0)
        self.assertLess(values["theta"], 0)

        put_values = calculate_greeks(100, 100, 1, 0.05, 0.20, "put")
        self.assertLess(put_values["theta"], 0)

    def test_vectorized_pricing(self):
        prices = black_scholes_price(
            np.array([90, 100, 110]), 100, 1, 0.05, 0.20, "call"
        )
        self.assertEqual(prices.shape, (3,))

    def test_iv_round_trip(self):
        price = black_scholes_price(100, 100, 0.5, 0.03, 0.27, "call")
        iv = implied_volatility(price, 100, 100, 0.5, 0.03, "call")
        self.assertAlmostEqual(iv, 0.27, places=8)

    def test_mc_reproducible_and_shape(self):
        first = simulate_gbm_paths(100, 0.05, 0.2, 1, 10, 100, seed=42)
        second = simulate_gbm_paths(100, 0.05, 0.2, 1, 10, 100, seed=42)
        self.assertEqual(first.paths.shape, (100, 11))
        np.testing.assert_allclose(first.paths, second.paths)

    def test_antithetic_requires_even_path_count(self):
        with self.assertRaises(ValueError):
            simulate_gbm_paths(100, 0.05, 0.2, 1, 10, 99, seed=1, antithetic=True)

    def test_mc_converges_to_bs_with_reasonable_tolerance(self):
        estimate = price_european_option_mc(
            100, 100, 1, 0.05, 0.20, "call",
            num_paths=150_000, seed=123, antithetic=True,
        )
        exact = black_scholes_price(100, 100, 1, 0.05, 0.20, "call")
        self.assertLess(abs(estimate["price"] - exact), 0.12)


if __name__ == "__main__":
    unittest.main()
