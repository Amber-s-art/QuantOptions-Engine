"""Geometric Brownian Motion simulation and Monte Carlo option valuation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MonteCarloResult:
    """Simulation output plus the assumptions needed to reproduce it."""

    paths: np.ndarray
    terminal_prices: np.ndarray
    metadata: dict


def simulate_gbm_paths(
    spot: float,
    drift: float,
    volatility: float,
    time_horizon: float,
    num_steps: int,
    num_paths: int,
    *,
    seed: Optional[int] = None,
    antithetic: bool = False,
) -> MonteCarloResult:
    """Simulate GBM paths using exact lognormal time-step transitions.

    dS = drift * S * dt + volatility * S * dW

    ``drift`` is the physical/assumed drift for scenario simulation. For
    risk-neutral option valuation, use ``drift = risk_free_rate - dividend_yield``.
    Returns a (num_paths, num_steps + 1) array including the initial spot.
    """
    numeric = [spot, drift, volatility, time_horizon]
    if not all(np.isfinite(value) for value in numeric):
        raise ValueError("Simulation inputs must be finite.")
    if spot <= 0 or volatility < 0 or time_horizon <= 0:
        raise ValueError("spot and horizon must be positive; volatility non-negative.")
    if num_steps < 1 or num_paths < 1:
        raise ValueError("num_steps and num_paths must be positive integers.")
    if int(num_steps) != num_steps or int(num_paths) != num_paths:
        raise ValueError("num_steps and num_paths must be integers.")
    if antithetic and int(num_paths) % 2:
        raise ValueError("num_paths must be even when antithetic=True.")

    num_steps, num_paths = int(num_steps), int(num_paths)
    rng = np.random.default_rng(seed)
    dt = time_horizon / num_steps
    shocks_count = num_paths // 2 if antithetic else num_paths
    shocks = rng.standard_normal((shocks_count, num_steps))

    if antithetic:
        shocks = np.concatenate([shocks, -shocks], axis=0)

    increments = (
        (drift - 0.5 * volatility**2) * dt
        + volatility * np.sqrt(dt) * shocks
    )
    log_paths = np.concatenate(
        [
            np.zeros((num_paths, 1), dtype=float),
            np.cumsum(increments, axis=1),
        ],
        axis=1,
    )
    paths = spot * np.exp(log_paths)
    return MonteCarloResult(
        paths=paths,
        terminal_prices=paths[:, -1].copy(),
        metadata={
            "model": "geometric_brownian_motion",
            "spot": float(spot),
            "drift": float(drift),
            "volatility": float(volatility),
            "time_horizon": float(time_horizon),
            "num_steps": num_steps,
            "num_paths": num_paths,
            "seed": seed,
            "antithetic": bool(antithetic),
            "time_step_years": dt,
        },
    )


def price_european_option_mc(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str,
    *,
    dividend_yield: float = 0.0,
    num_paths: int = 100_000,
    num_steps: int = 1,
    seed: Optional[int] = None,
    antithetic: bool = True,
) -> dict[str, float | int | str]:
    """Estimate a European option value under the risk-neutral GBM measure.

    The payoff is discounted at the continuously compounded risk-free rate.
    With one time step, terminal prices are sufficient for a vanilla European
    payoff; more steps are accepted for consistency with the path simulator.
    Standard error is estimated from the discounted path payoffs.
    """
    kind = option_type.lower() if isinstance(option_type, str) else ""
    if kind not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")
    if strike <= 0 or time_to_expiry <= 0:
        raise ValueError("strike and time_to_expiry must be positive.")
    if not np.isfinite(risk_free_rate) or not np.isfinite(dividend_yield):
        raise ValueError("Rates must be finite.")

    simulation = simulate_gbm_paths(
        spot=spot,
        drift=risk_free_rate - dividend_yield,
        volatility=volatility,
        time_horizon=time_to_expiry,
        num_steps=num_steps,
        num_paths=num_paths,
        seed=seed,
        antithetic=antithetic,
    )
    terminal = simulation.terminal_prices
    if kind == "call":
        payoffs = np.maximum(terminal - strike, 0.0)
    else:
        payoffs = np.maximum(strike - terminal, 0.0)

    discounted_payoffs = np.exp(-risk_free_rate * time_to_expiry) * payoffs
    estimate = float(discounted_payoffs.mean())
    if antithetic:
        # Each +Z/-Z pair is one independent Monte Carlo observation.
        pair_count = discounted_payoffs.size // 2
        pair_payoffs = (
            discounted_payoffs[:pair_count]
            + discounted_payoffs[pair_count:]
        ) / 2.0
        standard_error = (
            float(pair_payoffs.std(ddof=1) / np.sqrt(pair_count))
            if pair_count > 1 else 0.0
        )
    else:
        standard_error = (
            float(discounted_payoffs.std(ddof=1) / np.sqrt(discounted_payoffs.size))
            if discounted_payoffs.size > 1 else 0.0
        )
    return {
        "price": estimate,
        "standard_error": standard_error,
        "confidence_interval_95_low": estimate - 1.96 * standard_error,
        "confidence_interval_95_high": estimate + 1.96 * standard_error,
        "num_paths": int(discounted_payoffs.size),
        "option_type": kind,
    }
