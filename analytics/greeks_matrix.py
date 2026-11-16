"""Scenario grids for Black-Scholes option Greeks."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd


GreeksFunction = Callable[..., dict]


def _default_greeks_function() -> GreeksFunction:
    """Load the shared core Greeks function lazily.

    Expected core interface:
        calculate_greeks(spot, strike, time_to_expiry, rate, volatility,
                         option_type, dividend_yield=0.0)

    It should return a mapping with delta, gamma, theta, vega and rho.
    Theta and Vega units must match core/pricing.py's documented convention.
    """
    try:
        from core.pricing import calculate_greeks
    except ImportError as exc:
        raise ImportError(
            "Provide greeks_function=... or expose "
            "core.pricing.calculate_greeks with the documented interface."
        ) from exc
    return calculate_greeks


def build_greeks_matrix(
    *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: str,
    greeks_function: Optional[GreeksFunction] = None,
    dividend_yield: float = 0.0,
    spot_change: float = 0.15,
    spot_steps: int = 31,
    time_steps: int = 25,
) -> dict[str, pd.DataFrame]:
    """Build Delta/Gamma/Theta/Vega grids over spot and remaining time.

    The spot axis spans ``spot * (1 ± spot_change)``. The time axis runs from
    current maturity down to expiry. The zero-time row is retained: the
    supplied core function must define its expiry convention.

    Returns a dictionary of DataFrames indexed by time-to-expiry and with
    spot prices as columns. No plotting is performed here.
    """
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive.")
    if time_to_expiry < 0:
        raise ValueError("time_to_expiry cannot be negative.")
    if volatility <= 0:
        raise ValueError("volatility must be positive.")
    if spot_change < 0 or spot_change >= 1:
        raise ValueError("spot_change must be in [0, 1).")
    if spot_steps < 2 or time_steps < 2:
        raise ValueError("spot_steps and time_steps must both be at least 2.")
    option_type = option_type.lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    greeks = greeks_function or _default_greeks_function()
    spots = np.linspace(
        spot * (1.0 - spot_change), spot * (1.0 + spot_change), spot_steps
    )
    maturities = np.linspace(time_to_expiry, 0.0, time_steps)
    greek_names = ("delta", "gamma", "theta", "vega")
    grids = {
        name: np.full((time_steps, spot_steps), np.nan, dtype=float)
        for name in greek_names
    }

    for i, maturity in enumerate(maturities):
        for j, scenario_spot in enumerate(spots):
            try:
                values = greeks(
                    spot=float(scenario_spot),
                    strike=float(strike),
                    time_to_expiry=float(maturity),
                    rate=float(rate),
                    volatility=float(volatility),
                    option_type=option_type,
                    dividend_yield=float(dividend_yield),
                )
                for name in greek_names:
                    grids[name][i, j] = float(values[name])
            except (KeyError, TypeError, ValueError, ArithmeticError):
                # Preserve an explicit NaN if a core model cannot value a point.
                continue

    columns = pd.Index(spots, name="spot")
    index = pd.Index(maturities, name="time_to_expiry")
    return {
        name: pd.DataFrame(values, index=index, columns=columns)
        for name, values in grids.items()
    }
