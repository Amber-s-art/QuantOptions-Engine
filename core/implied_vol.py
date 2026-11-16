"""Implied-volatility inversion using no-arbitrage bounds and Brent's method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .pricing import black_scholes_price


class ImpliedVolatilityError(ValueError):
    """Raised when an implied volatility cannot be determined reliably."""


@dataclass(frozen=True)
class IVResult:
    volatility: Optional[float]
    status: str
    iterations: int
    model_price: Optional[float]
    price_error: Optional[float]
    message: str = ""


def _price_bounds(spot, strike, time_to_expiry, rate, option_type, dividend_yield):
    """European option lower/upper bounds with continuous dividend yield."""
    discounted_spot = spot * np.exp(-dividend_yield * time_to_expiry)
    discounted_strike = strike * np.exp(-rate * time_to_expiry)
    if option_type == "call":
        return max(discounted_spot - discounted_strike, 0.0), discounted_spot
    return max(discounted_strike - discounted_spot, 0.0), discounted_strike


def solve_implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    option_type: str,
    *,
    dividend_yield: float = 0.0,
    lower_volatility: float = 1e-6,
    upper_volatility: float = 5.0,
    price_tolerance: float = 1e-8,
    max_iterations: int = 200,
) -> IVResult:
    """Solve for volatility consistent with a European option market price.

    Brent's bracketed method is used because it is more robust than a
    stand-alone Newton iteration when Vega is small. The solver reports a
    status and never returns a fabricated volatility for an invalid price.
    """
    kind = option_type.lower() if isinstance(option_type, str) else ""
    if kind not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    inputs = [market_price, spot, strike, time_to_expiry, rate, dividend_yield]
    if not all(np.isfinite(value) for value in inputs):
        raise ValueError("All implied-volatility inputs must be finite.")
    if market_price < 0 or spot <= 0 or strike <= 0 or time_to_expiry <= 0:
        raise ValueError("Price must be non-negative; spot, strike and time must be positive.")
    if lower_volatility <= 0 or upper_volatility <= lower_volatility:
        raise ValueError("Invalid volatility search interval.")
    if price_tolerance <= 0 or max_iterations < 1:
        raise ValueError("Invalid solver tolerance or iteration limit.")

    lower_price, upper_price = _price_bounds(
        spot, strike, time_to_expiry, rate, kind, dividend_yield
    )
    if market_price < lower_price - price_tolerance:
        raise ImpliedVolatilityError(
            f"Market price {market_price:.8g} is below the no-arbitrage lower "
            f"bound {lower_price:.8g}."
        )
    if market_price >= upper_price - price_tolerance:
        raise ImpliedVolatilityError(
            "Market price is at or too close to the theoretical upper bound; "
            "a finite implied volatility is not identifiable."
        )
    if abs(market_price - lower_price) <= price_tolerance:
        return IVResult(
            volatility=0.0,
            status="at_lower_bound",
            iterations=0,
            model_price=lower_price,
            price_error=lower_price - market_price,
            message="Market price is at the theoretical lower bound.",
        )

    def objective(volatility):
        return black_scholes_price(
            spot, strike, time_to_expiry, rate, volatility, kind, dividend_yield
        ) - market_price

    low_value = objective(lower_volatility)
    high_value = objective(upper_volatility)
    if low_value > 0 or high_value < 0:
        raise ImpliedVolatilityError(
            "Could not bracket an implied-volatility root within the configured "
            "volatility interval."
        )

    try:
        root, result = brentq(
            objective,
            lower_volatility,
            upper_volatility,
            xtol=1e-12,
            rtol=1e-12,
            maxiter=max_iterations,
            full_output=True,
            disp=True,
        )
    except (ValueError, RuntimeError) as exc:
        raise ImpliedVolatilityError(
            f"Implied-volatility solver did not converge: {exc}"
        ) from exc

    model_price = float(
        black_scholes_price(
            spot, strike, time_to_expiry, rate, root, kind, dividend_yield
        )
    )
    return IVResult(
        volatility=float(root),
        status="converged" if result.converged else "failed",
        iterations=int(result.iterations),
        model_price=model_price,
        price_error=model_price - market_price,
        message="Brent solver converged." if result.converged else "Solver did not converge.",
    )


def implied_volatility(*args, **kwargs) -> float:
    """Convenience wrapper returning only the volatility estimate."""
    result = solve_implied_volatility(*args, **kwargs)
    if result.volatility is None:
        raise ImpliedVolatilityError(result.message or result.status)
    return float(result.volatility)


def implied_volatility_chain(
    options: pd.DataFrame,
    *,
    spot: float,
    rate: float,
    price_column: str = "market_mid",
    dividend_yield: float = 0.0,
) -> pd.DataFrame:
    """Calculate IV for each option row and retain per-contract status.

    Required columns: strike, time_to_expiry, option_type and ``price_column``.
    If ``market_mid`` is absent but bid and ask exist, a valid midpoint is
    calculated. Invalid rows remain in the result with an explanatory status.
    """
    required = {"strike", "time_to_expiry", "option_type"}
    missing = required.difference(options.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = options.copy()
    if price_column not in result.columns:
        if {"bid", "ask"}.issubset(result.columns) and price_column == "market_mid":
            bid = pd.to_numeric(result["bid"], errors="coerce")
            ask = pd.to_numeric(result["ask"], errors="coerce")
            valid = bid.gt(0) & ask.gt(0) & ask.ge(bid)
            result[price_column] = ((bid + ask) / 2).where(valid)
        else:
            raise ValueError(f"Missing market price column {price_column!r}.")

    estimates = []
    statuses = []
    errors = []
    iterations = []

    for row in result.itertuples(index=False):
        values = row._asdict()
        market_price = values.get(price_column)
        try:
            if market_price is None or not np.isfinite(float(market_price)):
                raise ImpliedVolatilityError("Market price is missing or invalid.")
            solved = solve_implied_volatility(
                market_price=float(market_price),
                spot=spot,
                strike=float(values["strike"]),
                time_to_expiry=float(values["time_to_expiry"]),
                rate=rate,
                option_type=str(values["option_type"]),
                dividend_yield=dividend_yield,
            )
            estimates.append(solved.volatility)
            statuses.append(solved.status)
            errors.append(solved.price_error)
            iterations.append(solved.iterations)
        except (ValueError, TypeError, KeyError) as exc:
            estimates.append(np.nan)
            statuses.append("failed")
            errors.append(np.nan)
            iterations.append(0)

    result["implied_volatility"] = estimates
    result["iv_status"] = statuses
    result["iv_price_error"] = errors
    result["iv_iterations"] = iterations
    return result
