"""Black-Scholes-Merton pricing and analytical Greeks for European options.

Conventions:
- Rates, dividend yield and volatility are annualized decimals.
- Time to expiry is measured in years.
- Theta is the derivative with respect to calendar time (per year).
- Vega and Rho are sensitivities to a 1.00 change in volatility/rate.
- The model supports a continuous dividend yield q.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _validate_inputs(spot, strike, time_to_expiry, rate, volatility, dividend_yield):
    values = np.broadcast_arrays(
        np.asarray(spot, dtype=float),
        np.asarray(strike, dtype=float),
        np.asarray(time_to_expiry, dtype=float),
        np.asarray(rate, dtype=float),
        np.asarray(volatility, dtype=float),
        np.asarray(dividend_yield, dtype=float),
    )
    s, k, t, r, sigma, q = values

    if not all(np.all(np.isfinite(value)) for value in values):
        raise ValueError("All pricing inputs must be finite.")
    if np.any(s <= 0) or np.any(k <= 0):
        raise ValueError("spot and strike must be positive.")
    if np.any(t < 0):
        raise ValueError("time_to_expiry cannot be negative.")
    if np.any(sigma < 0):
        raise ValueError("volatility cannot be negative.")
    return s, k, t, r, sigma, q


def _return_scalar_if_scalar(value):
    array = np.asarray(value)
    return float(array) if array.ndim == 0 else array


def _validate_option_type(option_type: str) -> str:
    if not isinstance(option_type, str):
        raise TypeError("option_type must be 'call' or 'put'.")
    normalized = option_type.lower()
    if normalized not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")
    return normalized


def black_scholes_price(
    spot,
    strike,
    time_to_expiry,
    rate,
    volatility,
    option_type: str,
    dividend_yield=0.0,
):
    """Calculate a European call or put price using Black-Scholes-Merton.

    Inputs may be scalars or broadcast-compatible NumPy arrays. A zero-time
    option is valued at intrinsic value. Zero volatility uses the deterministic
    discounted-forward limit rather than dividing by zero.
    """
    kind = _validate_option_type(option_type)
    s, k, t, r, sigma, q = _validate_inputs(
        spot, strike, time_to_expiry, rate, volatility, dividend_yield
    )

    discount_s = s * np.exp(-q * t)
    discount_k = k * np.exp(-r * t)

    deterministic_call = np.maximum(discount_s - discount_k, 0.0)
    deterministic_put = np.maximum(discount_k - discount_s, 0.0)

    regular = (t > 0) & (sigma > 0)
    safe_t = np.where(regular, t, 1.0)
    safe_sigma = np.where(regular, sigma, 1.0)
    sqrt_t = np.sqrt(safe_t)

    d1 = (
        np.log(s / k) + (r - q + 0.5 * safe_sigma**2) * safe_t
    ) / (safe_sigma * sqrt_t)
    d2 = d1 - safe_sigma * sqrt_t

    if kind == "call":
        regular_price = discount_s * norm.cdf(d1) - discount_k * norm.cdf(d2)
        price = np.where(regular, regular_price, deterministic_call)
        price = np.where(t == 0, np.maximum(s - k, 0.0), price)
    else:
        regular_price = discount_k * norm.cdf(-d2) - discount_s * norm.cdf(-d1)
        price = np.where(regular, regular_price, deterministic_put)
        price = np.where(t == 0, np.maximum(k - s, 0.0), price)

    return _return_scalar_if_scalar(np.maximum(price, 0.0))


def calculate_greeks(
    spot,
    strike,
    time_to_expiry,
    rate,
    volatility,
    option_type: str,
    dividend_yield=0.0,
) -> dict[str, object]:
    """Return analytical Delta, Gamma, Theta, Vega and Rho.

    Theta is annualized, Vega is per 1.00 volatility unit, and Rho is per
    1.00 rate unit. Multiply Theta by 1/365 for an approximate calendar-day
    theta, or Vega/Rho by 0.01 for a one-percentage-point sensitivity.

    At expiry or exactly zero volatility, singular Greeks are set to zero;
    Delta uses a simple intrinsic-value convention at expiry.
    """
    kind = _validate_option_type(option_type)
    s, k, t, r, sigma, q = _validate_inputs(
        spot, strike, time_to_expiry, rate, volatility, dividend_yield
    )

    regular = (t > 0) & (sigma > 0)
    safe_t = np.where(regular, t, 1.0)
    safe_sigma = np.where(regular, sigma, 1.0)
    sqrt_t = np.sqrt(safe_t)

    d1 = (
        np.log(s / k) + (r - q + 0.5 * safe_sigma**2) * safe_t
    ) / (safe_sigma * sqrt_t)
    d2 = d1 - safe_sigma * sqrt_t
    discounted_spot = np.exp(-q * t)
    discounted_strike = np.exp(-r * t)
    pdf_d1 = norm.pdf(d1)

    gamma_regular = discounted_spot * pdf_d1 / (s * safe_sigma * sqrt_t)
    vega_regular = s * discounted_spot * pdf_d1 * sqrt_t

    if kind == "call":
        delta_regular = discounted_spot * norm.cdf(d1)
        theta_regular = (
            -s * discounted_spot * pdf_d1 * safe_sigma / (2.0 * sqrt_t)
            - r * k * discounted_strike * norm.cdf(d2)
            + q * s * discounted_spot * norm.cdf(d1)
        )
        rho_regular = k * t * discounted_strike * norm.cdf(d2)
        delta_expiry = np.where(s > k, 1.0, np.where(s < k, 0.0, 0.5))
    else:
        delta_regular = discounted_spot * (norm.cdf(d1) - 1.0)
        theta_regular = (
            -s * discounted_spot * pdf_d1 * safe_sigma / (2.0 * sqrt_t)
            + q * s * discounted_spot * norm.cdf(-d1)
            - r * k * discounted_strike * norm.cdf(-d2)
        )
        rho_regular = -k * t * discounted_strike * norm.cdf(-d2)
        delta_expiry = np.where(s < k, -1.0, np.where(s > k, 0.0, -0.5))

    delta = np.where(regular, delta_regular, np.where(t == 0, delta_expiry, 0.0))
    gamma = np.where(regular, gamma_regular, 0.0)
    theta = np.where(regular, theta_regular, 0.0)
    vega = np.where(regular, vega_regular, 0.0)
    rho = np.where(regular, rho_regular, 0.0)

    return {
        "delta": _return_scalar_if_scalar(delta),
        "gamma": _return_scalar_if_scalar(gamma),
        "theta": _return_scalar_if_scalar(theta),
        "vega": _return_scalar_if_scalar(vega),
        "rho": _return_scalar_if_scalar(rho),
    }
