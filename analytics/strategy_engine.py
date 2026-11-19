"""Consolidated multi-leg option strategy analytics and Hull strategy builders (European BSM baseline)."""
from __future__ import annotations

from typing import Callable
import numpy as np
import pandas as pd
from scipy.stats import norm

from core.pricing import black_scholes_price, calculate_greeks


REQUIRED_COLUMNS = {
    "option_type",
    "strike",
    "time_to_expiry",
    "volatility",
    "quantity",
    "entry_price",
}
VALID_OPTION_TYPES = {"call", "put"}


def _validate_legs(legs: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(legs.columns)
    if missing:
        raise ValueError(f"Missing strategy columns: {sorted(missing)}")
    if legs.empty:
        raise ValueError("Add at least one option leg.")

    clean = legs.copy()
    clean["option_type"] = clean["option_type"].astype(str).str.strip().str.lower()
    invalid_types = set(clean["option_type"]) - VALID_OPTION_TYPES
    if invalid_types:
        raise ValueError(f"Invalid option_type values: {sorted(invalid_types)}. Expected 'call' or 'put'.")

    numeric_cols = ["strike", "time_to_expiry", "volatility", "quantity", "entry_price"]
    for col in numeric_cols:
        clean[col] = pd.to_numeric(clean[col], errors="raise")

    if (clean["strike"] <= 0).any() or (clean["time_to_expiry"] <= 0).any() or (clean["volatility"] <= 0).any():
        raise ValueError("Strike, time_to_expiry, and volatility must be strictly positive.")

    return clean


def _intrinsic_value(spot_val: float | np.ndarray, strike: float, option_type: str) -> float | np.ndarray:
    if option_type == "call":
        return np.maximum(spot_val - strike, 0.0)
    return np.maximum(strike - spot_val, 0.0)


def _default_price(spot: float, strike: float, tte: float, rate: float, vol: float,
                   option_type: str, dividend_yield: float, entry_price: float | None) -> float:
    if entry_price is not None:
        return float(entry_price)
    return float(black_scholes_price(spot, strike, tte, rate, vol, option_type, dividend_yield))


def analyze_strategy(
    legs: pd.DataFrame,
    *,
    spot: float,
    rate: float,
    dividend_yield: float = 0.0,
    spot_grid: np.ndarray | None = None,
    iv_shift: float = 0.0,
    days_forward: int = 0,
    multiplier: int = 1,
) -> dict:
    """Aggregate signed leg value/Greeks, expiry P&L, and forward scenario P&L."""
    if spot <= 0:
        raise ValueError("Spot price must be strictly positive.")
    if multiplier <= 0:
        raise ValueError("Contract multiplier must be strictly positive.")

    clean = _validate_legs(legs)
    if spot_grid is None:
        spot_grid = np.linspace(spot * 0.7, spot * 1.3, 121)
    else:
        spot_grid = np.asarray(spot_grid, dtype=float)

    total_value = total_delta = total_gamma = total_theta = total_vega = total_rho = 0.0
    entry_cash = 0.0
    leg_rows = []

    for leg in clean.itertuples(index=False):
        kind = leg.option_type
        qty = float(leg.quantity)
        k = float(leg.strike)
        t = float(leg.time_to_expiry)
        vol = float(leg.volatility)
        entry = float(leg.entry_price)

        mark = float(black_scholes_price(spot, k, t, rate, vol, kind, dividend_yield))
        g = calculate_greeks(spot, k, t, rate, vol, kind, dividend_yield)

        total_value += qty * mark * multiplier
        entry_cash += qty * entry * multiplier
        total_delta += qty * float(g["delta"]) * multiplier
        total_gamma += qty * float(g["gamma"]) * multiplier
        total_theta += qty * float(g["theta"]) * multiplier
        total_vega += qty * float(g["vega"]) * multiplier
        total_rho += qty * float(g["rho"]) * multiplier

        leg_rows.append({
            "option_type": kind,
            "strike": k,
            "time_to_expiry": t,
            "volatility": vol,
            "quantity": qty,
            "entry_price": entry,
            "model_mark": mark,
            "delta": float(g["delta"]),
            "gamma": float(g["gamma"]),
            "theta_per_year": float(g["theta"]),
            "vega_per_1pct": float(g["vega"]) * 0.01,
        })

    # For multi-expiry legs (e.g., calendar/diagonal), evaluate at the earliest expiration.
    min_expiry = float(clean["time_to_expiry"].min())
    expiry_pnl = np.zeros_like(spot_grid, dtype=float)
    for leg in clean.itertuples(index=False):
        rem_t = max(float(leg.time_to_expiry) - min_expiry, 0.0)
        if rem_t == 0.0:
            end_val = _intrinsic_value(spot_grid, float(leg.strike), leg.option_type)
        else:
            end_val = np.array([
                float(black_scholes_price(float(s), float(leg.strike), rem_t, rate, float(leg.volatility), leg.option_type, dividend_yield))
                for s in spot_grid
            ])
        expiry_pnl += float(leg.quantity) * (end_val - float(leg.entry_price)) * multiplier

    horizon = max(0, int(days_forward)) / 365.0
    scenario_rows = []
    scenario_spots = spot * np.array([0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15])
    for s_scn in scenario_spots:
        pnl = 0.0
        for leg in clean.itertuples(index=False):
            t2 = max(float(leg.time_to_expiry) - horizon, 0.0)
            vol2 = max(float(leg.volatility) + iv_shift, 1e-4)
            if t2 == 0.0:
                mark2 = float(_intrinsic_value(float(s_scn), float(leg.strike), leg.option_type))
            else:
                mark2 = float(black_scholes_price(float(s_scn), float(leg.strike), t2, rate, vol2, leg.option_type, dividend_yield))
            pnl += float(leg.quantity) * (mark2 - float(leg.entry_price)) * multiplier
        scenario_rows.append({
            "spot": float(s_scn),
            "spot_change_pct": float((s_scn / spot - 1.0) * 100.0),
            "pnl": float(pnl),
        })

    return {
        "legs": pd.DataFrame(leg_rows),
        "spot_grid": spot_grid,
        "expiry_pnl": expiry_pnl,
        "scenario": pd.DataFrame(scenario_rows),
        "net_model_value": float(total_value),
        "net_entry_value": float(entry_cash),
        "net_delta": float(total_delta),
        "net_gamma": float(total_gamma),
        "net_theta": float(total_theta),
        "net_vega": float(total_vega),
        "net_rho": float(total_rho),
        "delta_hedge_shares": float(-total_delta),
    }


def summarize_strategy_payoff(
    analysis: dict,
    *,
    spot: float,
    rate: float,
    dividend_yield: float = 0.0,
    time_to_expiry: float | None = None,
    volatility: float | None = None,
) -> dict:
    """Extract breakevens, max profit/loss, and lognormal risk-neutral expected payoff."""
    grid = np.asarray(analysis["spot_grid"], dtype=float)
    pnl = np.asarray(analysis["expiry_pnl"], dtype=float)

    breakevens: list[float] = []
    for i in range(len(grid) - 1):
        y1, y2 = pnl[i], pnl[i + 1]
        if y1 == 0.0:
            breakevens.append(float(grid[i]))
        elif y1 * y2 < 0.0:
            x_be = grid[i] - y1 * (grid[i + 1] - grid[i]) / (y2 - y1)
            breakevens.append(float(x_be))

    # Detect whether payoff trends monotonically at boundaries (potentially unbounded)
    left_slope = pnl[1] - pnl[0]
    right_slope = pnl[-1] - pnl[-2]
    max_profit = float(np.max(pnl)) if right_slope <= 0 and left_slope >= 0 else (
        float("inf") if (right_slope > 1e-6 or left_slope < -1e-6) else float(np.max(pnl))
    )
    max_loss = float(np.min(pnl)) if right_slope >= 0 and left_slope <= 0 else (
        float("-inf") if (right_slope < -1e-6 or left_slope > 1e-6) else float(np.min(pnl))
    )

    expected_payoff = None
    if time_to_expiry is not None and volatility is not None and time_to_expiry > 0 and volatility > 0:
        mu = np.log(spot) + (rate - dividend_yield - 0.5 * volatility**2) * time_to_expiry
        sigma = volatility * np.sqrt(time_to_expiry)
        log_probs = norm.pdf(np.log(grid), loc=mu, scale=sigma) / grid
        weights = log_probs / np.sum(log_probs)
        expected_payoff = float(np.sum(weights * pnl))

    return {
        "breakeven_spots": breakevens,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "expected_payoff": expected_payoff,
        "net_debit_or_credit": analysis["net_entry_value"],
        "delta_hedge_shares": analysis["delta_hedge_shares"],
    }


# --- Hull Option Strategy Builders (European BSM Baseline) ---

def _legs_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(REQUIRED_COLUMNS))


def build_bull_call_spread(spot: float, k_low: float, k_high: float, tte: float, vol: float,
                           rate: float, dividend_yield: float = 0.0, qty: float = 1.0,
                           p_low: float | None = None, p_high: float | None = None) -> pd.DataFrame:
    return _legs_df([
        {"option_type": "call", "strike": k_low, "time_to_expiry": tte, "volatility": vol,
         "quantity": qty, "entry_price": _default_price(spot, k_low, tte, rate, vol, "call", dividend_yield, p_low)},
        {"option_type": "call", "strike": k_high, "time_to_expiry": tte, "volatility": vol,
         "quantity": -qty, "entry_price": _default_price(spot, k_high, tte, rate, vol, "call", dividend_yield, p_high)},
    ])


def build_bear_put_spread(spot: float, k_low: float, k_high: float, tte: float, vol: float,
                          rate: float, dividend_yield: float = 0.0, qty: float = 1.0,
                          p_low: float | None = None, p_high: float | None = None) -> pd.DataFrame:
    return _legs_df([
        {"option_type": "put", "strike": k_high, "time_to_expiry": tte, "volatility": vol,
         "quantity": qty, "entry_price": _default_price(spot, k_high, tte, rate, vol, "put", dividend_yield, p_high)},
        {"option_type": "put", "strike": k_low, "time_to_expiry": tte, "volatility": vol,
         "quantity": -qty, "entry_price": _default_price(spot, k_low, tte, rate, vol, "put", dividend_yield, p_low)},
    ])


def build_bull_put_spread(spot: float, k_low: float, k_high: float, tte: float, vol: float,
                          rate: float, dividend_yield: float = 0.0, qty: float = 1.0,
                          p_low: float | None = None, p_high: float | None = None) -> pd.DataFrame:
    return _legs_df([
        {"option_type": "put", "strike": k_high, "time_to_expiry": tte, "volatility": vol,
         "quantity": -qty, "entry_price": _default_price(spot, k_high, tte, rate, vol, "put", dividend_yield, p_high)},
        {"option_type": "put", "strike": k_low, "time_to_expiry": tte, "volatility": vol,
         "quantity": qty, "entry_price": _default_price(spot, k_low, tte, rate, vol, "put", dividend_yield, p_low)},
    ])


def build_bear_call_spread(spot: float, k_low: float, k_high: float, tte: float, vol: float,
                           rate: float, dividend_yield: float = 0.0, qty: float = 1.0,
                           p_low: float | None = None, p_high: float | None = None) -> pd.DataFrame:
    return _legs_df([
        {"option_type": "call", "strike": k_low, "time_to_expiry": tte, "volatility": vol,
         "quantity": -qty, "entry_price": _default_price(spot, k_low, tte, rate, vol, "call", dividend_yield, p_low)},
        {"option_type": "call", "strike": k_high, "time_to_expiry": tte, "volatility": vol,
         "quantity": qty, "entry_price": _default_price(spot, k_high, tte, rate, vol, "call", dividend_yield, p_high)},
    ])


def build_butterfly_call_spread(spot: float, k_low: float, k_mid: float, k_high: float, tte: float,
                                vol: float, rate: float, dividend_yield: float = 0.0,
                                qty: float = 1.0) -> pd.DataFrame:
    return _legs_df([
        {"option_type": "call", "strike": k_low, "time_to_expiry": tte, "volatility": vol,
         "quantity": qty, "entry_price": _default_price(spot, k_low, tte, rate, vol, "call", dividend_yield, None)},
        {"option_type": "call", "strike": k_mid, "time_to_expiry": tte, "volatility": vol,
         "quantity": -2.0 * qty, "entry_price": _default_price(spot, k_mid, tte, rate, vol, "call", dividend_yield, None)},
        {"option_type": "call", "strike": k_high, "time_to_expiry": tte, "volatility": vol,
         "quantity": qty, "entry_price": _default_price(spot, k_high, tte, rate, vol, "call", dividend_yield, None)},
    ])


def build_straddle(spot: float, strike: float, tte: float, vol: float, rate: float,
                   dividend_yield: float = 0.0, qty: float = 1.0, is_long: bool = True) -> pd.DataFrame:
    sign = 1.0 if is_long else -1.0
    return _legs_df([
        {"option_type": "call", "strike": strike, "time_to_expiry": tte, "volatility": vol,
         "quantity": sign * qty, "entry_price": _default_price(spot, strike, tte, rate, vol, "call", dividend_yield, None)},
        {"option_type": "put", "strike": strike, "time_to_expiry": tte, "volatility": vol,
         "quantity": sign * qty, "entry_price": _default_price(spot, strike, tte, rate, vol, "put", dividend_yield, None)},
    ])


def build_strangle(spot: float, k_put: float, k_call: float, tte: float, vol: float, rate: float,
                   dividend_yield: float = 0.0, qty: float = 1.0, is_long: bool = True) -> pd.DataFrame:
    sign = 1.0 if is_long else -1.0
    return _legs_df([
        {"option_type": "put", "strike": k_put, "time_to_expiry": tte, "volatility": vol,
         "quantity": sign * qty, "entry_price": _default_price(spot, k_put, tte, rate, vol, "put", dividend_yield, None)},
        {"option_type": "call", "strike": k_call, "time_to_expiry": tte, "volatility": vol,
         "quantity": sign * qty, "entry_price": _default_price(spot, k_call, tte, rate, vol, "call", dividend_yield, None)},
    ])


def build_strip(spot: float, strike: float, tte: float, vol: float, rate: float,
                dividend_yield: float = 0.0, qty: float = 1.0) -> pd.DataFrame:
    """Long 1 call + 2 puts at the same strike and expiry (Hull Ch. 11)."""
    return _legs_df([
        {"option_type": "call", "strike": strike, "time_to_expiry": tte, "volatility": vol,
         "quantity": qty, "entry_price": _default_price(spot, strike, tte, rate, vol, "call", dividend_yield, None)},
        {"option_type": "put", "strike": strike, "time_to_expiry": tte, "volatility": vol,
         "quantity": 2.0 * qty, "entry_price": _default_price(spot, strike, tte, rate, vol, "put", dividend_yield, None)},
    ])


def build_strap(spot: float, strike: float, tte: float, vol: float, rate: float,
                dividend_yield: float = 0.0, qty: float = 1.0) -> pd.DataFrame:
    """Long 2 calls + 1 put at the same strike and expiry (Hull Ch. 11)."""
    return _legs_df([
        {"option_type": "call", "strike": strike, "time_to_expiry": tte, "volatility": vol,
         "quantity": 2.0 * qty, "entry_price": _default_price(spot, strike, tte, rate, vol, "call", dividend_yield, None)},
        {"option_type": "put", "strike": strike, "time_to_expiry": tte, "volatility": vol,
         "quantity": qty, "entry_price": _default_price(spot, strike, tte, rate, vol, "put", dividend_yield, None)},
    ])


def build_calendar_call_spread(spot: float, strike: float, t_near: float, t_far: float, vol: float,
                             rate: float, dividend_yield: float = 0.0, qty: float = 1.0) -> pd.DataFrame:
    if t_far <= t_near:
        raise ValueError("Far expiration must be strictly greater than near expiration.")
    return _legs_df([
        {"option_type": "call", "strike": strike, "time_to_expiry": t_near, "volatility": vol,
         "quantity": -qty, "entry_price": _default_price(spot, strike, t_near, rate, vol, "call", dividend_yield, None)},
        {"option_type": "call", "strike": strike, "time_to_expiry": t_far, "volatility": vol,
         "quantity": qty, "entry_price": _default_price(spot, strike, t_far, rate, vol, "call", dividend_yield, None)},
    ])


STRATEGY_REGISTRY: dict[str, Callable[..., pd.DataFrame]] = {
    "Bull Call Spread": build_bull_call_spread,
    "Bear Put Spread": build_bear_put_spread,
    "Bull Put Spread": build_bull_put_spread,
    "Bear Call Spread": build_bear_call_spread,
    "Butterfly Call Spread": build_butterfly_call_spread,
    "Long Straddle": lambda **kw: build_straddle(**kw, is_long=True),
    "Short Straddle": lambda **kw: build_straddle(**kw, is_long=False),
    "Long Strangle": lambda **kw: build_strangle(**kw, is_long=True),
    "Short Strangle": lambda **kw: build_strangle(**kw, is_long=False),
    "Strip": build_strip,
    "Strap": build_strap,
    "Calendar Call Spread": build_calendar_call_spread,
}


def build_strategy(name: str, **kwargs) -> pd.DataFrame:
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY)}")
    return STRATEGY_REGISTRY[name](**kwargs)
