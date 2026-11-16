"""Transparent multi-leg option strategy analytics (European BSM baseline)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from core.pricing import black_scholes_price, calculate_greeks


def analyze_strategy(legs: pd.DataFrame, *, spot: float, rate: float, dividend_yield: float = 0.0,
                     spot_grid: np.ndarray | None = None, iv_shift: float = 0.0,
                     days_forward: int = 0) -> dict:
    """Aggregate signed leg value/Greeks and expiry payoff for a multi-leg book.

    quantity is signed: positive=long, negative=short. Premiums are per share.
    The scenario P&L re-marks each leg with BSM at a common forward horizon.
    """
    required = {"option_type", "strike", "time_to_expiry", "volatility", "quantity", "entry_price"}
    missing = required - set(legs.columns)
    if missing:
        raise ValueError(f"Missing strategy columns: {sorted(missing)}")
    clean = legs.copy()
    if clean.empty:
        raise ValueError("Add at least one option leg.")
    for c in ["strike", "time_to_expiry", "volatility", "quantity", "entry_price"]:
        clean[c] = pd.to_numeric(clean[c], errors="raise")
    if (clean.strike <= 0).any() or (clean.time_to_expiry <= 0).any() or (clean.volatility <= 0).any():
        raise ValueError("Strike, time to expiry and volatility must be positive.")
    if spot_grid is None:
        spot_grid = np.linspace(spot * 0.7, spot * 1.3, 121)
    total_value = total_delta = total_gamma = total_theta = total_vega = total_rho = 0.0
    entry_cash = 0.0
    leg_rows = []
    horizon = max(0, int(days_forward)) / 365.0
    for leg in clean.itertuples(index=False):
        kind = str(leg.option_type).lower()
        qty = float(leg.quantity)
        k, t, vol, entry = float(leg.strike), float(leg.time_to_expiry), float(leg.volatility), float(leg.entry_price)
        mark = float(black_scholes_price(spot, k, t, rate, vol, kind, dividend_yield))
        g = calculate_greeks(spot, k, t, rate, vol, kind, dividend_yield)
        total_value += qty * mark
        entry_cash += qty * entry
        total_delta += qty * float(g["delta"])
        total_gamma += qty * float(g["gamma"])
        total_theta += qty * float(g["theta"])
        total_vega += qty * float(g["vega"])
        total_rho += qty * float(g["rho"])
        leg_rows.append({"option_type": kind, "strike": k, "quantity": qty, "entry_price": entry,
                         "model_mark": mark, "delta": g["delta"], "gamma": g["gamma"],
                         "theta_per_year": g["theta"], "vega_per_1pct": float(g["vega"])*0.01})
    # At expiry each leg has intrinsic value. Entry cashflow is positive for credit (short legs).
    payoff = np.zeros_like(spot_grid, dtype=float)
    for leg in clean.itertuples(index=False):
        intrinsic = np.maximum(spot_grid - float(leg.strike), 0.0) if str(leg.option_type).lower() == "call" else np.maximum(float(leg.strike) - spot_grid, 0.0)
        payoff += float(leg.quantity) * (intrinsic - float(leg.entry_price))
    # Horizon scenario revaluation; expired legs settle to intrinsic.
    scenario_rows = []
    scenario_spots = spot * np.array([0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15])
    for s_scn in scenario_spots:
        pnl = 0.0
        for leg in clean.itertuples(index=False):
            t2 = max(float(leg.time_to_expiry) - horizon, 0.0)
            vol2 = max(float(leg.volatility) + iv_shift, 1e-4)
            if t2 == 0:
                mark2 = max(s_scn-float(leg.strike),0.0) if str(leg.option_type).lower()=="call" else max(float(leg.strike)-s_scn,0.0)
            else:
                mark2 = float(black_scholes_price(s_scn, float(leg.strike), t2, rate, vol2, str(leg.option_type).lower(), dividend_yield))
            pnl += float(leg.quantity) * (mark2 - float(leg.entry_price))
        scenario_rows.append({"spot": s_scn, "spot_change_pct": (s_scn/spot-1)*100, "pnl_per_share": pnl})
    return {"legs": pd.DataFrame(leg_rows), "spot_grid": spot_grid, "expiry_pnl": payoff,
            "scenario": pd.DataFrame(scenario_rows), "net_model_value": total_value,
            "net_entry_value": entry_cash, "net_delta": total_delta, "net_gamma": total_gamma,
            "net_theta": total_theta, "net_vega": total_vega, "net_rho": total_rho,
            "delta_hedge_shares": -total_delta}
