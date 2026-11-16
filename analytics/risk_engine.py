"""Loss-distribution analytics and transparent portfolio stress testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskSummary:
    """Historical/simulated loss statistics in the same currency units as input."""

    confidence_level: float
    var: float
    cvar: float
    observations: int
    mean_loss: float
    worst_loss: float


def calculate_var_cvar(
    losses: Sequence[float] | np.ndarray,
    confidence_level: float = 0.95,
) -> RiskSummary:
    """Calculate historical/simulation VaR and CVaR from a loss sample.

    Positive values represent losses; negative values represent gains.
    VaR is the empirical confidence-level quantile. CVaR is the arithmetic
    mean of observations at or above that quantile (a transparent empirical
    tail convention).
    """
    values = np.asarray(losses, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("losses must contain at least one finite observation.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1.")

    var_value = float(np.quantile(values, confidence_level, method="higher"))
    tail = values[values >= var_value]
    cvar_value = float(tail.mean())
    return RiskSummary(
        confidence_level=float(confidence_level),
        var=var_value,
        cvar=cvar_value,
        observations=int(values.size),
        mean_loss=float(values.mean()),
        worst_loss=float(values.max()),
    )


def calculate_portfolio_pnl(
    positions: pd.DataFrame,
    *,
    scenario_spots: Sequence[float] | np.ndarray,
    pricing_function: Callable[..., float],
    scenario_volatility_changes: Optional[Sequence[float] | np.ndarray] = None,
    dividend_yield: float = 0.0,
) -> pd.DataFrame:
    """Reprice a simple European-option portfolio under spot/vol scenarios.

    Required position columns:
      ``quantity``, ``multiplier``, ``option_type``, ``strike``,
      ``time_to_expiry``, ``rate``, ``volatility``, ``entry_price``.

    ``quantity`` is signed: positive for long, negative for short.
    ``entry_price`` is the current per-unit option mark. The result reports
    scenario portfolio value and P&L relative to that mark.

    This function intentionally requires the caller to pass the shared core
    pricing function. That avoids silently using a different pricing model.
    """
    required = {
        "quantity", "multiplier", "option_type", "strike",
        "time_to_expiry", "rate", "volatility", "entry_price",
    }
    missing = required.difference(positions.columns)
    if missing:
        raise ValueError(f"Missing position columns: {sorted(missing)}")
    if positions.empty:
        raise ValueError("positions cannot be empty.")

    spots = np.asarray(scenario_spots, dtype=float).reshape(-1)
    if spots.size == 0 or not np.all(np.isfinite(spots)) or np.any(spots <= 0):
        raise ValueError("scenario_spots must be finite positive prices.")
    if scenario_volatility_changes is None:
        vol_changes = np.zeros(spots.size, dtype=float)
    else:
        vol_changes = np.asarray(
            scenario_volatility_changes, dtype=float
        ).reshape(-1)
        if vol_changes.size != spots.size or not np.all(np.isfinite(vol_changes)):
            raise ValueError(
                "scenario_volatility_changes must match scenario_spots in length."
            )

    rows = []
    base_value = float(
        (
            positions["quantity"].astype(float)
            * positions["multiplier"].astype(float)
            * positions["entry_price"].astype(float)
        ).sum()
    )

    for scenario_id, (scenario_spot, vol_change) in enumerate(zip(spots, vol_changes)):
        scenario_value = 0.0
        for position in positions.itertuples(index=False):
            scenario_vol = float(position.volatility) + float(vol_change)
            if scenario_vol <= 0:
                raise ValueError(
                    f"Scenario {scenario_id} produces non-positive volatility."
                )
            option_value = pricing_function(
                spot=float(scenario_spot),
                strike=float(position.strike),
                time_to_expiry=float(position.time_to_expiry),
                rate=float(position.rate),
                volatility=scenario_vol,
                option_type=str(position.option_type).lower(),
                dividend_yield=float(dividend_yield),
            )
            scenario_value += (
                float(position.quantity)
                * float(position.multiplier)
                * float(option_value)
            )

        rows.append(
            {
                "scenario_id": scenario_id,
                "spot": float(scenario_spot),
                "volatility_change": float(vol_change),
                "portfolio_value": float(scenario_value),
                "pnl": float(scenario_value - base_value),
                "loss": float(base_value - scenario_value),
            }
        )
    return pd.DataFrame(rows)


def run_stress_test(
    positions: pd.DataFrame,
    scenarios: Iterable[Mapping[str, float]],
    *,
    pricing_function: Callable[..., float],
    dividend_yield: float = 0.0,
) -> pd.DataFrame:
    """Reprice a portfolio for named spot/volatility shock scenarios.

    Each scenario mapping must contain ``name``, ``spot_change`` and
    ``volatility_change``. Shock values are decimal changes, e.g. -0.10 means
    a 10% spot decline and +0.05 means a five-volatility-point increase.
    """
    if "spot" not in positions.columns:
        raise ValueError("positions must include a current 'spot' column.")
    current_spots = pd.to_numeric(positions["spot"], errors="coerce").dropna().unique()
    if len(current_spots) != 1:
        raise ValueError(
            "All positions must share one underlying spot for this stress test."
        )
    current_spot = float(current_spots[0])

    results = []
    for scenario in scenarios:
        try:
            name = str(scenario["name"])
            spot_change = float(scenario["spot_change"])
            vol_change = float(scenario["volatility_change"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Each scenario needs name, spot_change and volatility_change."
            ) from exc

        scenario_spot = current_spot * (1.0 + spot_change)
        if scenario_spot <= 0:
            raise ValueError(f"Scenario {name!r} produces a non-positive spot.")

        repriced = calculate_portfolio_pnl(
            positions,
            scenario_spots=[scenario_spot],
            scenario_volatility_changes=[vol_change],
            pricing_function=pricing_function,
            dividend_yield=dividend_yield,
        )
        row = repriced.iloc[0].to_dict()
        row.update(
            {
                "scenario": name,
                "spot_change": spot_change,
            }
        )
        results.append(row)

    return pd.DataFrame(results)
