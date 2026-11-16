"""Market-versus-model price comparison for listed options.

This module flags possible relative-value candidates. A model residual is not
proof of an executable arbitrage opportunity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd


PriceFunction = Callable[..., float]


@dataclass(frozen=True)
class MispricingFilters:
    """Configurable quote-quality and liquidity filters.

    ``max_spread_pct`` is measured against the midpoint, not the ask.
    Set a threshold to ``None`` to disable that filter.
    """

    min_open_interest: Optional[int] = 1
    min_volume: Optional[int] = 0
    max_spread_pct: Optional[float] = 0.25
    min_abs_residual: float = 0.10
    min_residual_pct: Optional[float] = None


def _default_pricer() -> PriceFunction:
    """Load the agreed core pricing function only when it is needed.

    Expected core interface:
        black_scholes_price(spot, strike, time_to_expiry, rate,
                            volatility, option_type, dividend_yield=0.0)

    Passing ``pricing_function`` explicitly is recommended when integrating
    with an existing pricing.py whose public function has a different name.
    """
    try:
        from core.pricing import black_scholes_price
    except ImportError as exc:
        raise ImportError(
            "Provide pricing_function=... or expose "
            "core.pricing.black_scholes_price with the documented interface."
        ) from exc
    return black_scholes_price


def _market_midpoint(bid: pd.Series, ask: pd.Series) -> pd.Series:
    """Return midpoint only for positive, non-crossed two-sided quotes."""
    valid = bid.gt(0) & ask.gt(0) & ask.ge(bid)
    return ((bid + ask) / 2).where(valid)


def scan_mispricing(
    options: pd.DataFrame,
    *,
    spot: float,
    rate: float,
    volatility: Optional[float] = None,
    pricing_function: Optional[PriceFunction] = None,
    filters: Optional[MispricingFilters] = None,
    dividend_yield: float = 0.0,
) -> pd.DataFrame:
    """Compare option mid-prices with Black-Scholes model prices.

    Required columns: ``strike``, ``bid``, ``ask``, ``option_type``,
    ``time_to_expiry``. Supply either a scalar ``volatility`` or a
    ``volatility`` column. Optional columns: ``openInterest``, ``volume``,
    ``contractSymbol``, ``expiration``.

    The returned DataFrame contains every input row plus model diagnostics
    and an ``eligible`` flag. This makes exclusions auditable.
    """
    if not isinstance(options, pd.DataFrame):
        raise TypeError("options must be a pandas DataFrame.")
    required = {"strike", "bid", "ask", "option_type", "time_to_expiry"}
    missing = required.difference(options.columns)
    if missing:
        raise ValueError(f"Missing required option columns: {sorted(missing)}")
    if not np.isfinite(spot) or spot <= 0:
        raise ValueError("spot must be a positive finite number.")
    if not np.isfinite(rate):
        raise ValueError("rate must be finite.")
    if not np.isfinite(dividend_yield):
        raise ValueError("dividend_yield must be finite.")

    rules = filters or MispricingFilters()
    price_option = pricing_function or _default_pricer()
    result = options.copy()

    for column in ("strike", "bid", "ask", "time_to_expiry"):
        result[column] = pd.to_numeric(result[column], errors="coerce")

    if volatility is not None:
        result["model_volatility"] = float(volatility)
    elif "volatility" in result.columns:
        result["model_volatility"] = pd.to_numeric(
            result["volatility"], errors="coerce"
        )
    else:
        raise ValueError("Provide volatility=... or a 'volatility' column.")

    result["option_type"] = result["option_type"].astype(str).str.lower()
    valid_inputs = (
        result["strike"].gt(0)
        & result["time_to_expiry"].gt(0)
        & result["model_volatility"].gt(0)
        & result["model_volatility"].notna()
        & result["option_type"].isin(["call", "put"])
    )

    result["market_mid"] = _market_midpoint(result["bid"], result["ask"])
    result["spread"] = result["ask"] - result["bid"]
    result["spread_pct"] = (
        result["spread"] / result["market_mid"]
    ).where(result["market_mid"].gt(0))

    model_prices = np.full(len(result), np.nan, dtype=float)
    for position, row in enumerate(result.itertuples(index=False)):
        if not valid_inputs.iloc[position]:
            continue
        try:
            model_prices[position] = float(
                price_option(
                    spot=spot,
                    strike=float(row.strike),
                    time_to_expiry=float(row.time_to_expiry),
                    rate=rate,
                    volatility=float(row.model_volatility),
                    option_type=row.option_type,
                    dividend_yield=dividend_yield,
                )
            )
        except (TypeError, ValueError, ArithmeticError):
            # Keep a failed model value missing; do not turn it into zero.
            continue

    result["theoretical_price"] = model_prices
    result["residual"] = result["market_mid"] - result["theoretical_price"]
    result["residual_pct"] = (
        result["residual"] / result["theoretical_price"]
    ).where(result["theoretical_price"].gt(0))

    eligible = (
        valid_inputs
        & result["market_mid"].notna()
        & result["theoretical_price"].notna()
    )
    reasons = pd.Series("", index=result.index, dtype="object")

    def exclude(mask: pd.Series, reason: str) -> None:
        nonlocal eligible, reasons
        failed = eligible & ~mask.fillna(False)
        reasons.loc[failed] = reasons.loc[failed].map(
            lambda old: f"{old}; {reason}".strip("; ")
        )
        eligible &= mask.fillna(False)

    if rules.min_open_interest is not None:
        if "openInterest" not in result.columns:
            exclude(pd.Series(False, index=result.index), "open interest unavailable")
        else:
            oi = pd.to_numeric(result["openInterest"], errors="coerce")
            exclude(oi.ge(rules.min_open_interest), "open interest below minimum")

    if rules.min_volume is not None:
        if "volume" not in result.columns:
            if rules.min_volume > 0:
                exclude(pd.Series(False, index=result.index), "volume unavailable")
        else:
            volume = pd.to_numeric(result["volume"], errors="coerce")
            exclude(volume.ge(rules.min_volume), "volume below minimum")

    if rules.max_spread_pct is not None:
        exclude(
            result["spread_pct"].le(rules.max_spread_pct),
            "bid-ask spread above maximum",
        )

    exclude(
        result["residual"].abs().ge(rules.min_abs_residual),
        "absolute residual below minimum",
    )
    if rules.min_residual_pct is not None:
        exclude(
            result["residual_pct"].abs().ge(rules.min_residual_pct),
            "relative residual below minimum",
        )

    reasons.loc[~valid_inputs] = "invalid model inputs"
    reasons.loc[result["market_mid"].isna() & valid_inputs] = (
        "invalid or incomplete bid-ask quote"
    )
    result["eligible"] = eligible
    result["exclusion_reason"] = reasons.replace("", pd.NA)
    result["signal"] = np.select(
        [result["residual"].gt(0), result["residual"].lt(0)],
        ["market_above_model", "market_below_model"],
        default="no_signal",
    )
    return result.sort_values(
        by="residual", key=lambda values: values.abs(), ascending=False,
        na_position="last"
    ).reset_index(drop=True)
