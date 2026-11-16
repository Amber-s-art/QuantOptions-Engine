"""Download and standardize equity option data through yfinance.

The data provider is isolated here. Core pricing and analytics modules should
consume the returned pandas DataFrames and must not call yfinance themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


class MarketDataError(RuntimeError):
    """Raised when market data cannot be retrieved or validated."""


@dataclass
class MarketSnapshot:
    """Consistent market-data bundle for one ticker and expiration."""

    ticker: str
    spot: float
    expiration: str
    risk_free_rate: Optional[float]
    calls: pd.DataFrame
    puts: pd.DataFrame
    retrieved_at: datetime

    def combined_options(self) -> pd.DataFrame:
        """Return calls and puts in one table."""
        return pd.concat([self.calls, self.puts], ignore_index=True)


OPTION_COLUMNS = [
    "contractSymbol",
    "option_type",
    "expiration",
    "strike",
    "bid",
    "ask",
    "lastPrice",
    "volume",
    "openInterest",
    "impliedVolatility",
    "time_to_expiry",
]


def _get_yfinance_ticker(ticker: str):
    """Import yfinance lazily so mathematical modules work without network tools."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for live market data. Install it with "
            "'pip install yfinance'."
        ) from exc
    try:
        return yf.Ticker(ticker)
    except Exception as exc:
        raise MarketDataError(f"Could not initialize ticker {ticker!r}.") from exc


def _normalize_ticker(ticker: str) -> str:
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("ticker must be a non-empty string.")
    return ticker.strip().upper()


def get_underlying_price(ticker: str) -> float:
    """Get the latest available underlying price.

    ``fast_info`` is tried first. Recent daily history is the fallback because
    fast_info fields are not consistently available for every instrument.
    """
    symbol = _normalize_ticker(ticker)
    instrument = _get_yfinance_ticker(symbol)

    try:
        fast_info = instrument.fast_info
        price = fast_info.get("last_price")
        if price is None:
            price = fast_info.get("lastPrice")
        if price is not None and np.isfinite(float(price)) and float(price) > 0:
            return float(price)
    except Exception:
        pass

    try:
        history = instrument.history(period="5d", auto_adjust=False)
        if history.empty or "Close" not in history:
            raise MarketDataError(f"No recent price history returned for {symbol}.")
        closes = pd.to_numeric(history["Close"], errors="coerce").dropna()
        closes = closes[closes > 0]
        if closes.empty:
            raise MarketDataError(f"No valid closing price returned for {symbol}.")
        return float(closes.iloc[-1])
    except MarketDataError:
        raise
    except Exception as exc:
        raise MarketDataError(f"Failed to retrieve the price for {symbol}.") from exc


def get_available_expirations(ticker: str) -> list[str]:
    """Return available option expirations as ISO date strings."""
    symbol = _normalize_ticker(ticker)
    instrument = _get_yfinance_ticker(symbol)
    try:
        expirations = list(instrument.options or [])
    except Exception as exc:
        raise MarketDataError(
            f"Could not retrieve option expirations for {symbol}."
        ) from exc

    valid = []
    for expiry in expirations:
        try:
            valid.append(pd.Timestamp(expiry).date().isoformat())
        except (TypeError, ValueError):
            continue
    return sorted(set(valid))


def _time_to_expiry(expiration: str, as_of: Optional[datetime] = None) -> float:
    """Convert expiry to ACT/365 year fraction.

    yfinance supplies date-only expirations, not exact exchange timestamps.
    We therefore use the expiry date at 16:00 US/Eastern as an approximation.
    This is a documented calendar convention, not an exchange calendar.
    """
    expiry_date = date.fromisoformat(expiration)
    now = as_of or datetime.now(timezone.utc)

    # yfinance provides a date, not the exact contract settlement timestamp.
    # Use the regular US equity market close in the exchange timezone so
    # daylight-saving changes are handled correctly.
    expiry_local = datetime(
        expiry_date.year, expiry_date.month, expiry_date.day,
        16, 0, 0, tzinfo=ZoneInfo("America/New_York")
    )
    expiry_utc = expiry_local.astimezone(timezone.utc)
    seconds = (expiry_utc - now).total_seconds()
    return max(seconds, 0.0) / (365.0 * 24.0 * 60.0 * 60.0)


def _clean_option_chain(
    frame: pd.DataFrame,
    option_type: str,
    expiration: str,
    time_to_expiry: float,
) -> pd.DataFrame:
    """Normalize provider columns without inventing missing market quotes."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=OPTION_COLUMNS)

    result = frame.copy()
    result["option_type"] = option_type
    result["expiration"] = expiration

    numeric_columns = [
        "strike", "bid", "ask", "lastPrice", "volume",
        "openInterest", "impliedVolatility",
    ]
    for column in numeric_columns:
        if column not in result.columns:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")

    if "contractSymbol" not in result.columns:
        result["contractSymbol"] = pd.NA
    result["time_to_expiry"] = float(time_to_expiry)

    # Preserve invalid quote values as NaN. Downstream code can then reject
    # them explicitly instead of accidentally treating them as zero prices.
    result.loc[result["bid"] < 0, "bid"] = np.nan
    result.loc[result["ask"] < 0, "ask"] = np.nan
    result.loc[result["lastPrice"] < 0, "lastPrice"] = np.nan
    result.loc[result["strike"] <= 0, "strike"] = np.nan

    return result[OPTION_COLUMNS].reset_index(drop=True)


def fetch_market_data(
    ticker: str,
    expiration: str,
    *,
    risk_free_rate: Optional[float] = None,
    as_of: Optional[datetime] = None,
) -> MarketSnapshot:
    """Fetch spot and call/put chains for one selected expiration.

    ``risk_free_rate`` is intentionally supplied by the caller. A robust
    production system should select a rate curve matching the option's
    maturity and currency; this fetcher does not silently substitute a
    mismatched Treasury yield.

    The returned rate is a decimal continuously compounded input convention
    for the model layer. The caller is responsible for sourcing/converting it.
    """
    symbol = _normalize_ticker(ticker)
    if risk_free_rate is not None and not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite or None.")

    requested_expiry = pd.Timestamp(expiration).date().isoformat()
    instrument = _get_yfinance_ticker(symbol)

    try:
        available = get_available_expirations(symbol)
        if requested_expiry not in available:
            raise MarketDataError(
                f"Expiration {requested_expiry} is not available for {symbol}. "
                f"Available expirations: {available}"
            )
        spot = get_underlying_price(symbol)
        chain = instrument.option_chain(requested_expiry)
    except MarketDataError:
        raise
    except Exception as exc:
        raise MarketDataError(
            f"Failed to retrieve the option chain for {symbol} {requested_expiry}."
        ) from exc

    if chain is None:
        raise MarketDataError("The provider returned no option-chain response.")

    maturity = _time_to_expiry(requested_expiry, as_of=as_of)
    calls = _clean_option_chain(chain.calls, "call", requested_expiry, maturity)
    puts = _clean_option_chain(chain.puts, "put", requested_expiry, maturity)

    if calls.empty and puts.empty:
        raise MarketDataError(
            f"The option chain for {symbol} {requested_expiry} is empty."
        )

    return MarketSnapshot(
        ticker=symbol,
        spot=spot,
        expiration=requested_expiry,
        risk_free_rate=float(risk_free_rate) if risk_free_rate is not None else None,
        calls=calls,
        puts=puts,
        retrieved_at=datetime.now(timezone.utc),
    )


def search_tickers(query: str, limit: int = 12) -> pd.DataFrame:
    """Search Yahoo Finance's symbol search for stocks/ETFs with listed options.

    Search results are discovery hints; availability is confirmed separately by
    get_available_expirations(). Yahoo Finance is an unofficial research feed.
    """
    if not isinstance(query, str) or len(query.strip()) < 1:
        return pd.DataFrame(columns=["symbol", "name", "exchange", "type"])
    try:
        import yfinance as yf
        result = yf.Search(query.strip(), max_results=max(1, min(int(limit), 25)), news_count=0)
        quotes = getattr(result, "quotes", None) or []
    except Exception as exc:
        raise MarketDataError(f"Ticker search failed for {query!r}: {exc}") from exc
    rows = []
    for item in quotes:
        if not isinstance(item, dict):
            continue
        rows.append({
            "symbol": item.get("symbol", ""),
            "name": item.get("shortname") or item.get("longname") or "",
            "exchange": item.get("exchange", ""),
            "type": item.get("quoteType", ""),
        })
    return pd.DataFrame(rows, columns=["symbol", "name", "exchange", "type"])


def fetch_multiple_expiries(ticker: str, expirations: list[str], *, risk_free_rate: float = 0.04) -> pd.DataFrame:
    """Fetch and normalize several expiries with one provider object and spot lookup."""
    symbol = _normalize_ticker(ticker)
    requested = [pd.Timestamp(x).date().isoformat() for x in expirations]
    if not requested:
        return pd.DataFrame(columns=OPTION_COLUMNS + ["spot", "retrieved_at"])
    instrument = _get_yfinance_ticker(symbol)
    try:
        available = get_available_expirations(symbol)
        invalid = [x for x in requested if x not in available]
        if invalid:
            raise MarketDataError(f"Unavailable expirations for {symbol}: {invalid}")
        spot = get_underlying_price(symbol)
        retrieved = datetime.now(timezone.utc)
        frames = []
        for expiry in requested:
            raw = instrument.option_chain(expiry)
            maturity = _time_to_expiry(expiry)
            calls = _clean_option_chain(raw.calls, "call", expiry, maturity)
            puts = _clean_option_chain(raw.puts, "put", expiry, maturity)
            frame = pd.concat([calls, puts], ignore_index=True)
            frame["spot"] = spot
            frame["retrieved_at"] = retrieved
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OPTION_COLUMNS + ["spot", "retrieved_at"])
    except MarketDataError:
        raise
    except Exception as exc:
        raise MarketDataError(f"Failed to retrieve multiple expiries for {symbol}: {exc}") from exc
