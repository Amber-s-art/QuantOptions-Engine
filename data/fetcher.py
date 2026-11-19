"""Fetch and standardize option-chain market data for analytics and paper trading."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional, Protocol
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


class OptionChainProvider(Protocol):
    """Interface for plugging in official exchange/broker feeds."""

    def get_spot(self, ticker: str) -> float: ...
    def get_expirations(self, ticker: str) -> list[str]: ...
    def get_chain(self, ticker: str, expiration: str) -> tuple[pd.DataFrame, pd.DataFrame]: ...


class YFinanceProvider:
    """Unofficial Yahoo Finance fallback/research provider."""

    def __init__(self) -> None:
        self._tickers: dict[str, object] = {}

    def _client(self, symbol: str):
        if symbol not in self._tickers:
            try:
                import yfinance as yf
            except ImportError as exc:
                raise ImportError(
                    "yfinance is required for YFinanceProvider. Install it with 'pip install yfinance'."
                ) from exc
            try:
                self._tickers[symbol] = yf.Ticker(symbol)
            except Exception as exc:
                raise MarketDataError(f"Could not initialize ticker {symbol!r}.") from exc
        return self._tickers[symbol]

    def get_spot(self, ticker: str) -> float:
        symbol = _normalize_ticker(ticker)
        instrument = self._client(symbol)

        try:
            fast_info = instrument.fast_info
            for key in ("last_price", "lastPrice"):
                val = fast_info.get(key)
                if val is not None and np.isfinite(float(val)) and float(val) > 0:
                    return float(val)
        except Exception:
            pass

        try:
            history = instrument.history(period="5d", auto_adjust=False)
            if history is None or history.empty or "Close" not in history:
                raise MarketDataError(f"No recent price history returned for {symbol}.")
            closes = pd.to_numeric(history["Close"], errors="coerce").dropna()
            closes = closes[closes > 0]
            if closes.empty:
                raise MarketDataError(f"No valid closing price returned for {symbol}.")
            return float(closes.iloc[-1])
        except MarketDataError:
            raise
        except Exception as exc:
            raise MarketDataError(f"Failed to retrieve spot price for {symbol}.") from exc

    def get_expirations(self, ticker: str) -> list[str]:
        symbol = _normalize_ticker(ticker)
        instrument = self._client(symbol)
        try:
            raw_expiries = list(instrument.options or [])
        except Exception as exc:
            raise MarketDataError(f"Could not retrieve expirations for {symbol}.") from exc

        valid: list[str] = []
        for exp in raw_expiries:
            try:
                valid.append(pd.Timestamp(exp).date().isoformat())
            except (TypeError, ValueError):
                continue
        return sorted(set(valid))

    def get_chain(self, ticker: str, expiration: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        symbol = _normalize_ticker(ticker)
        instrument = self._client(symbol)
        try:
            chain = instrument.option_chain(expiration)
        except Exception as exc:
            raise MarketDataError(
                f"Failed to retrieve option chain for {symbol} ({expiration})."
            ) from exc
        if chain is None:
            raise MarketDataError(f"Provider returned empty chain for {symbol} ({expiration}).")
        return chain.calls, chain.puts


_DEFAULT_PROVIDER = YFinanceProvider()


def _normalize_ticker(ticker: str) -> str:
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("ticker must be a non-empty string.")
    return ticker.strip().upper()


def _normalize_expiry(expiration: str) -> str:
    try:
        return pd.Timestamp(expiration).date().isoformat()
    except Exception as exc:
        raise ValueError(f"Invalid expiration date: {expiration!r}") from exc


def get_available_expirations(ticker: str, provider: Optional[OptionChainProvider] = None) -> list[str]:
    """Return sorted ISO expiration dates for a ticker."""
    src = provider or _DEFAULT_PROVIDER
    return src.get_expirations(_normalize_ticker(ticker))


def search_tickers(query: str, limit: int = 12) -> pd.DataFrame:
    """Search Yahoo Finance symbols for discovery hints."""
    cols = ["symbol", "name", "exchange", "type"]
    if not isinstance(query, str) or not query.strip():
        return pd.DataFrame(columns=cols)
    try:
        import yfinance as yf

        result = yf.Search(query.strip(), max_results=max(1, min(int(limit), 25)), news_count=0)
        quotes = getattr(result, "quotes", None) or []
    except Exception as exc:
        raise MarketDataError(f"Ticker search failed for {query!r}: {exc}") from exc

    rows = [
        {
            "symbol": item.get("symbol", ""),
            "name": item.get("shortname") or item.get("longname") or "",
            "exchange": item.get("exchange", ""),
            "type": item.get("quoteType", ""),
        }
        for item in quotes
        if isinstance(item, dict)
    ]
    return pd.DataFrame(rows, columns=cols)


def time_to_expiry(
    expiration: str,
    *,
    as_of: Optional[datetime] = None,
    market_tz: str = "Asia/Kolkata",
    close_hour: int = 15,
    close_minute: int = 30,
) -> float:
    """Convert expiration date to ACT/365 year fraction using exchange close time."""
    expiry_date = date.fromisoformat(_normalize_expiry(expiration))
    now_utc = (as_of or datetime.now(timezone.utc)).astimezone(timezone.utc)

    expiry_local = datetime(
        expiry_date.year,
        expiry_date.month,
        expiry_date.day,
        close_hour,
        close_minute,
        0,
        tzinfo=ZoneInfo(market_tz),
    )
    seconds = (expiry_local.astimezone(timezone.utc) - now_utc).total_seconds()
    return max(seconds, 0.0) / (365.0 * 24.0 * 60.0 * 60.0)


def _clean_option_chain(
    frame: Optional[pd.DataFrame],
    option_type: str,
    expiration: str,
    tte: float,
) -> pd.DataFrame:
    """Normalize provider option columns without fabricating missing quotes."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=OPTION_COLUMNS)

    df = frame.copy()
    df["option_type"] = option_type
    df["expiration"] = expiration

    for col in ("strike", "bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "contractSymbol" not in df.columns:
        df["contractSymbol"] = pd.NA

    for col in ("bid", "ask", "lastPrice"):
        df.loc[df[col] < 0, col] = np.nan

    df = df[df["strike"].notna() & (df["strike"] > 0)].copy()
    df["time_to_expiry"] = float(tte)
    return df[OPTION_COLUMNS].sort_values("strike").reset_index(drop=True)


def fetch_market_data(
    ticker: str,
    expiration: str,
    *,
    risk_free_rate: Optional[float] = None,
    as_of: Optional[datetime] = None,
    provider: Optional[OptionChainProvider] = None,
    market_tz: str = "Asia/Kolkata",
    close_hour: int = 15,
    close_minute: int = 30,
) -> MarketSnapshot:
    """Fetch spot and call/put chains for one selected expiration."""
    symbol = _normalize_ticker(ticker)
    if risk_free_rate is not None and not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite or None.")

    req_expiry = _normalize_expiry(expiration)
    src = provider or _DEFAULT_PROVIDER
    retrieved_at = datetime.now(timezone.utc)

    available = src.get_expirations(symbol)
    if req_expiry not in available:
        raise MarketDataError(
            f"Expiration {req_expiry} not available for {symbol}. Available: {available}"
        )

    spot = src.get_spot(symbol)
    raw_calls, raw_puts = src.get_chain(symbol, req_expiry)

    tte = time_to_expiry(
        req_expiry,
        as_of=as_of,
        market_tz=market_tz,
        close_hour=close_hour,
        close_minute=close_minute,
    )
    calls = _clean_option_chain(raw_calls, "call", req_expiry, tte)
    puts = _clean_option_chain(raw_puts, "put", req_expiry, tte)

    if calls.empty and puts.empty:
        raise MarketDataError(f"Option chain for {symbol} ({req_expiry}) is empty.")

    return MarketSnapshot(
        ticker=symbol,
        spot=float(spot),
        expiration=req_expiry,
        risk_free_rate=float(risk_free_rate) if risk_free_rate is not None else None,
        calls=calls,
        puts=puts,
        retrieved_at=retrieved_at,
    )


def fetch_multiple_expiries(
    ticker: str,
    expirations: list[str],
    *,
    risk_free_rate: Optional[float] = None,
    as_of: Optional[datetime] = None,
    provider: Optional[OptionChainProvider] = None,
    market_tz: str = "Asia/Kolkata",
    close_hour: int = 15,
    close_minute: int = 30,
) -> pd.DataFrame:
    """Fetch and normalize multiple expiries using a single spot and expiration lookup."""
    symbol = _normalize_ticker(ticker)
    requested = [_normalize_expiry(x) for x in expirations]
    if not requested:
        return pd.DataFrame(columns=OPTION_COLUMNS + ["spot", "retrieved_at"])

    src = provider or _DEFAULT_PROVIDER
    available = set(src.get_expirations(symbol))
    missing = [x for x in requested if x not in available]
    if missing:
        raise MarketDataError(f"Unavailable expirations for {symbol}: {missing}")

    spot = float(src.get_spot(symbol))
    retrieved_at = datetime.now(timezone.utc)
    frames: list[pd.DataFrame] = []

    for exp in requested:
        raw_calls, raw_puts = src.get_chain(symbol, exp)
        tte = time_to_expiry(
            exp,
            as_of=as_of,
            market_tz=market_tz,
            close_hour=close_hour,
            close_minute=close_minute,
        )
        calls = _clean_option_chain(raw_calls, "call", exp, tte)
        puts = _clean_option_chain(raw_puts, "put", exp, tte)
        combined = pd.concat([calls, puts], ignore_index=True)
        if not combined.empty:
            combined["spot"] = spot
            combined["retrieved_at"] = retrieved_at
            frames.append(combined)

    if not frames:
        return pd.DataFrame(columns=OPTION_COLUMNS + ["spot", "retrieved_at"])
    return pd.concat(frames, ignore_index=True)
