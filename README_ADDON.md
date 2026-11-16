# QuantOptions-Engine — Streamlit integration add-on

This package combines the previously prepared `data/`, `core/`, and `analytics/` modules with a Streamlit research interface. It is supplied separately; it does not modify or push to your GitHub repository.

## Run locally

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Dashboard sections

- Market chain: standardized calls and puts, export to CSV.
- Pricing & Greeks: Black–Scholes–Merton model value and analytical Greeks.
- Implied volatility: numerical IV estimates plotted by strike.
- Mispricing scan: market midpoint versus a constant-volatility model baseline, with quote-quality/liquidity filters.
- Sensitivity: Delta/Gamma/Theta/Vega grids across spot and maturity.
- Monte Carlo & Risk: GBM paths and underlying-price VaR/CVaR.

## Current market-data status

The included `data/fetcher.py` adapter uses `yfinance`. This is convenient for a prototype, but it is **not an official exchange feed and should not be represented as guaranteed real-time data**. The app labels the provider and timestamps each snapshot. Do not use it for live execution decisions.

For India-focused official/authorized real-time data, integrate a broker's authenticated API or obtain a licensed NSE data feed. Examples of documented broker APIs include DhanHQ (REST quotes and WebSocket market feed) and Zerodha Kite Connect (market quotes, instrument master and WebSocket streaming). These require account/API credentials and may have subscriptions, rate limits and exchange data-use terms. Choose your broker before implementing its adapter; the app should consume a provider-neutral `MarketSnapshot` so the UI and pricing engine do not need rewriting.

Official documentation:
- DhanHQ market quotes: https://dhanhq.co/docs/v2/market-quote/
- DhanHQ live market feed: https://dhanhq.co/docs/v2/live-market-feed/
- Zerodha Kite Connect market quotes: https://kite.trade/docs/connect/v3/market-quotes/
- NSE real-time data subscription: https://www.nseindia.com/market-data/real-time-data-subscription

## Paper-trading boundary

This version does not connect to a broker, submit orders, simulate fills, or maintain a paper portfolio. Before adding paper trading, implement a separate broker interface with explicit `paper` mode, order/position ledger, fees, slippage, partial fills, market-hours checks, idempotent order IDs, audit logs, and kill switch. Keep market-data credentials in environment variables or Streamlit secrets; never commit tokens or passwords.

## Mathematical conventions

- Rates, dividend yield and volatility are annualized decimals internally.
- Time to expiry is ACT/365 years. For Yahoo Finance date-only expiries, the fetcher approximates expiry at 16:00 America/New_York; this is not an exchange settlement calendar.
- Black–Scholes–Merton values are European-option model estimates.
- Theta is annualized; Vega and Rho are per 1.00 decimal change.
- The mispricing scan uses the sidebar's constant volatility as a research baseline. It does not prove arbitrage.
- Monte Carlo risk tab computes VaR/CVaR on underlying price changes, not a complete options portfolio.

## Validation

Run:

```bash
python -m compileall .
python -m unittest discover -s tests -v
```

The mathematical smoke tests are designed to run without a live market-data request. The Streamlit app itself requires installed dependencies and network access for Yahoo Finance retrieval.
