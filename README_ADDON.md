# QuantOptions-Engine — Volatility & Strategy Workbench

A Streamlit research dashboard for listed options. It combines live-ish market snapshots from Yahoo Finance/yfinance with a transparent European Black–Scholes–Merton (BSM) model. It is designed for analysis and paper-trading preparation; it does **not** connect to a broker or submit orders.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Dashboard features

- **Ticker discovery:** search company names or symbols through Yahoo Finance, select a result, then load its listed option expirations and chain.
- **Market chain:** calls and puts with bid, ask, midpoint, last, volume and open interest. Download the selected chain as CSV.
- **Option calculator:** calculate a BSM theoretical price and Greeks from user-entered inputs, or solve implied volatility from an observed option price and contract specifications.
- **IV smile/skew:** solve IV for each contract from its bid/ask midpoint and plot IV against strike across expiries.
- **IV term structure / curve:** plot near-ATM solved IV against expiry.
- **IV surface:** interactive 3D view of solved IV against moneyness and days to expiry when sufficient valid data exists.
- **Greeks through time:** select a listed contract and see Delta, Gamma, Theta, Vega, Rho and model value as time to maturity declines. Spot and IV shocks can be adjusted.
- **Multi-leg strategy lab:** editable option legs, signed quantities (long positive, short negative), entry premiums, volatility/spot scenarios, expiry payoff, scenario repricing and aggregate Greeks. Includes a first-order delta hedge diagnostic.
- **Paper portfolio ledger:** save an analyzed strategy to a session-only simulated book, see model-marked P&L and aggregate Greeks, and remove positions. No broker order is sent; the ledger is not persistent after the browser session ends.
- **Research commentary and playbook:** contextual IV observations plus educational summaries of covered calls, protective puts, vertical spreads, straddles, iron condors, calendars and delta hedging.

## Implied volatility methodology

For every contract, the app calculates a valid bid/ask midpoint and solves for volatility `sigma` such that:

`BSM_price(spot, strike, time, rate, sigma, call_or_put, dividend_yield) = observed_mid_price`.

The numerical solver uses Brent's bracketed root-finding method and checks European no-arbitrage price bounds. Invalid, stale, crossed, missing, or non-identifiable quotes remain flagged rather than receiving a fabricated IV. The provider's displayed IV field is retained for reference only; the app's primary smile/curve/surface uses its own BSM-inverted IV.

## Model conventions and limitations

- Rates, dividend yield and volatility are annualized decimals internally.
- Time to expiry uses ACT/365. Yahoo date-only expiries are approximated at 16:00 America/New_York; this is not an exchange settlement calendar.
- BSM assumes European exercise and continuous dividend yield. Many US single-stock options are American-style; early exercise and discrete dividends can make BSM IV a model-implied comparison rather than an exact market-consistent volatility.
- The app uses one flat risk-free rate and continuous dividend yield per analysis, not a full maturity-matched yield/dividend curve.
- Yahoo Finance/yfinance is an unofficial data interface; data may be delayed, incomplete, rate-limited, adjusted or unavailable. Verify all quotes and contract details with a broker/exchange before any decision.
- The IV surface is built from noisy discrete option quotes and median-binned observations; it is a visualization, not a calibrated arbitrage-free volatility surface.
- Greeks are local BSM sensitivities. In the time-decay chart, spot, rate, dividend yield and volatility are held constant while time falls; real markets do not hold them constant.
- Strategy P&L and Greeks are per underlying share/one strategy unit. Contract multipliers, fees, bid/ask execution, margin, borrow, assignment, exercise, taxes and portfolio cross-margin are not automatically modeled.
- The strategy engine re-marks using BSM and scenario assumptions; it is not a broker-grade backtester. Hedge share counts are theoretical first-order delta offsets.
- Signals and strategy descriptions are descriptive educational rules, not forecasts or personalized financial advice.

## Tests

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

Network access to Yahoo Finance is intentionally not part of the unit tests; provider availability and live quote quality must be checked separately.
