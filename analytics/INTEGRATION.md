# Analytics module integration contract

These files follow the QuantOptions-Engine README. The README defines module
responsibilities but does not specify exact public function signatures for
`core/pricing.py` or `core/monte_carlo.py`. To avoid guessing those signatures,
the analytics modules accept the relevant core function as an optional
callable (and use a documented default import where appropriate).

## Expected core pricing interfaces

For default imports, expose these functions in `core/pricing.py`:

```python
def black_scholes_price(
    spot, strike, time_to_expiry, rate, volatility, option_type,
    dividend_yield=0.0
) -> float:
    ...

def calculate_greeks(
    spot, strike, time_to_expiry, rate, volatility, option_type,
    dividend_yield=0.0
) -> dict:
    # keys: delta, gamma, theta, vega, rho
    ...
```

The analytics modules do not implement a second copy of Black-Scholes. If
your core functions use different names or parameter conventions, pass them
through `pricing_function=` or `greeks_function=`.

## Common conventions

- Spot, strike, and option prices use the same currency/unit.
- Rates, volatility, dividend yield, and relative shocks are decimals.
- Time to expiry is in years.
- `option_type` is `"call"` or `"put"`.
- In portfolio positions, positive quantity means long and negative means short.
- `multiplier` is normally 100 for a standard US equity option; use the actual
  contract multiplier for the instrument being analyzed.
- The risk engine's VaR/CVaR inputs are **losses**, so positive means loss.
- Theta and Vega retain the units returned by `core/pricing.py`; document those
  units there and keep them consistent across the application.

## Quick usage

```python
from analytics import (
    MispricingFilters,
    scan_mispricing,
    build_greeks_matrix,
    calculate_var_cvar,
)

candidates = scan_mispricing(
    option_chain,
    spot=100.0,
    rate=0.04,
    volatility=0.25,
    filters=MispricingFilters(
        min_open_interest=100,
        max_spread_pct=0.10,
        min_abs_residual=0.15,
    ),
    pricing_function=black_scholes_price,
)

surfaces = build_greeks_matrix(
    spot=100, strike=100, time_to_expiry=0.5, rate=0.04,
    volatility=0.25, option_type="call",
    greeks_function=calculate_greeks,
)

risk = calculate_var_cvar(simulated_losses, confidence_level=0.95)
```

## Important modeling notes

1. The scanner uses a valid bid/ask midpoint, not the last-traded price.
   It rejects missing, non-positive, or crossed quotes.
2. The scanner's volatility input must be chosen deliberately. A single
   constant volatility is useful for a controlled baseline, but it can make
   market/model residuals misleading when the market has a volatility smile.
3. A market/model residual is a research flag, not an arbitrage proof.
   Executable bid/ask prices, fees, dividends, exercise style, borrow/financing,
   and hedge costs still matter.
4. `calculate_portfolio_pnl` fully reprices the option positions for each
   supplied spot/volatility scenario. It does not pretend that simulated
   underlying prices alone are portfolio P&L.
5. The risk engine expects a loss distribution. If Monte Carlo currently
   produces only underlying price paths, first define a portfolio valuation
   rule and convert each path/scenario into portfolio P&L.
6. VaR is the empirical quantile; CVaR is the average of losses at or beyond
   that empirical quantile. This tail convention is stated explicitly because
   finite samples can produce slightly different ES estimates under other
   quantile conventions.
