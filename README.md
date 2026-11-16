# QuantOptions-Engine 🚀

**A modular Python platform for options pricing, implied-volatility analysis, mispricing research, sensitivity analysis, and portfolio risk simulation.**

QuantOptions-Engine brings several quantitative-options workflows into one project. It is designed to take market data for an underlying asset and its listed options, transform that data into model inputs, calculate theoretical prices and risk sensitivities, estimate implied volatility, identify potential market/model price discrepancies, and explore portfolio risk under simulated market scenarios.

The project is organized into separate data, core-model, and analytics modules so that each area can be developed and tested independently while following shared input and output conventions.

> **Important:** This repository is an analytical and research platform. A model-price discrepancy is not automatically a risk-free arbitrage opportunity, and the project should not be treated as an automated trading or execution system. Market data may be delayed or incomplete, and model assumptions may not match market conditions.

---

## Table of contents

1. [Project goals](#1-project-goals)
2. [What the platform does](#2-what-the-platform-does)
3. [Repository structure](#3-repository-structure)
4. [System architecture and data flow](#4-system-architecture-and-data-flow)
5. [Detailed module documentation](#5-detailed-module-documentation)
   - [Data ingestion](#module-a--market-data-ingestion)
   - [Black–Scholes pricing and Greeks](#module-b--options-pricing-and-greeks)
   - [Implied volatility and smile analysis](#module-c--implied-volatility-and-smile-calibration)
   - [Mispricing and relative-value scanner](#module-d--mispricing-and-relative-value-scanner)
   - [Greeks sensitivity matrix](#module-e--greeks-sensitivity-matrix)
   - [Monte Carlo and risk engine](#module-f--monte-carlo-simulation-and-risk-engine)
6. [End-to-end workflow](#6-end-to-end-workflow)
7. [Team responsibilities and work division](#7-team-responsibilities-and-work-division)
8. [Module interfaces and integration expectations](#8-module-interfaces-and-integration-expectations)
9. [Technology stack](#9-technology-stack)
10. [Installation and setup](#10-installation-and-setup)
11. [Running the project and tests](#11-running-the-project-and-tests)
12. [Validation and testing plan](#12-validation-and-testing-plan)
13. [Assumptions, limitations, and risk considerations](#13-assumptions-limitations-and-risk-considerations)
14. [Suggested development sequence](#14-suggested-development-sequence)
15. [Contributing](#15-contributing)

---

## 1. Project goals

The main goal of QuantOptions-Engine is to create a unified workflow for quantitative options analysis rather than a collection of unrelated scripts.

The platform is intended to:

- Retrieve underlying-asset prices and listed options-chain data.
- Standardize market quotes so that downstream modules use consistent inputs.
- Calculate theoretical European call and put prices using the Black–Scholes model.
- Calculate the main option Greeks to describe how theoretical value changes as market inputs change.
- Infer implied volatility from observed option prices.
- Analyze volatility by strike to visualize a volatility smile or skew.
- Compare observed option prices with model estimates and flag possible relative-value discrepancies.
- Generate sensitivity grids to show how Greeks change across spot-price and time-to-expiry scenarios.
- Simulate possible underlying-price paths and estimate loss-distribution measures such as Value at Risk (VaR) and Conditional Value at Risk (CVaR).
- Divide development into five areas with clear ownership, deliverables, and integration responsibilities.

### Intended users

The project is suitable as a quantitative-finance engineering project for developers and students who want to understand how market-data handling, derivatives mathematics, numerical methods, analytics, and risk simulation fit together in a single Python codebase.

### Expected result

Given a supported ticker and option expiry, the intended workflow produces a structured set of analytical outputs, including:

1. Clean underlying and option-chain data.
2. Theoretical prices and Greek values.
3. Implied-volatility estimates and strike-level volatility information.
4. A filtered list of contracts whose market prices differ from model estimates.
5. Greek sensitivity data across selected scenarios.
6. Simulated price paths and risk statistics.

The exact outputs available depend on which modules have been implemented and integrated in the working version of the repository.

---

## 2. What the platform does

| Capability | Purpose | Main module |
|---|---|---|
| Market-data retrieval | Obtain underlying prices, expirations, option chains, and rate inputs | `data/fetcher.py` |
| Theoretical pricing | Estimate European option values from model inputs | `core/pricing.py` |
| Greeks calculation | Measure model sensitivity to spot, volatility, time, and rates | `core/pricing.py` |
| Implied-volatility solving | Infer volatility consistent with an observed option price | `core/implied_vol.py` |
| Volatility smile/skew | Examine how implied volatility varies by strike | `core/implied_vol.py` |
| Mispricing scan | Compare market mid-prices with theoretical estimates | `analytics/arbitrage.py` |
| Sensitivity surfaces | Examine Greeks over spot and time scenarios | `analytics/greeks_matrix.py` |
| Monte Carlo simulation | Generate possible future price paths | `core/monte_carlo.py` |
| Risk and stress analysis | Summarize simulated losses and evaluate shock scenarios | `analytics/risk_engine.py` |

These components are related, but they have different responsibilities. The `core/` package contains reusable mathematical and numerical calculations. The `data/` package obtains and prepares external data. The `analytics/` package combines inputs and core calculations to answer higher-level analytical questions.

---

## 3. Repository structure

```text
QuantOptions-Engine/
│
├── data/
│   ├── __init__.py
│   └── fetcher.py
│
├── core/
│   ├── __init__.py
│   ├── pricing.py
│   ├── implied_vol.py
│   └── monte_carlo.py
│
├── analytics/
│   ├── __init__.py
│   ├── arbitrage.py
│   ├── greeks_matrix.py
│   └── risk_engine.py
│
├── requirements.txt
└── README.md
```

### Directory responsibilities

- **`data/` — external data and preparation:** communicates with the market-data provider and converts retrieved information into a consistent format.
- **`core/` — quantitative calculations:** contains pricing formulas, numerical solvers, simulation routines, and other reusable mathematical functions.
- **`analytics/` — analysis built on top of core calculations:** combines market data and model outputs to create scans, sensitivity grids, and risk summaries.
- **`requirements.txt` — dependencies:** lists the Python packages needed by the project.
- **`README.md` — project documentation:** explains the system, its modules, team ownership, setup, and development expectations.

The structure is intentionally modular. A developer should be able to work on one module without changing the internal implementation of every other module, provided the agreed input and output interfaces remain stable.

---

## 4. System architecture and data flow

At a high level, data moves through the system in the following order:

```text
Ticker + Expiration
        |
        v
Market Data Ingestion
(data/fetcher.py)
        |
        v
Clean Underlying Data + Options Chain
        |
        +------------------------------+
        |                              |
        v                              v
Black–Scholes Pricing             Implied Volatility
(core/pricing.py)                 (core/implied_vol.py)
        |                              |
        v                              v
Theoretical Prices + Greeks       IV by Strike / Smile Data
        |                              |
        +---------------+--------------+
                        |
          +-------------+------------------+
          |                |               |
          v                v               v
    Mispricing Scan   Greeks Matrix   Monte Carlo Simulation
    (arbitrage.py)    (greeks_matrix) (monte_carlo.py)
          |                |               |
          v                v               v
    Candidate List    Sensitivity Data  Simulated Paths
                                           |
                                           v
                                      Risk Engine
                                      (risk_engine.py)
                                           |
                                           v
                                  VaR / CVaR / Stress Results
```

### Workflow stages

1. **Request inputs:** the user or calling script supplies an underlying ticker and an option expiration date. Additional parameters may include a risk-free rate, simulation settings, and filtering thresholds.
2. **Retrieve market data:** the data layer obtains the underlying price, available expirations, and call/put option-chain quotes.
3. **Normalize the data:** the data layer standardizes column names and numeric types, handles missing values, and prepares the chain for analysis.
4. **Calculate model values:** the pricing module calculates theoretical call/put values and Greeks for valid contracts.
5. **Estimate implied volatility:** the IV module solves for the volatility that makes the model price match the observed market price.
6. **Run analytics:** the arbitrage scanner, sensitivity-matrix generator, and risk modules use the prepared data and core calculations for their separate analyses.
7. **Review outputs:** the resulting tables, arrays, plots, and risk statistics can be used for research and further decision-making.

The architecture separates data retrieval from calculations. This makes it easier to test mathematical functions with known inputs without requiring a live market-data request.

---

## 5. Detailed module documentation

### Module A — Market data ingestion

**File:** `data/fetcher.py`

**Owner:** Developer 1 — Data Pipeline Engineer

#### Purpose

This module is the entry point for external market information. It retrieves the data required by the pricing and analytics components and prepares it in a consistent structure.

#### Responsibilities

- Accept an underlying ticker symbol.
- Retrieve the current or latest available underlying spot price, \(S\).
- Retrieve available option expiration dates.
- Retrieve call and put option chains for a selected expiration.
- Preserve relevant contract fields, such as strike, bid, ask, last price, volume, and open interest, when supplied by the data provider.
- Obtain or prepare a risk-free-rate input, \(r\), using the rate source selected for the project.
- Convert raw provider output into clean pandas DataFrames.
- Handle unavailable tickers, invalid expiration dates, empty chains, missing values, and provider errors.

#### Data flow

**Inputs:**
- Ticker symbol, for example `AAPL` or `SPY`.
- Selected expiration date.

**Expected outputs:**
- Underlying spot-price information.
- The selected expiration.
- A call-options DataFrame.
- A put-options DataFrame.
- A risk-free-rate value or clearly identified rate input.

The precise return signature should be agreed by the team and documented in the function docstrings before the modules are integrated.

#### Data-quality requirements

The module should:

- Convert price, strike, volume, and open-interest columns to appropriate numeric types.
- Identify missing or invalid quotes rather than silently treating them as valid prices.
- Avoid using a bid/ask midpoint when the quote is incomplete or invalid.
- Retain enough information to identify the original option contract.
- Make it clear whether a price is a last-traded price, bid, ask, or calculated midpoint.
- Handle market-data delays and API failures gracefully.

#### Why this module matters

All later calculations depend on the quality of the inputs. A bad quote, incorrect expiry, or inconsistent rate can produce misleading theoretical prices, implied volatilities, and apparent mispricing signals. The data layer should therefore validate and document its output rather than merely download raw data.

---

### Module B — Options pricing and Greeks

**File:** `core/pricing.py`

**Owner:** Developer 2 — Pricing and Core Greeks Quantitative Developer

#### Purpose

This module calculates theoretical European option values and their principal risk sensitivities using the Black–Scholes framework.

#### Inputs

- \(S\): current underlying spot price.
- \(K\): option strike price.
- \(T\): time to expiration, expressed in years.
- \(r\): continuously compounded risk-free interest rate.
- \(\sigma\): annualized volatility.
- Option type: call or put.

All modules must use consistent units. In particular, time must be in years and volatility must be annualized in decimal form (for example, `0.20` for 20%).

#### Black–Scholes calculations

For a non-dividend-paying underlying under the standard Black–Scholes assumptions:

\[
d_1 =
\frac{\ln(S/K)+(r+\frac{1}{2}\sigma^2)T}
{\sigma\sqrt{T}}
\]

\[
d_2 = d_1-\sigma\sqrt{T}
\]

The European call price is:

\[
C = S N(d_1)-K e^{-rT}N(d_2)
\]

The European put price is:

\[
P = K e^{-rT}N(-d_2)-S N(-d_1)
\]

Here, \(N(x)\) is the cumulative distribution function of the standard normal distribution.

#### Greeks

The module calculates the standard Greeks used to describe how option value responds to changes in model inputs.

- **Delta (\(\Delta\)):** sensitivity of option value to a small change in the underlying price.
- **Gamma (\(\Gamma\)):** rate of change of Delta with respect to the underlying price.
- **Theta (\(\Theta\)):** sensitivity to the passage of time. The implementation must document whether it reports calendar-day or annualized Theta and its sign convention.
- **Vega (\(\mathcal{V}\)):** sensitivity to a change in volatility. The implementation must document whether Vega is reported per 1.00 volatility unit or per one percentage point.
- **Rho (\(\rho\)):** sensitivity to a change in the risk-free interest rate.

For a standard European call in the no-dividend Black–Scholes model:

\[
\Delta_{\text{call}}=N(d_1)
\]

\[
\Gamma=\frac{n(d_1)}{S\sigma\sqrt{T}}
\]

\[
\text{Vega}=S n(d_1)\sqrt{T}
\]

where \(n(x)\) is the standard normal probability density function. Put Greeks and the remaining call Greeks should be implemented using the corresponding analytical formulas.

#### Expected outputs

- Theoretical call or put price.
- Delta, Gamma, Theta, Vega, and Rho.
- Results that can be calculated for individual contracts or efficiently across arrays of strikes.

#### Implementation considerations

- Use `scipy.stats.norm` for the normal CDF and PDF where appropriate.
- Use NumPy operations to support vectorized calculations.
- Validate positive spot and strike prices, non-negative time to expiry, and valid volatility inputs.
- Handle expiry and near-zero-volatility cases explicitly to avoid division-by-zero errors.
- Keep the pricing formulas independent of the data provider so they can be unit-tested with fixed inputs.

#### Why this module matters

The pricing module is the mathematical foundation for several other parts of the project. The mispricing scanner needs theoretical values, the sensitivity module needs Greeks, and other analysis may use the pricing model as a reference.

---

### Module C — Implied volatility and smile calibration

**File:** `core/implied_vol.py`

**Owner:** Developer 3 — Implied Volatility and Smile Calibration Specialist

#### Purpose

This module estimates the volatility implied by an observed market option price and analyzes how that volatility varies across strike prices.

The Black–Scholes pricing equation gives an option price when volatility is supplied. Implied-volatility calculation works in the opposite direction: it searches for the volatility that makes the model price equal the observed market price.

#### Mathematical objective

Find \(\sigma_{\text{implied}}\) such that:

\[
f(\sigma)=
\text{BlackScholesPrice}(S,K,T,r,\sigma)
-\text{MarketPrice}=0
\]

#### Numerical methods

The planned numerical approach uses:

1. **Newton–Raphson:** an iterative method that updates the volatility estimate using the pricing error and its derivative.
2. **Brent's method:** a bracketed root-finding method available through `scipy.optimize`, used as a fallback when a valid bracket can be established and Newton–Raphson does not converge.

The solver should return a clear status for each contract. It should not report a numerical value as a valid implied volatility when the input price is outside the model's feasible bounds or the solver has failed.

#### Smile and skew analysis

After estimating implied volatility for valid contracts, the module organizes the results by strike. Plotting implied volatility against strike helps reveal the market's volatility smile or skew.

The analysis can help users observe:

- Differences between implied volatility at lower and higher strikes.
- The relationship between implied volatility and moneyness.
- Whether call and put observations produce consistent patterns.
- Potential outliers caused by stale, wide, or illiquid quotes.

#### Expected inputs and outputs

**Inputs:**
- Spot price.
- Strike prices.
- Time to expiration.
- Risk-free rate.
- Option type.
- Observed market option prices.

**Expected outputs:**
- Implied-volatility estimate for each valid contract.
- Convergence or failure status.
- Strike-level data suitable for plotting a smile or skew.
- Optional fitted or smoothed curve data, if the calibration method is implemented.

#### Implementation considerations

- Use a consistent market-price convention, preferably a validated midpoint where both bid and ask are usable.
- Check no-arbitrage price bounds before attempting numerical inversion.
- Handle very low time value and deep in-the-money or out-of-the-money contracts carefully.
- Set iteration limits and convergence tolerances.
- Reject or label failed estimates instead of silently filling them with arbitrary values.
- Keep raw IV estimates available so that smoothing does not hide unusual observations.

#### Why this module matters

Implied volatility is a market-implied measure of uncertainty. Its variation across strikes provides information that a single constant-volatility model input cannot represent.

---

### Module D — Mispricing and relative-value scanner

**File:** `analytics/arbitrage.py`

**Owner:** Developer 4 — Quantitative Arbitrage and Mispricing Analyst

#### Purpose

This module compares observed market option prices with theoretical Black–Scholes prices and identifies contracts that meet configurable discrepancy and liquidity criteria.

The scanner is intended to identify **potential mispricing or relative-value candidates** for further investigation. A price difference alone does not prove that an executable or risk-free arbitrage exists.

#### Main calculation

For each option contract:

\[
\text{Residual}
=
\text{Market Mid Price}
-
\text{Theoretical Black–Scholes Price}
\]

Interpretation:

- A positive residual means the observed midpoint is above the model estimate.
- A negative residual means the observed midpoint is below the model estimate.
- A residual close to zero means the observed midpoint is close to the model estimate.

The residual is meaningful only when the market quote and model inputs are valid and comparable.

#### Processing steps

1. Receive cleaned option-chain data.
2. Obtain the required model inputs, including spot, strike, expiry, rate, and volatility.
3. Calculate theoretical prices using the pricing module.
4. Calculate a valid market midpoint from bid and ask quotes.
5. Calculate the price residual for each contract.
6. Apply configurable liquidity and quote-quality filters.
7. Rank or return contracts that pass the selected thresholds.

The volatility input used for theoretical pricing must be documented. The project description allows for historical volatility or a smoothed at-the-money volatility baseline; these are different assumptions and can produce different residuals. The implementation should make the selected method explicit.

#### Potential filters

- Minimum open interest.
- Minimum trading volume, if used.
- Maximum bid–ask spread in absolute or percentage terms.
- Valid positive bid/ask quotes.
- Minimum absolute or relative price residual.
- Exclusion of expired or invalid contracts.

Thresholds should be configurable rather than hidden inside the calculation.

#### Expected outputs

A structured table or DataFrame containing, where available:

- Contract identifier and option type.
- Strike and expiration.
- Bid, ask, and market midpoint.
- Theoretical model price.
- Price residual.
- Liquidity fields such as volume and open interest.
- Filter status or reason for exclusion.

#### Important interpretation

A model-price residual can result from differences in volatility assumptions, dividends, exercise style, transaction costs, stale quotes, wide spreads, or model limitations. Before treating a candidate as a trading opportunity, users must investigate executable bid/ask prices, contract specifications, hedging costs, and relevant market risks.

---

### Module E — Greeks sensitivity matrix

**File:** `analytics/greeks_matrix.py`

**Owner:** Developer 2 — Pricing and Core Greeks Quantitative Developer

#### Purpose

This module shows how option Greeks change when important market variables change. It builds on the pricing functions rather than duplicating the Black–Scholes formulas.

#### Scenario design

The project plan calls for a two-dimensional grid that varies:

- Underlying spot price around its current level, with an initial range of approximately \(\pm 15\%\).
- Time to expiration, moving from the current maturity toward expiry.

For each point in the grid, the module recalculates the option Greeks using the same pricing implementation.

#### Outputs

The module should provide structured grid data for:

- Delta surface.
- Gamma surface.
- Vega surface.
- Theta surface.

These outputs can be used to generate heatmaps or three-dimensional surfaces with a plotting library such as Plotly, if visualization is included in the implementation.

#### Analytical value

Sensitivity surfaces help users examine how risk exposures change across scenarios. For example:

- Gamma can become concentrated around at-the-money strikes as expiry approaches.
- Delta changes as the underlying moves relative to the strike.
- Vega generally changes with maturity and moneyness.
- Theta behavior changes as the option approaches expiration.

The module should describe these as model sensitivities, not guaranteed future outcomes.

#### Implementation considerations

- Reuse the functions from `core/pricing.py`.
- Keep scenario ranges and grid resolution configurable.
- Define how the module handles zero time to expiry.
- Use consistent units and Greek conventions.
- Return the numerical grid separately from plotting code so it can be tested without rendering a chart.

---

### Module F — Monte Carlo simulation and risk engine

**Files:** `core/monte_carlo.py` and `analytics/risk_engine.py`

**Owner:** Developer 5 — Stochastic Modeling and Risk Quant

These files work together but have separate responsibilities: the core module generates simulated market paths, while the analytics module summarizes risk from the simulated outcomes and applies stress scenarios.

#### Part 1: Monte Carlo simulation

**File:** `core/monte_carlo.py`

**Purpose:** Generate many possible underlying-price paths using a specified stochastic model.

The project plan identifies Geometric Brownian Motion (GBM) as the baseline model. A GBM process can be expressed as:

\[
dS_t=\mu S_t\,dt+\sigma S_t\,dW_t
\]

where:

- \(S_t\) is the underlying price at time \(t\).
- \(\mu\) is the assumed drift.
- \(\sigma\) is annualized volatility.
- \(W_t\) is a standard Brownian motion.

The module should accept simulation parameters such as the starting spot, drift, volatility, time horizon, number of time steps, and number of paths.

**Expected outputs:**
- Simulated price paths.
- Terminal simulated prices.
- Simulation metadata, including the parameters used.

**Performance:** Numba can be used to accelerate computationally intensive loops. Performance improvements should be measured against a correct NumPy or Python baseline.

The original project plan also mentions Heston stochastic volatility. Treat Heston as an additional model only if it is implemented; the repository structure currently names `monte_carlo.py` but does not establish a separate Heston module or confirm that Heston simulation is complete.

#### Part 2: Risk analytics and stress testing

**File:** `analytics/risk_engine.py`

**Purpose:** Convert simulated outcomes into risk summaries and scenario-based stress results.

The planned risk measures include:

- **Value at Risk (VaR):** a loss threshold at a chosen confidence level. For example, 95% VaR is commonly interpreted as a loss threshold that is exceeded in approximately 5% of outcomes, subject to the chosen loss convention and estimation method.
- **Conditional Value at Risk (CVaR) / Expected Shortfall:** the average loss in the tail beyond the selected VaR threshold, using a clearly documented quantile convention.

The risk engine may also evaluate event shocks that combine an underlying-price move with an implied-volatility change. The shock sizes and direction should be explicit inputs rather than assumed market forecasts.

**Expected outputs:**
- Loss or profit-and-loss distribution.
- VaR at selected confidence levels.
- CVaR / Expected Shortfall at matching levels.
- Stress-scenario results and the parameters used to create them.

#### Important modeling considerations

- GBM assumes constant drift and volatility and does not reproduce all real-market features.
- Simulated paths are conditional on model assumptions and input parameters.
- VaR and CVaR estimates depend on the number of simulations, horizon, confidence level, and P&L definition.
- A price-path simulator is not automatically a complete options-portfolio risk engine. Portfolio valuation under each simulated scenario must be defined if the project is to report portfolio-level P&L.
- Event shocks are scenarios, not probability forecasts.

---

## 6. End-to-end workflow

A typical analysis is intended to follow this sequence:

1. **Select an underlying and expiration.** The user chooses a ticker such as `AAPL` or `SPY` and an available option expiration.
2. **Fetch market data.** `data/fetcher.py` retrieves the underlying price and the selected calls and puts.
3. **Validate and standardize quotes.** The data layer checks fields, types, missing values, and quote quality.
4. **Calculate theoretical prices and Greeks.** `core/pricing.py` evaluates the Black–Scholes model for eligible contracts.
5. **Calculate implied volatility.** `core/implied_vol.py` estimates the volatility implied by each valid market option price and prepares strike-level smile data.
6. **Scan for price discrepancies.** `analytics/arbitrage.py` compares market midpoints with theoretical prices and applies liquidity and residual filters.
7. **Generate sensitivity grids.** `analytics/greeks_matrix.py` recalculates Greeks over the selected spot and time scenarios.
8. **Simulate market paths.** `core/monte_carlo.py` generates paths using the selected stochastic assumptions.
9. **Summarize risk.** `analytics/risk_engine.py` calculates loss-distribution measures and evaluates configured stress scenarios.
10. **Review and interpret results.** The user examines outputs, assumptions, data quality, and limitations before drawing conclusions.

### Example analysis scenario

A user selects `AAPL` and an available expiration date. The platform retrieves the current underlying price and the corresponding option chain. For each eligible contract, it calculates a theoretical price and Greeks, then attempts to infer implied volatility from the market quote. The scanner compares the observed midpoint with the model estimate, excluding contracts that fail quote-quality or liquidity filters. Separately, the sensitivity module shows how the Greeks change under spot and time scenarios, while the Monte Carlo and risk modules estimate the range of outcomes under the configured simulation assumptions.

This example describes the intended system workflow; it does not claim that a live trading signal or profitable strategy is produced.

---

## 7. Team responsibilities and work division

The project is divided among **five developers**. Each developer owns a defined area, its implementation, and its unit tests. Shared interfaces and integration tests are team responsibilities.

### Developer 1 — Data Pipeline Engineer

**Primary ownership:**
- `data/fetcher.py`
- `data/__init__.py`

**Objective:** Deliver reliable, standardized market data for every downstream module.

**Tasks:**
1. Build the `yfinance` integration for underlying prices and option-chain retrieval.
2. Retrieve available expiration dates and validate the selected expiry.
3. Retrieve call and put chains, retaining relevant strike, bid, ask, last-price, volume, and open-interest fields when available.
4. Obtain or prepare the risk-free-rate input and document its source and convention.
5. Clean column names, numeric types, missing values, and invalid quotes.
6. Return a documented and consistent data structure.
7. Add error handling for invalid tickers, unavailable expirations, empty chains, network failures, and provider limitations.
8. Write tests using mocked provider responses so tests do not depend on a live API.

**Inputs:**
- Ticker symbol.
- Expiration date.

**Deliverables:**
- Reusable market-data fetching functions.
- Standardized underlying and option-chain outputs.
- Documented errors and data-quality behavior.
- Unit tests for successful and unsuccessful retrieval cases.

**Implementation focus:** Keep the data provider isolated in this module. Other modules should receive prepared data rather than calling `yfinance` directly.

---

### Developer 2 — Pricing and Core Greeks Quantitative Developer

**Primary ownership:**
- `core/pricing.py`
- Coordination with `analytics/greeks_matrix.py`

**Objective:** Implement and validate the mathematical pricing foundation used throughout the platform.

**Tasks:**
1. Implement Black–Scholes European call pricing.
2. Implement Black–Scholes European put pricing.
3. Implement analytical Delta, Gamma, Theta, Vega, and Rho.
4. Support scalar and vectorized NumPy inputs where practical.
5. Define and document units, sign conventions, and edge-case behavior.
6. Expose reusable functions for the sensitivity-matrix module.
7. Validate results against independently calculated reference cases.
8. Add unit tests for calls, puts, different maturities, volatility levels, and boundary conditions.

**Inputs:**
- Spot price.
- Strike price.
- Time to expiration.
- Risk-free rate.
- Volatility.
- Option type.

**Deliverables:**
- Pricing functions.
- Greek calculation functions.
- Documented input and output conventions.
- Unit tests and reference examples.

**Implementation focus:** Use `scipy.stats.norm` for normal-distribution calculations and NumPy for vectorized processing. Make sure the formulas and reported Greek units are consistent across the entire project.

---

### Developer 3 — Implied Volatility and Smile Calibration Specialist

**Primary ownership:**
- `core/implied_vol.py`

**Objective:** Build a robust numerical solver that infers volatility from observed option prices and prepares volatility-smile data.

**Tasks:**
1. Implement the implied-volatility objective function using the shared pricing module.
2. Implement Newton–Raphson with a defined tolerance and iteration limit.
3. Add Brent's method as a fallback when a valid bracket is available.
4. Validate market prices against theoretical no-arbitrage bounds before solving.
5. Handle non-convergence and invalid contracts with explicit status information.
6. Calculate implied volatility across eligible strikes.
7. Prepare strike-level smile/skew data.
8. If curve fitting is included, document the selected fitting method and keep raw estimates available for comparison.
9. Add tests for normal convergence, fallback behavior, invalid prices, and difficult low-time-value contracts.

**Inputs:**
- Spot price.
- Strike prices.
- Time to expiration.
- Risk-free rate.
- Option type.
- Observed market option prices.

**Deliverables:**
- Implied-volatility solver.
- Per-contract result and convergence status.
- Strike-level IV dataset.
- Tests for numerical edge cases.

**Implementation focus:** Newton–Raphson can fail when Vega is very small or the starting estimate is poor. The fallback must be controlled and must not conceal invalid inputs or an unbracketed root.

---

### Developer 4 — Quantitative Arbitrage and Mispricing Analyst

**Primary ownership:**
- `analytics/arbitrage.py`

**Objective:** Build a transparent scanner that identifies option contracts with significant market/model price differences after quote-quality and liquidity checks.

**Tasks:**
1. Accept standardized market data and model outputs.
2. Obtain theoretical prices using the shared pricing functions.
3. Calculate market midpoints from valid bid/ask quotes.
4. Calculate signed price residuals.
5. Define configurable liquidity and quote-quality filters.
6. Apply minimum open-interest and maximum bid–ask-spread thresholds.
7. Support configurable residual thresholds and ranking.
8. Return a structured table that explains why each contract was included or excluded.
9. Test positive, negative, and negligible residuals as well as invalid and illiquid quotes.

**Inputs:**
- Clean market option-chain data.
- Theoretical model prices.
- Volatility input or a documented volatility baseline.
- Filter thresholds.

**Deliverables:**
- Mispricing-scanner functions.
- Filtered candidate DataFrame.
- Documented residual and filter definitions.
- Unit tests for filtering and residual calculations.

**Implementation focus:** Avoid labeling every model-price difference as arbitrage. The scanner should distinguish a mathematical discrepancy from a potentially executable opportunity and should make its assumptions visible.

---

### Developer 5 — Stochastic Modeling and Risk Quant

**Primary ownership:**
- `core/monte_carlo.py`
- `analytics/risk_engine.py`

**Objective:** Build the simulation engine and convert its outputs into interpretable risk measures and stress-test results.

**Tasks:**
1. Implement GBM price-path simulation.
2. Define simulation inputs, including spot, drift, volatility, horizon, number of steps, and number of paths.
3. Support reproducible simulations through a configurable random seed where practical.
4. Use Numba to accelerate computationally expensive routines where it provides a measurable benefit.
5. Calculate terminal-price and, where defined, portfolio-loss distributions.
6. Implement VaR and CVaR / Expected Shortfall with explicit confidence-level and loss conventions.
7. Implement configurable event-shock scenarios for spot and volatility changes.
8. Validate simulation output shape, numerical stability, and risk-statistic calculations.
9. Document which stochastic models are actually implemented; treat Heston as an extension unless it is completed and tested.

**Inputs:**
- Initial spot price.
- Drift assumption.
- Volatility assumption.
- Simulation horizon and time steps.
- Number of paths.
- Optional random seed and stress-scenario parameters.
- Portfolio valuation/P&L definition if portfolio-level risk is calculated.

**Deliverables:**
- Monte Carlo path simulator.
- Risk-statistics functions.
- Stress-scenario functions.
- Performance and correctness tests.
- Documentation of model assumptions.

**Implementation focus:** Correctness and reproducibility come before speed. Numba optimization should preserve the behavior of the reference implementation.

---

### Shared responsibilities and integration

The five ownership areas are separate, but the platform will only work correctly if the modules agree on shared conventions.

All developers should coordinate on:

- Function names and signatures.
- DataFrame column names and data types.
- Time-to-expiry units.
- Interest-rate and volatility conventions.
- Option-type representation (`call`/`put`, or another agreed standard).
- Handling of missing and invalid data.
- Error and convergence reporting.
- Greek units and sign conventions.
- Test fixtures and integration-test expectations.

Each developer should write unit tests for their own module. The team should also create integration tests that pass representative data from the fetcher through pricing, implied volatility, and analytics.

---

## 8. Module interfaces and integration expectations

The repository structure identifies module ownership, but the exact function signatures are not specified in the project description. The team should agree on these before parallel implementation is completed.

The following are **recommended interface expectations**, not claims that these exact functions already exist:

| Module | Receives | Provides |
|---|---|---|
| `data/fetcher.py` | Ticker and expiry | Spot, rate input, standardized calls and puts |
| `core/pricing.py` | Spot, strike, time, rate, volatility, option type | Theoretical price and Greeks |
| `core/implied_vol.py` | Market price and contract/model inputs | IV estimate, status, strike-level IV data |
| `analytics/arbitrage.py` | Market quotes, model prices, filter settings | Residuals and filtered candidate table |
| `analytics/greeks_matrix.py` | Contract inputs and scenario ranges | Greek grids for each scenario |
| `core/monte_carlo.py` | Initial price and simulation settings | Simulated paths and terminal values |
| `analytics/risk_engine.py` | Simulated outcomes, P&L definition, confidence levels, shocks | VaR, CVaR, and stress results |

### Recommended shared conventions

- **Prices:** use the same currency and price units throughout a calculation.
- **Volatility:** use annualized decimal values, such as `0.25` for 25%.
- **Rates:** use decimal rates, such as `0.05` for 5%, and document compounding conventions.
- **Time:** express time to expiration in years for Black–Scholes calculations.
- **Option type:** use one consistent representation throughout the project.
- **Missing values:** represent unavailable data explicitly; do not silently replace it with zero.
- **Market price:** distinguish bid, ask, last price, and midpoint.
- **Solver results:** include convergence status alongside numerical values.
- **Randomness:** allow a seed to be specified for reproducible simulation tests.

---

## 9. Technology stack

The project is designed around the following Python ecosystem:

| Technology | Intended use |
|---|---|
| Python | Main programming language |
| NumPy | Numerical arrays and vectorized calculations |
| SciPy | Statistical distributions and numerical optimization/root finding |
| pandas | Market-data tables and analytical outputs |
| yfinance | Retrieval of underlying and options-chain information |
| Numba | Optional acceleration of simulation routines |
| Plotly | Interactive charts and sensitivity visualizations, where implemented |
| unittest | Automated unit-test framework included with Python |

The dependencies should be maintained in `requirements.txt`. The actual installed versions should be pinned or constrained there to support reproducible environments.

---

## 10. Installation and setup

### Prerequisites

- Python installed and available from the command line.
- Git installed.
- Internet access for market-data retrieval.
- A terminal or command prompt.

### Step 1 — Clone the repository

```bash
git clone https://github.com/Amber-s-art/QuantOptions-Engine.git
cd QuantOptions-Engine
```

### Step 2 — Create and activate a virtual environment

```bash
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```cmd
.venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### Step 3 — Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If a dependency is missing from `requirements.txt`, add it to that file rather than relying on a developer's manually configured environment.

### Step 4 — Check the project structure

Confirm that the `data/`, `core/`, and `analytics/` packages and their expected Python files are present.

---

## 11. Running the project and tests

The current project description does not define a single command-line entry point or a user interface. Run modules through the entry point or example scripts provided by the implementation. Avoid assuming that a command exists until it has been added to the repository.

### Run the unit-test suite

```bash
python -m unittest discover
```

If tests are stored in a dedicated `tests/` directory, the team can use:

```bash
python -m unittest discover -s tests
```

### Recommended development checks

- Run the relevant module's unit tests after each change.
- Run the complete test suite before merging changes.
- Test mathematical functions with fixed inputs.
- Mock market-data responses so routine tests do not depend on external availability.
- Use fixed random seeds in simulation tests where reproducibility is required.
- Test the full data-to-analytics workflow with a small, known fixture.

---

## 12. Validation and testing plan

A quantitative platform needs more than tests that check whether functions run. It should also check mathematical properties, input validation, and integration behavior.

### Data ingestion tests

- Valid ticker and available expiration.
- Invalid ticker or unavailable expiration.
- Empty or incomplete option chain.
- Missing bid, ask, volume, or open-interest values.
- Invalid numeric fields and provider/network errors.

### Pricing and Greeks tests

- Compare selected call and put prices with trusted reference calculations.
- Verify put-call parity under the model's stated assumptions.
- Check that prices are non-negative and respect relevant theoretical bounds.
- Check expected relationships such as Gamma being non-negative in the standard Black–Scholes model.
- Test zero time to expiry and other boundary conditions.
- Confirm the documented units and signs of Theta, Vega, and Rho.

### Implied-volatility tests

- Generate a model price from a known volatility and check that the solver recovers it within tolerance.
- Test Newton–Raphson convergence.
- Test fallback behavior.
- Test prices outside valid bounds.
- Test cases where Vega is very small and convergence is difficult.

### Mispricing scanner tests

- Verify residual sign and magnitude.
- Verify midpoint calculations.
- Verify liquidity filters.
- Verify behavior for crossed, missing, or excessively wide quotes.
- Confirm that filter thresholds are configurable.

### Sensitivity-matrix tests

- Check grid dimensions and scenario values.
- Confirm that each grid point uses the shared pricing implementation.
- Test behavior at or near expiry.
- Verify that output arrays contain valid values or explicit invalid markers.

### Monte Carlo and risk tests

- Check path-array dimensions and starting values.
- Check reproducibility with a fixed seed.
- Compare a small simulation against a simple reference implementation.
- Verify VaR and CVaR calculations on a known sample loss array.
- Check stress-scenario inputs and outputs.
- Benchmark optimized code against the reference implementation.

### Integration tests

At minimum, test the following flow with fixed or mocked data:

```text
Prepared option chain
       |
       v
Pricing and Greeks
       |
       v
Implied volatility
       |
       v
Mispricing scan
```

Test the sensitivity and Monte Carlo/risk workflows separately, then add broader integration coverage as the project grows.

---

## 13. Assumptions, limitations, and risk considerations

The outputs of this platform depend on both the quality of market data and the assumptions of the selected models.

### Black–Scholes assumptions

The standard Black–Scholes formulas used here are based on simplifying assumptions, including a lognormal underlying-price process, constant volatility and interest rate over the option's life, and European exercise. The basic formulas shown above assume no dividends. If dividend-paying underlyings are supported, the model must account for the dividend yield or other appropriate dividend treatment.

### Market-data limitations

- Data supplied by an external provider may be delayed, incomplete, or unavailable.
- Last-traded prices may be stale and should not automatically be treated as executable prices.
- Bid/ask spreads can be wide, particularly for illiquid contracts.
- Open interest and volume may not fully describe current liquidity.
- The rate input and its maturity should be appropriate for the option's time horizon.

### Model and scanner limitations

- A Black–Scholes price is a model estimate, not a guaranteed market value.
- Implied volatility can be unstable for illiquid options or prices with little time value.
- A volatility smile fit may be sensitive to outliers and the chosen fitting method.
- A market/model residual may reflect model assumptions or quote quality rather than a tradable opportunity.
- Transaction costs, slippage, financing, dividends, exercise style, and hedging costs can materially affect a strategy.

### Simulation limitations

- GBM does not fully capture jumps, volatility clustering, changing correlations, or all market stress behavior.
- Monte Carlo results depend on assumptions and simulation settings.
- VaR does not describe the size of losses beyond its quantile threshold; CVaR is intended to summarize that tail.
- Stress tests are conditional scenarios and should not be interpreted as forecasts.

**Use the platform for research, education, and analytical support. It does not provide financial advice or guarantee profitable trades.**

---

## 14. Suggested development sequence

A practical implementation order is:

1. **Agree on interfaces:** settle common data fields, function signatures, units, and error conventions.
2. **Build the data layer:** provide reliable sample and live-data structures.
3. **Build and validate pricing:** establish the shared Black–Scholes and Greeks foundation.
4. **Build implied volatility:** use the pricing module and validate numerical convergence.
5. **Build analytics:** implement the mispricing scanner and sensitivity grids using shared outputs.
6. **Build simulation and risk:** implement the baseline GBM simulator, then risk summaries and stress scenarios.
7. **Integrate the modules:** run the end-to-end workflow on a controlled sample dataset.
8. **Validate and document:** run the complete test suite, record model assumptions, and update examples to match the actual implementation.

The team can work in parallel after the shared interfaces are agreed. Integration should happen continuously rather than being postponed until every module is complete.

---

## 15. Contributing

When contributing to QuantOptions-Engine:

1. Work within the module assigned to you unless a shared change has been agreed.
2. Preserve the documented interfaces or coordinate interface changes with the team.
3. Add or update unit tests for every behavior change.
4. Keep market-data retrieval separate from mathematical calculations.
5. Document assumptions, units, conventions, and failure behavior.
6. Do not silently replace invalid market data with fabricated values.
7. Run the test suite before submitting changes.
8. Update this README whenever the architecture, module responsibilities, setup, or supported functionality changes.

### Project status and implementation accuracy

This README documents the intended architecture, responsibilities, and workflow based on the project plan. A listed capability should be considered complete only when its implementation, tests, and integration are present in the repository. In particular, optional extensions such as Heston simulation, fitted volatility curves, and portfolio-level P&L stress testing should be labeled according to their actual implementation status.

---

**QuantOptions-Engine** — connecting options data, quantitative models, market analysis, and risk simulation in one modular Python project.
