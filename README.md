```markdown
# QuantOptions-Engine 🚀
*A Unified Quantitative Options Pricing, Volatility Calibration, Arbitrage Scanning, and Risk Management Platform*

---

## 1. Project Overview
**QuantOptions-Engine** is a production-grade, modular quantitative finance toolkit designed to bridge the gap between theoretical derivative pricing models and real-world market execution. 

While most academic repositories isolate individual components, this platform unifies **Black-Scholes pricing**, exact **analytical Greeks**, **implied volatility smile calibration**, **empirical mispricing arbitrage scanning**, and **stochastic Monte Carlo risk stress-testing** into a single, cohesive, decoupled Python engine powered by live market feeds.

---

## 2. Core Architectural Modules & Directory Structure

The repository is organized into a clean, modular structure separating data ingestion, mathematical core logic, advanced analytics, and execution workflows:

```text
QuantOptions-Engine/
│
├── data/
│   ├── __init__.py
│   └── fetcher.py             # yfinance API wrapper for live spot, options chains, & rates
│
├── core/
│   ├── __init__.py
│   ├── pricing.py             # Analytical Black-Scholes formulas & exact Greeks
│   ├── implied_vol.py         # Newton-Raphson IV solver & Volatility Smile/SVI fitting
│   └── monte_carlo.py         # Numba-accelerated GBM simulation & VaR/CVaR risk engine
│
├── analytics/
│   ├── __init__.py
│   ├── arbitrage.py           # Actual vs. Theoretical price discrepancy scanner
│   ├── greeks_matrix.py       # 2D sensitivity surface generator (Delta, Gamma, Vega, Theta)
│   └── risk_engine.py         # IV Rank/Percentile tracker & event-shock stress testing
│
├── requirements.txt           # Project dependencies (numpy, scipy, pandas, yfinance, numba, plotly)
└── README.md                  # Comprehensive project documentation

```

---

## 3. Module-by-Module Technical Breakdown

### Module A: Data Ingestion Pipeline (`data/fetcher.py`)

* **Objective:** Pull real-time market data to feed the analytical engine.
* **Mechanism:** Utilizes `yfinance` to fetch the underlying spot price ($S$), available expiration dates, and full option chains (Calls and Puts with Strikes, Bids, Asks, Volume, and Open Interest). Programmatically estimates the risk-free rate ($r$) using benchmark Treasury yields or short-term bills.
* **Key Output:** Standardized, cleaned pandas DataFrames containing strike-level market quotes ready for volatility inversion and arbitrage comparison.

### Module B: Core Pricing & Greeks Engine (`core/pricing.py`)

* **Objective:** Compute exact theoretical fair value and risk sensitivities.
* **Mathematical Formulation:**

$$d_1 = \frac{\ln(S/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$


$$\text{Call Price } (C) = S N(d_1) - K e^{-rT} N(d_2)$$


* **Analytical Greeks:**
* **Delta ($\Delta$):** $\frac{\partial C}{\partial S} = N(d_1)$ (Call sensitivity to spot price movement)
* **Gamma ($\Gamma$):** $\frac{\partial^2 C}{\partial S^2} = \frac{N'(d_1)}{S \sigma \sqrt{T}}$ (Rate of change of Delta)
* **Theta ($\Theta$):** Daily time decay profile
* **Vega ($\mathcal{V}$):** Sensitivity to volatility shifts: $S N'(d_1) \sqrt{T}$



### Module C: Implied Volatility & Smile Calibration (`core/implied_vol.py`)

* **Objective:** Back out market expectations and map the volatility smile/skew.
* **Mechanism:** Since Black-Scholes cannot be inverted algebraically for volatility, the module implements a robust root-finding algorithm (**Newton-Raphson** with fallback to **Brent's method** via `scipy.optimize`) to solve for $\sigma_{\text{implied}}$ where $\text{BS\_Price}(\sigma) - \text{Market\_Price} = 0$.
* **Analysis:** Plots implied volatility across strikes ($K$) to visualize the volatility smile/skew, capturing market tail-risk pricing vs. flat log-normal expectations.

### Module D: Empirical Arbitrage & Mispricing Scanner (`analytics/arbitrage.py`)

* **Objective:** Spot statistical edge between model theory and market reality.
* **Mechanism:**
1. Computes theoretical Black-Scholes prices using historical volatility or a smoothed ATM volatility baseline.
2. Calculates the **Price Residual**: $\text{Residual} = \text{Market Mid Price} - \text{Theoretical BS Price}$.
3. Filters options using strict liquidity and statistical thresholds (e.g., minimum open interest, maximum bid-ask spread percentage) to flag mispriced contracts and potential relative-value arbitrage opportunities.



### Module E: Greeks Sensitivity Matrix (`analytics/greeks_matrix.py`)

* **Objective:** Model how risk exposures mutate across market shifts.
* **Mechanism:** Generates a 2D grid varying underlying spot price ($\pm 15\%$) and time to expiration ($T \to 0$).
* **Trading Value:** Identifies **Gamma inflection points**—showing precisely when a delta-hedged portfolio's tracking error explodes or when time decay ($\Theta$) accelerates exponentially.

### Module F: Stochastic Simulation & Risk Engine (`core/monte_carlo.py` & `analytics/risk_engine.py`)

* **Objective:** Stress-test positions against tail events.
* **Mechanism:** Simulates thousands of price paths using Numba-accelerated **Geometric Brownian Motion (GBM)** or Heston stochastic volatility. Computes tail-risk metrics: **Value at Risk (VaR)** and **Expected Shortfall (CVaR)**.

---

## 4. End-to-End Execution Workflow

1. **Ticker & Expiry Selection:** User specifies the target asset (e.g., AAPL, SPY) and expiration date.
2. **Data Pull (`fetcher.py`):** The pipeline fetches live spot prices and option chains.
3. **IV Calculation (`implied_vol.py`):** Solves for implied volatility across all strikes to construct the live Volatility Smile.
4. **Arbitrage Scan (`arbitrage.py`):** Compares market mid-prices against theoretical prices to isolate mispriced options.
5. **Risk & Sensitivity Rendering (`greeks_matrix.py` & `monte_carlo.py`):** Computes Greek sensitivity matrices and Monte Carlo path distributions.
6. **Execution Edge:** Flags actionable setups (e.g., overvalued IV rank + negative pricing residual = optimal credit spread candidate).

---

## 5. Team Developer Workload & Task Distribution

To ensure clean execution and independent unit testing, the engineering workload is divided among **5 specialized developers**:

### Dev 1: Data Pipeline Engineer

* **Primary Module:** `data/fetcher.py`, `data/__init__.py`
* **Core Responsibilities:**
* Build API wrappers for live market data ingestion via `yfinance`.
* Clean and structure option chains (Calls/Puts, Strikes, Bids, Asks, Volume, Open Interest).
* Programmatically fetch risk-free rates ($r$) and underlying spot prices ($S$).


* **Inputs / Outputs:**
* *Input:* Ticker symbol, expiration date.
* *Output:* Standardized pandas DataFrames of clean market data.


* **Implementation Focus:** Write robust error handling for missing option chain dates, filter illiquid strikes with zero bid/ask spreads, and ensure rate-limit compliance.

### Dev 2: Pricing & Core Greeks Quantitative Developer

* **Primary Module:** `core/pricing.py`
* **Core Responsibilities:**
* Implement analytical Black-Scholes pricing formulas for European calls and puts.
* Derive and code exact analytical formulas for primary Greeks ($\Delta, \Gamma, \Theta, \mathcal{V}, \rho$).
* Build parameter sensitivity grid generators to track Greek mutation.


* **Inputs / Outputs:**
* *Input:* Spot, Strike, Time to Expiry, Rate, Volatility.
* *Output:* Theoretical fair value and Greek risk vectors.


* **Implementation Focus:** Code the cumulative normal distribution function ($N(x)$) accurately using `scipy.stats.norm` and vectorize operations using `numpy` for high-speed chain processing.

### Dev 3: Implied Volatility & Smile Calibration Specialist

* **Primary Module:** `core/implied_vol.py`
* **Core Responsibilities:**
* Implement robust root-finding algorithms (Newton-Raphson and Brent's method via `scipy.optimize`) to solve for Implied Volatility.
* Construct strike-level Volatility Smile/Skew curves.
* Fit baseline parametric curves to eliminate arbitrage outliers.


* **Inputs / Outputs:**
* *Input:* Market option prices, strike vector, spot price.
* *Output:* Strike-wise Implied Volatility array and smile curve data.


* **Implementation Focus:** Handle edge cases where Newton-Raphson fails to converge (e.g., deep OTM options with zero time value) by falling back to Brent's bracketed root-finding method.

### Dev 4: Quantitative Arbitrage & Mispricing Strategist

* **Primary Module:** `analytics/arbitrage.py`
* **Core Responsibilities:**
* Build the Actual vs. Theoretical price discrepancy scanner.
* Compute price residuals ($\text{Market Price} - \text{BS Model Price}$).
* Implement statistical threshold filters and rolling IV Rank / Percentile calculations to flag relative-value trading opportunities.


* **Inputs / Outputs:**
* *Input:* Live market prices, computed BS prices, historical IV series.
* *Output:* Flagged list of mispriced options and spread candidates.


* **Implementation Focus:** Define strict quantitative filters (minimum open interest, maximum bid-ask spread percentage) to ensure flagged mispricings are genuine arbitrage edges rather than illiquid quotes.

### Dev 5: Stochastic Modeling & Risk Quant

* **Primary Module:** `core/monte_carlo.py`, `analytics/risk_engine.py`
* **Core Responsibilities:**
* Build Numba-accelerated Geometric Brownian Motion (GBM) Monte Carlo path simulators.
* Implement event-shock generators (simulating spot gaps + IV spikes simultaneously).
* Calculate tail-risk metrics: Value at Risk (VaR) and Expected Shortfall (CVaR).


* **Inputs / Outputs:**
* *Input:* Spot, volatility, drift, simulation parameters.
* *Output:* Simulated price paths and portfolio loss distribution metrics.


* **Implementation Focus:** Leverage `numba.jit` compilation loops to optimize random normal number generation across tens of thousands of simulated paths for real-time stress testing.

---

## 6. Getting Started & Installation

1. **Clone the Repository:**
```bash
git clone [https://github.com/Amber-s-art/QuantOptions-Engine.git](https://github.com/Amber-s-art/QuantOptions-Engine.git)
cd QuantOptions-Engine

```


2. **Install Dependencies:**
```bash
pip install -r requirements.txt

```


3. **Run Unit Tests & Modules:**
```bash
python -m unittest discover

```



```

```
