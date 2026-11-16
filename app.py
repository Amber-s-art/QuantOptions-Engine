"""Streamlit interface for QuantOptions-Engine.

Run from the repository root with: streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from data.fetcher import MarketDataError, fetch_market_data, get_available_expirations
from core.pricing import black_scholes_price, calculate_greeks
from core.implied_vol import solve_implied_volatility
from core.monte_carlo import simulate_gbm_paths, price_european_option_mc
from analytics.arbitrage import MispricingFilters, scan_mispricing
from analytics.greeks_matrix import build_greeks_matrix
from analytics.risk_engine import calculate_var_cvar

st.set_page_config(page_title="QuantOptions-Engine", page_icon="📈", layout="wide")

st.markdown("""
<style>
:root { --qoe-amber: #B8860B; }
.block-container {padding-top: 1.6rem; max-width: 1500px;}
[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
div[data-testid="stAlert"] {border-radius: 4px;}
</style>
""", unsafe_allow_html=True)

st.title("QuantOptions-Engine")
st.caption("Options pricing · Implied volatility · Mispricing research · Sensitivity · Monte Carlo risk")
st.info("Research interface only. Market/model differences are not guaranteed arbitrage. No orders are placed by this application.")

with st.sidebar:
    st.header("Market inputs")
    ticker = st.text_input("Ticker", value="AAPL", help="Examples: AAPL, MSFT, SPY. Yahoo Finance symbols are used by the current data adapter.").strip().upper()
    rate_pct = st.number_input("Risk-free rate (%)", min_value=-5.0, max_value=30.0, value=4.0, step=0.25)
    dividend_pct = st.number_input("Continuous dividend yield (%)", min_value=-5.0, max_value=30.0, value=0.0, step=0.25)
    baseline_vol_pct = st.number_input("Baseline model volatility (%)", min_value=0.1, max_value=300.0, value=25.0, step=1.0)
    st.caption("Rates and volatility are entered as percentages and converted to decimals internally.")
    load = st.button("Load market data", type="primary", use_container_width=True)

rate = rate_pct / 100.0
dividend_yield = dividend_pct / 100.0
baseline_vol = baseline_vol_pct / 100.0

@st.cache_data(ttl=60, show_spinner=False)
def load_expiries(symbol: str):
    return get_available_expirations(symbol)

@st.cache_data(ttl=45, show_spinner=False)
def load_snapshot(symbol: str, expiry: str):
    return fetch_market_data(symbol, expiry, risk_free_rate=rate)

if "snapshot" not in st.session_state:
    st.session_state.snapshot = None
if "snapshot_key" not in st.session_state:
    st.session_state.snapshot_key = None

if load:
    try:
        with st.spinner(f"Retrieving {ticker} option expirations..."):
            expiries = load_expiries(ticker)
        if not expiries:
            st.error(f"No listed option expirations were returned for {ticker}.")
        else:
            st.session_state.available_expiries = expiries
            st.session_state.selected_expiry = expiries[0]
    except Exception as exc:
        st.error(f"Market-data request failed: {exc}")

expiries = st.session_state.get("available_expiries", [])
if expiries:
    with st.sidebar:
        selected_expiry = st.selectbox("Option expiration", expiries, index=0)
        fetch_selected = st.button("Fetch selected chain", use_container_width=True)
    if fetch_selected or st.session_state.get("snapshot_key") != (ticker, selected_expiry):
        try:
            with st.spinner(f"Loading {ticker} options for {selected_expiry}..."):
                st.session_state.snapshot = load_snapshot(ticker, selected_expiry)
                st.session_state.snapshot_key = (ticker, selected_expiry)
        except (MarketDataError, ValueError, Exception) as exc:
            st.session_state.snapshot = None
            st.error(f"Could not load option chain: {exc}")

snapshot = st.session_state.snapshot
if snapshot is None:
    st.markdown("### Start an analysis")
    st.write("Enter a ticker in the sidebar and select **Load market data**. The current adapter uses Yahoo Finance through `yfinance`; it is a convenient research feed, not an exchange-authorized real-time feed.")
    st.stop()

spot = float(snapshot.spot)
chain = snapshot.combined_options().copy()
chain["market_mid"] = np.where(
    (chain["bid"] > 0) & (chain["ask"] >= chain["bid"]),
    (chain["bid"] + chain["ask"]) / 2.0,
    np.nan,
)
chain["spread_pct"] = (chain["ask"] - chain["bid"]) / chain["market_mid"]
chain["theoretical_price"] = np.nan
chain["delta"] = np.nan
chain["gamma"] = np.nan
chain["theta"] = np.nan
chain["vega"] = np.nan
chain["rho"] = np.nan
chain["implied_volatility_solved"] = np.nan

for idx, row in chain.iterrows():
    t = float(row["time_to_expiry"])
    k = float(row["strike"])
    kind = str(row["option_type"]).lower()
    if t <= 0 or not np.isfinite(k) or k <= 0:
        continue
    try:
        chain.at[idx, "theoretical_price"] = black_scholes_price(
            spot, k, t, rate, baseline_vol, kind, dividend_yield
        )
        greeks = calculate_greeks(spot, k, t, rate, baseline_vol, kind, dividend_yield)
        for greek in ("delta", "gamma", "theta", "vega", "rho"):
            chain.at[idx, greek] = float(greeks[greek])
        market_price = row["market_mid"]
        if np.isfinite(market_price) and market_price > 0:
            result = solve_implied_volatility(
                market_price, spot, k, t, rate, kind,
                dividend_yield=dividend_yield,
            )
            chain.at[idx, "implied_volatility_solved"] = float(result.volatility)
    except (ValueError, ArithmeticError, TypeError):
        continue

st.caption(f"Underlying: {snapshot.ticker} · Expiry: {snapshot.expiration} · Retrieved: {snapshot.retrieved_at:%Y-%m-%d %H:%M:%S %Z} · Provider: Yahoo Finance/yfinance")
metric_cols = st.columns(4)
metric_cols[0].metric("Spot price", f"{spot:,.2f}")
metric_cols[1].metric("Contracts", f"{len(chain):,}")
metric_cols[2].metric("Calls", f"{len(snapshot.calls):,}")
metric_cols[3].metric("Puts", f"{len(snapshot.puts):,}")

tabs = st.tabs(["Market chain", "Pricing & Greeks", "Implied volatility", "Mispricing scan", "Sensitivity", "Monte Carlo & Risk"])

with tabs[0]:
    st.subheader("Listed options chain")
    side = st.radio("Option side", ["All", "Calls", "Puts"], horizontal=True)
    shown = chain if side == "All" else chain[chain.option_type == ("call" if side == "Calls" else "put")]
    st.dataframe(shown, use_container_width=True, hide_index=True)
    st.download_button("Download chain CSV", shown.to_csv(index=False), file_name=f"{ticker}_{snapshot.expiration}_chain.csv", mime="text/csv")

with tabs[1]:
    st.subheader("Black–Scholes–Merton valuation")
    valid = chain.dropna(subset=["strike", "time_to_expiry"])
    if valid.empty:
        st.warning("No valid contracts are available for pricing.")
    else:
        chosen = st.selectbox("Contract", valid.index, format_func=lambda i: f"{valid.loc[i, 'contractSymbol']} · {valid.loc[i, 'option_type'].upper()} · K={valid.loc[i, 'strike']:.2f}")
        row = chain.loc[chosen]
        theoretical = float(row["theoretical_price"])
        gcols = st.columns(6)
        gcols[0].metric("Model price", f"{theoretical:.4f}")
        for col, greek in zip(gcols[1:], ["delta", "gamma", "theta", "vega", "rho"]):
            gcols[list(["delta", "gamma", "theta", "vega", "rho"]).index(greek)+1].metric(greek.title(), f"{row[greek]:.5f}" if pd.notna(row[greek]) else "—")
        st.caption("Theta is per year; Vega and Rho are per 1.00 change in decimal volatility/rate. Divide by 365 for approximate daily theta; multiply Vega/Rho by 0.01 for a one-percentage-point move.")

with tabs[2]:
    st.subheader("Implied volatility by strike")
    iv_data = chain.dropna(subset=["implied_volatility_solved"]).copy()
    if iv_data.empty:
        st.warning("No contracts produced a valid implied-volatility solution. Check quotes, expiry and model inputs.")
    else:
        iv_data["IV (%)"] = iv_data["implied_volatility_solved"] * 100
        fig = px.scatter(iv_data, x="strike", y="IV (%)", color="option_type", hover_data=["contractSymbol", "market_mid", "openInterest"], title="Solved implied volatility smile / skew")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(iv_data[["contractSymbol", "option_type", "strike", "market_mid", "IV (%)", "volume", "openInterest"]], use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Market-versus-model discrepancy scan")
    st.caption("The model uses the baseline volatility entered in the sidebar. This is a model residual, not a guaranteed arbitrage or executable trade.")
    min_oi = st.number_input("Minimum open interest", min_value=0, value=10, step=10)
    max_spread = st.slider("Maximum bid/ask spread as fraction of midpoint", 0.01, 1.0, 0.30, 0.01)
    min_edge = st.number_input("Minimum absolute price residual", min_value=0.0, value=0.10, step=0.05)
    try:
        candidates = scan_mispricing(
            chain, spot=spot, rate=rate, volatility=baseline_vol,
            filters=MispricingFilters(min_open_interest=int(min_oi), min_volume=0,
                                      max_spread_pct=float(max_spread), min_abs_residual=float(min_edge)),
            dividend_yield=dividend_yield,
        )
        st.metric("Eligible candidates", int(candidates["eligible"].sum()))
        st.dataframe(candidates.sort_values("residual", ascending=False), use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Scanner could not run: {exc}")

with tabs[4]:
    st.subheader("Greek sensitivity matrix")
    valid = chain.dropna(subset=["strike", "time_to_expiry"])
    if not valid.empty:
        selected = st.selectbox("Sensitivity contract", valid.index, key="matrix_contract", format_func=lambda i: f"{valid.loc[i, 'contractSymbol']} · K={valid.loc[i, 'strike']:.2f}")
        row = chain.loc[selected]
        grids = build_greeks_matrix(
            spot=spot, strike=float(row.strike), time_to_expiry=float(row.time_to_expiry),
            rate=rate, volatility=baseline_vol, option_type=str(row.option_type),
            dividend_yield=dividend_yield, spot_steps=25, time_steps=20,
        )
        greek_choice = st.selectbox("Greek", list(grids.keys()))
        fig = px.imshow(grids[greek_choice], aspect="auto", origin="lower",
                        labels={"x": "Spot price", "y": "Time to expiry (years)", "color": greek_choice.title()},
                        title=f"{greek_choice.title()} across spot and time scenarios")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No valid contract is available for sensitivity analysis.")

with tabs[5]:
    st.subheader("GBM Monte Carlo scenario analysis")
    c1, c2, c3 = st.columns(3)
    horizon_days = c1.slider("Simulation horizon (days)", 1, 90, 30)
    paths_count = c2.select_slider("Simulation paths", options=[1000, 2500, 5000, 10000, 20000], value=5000)
    annual_drift_pct = c3.number_input("Assumed annual drift (%)", min_value=-100.0, max_value=200.0, value=0.0, step=1.0)
    sim = simulate_gbm_paths(
        spot=spot, drift=annual_drift_pct / 100.0, volatility=baseline_vol,
        time_horizon=horizon_days / 365.0, num_steps=min(horizon_days, 30),
        num_paths=int(paths_count), seed=42,
    )
    sample_paths = pd.DataFrame(sim.paths[: min(100, len(sim.paths))].T)
    sample_paths.index.name = "Step"
    fig = px.line(sample_paths, title="Sample simulated underlying-price paths", labels={"value": "Underlying price", "index": "Step"})
    st.plotly_chart(fig, use_container_width=True)
    terminal = sim.terminal_prices
    losses = spot - terminal
    summary = calculate_var_cvar(losses, 0.95)
    rcols = st.columns(4)
    rcols[0].metric("Mean terminal price", f"{terminal.mean():,.2f}")
    rcols[1].metric("5th percentile price", f"{np.quantile(terminal, 0.05):,.2f}")
    rcols[2].metric("95% VaR (underlying loss)", f"{summary.var:,.2f}")
    rcols[3].metric("95% CVaR (underlying loss)", f"{summary.cvar:,.2f}")
    st.caption("VaR/CVaR here are calculated on the underlying's price-change loss distribution (spot minus terminal spot), not on an options portfolio. GBM is a simplified constant-volatility model.")

st.divider()
st.caption("QuantOptions-Engine · Research and educational use · No broker connection or order execution is enabled.")
