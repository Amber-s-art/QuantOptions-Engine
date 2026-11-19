"""QuantOptions-Engine: Dark-themed options research, IV solver, and paper-trading workbench."""
from __future__ import annotations

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics.strategy_engine import (
    STRATEGY_REGISTRY,
    analyze_strategy,
    build_strategy,
    summarize_strategy_payoff,
)
from core.implied_vol import implied_volatility_chain, solve_implied_volatility
from core.pricing import black_scholes_price, calculate_greeks
from data.fetcher import (
    MarketDataError,
    fetch_market_data,
    fetch_multiple_expiries,
    get_available_expirations,
    search_tickers,
)

PLOTLY_TEMPLATE = "plotly_dark"

st.set_page_config(
    page_title="QuantOptions Engine | Quant & Volatility Workbench",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
    --bg-card: rgba(22, 27, 38, 0.85);
    --border-subtle: rgba(148, 163, 184, 0.18);
    --accent: #38bdf8;
    --accent-2: #22c55e;
}

@keyframes fadeSlideUp {
    from {
        opacity: 0;
        transform: translateY(8px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.block-container {
    padding-top: 1.15rem;
    padding-bottom: 2.5rem;
    max-width: 1680px;
    animation: fadeSlideUp 0.35s ease-out;
}

div[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 0.75rem 0.95rem;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: rgba(56, 189, 248, 0.45);
}

[data-testid="stMetricValue"] {
    font-variant-numeric: tabular-nums;
}

.stButton > button, .stDownloadButton > button {
    border-radius: 10px;
    transition: transform 0.15s ease, filter 0.15s ease;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px);
    filter: brightness(1.06);
}

.hero-banner {
    background: linear-gradient(125deg, rgba(15, 23, 42, 0.95), rgba(17, 36, 58, 0.85));
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.85rem;
    animation: fadeSlideUp 0.3s ease-out;
}

.hero-title {
    font-size: clamp(1.35rem, 2.5vw, 2rem);
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
}

.hero-sub {
    color: #94a3b8;
    font-size: clamp(0.82rem, 1.3vw, 0.95rem);
    margin-top: 0.25rem;
}

@media (max-width: 768px) {
    .block-container {
        padding-left: 0.75rem;
        padding-right: 0.75rem;
        padding-top: 0.75rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


def _style_fig(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.45)",
        margin=dict(l=16, r=16, t=46, b=16),
        hovermode="x unified",
        transition=dict(duration=250, easing="cubic-in-out"),
    )
    return fig


@st.cache_data(ttl=90, show_spinner=False)
def cached_search(query: str) -> pd.DataFrame:
    return search_tickers(query)


@st.cache_data(ttl=90, show_spinner=False)
def cached_expiries(symbol: str) -> list[str]:
    return get_available_expirations(symbol)


@st.cache_data(ttl=60, show_spinner=False)
def cached_chain(symbol: str, expiry: str, rate: float):
    return fetch_market_data(symbol, expiry, risk_free_rate=rate)


@st.cache_data(ttl=120, show_spinner=False)
def cached_multi(symbol: str, expiries: tuple[str, ...], rate: float) -> pd.DataFrame:
    return fetch_multiple_expiries(symbol, list(expiries), risk_free_rate=rate)


def solve_chain_iv_from_market(frame: pd.DataFrame, spot: float, rate: float, q: float) -> pd.DataFrame:
    """Solve BSM implied volatility directly from observed market bid/ask mid (or fallback price)."""
    df = frame.copy()
    valid_two_way = df["bid"].gt(0) & df["ask"].gt(0) & df["ask"].ge(df["bid"])
    mid = ((df["bid"] + df["ask"]) / 2.0).where(valid_two_way)
    # Use lastPrice only as secondary reference if bid/ask mid is unavailable
    fallback_px = df["lastPrice"].where(df["lastPrice"].gt(0))
    df["market_mid"] = mid
    df["quote_price_used"] = mid.combine_first(fallback_px)
    solved_df = implied_volatility_chain(df, spot=spot, rate=rate, dividend_yield=q)
    solved_df["market_mid"] = df["market_mid"]
    solved_df["quote_price_used"] = df["quote_price_used"]
    return solved_df


def fmt_pct(val: float | None) -> str:
    return f"{100.0 * val:.2f}%" if val is not None and pd.notna(val) else "—"


def _default_legs_from_chain(chain: pd.DataFrame, spot: float, rate: float, q: float) -> pd.DataFrame:
    calls = chain[chain["option_type"] == "call"].dropna(subset=["strike"]).copy()
    calls = calls.sort_values("strike", key=lambda s: (s - spot).abs())
    rows = []
    for idx, (_, r) in enumerate(calls.head(2).iterrows()):
        tte = max(float(r["time_to_expiry"]), 1.0 / 365.0)
        vol = float(r["implied_volatility"]) if pd.notna(r.get("implied_volatility")) and float(r["implied_volatility"]) > 0 else 0.25
        px = (
            float(r["quote_price_used"])
            if pd.notna(r.get("quote_price_used")) and float(r["quote_price_used"]) > 0
            else float(black_scholes_price(spot, float(r["strike"]), tte, rate, vol, "call", q))
        )
        rows.append(
            {
                "option_type": "call",
                "strike": float(r["strike"]),
                "time_to_expiry": tte,
                "volatility": vol,
                "quantity": 1.0 if idx == 0 else -1.0,
                "entry_price": px,
            }
        )
    if not rows:
        tte = 30.0 / 365.0
        vol = 0.25
        rows.append(
            {
                "option_type": "call",
                "strike": float(spot),
                "time_to_expiry": tte,
                "volatility": vol,
                "quantity": 1.0,
                "entry_price": float(black_scholes_price(spot, float(spot), tte, rate, vol, "call", q)),
            }
        )
    return pd.DataFrame(rows)


# --- Header & Navigation ---
st.markdown(
    """
<div class="hero-banner">
    <div class="hero-title">QuantOptions Engine</div>
    <div class="hero-sub">European BSM Valuation · Market-Implied Volatility Solver · Vol Surface · Multi-Leg Payoff & Paper Book</div>
</div>
""",
    unsafe_allow_html=True,
)
st.caption(
    "Research and paper-trading workbench only. Market quotes may be delayed or incomplete; "
    "no broker execution or personalized financial advice is provided."
)

# --- Sidebar: Global Market & Model Assumptions ---
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Workspace module",
        [
            "Option Pricer",
            "IV Solver & Chain",
            "Smile & Surface",
            "Greeks Decay",
            "Strategy & Paper Book",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.header("Model & Market Inputs")
    rate_pct = st.number_input("Risk-free rate (%)", min_value=-10.0, max_value=40.0, value=6.0, step=0.25)
    q_pct = st.number_input("Continuous dividend yield (%)", min_value=-10.0, max_value=40.0, value=0.0, step=0.25)
    rate = rate_pct / 100.0
    q = q_pct / 100.0

    st.divider()
    st.subheader("Market Ticker & Chain")
    search_q = st.text_input("Search symbol / company", value="SPY", help="Search tickers via market adapter.")
    if st.button("Search tickers", width="stretch"):
        try:
            st.session_state["search_results"] = cached_search(search_q)
        except Exception as exc:
            st.error(f"Search failed: {exc}")

    results = st.session_state.get("search_results", pd.DataFrame())
    if isinstance(results, pd.DataFrame) and not results.empty:
        symbols = results["symbol"].dropna().astype(str).tolist()
        labels = {str(r.symbol): f"{r.symbol} — {r.name} ({r.exchange})" for r in results.itertuples()}
        picked = st.selectbox("Matched symbols", symbols, format_func=lambda s: labels.get(s, s))
        default_sym = picked or (symbols[0] if symbols else "SPY")
    else:
        default_sym = search_q.strip().upper() or "SPY"

    ticker = st.text_input("Active ticker", value=st.session_state.get("ticker", default_sym)).strip().upper()
    st.session_state["ticker"] = ticker
    refresh_exp = st.button("Fetch expirations", type="primary", width="stretch")


# Fetch expirations when requested or ticker changes
if refresh_exp or "expiries" not in st.session_state or st.session_state.get("expiry_ticker") != ticker:
    try:
        with st.spinner(f"Fetching expirations for {ticker}..."):
            st.session_state["expiries"] = cached_expiries(ticker)
            st.session_state["expiry_ticker"] = ticker
    except Exception as exc:
        st.session_state["expiries"] = []
        st.warning(f"Live chain unavailable for {ticker}: {exc}")

expiries: list[str] = st.session_state.get("expiries", [])
snap = None
chain = pd.DataFrame()
spot: float | None = None

if expiries:
    with st.sidebar:
        expiry = st.selectbox("Expiration date", expiries, index=0)
        max_curve_exp = (
            st.slider("Expiries for vol surface", 2, min(10, len(expiries)), min(5, len(expiries)))
            if len(expiries) >= 2
            else 1
        )
        load_chain_btn = st.button("Load option chain", width="stretch")

    chain_key = (ticker, expiry, rate)
    if load_chain_btn or st.session_state.get("chain_key") != chain_key:
        try:
            with st.spinner(f"Loading {ticker} ({expiry}) chain..."):
                st.session_state["snapshot"] = cached_chain(ticker, expiry, rate)
                st.session_state["chain_key"] = chain_key
        except MarketDataError as exc:
            st.session_state["snapshot"] = None
            st.error(f"Chain load error: {exc}")

    snap = st.session_state.get("snapshot")
    if snap is not None:
        spot = float(snap.spot)
        raw_chain = snap.combined_options()
        chain = solve_chain_iv_from_market(raw_chain, spot=spot, rate=rate, q=q)


# ==============================================================================
# PAGE 1: OPTION PRICER (NO OPTION PRICE INPUT — SOLVES THEORETICAL PRICE & GREEKS)
# ==============================================================================
if page == "Option Pricer":
    st.subheader("Theoretical Option Pricer (BSM)")
    st.caption("Calculates European option fair value and Greeks from model & contract parameters—no option price input is used here.")

    default_spot = float(spot) if spot is not None else 100.0
    default_strike = round(default_spot, 2)

    with st.form("bsm_price_only_form"):
        c1, c2, c3 = st.columns(3)
        p_type = c1.selectbox("Option type", ["call", "put"], index=0)
        p_spot = c1.number_input("Underlying spot price (S)", min_value=0.0001, value=default_spot, format="%.4f")
        p_strike = c2.number_input("Strike price (K)", min_value=0.0001, value=default_strike, format="%.4f")
        p_days = c2.number_input("Days to expiration", min_value=0.5, max_value=3650.0, value=30.0, step=1.0)
        p_vol_pct = c3.number_input("Volatility σ (%)", min_value=0.1, max_value=500.0, value=22.0, step=0.5)
        p_qty = c3.number_input("Contracts / units", min_value=1, max_value=100000, value=1, step=1)
        calc_btn = st.form_submit_button("Compute Theoretical Option Price", type="primary", width="stretch")

    if calc_btn:
        tte = float(p_days) / 365.0
        vol = float(p_vol_pct) / 100.0
        unit_price = float(black_scholes_price(p_spot, p_strike, tte, rate, vol, p_type, q))
        greeks = calculate_greeks(p_spot, p_strike, tte, rate, vol, p_type, q)

        m_cols = st.columns(4)
        m_cols[0].metric("Option Price (per unit)", f"{unit_price:,.4f}")
        m_cols[1].metric(f"Total Value ({p_qty} units)", f"{unit_price * p_qty:,.4f}")
        m_cols[2].metric("Intrinsic Value", f"{max(p_spot - p_strike, 0.0) if p_type == 'call' else max(p_strike - p_spot, 0.0):,.4f}")
        m_cols[3].metric("Time Value", f"{max(unit_price - (max(p_spot - p_strike, 0.0) if p_type == 'call' else max(p_strike - p_spot, 0.0)), 0.0):,.4f}")

        g_cols = st.columns(5)
        for col, key in zip(g_cols, ["delta", "gamma", "theta", "vega", "rho"]):
            col.metric(key.capitalize(), f"{greeks[key]:.6f}")

        spot_range = np.linspace(p_spot * 0.75, p_spot * 1.25, 100)
        curve_prices = [
            float(black_scholes_price(float(s), p_strike, tte, rate, vol, p_type, q))
            for s in spot_range
        ]
        payoff_expiry = [
            max(float(s) - p_strike, 0.0) if p_type == "call" else max(p_strike - float(s), 0.0)
            for s in spot_range
        ]
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=spot_range, y=curve_prices, mode="lines", name="BSM Fair Value (Now)"))
        fig_p.add_trace(go.Scatter(x=spot_range, y=payoff_expiry, mode="lines", line=dict(dash="dash"), name="Intrinsic Payoff at Expiry"))
        fig_p.update_layout(title="Option Value vs. Underlying Spot", xaxis_title="Spot Price (S)", yaxis_title="Price")
        st.plotly_chart(_style_fig(fig_p), width="stretch")


# ==============================================================================
# PAGE 2: IV SOLVER & LIVE CHAIN (NO IV INPUT PARAMETER — SOLVES FROM MARKET PRICE)
# ==============================================================================
elif page == "IV Solver & Chain":
    st.subheader("Live Option Chain & Implied Volatility Solver")
    st.caption("Fetches real option chain quotes and solves BSM implied volatility from market prices—no IV input parameter is accepted.")

    if chain.empty or spot is None:
        st.info("Select a ticker and load an option chain from the sidebar to inspect live quotes and solve implied volatility.")
    else:
        k_cols = st.columns(4)
        k_cols[0].metric("Underlying Spot", f"{spot:,.2f}")
        k_cols[1].metric("Contracts Loaded", f"{len(chain):,}")
        k_cols[2].metric("Calls / Puts", f"{len(snap.calls)} / {len(snap.puts)}")
        k_cols[3].metric("Solved IVs", f"{chain['implied_volatility'].notna().sum():,}")
        st.caption(f"{ticker} · Expiry {snap.expiration} · Retrieved {snap.retrieved_at:%Y-%m-%d %H:%M:%S UTC}")

        st.markdown("#### Single-Contract IV Solver (From Market Price)")
        iv_col1, iv_col2, iv_col3 = st.columns(3)
        side_filter = iv_col1.radio("Side filter", ["All", "Calls", "Puts"], horizontal=True, key="iv_solver_side")
        filtered = (
            chain
            if side_filter == "All"
            else chain[chain["option_type"] == ("call" if side_filter == "Calls" else "put")]
        ).reset_index(drop=True)

        idx = iv_col2.selectbox(
            "Select contract",
            list(range(len(filtered))),
            format_func=lambda i: (
                f"{filtered.loc[i, 'contractSymbol']} · {filtered.loc[i, 'option_type'].upper()} · K={filtered.loc[i, 'strike']:g}"
            ),
        )
        sel = filtered.loc[idx]
        default_mkt_px = (
            float(sel["quote_price_used"])
            if pd.notna(sel.get("quote_price_used")) and float(sel["quote_price_used"]) > 0
            else 1.0
        )
        mkt_price_input = iv_col3.number_input(
            "Market option price (input)",
            min_value=0.0001,
            value=default_mkt_px,
            format="%.4f",
            key="iv_solver_mkt_px",
        )

        if st.button("Solve Implied Volatility", type="primary"):
            try:
                tte = max(float(sel["time_to_expiry"]), 1e-6)
                solved = solve_implied_volatility(
                    mkt_price_input,
                    spot=spot,
                    strike=float(sel["strike"]),
                    time_to_expiry=tte,
                    rate=rate,
                    option_type=str(sel["option_type"]),
                    dividend_yield=q,
                )
                r_cols = st.columns(4)
                r_cols[0].metric("Solved Implied Vol", f"{solved.volatility * 100:.2f}%")
                r_cols[1].metric("Solver Status", str(solved.status))
                r_cols[2].metric("Model Re-Price", f"{solved.model_price:.4f}")
                r_cols[3].metric("Pricing Error", f"{solved.price_error:.2e}")
            except Exception as exc:
                st.error(f"Could not solve IV for this quote: {exc}")

        st.markdown("#### Full Chain Table (Solved Market IV)")
        display_cols = [
            c
            for c in [
                "contractSymbol",
                "option_type",
                "expiration",
                "strike",
                "bid",
                "ask",
                "market_mid",
                "lastPrice",
                "volume",
                "openInterest",
                "implied_volatility",
                "iv_status",
                "iv_price_error",
            ]
            if c in filtered.columns
        ]
        st.dataframe(filtered[display_cols], width="stretch", hide_index=True)
        st.download_button(
            "Download chain CSV",
            filtered[display_cols].to_csv(index=False),
            file_name=f"{ticker}_{snap.expiration}_chain_iv.csv",
            mime="text/csv",
        )


# ==============================================================================
# PAGE 3: SMILE & SURFACE
# ==============================================================================
elif page == "Smile & Surface":
    st.subheader("Volatility Smile, Term Structure & Surface")
    if spot is None or len(expiries) < 2:
        st.info("Load a ticker with at least two expirations to view cross-expiry smiles and surfaces.")
    else:
        try:
            selected_exps = tuple(expiries[:max_curve_exp])
            with st.spinner("Computing cross-expiry implied volatility..."):
                multi = cached_multi(ticker, selected_exps, rate)
            solved_multi = solve_chain_iv_from_market(multi, spot=spot, rate=rate, q=q)
            valid_iv = solved_multi.dropna(subset=["implied_volatility"]).copy()
            valid_iv = valid_iv[(valid_iv["implied_volatility"] > 0.0) & (valid_iv["implied_volatility"] < 5.0)]
            valid_iv["iv_pct"] = valid_iv["implied_volatility"] * 100.0
            valid_iv["moneyness"] = valid_iv["strike"] / spot
            valid_iv["dte"] = valid_iv["time_to_expiry"] * 365.0
        except Exception as exc:
            st.error(f"Volatility surface error: {exc}")
            valid_iv = pd.DataFrame()

        if valid_iv.empty:
            st.info("No valid solved IV observations across the selected expirations.")
        else:
            side_choice = st.radio("Option side", ["Both", "Calls", "Puts"], horizontal=True, key="surf_side")
            if side_choice != "Both":
                valid_iv = valid_iv[valid_iv["option_type"] == ("call" if side_choice == "Calls" else "put")]

            fig_smile = px.scatter(
                valid_iv,
                x="strike",
                y="iv_pct",
                color="expiration",
                symbol="option_type",
                hover_data=["contractSymbol", "market_mid", "volume", "openInterest"],
                title="Solved Implied Volatility Smile Across Expirations",
            )
            fig_smile.update_yaxes(title="Implied Volatility (%)")
            st.plotly_chart(_style_fig(fig_smile), width="stretch")

            valid_iv["atm_gap"] = (valid_iv["moneyness"] - 1.0).abs()
            atm_ts = valid_iv.sort_values("atm_gap").groupby("expiration", as_index=False).first()
            fig_ts = px.line(
                atm_ts,
                x="expiration",
                y="iv_pct",
                markers=True,
                title="Near-ATM Implied Volatility Term Structure",
            )
            fig_ts.update_yaxes(title="Near-ATM IV (%)")
            st.plotly_chart(_style_fig(fig_ts, height=360), width="stretch")

            surf_df = valid_iv.copy()
            surf_df["m_bin"] = (surf_df["moneyness"] * 100.0).round() / 100.0
            surf_df["dte_bin"] = (surf_df["dte"] / 7.0).round() * 7.0
            pivot = surf_df.pivot_table(index="dte_bin", columns="m_bin", values="iv_pct", aggfunc="median")
            if pivot.shape[0] >= 2 and pivot.shape[1] >= 2:
                fig_3d = go.Figure(
                    data=[
                        go.Surface(
                            x=pivot.columns.to_numpy(),
                            y=pivot.index.to_numpy(),
                            z=pivot.to_numpy(),
                            colorbar_title="IV (%)",
                        )
                    ]
                )
                fig_3d.update_layout(
                    template=PLOTLY_TEMPLATE,
                    title="3D Implied Volatility Surface",
                    scene=dict(
                        xaxis_title="Moneyness (K / S)",
                        yaxis_title="Days to Expiry",
                        zaxis_title="IV (%)",
                    ),
                    height=580,
                )
                st.plotly_chart(fig_3d, width="stretch")


# ==============================================================================
# PAGE 4: GREEKS DECAY
# ==============================================================================
elif page == "Greeks Decay":
    st.subheader("Greeks & Option Value Time-Decay Profile")
    if chain.empty or spot is None:
        st.info("Load an option chain first to inspect time decay for a listed contract.")
    else:
        valid = chain.dropna(subset=["strike", "time_to_expiry"]).reset_index(drop=True)
        c_idx = st.selectbox(
            "Contract",
            list(range(len(valid))),
            format_func=lambda i: f"{valid.loc[i, 'contractSymbol']} · {valid.loc[i, 'option_type'].upper()} · K={valid.loc[i, 'strike']:g}",
        )
        row = valid.loc[c_idx]

        s_col, v_col = st.columns(2)
        spot_shock = s_col.slider("Spot shock (%)", -30, 30, 0, step=1) / 100.0
        vol_shock = v_col.slider("Vol shock (percentage points)", -30, 30, 0, step=1) / 100.0

        s_eff = spot * (1.0 + spot_shock)
        base_vol = float(row["implied_volatility"]) if pd.notna(row.get("implied_volatility")) and float(row["implied_volatility"]) > 0 else 0.25
        vol_eff = max(0.001, base_vol + vol_shock)
        strike_val = float(row["strike"])
        t0 = max(float(row["time_to_expiry"]), 2.0 / 365.0)
        kind = str(row["option_type"])

        t_grid = np.linspace(t0, 1.0 / 365.0, 80)
        decay_rows = []
        for tt in t_grid:
            g_vals = calculate_greeks(s_eff, strike_val, float(tt), rate, vol_eff, kind, q)
            px_val = float(black_scholes_price(s_eff, strike_val, float(tt), rate, vol_eff, kind, q))
            decay_rows.append({"days_to_expiry": float(tt) * 365.0, "model_price": px_val, **g_vals})

        decay_df = pd.DataFrame(decay_rows).sort_values("days_to_expiry")
        metrics_sel = st.multiselect(
            "Metrics to plot",
            ["delta", "gamma", "theta", "vega", "rho", "model_price"],
            default=["delta", "gamma", "theta", "vega"],
        )
        if metrics_sel:
            melted = decay_df.melt(id_vars="days_to_expiry", value_vars=metrics_sel, var_name="metric", value_name="value")
            fig_g = px.line(melted, x="days_to_expiry", y="value", color="metric", title="Sensitivity & Value vs. Time to Expiry")
            st.plotly_chart(_style_fig(fig_g), width="stretch")


# ==============================================================================
# PAGE 5: STRATEGY & PAPER BOOK
# ==============================================================================
elif page == "Strategy & Paper Book":
    st.subheader("Multi-Leg Strategy Lab & Session Paper Portfolio")
    if spot is None:
        st.info("Load a ticker chain first so strategy legs can seed from current spot and market quotes.")
    else:
        preset_names = ["Custom Legs"] + list(STRATEGY_REGISTRY.keys())
        preset = st.selectbox("Strategy template", preset_names)

        leg_state_key = f"legs_{ticker}_{snap.expiration}"
        if leg_state_key not in st.session_state:
            st.session_state[leg_state_key] = _default_legs_from_chain(chain, spot=spot, rate=rate, q=q)

        if preset != "Custom Legs":
            p_col1, p_col2, p_col3 = st.columns(3)
            p_tte = p_col1.number_input("Template expiry (years)", min_value=0.005, max_value=5.0, value=max(float(chain["time_to_expiry"].iloc[0]), 0.05), step=0.01)
            p_vol = p_col2.number_input("Template vol (decimal)", min_value=0.01, max_value=3.0, value=0.22, step=0.01)
            p_qty = p_col3.number_input("Base quantity", min_value=1.0, max_value=1000.0, value=1.0, step=1.0)

            k_step = max(round(spot * 0.02, 2), 1.0)
            k1 = round(spot - k_step, 2)
            k2 = round(spot, 2)
            k3 = round(spot + k_step, 2)

            if st.button("Apply template to legs table"):
                kwargs_map = {
                    "Bull Call Spread": dict(spot=spot, k_low=k1, k_high=k3, tte=p_tte, vol=p_vol, rate=rate, dividend_yield=q, qty=p_qty),
                    "Bear Put Spread": dict(spot=spot, k_low=k1, k_high=k3, tte=p_tte, vol=p_vol, rate=rate, dividend_yield=q, qty=p_qty),
                    "Bull Put Spread": dict(spot=spot, k_low=k1, k_high=k3, tte=p_tte, vol=p_vol, rate=rate, dividend_yield=q, qty=p_qty),
                    "Bear Call Spread": dict(spot=spot, k_low=k1, k_high=k3, tte=p_tte, vol=p_vol, rate=rate, dividend_yield=q, qty=p_qty),
                    "Butterfly Call Spread": dict(spot=spot, k_low=k1, k_mid=k2, k_high=k3, tte=p_tte, vol=p_vol, rate=rate, dividend_yield=q, qty=p_qty),
                    "Long Straddle": dict(spot=spot, strike=k2, tte=p_tte, vol=p_vol, rate=rate, dividend_yield=q, qty=p_qty, is_long=True),
                    "Short Straddle": dict(spot=spot, strike=k2, tte=p_tte, vol=p_vol, rate=rate, dividend_yield=q, qty=p_qty, is_long=False),
                    "Long Strangle": dict(spot=spot, k_put=k1, k_call=k3, tte=p_tte, vol=p_vol, rate=rate, dividend_yield=q, qty=p_qty, is_long=True),
                    "Short Strangle": dict(spot=spot, k_put=k1, k_call=k3, tte=p_tte, vol=p_vol, rate=rate, dividend_yield=q, qty=p_qty, is_long=False),
                    "Strip": dict(spot=spot, strike=k2, tte=p_tte, vol=p_vol, rate=rate, dividend_yield=q, qty=p_qty),
                    "Strap": dict(spot=spot, strike=k2, tte=p_tte, vol=p_vol, rate=rate, dividend_yield=q, qty=p_qty),
                    "Calendar Call Spread": dict(spot=spot, strike=k2, t_near=max(p_tte * 0.5, 0.005), t_far=max(p_tte, 0.02), vol=p_vol, rate=rate, dividend_yield=q, qty=p_qty),
                }
                st.session_state[leg_state_key] = build_strategy(preset, **kwargs_map[preset])

        edited_legs = st.data_editor(
            st.session_state[leg_state_key],
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            key=f"editor_{leg_state_key}",
        )
        st.session_state[leg_state_key] = edited_legs

        s_col1, s_col2, s_col3 = st.columns(3)
        days_fwd = s_col1.slider("Scenario horizon (days)", 0, 90, 7)
        iv_shift_pct = s_col2.slider("IV shock (pp)", -30, 30, 0)
        mult = s_col3.number_input("Contract multiplier", min_value=1, max_value=10000, value=1, step=1)

        if st.button("Analyze Strategy & Expected Payoff", type="primary"):
            try:
                res = analyze_strategy(
                    edited_legs,
                    spot=spot,
                    rate=rate,
                    dividend_yield=q,
                    days_forward=days_fwd,
                    iv_shift=iv_shift_pct / 100.0,
                    multiplier=int(mult),
                )
                avg_tte = float(pd.to_numeric(edited_legs["time_to_expiry"]).mean())
                avg_vol = float(pd.to_numeric(edited_legs["volatility"]).mean())
                summary = summarize_strategy_payoff(
                    res,
                    spot=spot,
                    rate=rate,
                    dividend_yield=q,
                    time_to_expiry=avg_tte,
                    volatility=avg_vol,
                )
                st.session_state["last_strategy_bundle"] = (res, summary, edited_legs.copy())
            except Exception as exc:
                st.error(f"Strategy evaluation failed: {exc}")

        bundle = st.session_state.get("last_strategy_bundle")
        if bundle:
            res, summary, saved_legs = bundle
            p_metrics = st.columns(4)
            p_metrics[0].metric(
                "Expected Payoff (BSM)",
                f"{summary['expected_payoff']:,.4f}" if summary["expected_payoff"] is not None else "—",
            )
            p_metrics[1].metric("Max Profit", f"{summary['max_profit']:,.4f}" if np.isfinite(summary["max_profit"]) else "Unbounded")
            p_metrics[2].metric("Max Loss", f"{summary['max_loss']:,.4f}" if np.isfinite(summary["max_loss"]) else "Unbounded")
            be_text = ", ".join(f"{x:,.2f}" for x in summary["breakeven_spots"][:3]) or "None in grid"
            p_metrics[3].metric("Breakeven Spot(s)", be_text)

            g_metrics = st.columns(5)
            g_metrics[0].metric("Net Delta", f"{res['net_delta']:,.4f}")
            g_metrics[1].metric("Net Gamma", f"{res['net_gamma']:,.4f}")
            g_metrics[2].metric("Net Theta / yr", f"{res['net_theta']:,.4f}")
            g_metrics[3].metric("Net Vega", f"{res['net_vega']:,.4f}")
            g_metrics[4].metric("Hedge Shares", f"{res['delta_hedge_shares']:+,.4f}")

            fig_payoff = go.Figure()
            fig_payoff.add_trace(go.Scatter(x=res["spot_grid"], y=res["expiry_pnl"], mode="lines", name="Payoff P&L"))
            fig_payoff.add_hline(y=0.0, line_dash="dash")
            fig_payoff.update_layout(title="Strategy Payoff Profile", xaxis_title="Spot Price", yaxis_title="P&L")
            st.plotly_chart(_style_fig(fig_payoff), width="stretch")

            scen_df = res["scenario"]
            scen_y_col = "pnl" if "pnl" in scen_df.columns else "pnl_per_share"
            fig_scen = px.line(
                scen_df,
                x="spot_change_pct",
                y=scen_y_col,
                markers=True,
                title=f"Repriced Scenario P&L (+{days_fwd}d, IV shift {iv_shift_pct:+d} pp)",
            )
            st.plotly_chart(_style_fig(fig_scen, height=350), width="stretch")

            st.dataframe(res["legs"], width="stretch", hide_index=True)

            if st.button("Save to Session Paper Portfolio"):
                book = st.session_state.setdefault("paper_book", [])
                pos_id = len(book) + 1
                book.append(
                    {
                        "id": pos_id,
                        "ticker": ticker,
                        "saved_at": datetime.now(timezone.utc).isoformat(),
                        "entry_spot": spot,
                        "multiplier": int(mult),
                        "legs": saved_legs.to_dict(orient="records"),
                    }
                )
                st.success(f"Saved position #{pos_id} for {ticker} to session paper book.")

    st.divider()
    st.markdown("#### Session Paper Portfolio")
    book = st.session_state.get("paper_book", [])
    ticker_positions = [p for p in book if p["ticker"] == ticker]
    if spot is not None and ticker_positions:
        now_utc = datetime.now(timezone.utc)
        rows = []
        for pos in ticker_positions:
            saved_dt = datetime.fromisoformat(pos["saved_at"])
            elapsed_yr = max((now_utc - saved_dt).total_seconds(), 0.0) / (365.0 * 24.0 * 3600.0)
            p_mult = float(pos.get("multiplier", 1))
            mtm = 0.0
            entry_total = 0.0
            for leg in pos["legs"]:
                qty = float(leg["quantity"])
                strike = float(leg["strike"])
                tte_rem = max(float(leg["time_to_expiry"]) - elapsed_yr, 0.0)
                vol = float(leg["volatility"])
                kind = str(leg["option_type"]).lower()
                entry = float(leg["entry_price"])
                if tte_rem > 0.0:
                    mark = float(black_scholes_price(spot, strike, tte_rem, rate, vol, kind, q))
                else:
                    mark = max(spot - strike, 0.0) if kind == "call" else max(strike - spot, 0.0)
                mtm += qty * mark * p_mult
                entry_total += qty * entry * p_mult
            rows.append(
                {
                    "id": pos["id"],
                    "ticker": pos["ticker"],
                    "entry_spot": pos["entry_spot"],
                    "current_spot": spot,
                    "paper_pnl": mtm - entry_total,
                }
            )
        p_df = pd.DataFrame(rows)
        st.dataframe(p_df, width="stretch", hide_index=True)

        remove_id = st.selectbox("Select position ID to close", p_df["id"].tolist(), key="remove_pos_id")
        if st.button("Close Selected Paper Position"):
            st.session_state["paper_book"] = [p for p in st.session_state["paper_book"] if p["id"] != remove_id]
            st.rerun()
    else:
        st.caption("No open session paper positions for this ticker.")
