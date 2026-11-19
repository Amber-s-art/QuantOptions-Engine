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


def _init_state() -> None:
    st.session_state.setdefault("ticker", "SPY")
    st.session_state.setdefault("search_results", pd.DataFrame(columns=["symbol", "name", "exchange", "type"]))
    st.session_state.setdefault("expiries", [])
    st.session_state.setdefault("expiry_ticker", None)
    st.session_state.setdefault("snapshot", None)
    st.session_state.setdefault("chain_key", None)
    st.session_state.setdefault("paper_book", [])
    st.session_state.setdefault("last_strategy_bundle", None)
    st.session_state.setdefault("leg_editor_ver", 0)


_init_state()


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


def fmt_pct(val: float | None) -> str:
    return f"{100.0 * val:.2f}%" if val is not None and pd.notna(val) else "—"


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
    if frame is None or frame.empty:
        return pd.DataFrame()
    df = frame.copy()
    for col in ("bid", "ask", "lastPrice", "strike", "time_to_expiry"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid_two_way = df["bid"].gt(0) & df["ask"].gt(0) & df["ask"].ge(df["bid"])
    mid = ((df["bid"] + df["ask"]) / 2.0).where(valid_two_way)
    fallback_px = df["lastPrice"].where(df["lastPrice"].gt(0))
    df["market_mid"] = mid
    df["quote_price_used"] = mid.combine_first(fallback_px)

    try:
        solved_df = implied_volatility_chain(df, spot=spot, rate=rate, dividend_yield=q)
    except Exception:
        solved_df = df.copy()
        solved_df["implied_volatility"] = np.nan

    solved_df["market_mid"] = df["market_mid"]
    solved_df["quote_price_used"] = df["quote_price_used"]
    return solved_df


def _clean_strategy_legs(legs_df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize edited strategy table rows before pricing/risk evaluation."""
    if legs_df is None or legs_df.empty:
        raise ValueError("Add at least one valid option leg before running strategy analysis.")

    df = legs_df.copy()
    required_cols = ["option_type", "strike", "time_to_expiry", "volatility", "quantity", "entry_price"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in strategy legs.")

    df["option_type"] = df["option_type"].astype(str).str.strip().str.lower()
    df = df[df["option_type"].isin(["call", "put"])].copy()

    for num_col in ["strike", "time_to_expiry", "volatility", "quantity", "entry_price"]:
        df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    df = df.dropna(subset=["strike", "time_to_expiry", "volatility", "quantity", "entry_price"]).copy()
    df = df[
        (df["strike"] > 0)
        & (df["time_to_expiry"] > 0)
        & (df["volatility"] > 0)
        & np.isfinite(df["strike"])
        & np.isfinite(df["time_to_expiry"])
        & np.isfinite(df["volatility"])
        & np.isfinite(df["quantity"])
        & np.isfinite(df["entry_price"])
    ].copy()

    if df.empty:
        raise ValueError("No valid rows remained after cleaning legs (check positive strike, TTE, and volatility).")
    return df.reset_index(drop=True)


def _default_legs_from_chain(chain: pd.DataFrame, spot: float, rate: float, q: float) -> pd.DataFrame:
    if chain is not None and not chain.empty and "option_type" in chain.columns and "strike" in chain.columns:
        calls = chain[chain["option_type"] == "call"].dropna(subset=["strike"]).copy()
        calls["strike"] = pd.to_numeric(calls["strike"], errors="coerce")
        calls = calls[calls["strike"] > 0]
        if not calls.empty:
            calls = calls.sort_values("strike", key=lambda s: (s - spot).abs())
            rows = []
            for idx, (_, r) in enumerate(calls.head(2).iterrows()):
                tte_raw = pd.to_numeric(r.get("time_to_expiry"), errors="coerce")
                tte = max(float(tte_raw), 1.0 / 365.0) if pd.notna(tte_raw) and float(tte_raw) > 0 else 30.0 / 365.0
                iv_raw = pd.to_numeric(r.get("implied_volatility"), errors="coerce")
                vol = float(iv_raw) if pd.notna(iv_raw) and 0.005 <= float(iv_raw) <= 5.0 else 0.25
                px_raw = pd.to_numeric(r.get("quote_price_used"), errors="coerce")
                px = (
                    float(px_raw)
                    if pd.notna(px_raw) and float(px_raw) > 0
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
            if rows:
                return pd.DataFrame(rows)

    tte = 30.0 / 365.0
    vol = 0.25
    safe_spot = max(float(spot), 0.01)
    return pd.DataFrame(
        [
            {
                "option_type": "call",
                "strike": safe_spot,
                "time_to_expiry": tte,
                "volatility": vol,
                "quantity": 1.0,
                "entry_price": float(black_scholes_price(safe_spot, safe_spot, tte, rate, vol, "call", q)),
            }
        ]
    )


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
    if isinstance(results, pd.DataFrame) and not results.empty and "symbol" in results.columns:
        symbols = results["symbol"].dropna().astype(str).tolist()
        labels = {
            str(r.symbol): f"{r.symbol} — {getattr(r, 'name', '')} ({getattr(r, 'exchange', '')})"
            for r in results.itertuples()
        }
        if symbols:
            picked = st.selectbox("Matched symbols", symbols, format_func=lambda s: labels.get(s, s))
            default_sym = picked or symbols[0]
        else:
            default_sym = search_q.strip().upper() or "SPY"
    else:
        default_sym = search_q.strip().upper() or "SPY"

    ticker = st.text_input("Active ticker", value=st.session_state.get("ticker", default_sym)).strip().upper() or "SPY"
    st.session_state["ticker"] = ticker
    refresh_exp = st.button("Fetch expirations", type="primary", width="stretch")


# Fetch expirations safely outside cache exception pickling
if refresh_exp or st.session_state.get("expiry_ticker") != ticker:
    try:
        with st.spinner(f"Fetching expirations for {ticker}..."):
            st.session_state["expiries"] = cached_expiries(ticker)
            st.session_state["expiry_ticker"] = ticker
            if st.session_state.get("snapshot") is not None and getattr(st.session_state["snapshot"], "ticker", None) != ticker:
                st.session_state["snapshot"] = None
                st.session_state["chain_key"] = None
    except Exception as exc:
        st.session_state["expiries"] = []
        st.session_state["expiry_ticker"] = ticker
        st.session_state["snapshot"] = None
        st.warning(f"Live chain unavailable for {ticker}: {exc}")

expiries: list[str] = st.session_state.get("expiries", [])
snap = None
chain = pd.DataFrame()
spot: float | None = None
max_curve_exp = 1

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
    if load_chain_btn or (st.session_state.get("snapshot") is None and st.session_state.get("chain_key") != chain_key):
        try:
            with st.spinner(f"Loading {ticker} ({expiry}) chain..."):
                st.session_state["snapshot"] = cached_chain(ticker, expiry, rate)
                st.session_state["chain_key"] = chain_key
        except Exception as exc:
            st.session_state["snapshot"] = None
            st.session_state["chain_key"] = chain_key
            st.error(f"Chain load error: {exc}")

    snap = st.session_state.get("snapshot")
    if snap is not None and getattr(snap, "ticker", None) == ticker:
        spot = float(snap.spot)
        raw_chain = snap.combined_options()
        chain = solve_chain_iv_from_market(raw_chain, spot=spot, rate=rate, q=q)
    else:
        snap = None


# ==============================================================================
# PAGE 1: OPTION PRICER (NO OPTION PRICE INPUT — SOLVES THEORETICAL PRICE & GREEKS)
# ==============================================================================
if page == "Option Pricer":
    st.subheader("Theoretical Option Pricer (BSM)")
    st.caption("Calculates European option fair value and Greeks from model & contract parameters—no option price input is used here.")

    default_spot = float(spot) if (spot is not None and spot > 0) else 100.0
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
        try:
            tte = float(p_days) / 365.0
            vol = float(p_vol_pct) / 100.0
            unit_price = float(black_scholes_price(p_spot, p_strike, tte, rate, vol, p_type, q))
            greeks = calculate_greeks(p_spot, p_strike, tte, rate, vol, p_type, q)

            intrinsic = max(p_spot - p_strike, 0.0) if p_type == "call" else max(p_strike - p_spot, 0.0)
            time_val = max(unit_price - intrinsic, 0.0)

            m_cols = st.columns(4)
            m_cols[0].metric("Option Price (per unit)", f"{unit_price:,.4f}")
            m_cols[1].metric(f"Total Value ({p_qty} units)", f"{unit_price * p_qty:,.4f}")
            m_cols[2].metric("Intrinsic Value", f"{intrinsic:,.4f}")
            m_cols[3].metric("Time Value", f"{time_val:,.4f}")

            g_cols = st.columns(5)
            for col, key in zip(g_cols, ["delta", "gamma", "theta", "vega", "rho"]):
                col.metric(key.capitalize(), f"{float(greeks.get(key, 0.0)):.6f}")

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
        except Exception as exc:
            st.error(f"Pricing calculation failed: {exc}")


# ==============================================================================
# PAGE 2: IV SOLVER & LIVE CHAIN (NO IV INPUT PARAMETER — SOLVES FROM MARKET PRICE)
# ==============================================================================
elif page == "IV Solver & Chain":
    st.subheader("Live Option Chain & Implied Volatility Solver")
    st.caption("Fetches real option chain quotes and solves BSM implied volatility from market prices—no IV input parameter is accepted.")

    if chain.empty or spot is None or snap is None:
        st.info("Select a ticker and load an option chain from the sidebar to inspect live quotes and solve implied volatility.")
    else:
        solved_count = int(chain["implied_volatility"].notna().sum()) if "implied_volatility" in chain.columns else 0
        k_cols = st.columns(4)
        k_cols[0].metric("Underlying Spot", f"{spot:,.2f}")
        k_cols[1].metric("Contracts Loaded", f"{len(chain):,}")
        k_cols[2].metric("Calls / Puts", f"{len(snap.calls)} / {len(snap.puts)}")
        k_cols[3].metric("Solved IVs", f"{solved_count:,}")
        st.caption(f"{ticker} · Expiry {snap.expiration} · Retrieved {snap.retrieved_at:%Y-%m-%d %H:%M:%S UTC}")

        st.markdown("#### Single-Contract IV Solver (From Market Price)")
        iv_col1, iv_col2, iv_col3 = st.columns(3)
        side_filter = iv_col1.radio("Side filter", ["All", "Calls", "Puts"], horizontal=True, key="iv_solver_side")
        filtered = (
            chain
            if side_filter == "All"
            else chain[chain["option_type"] == ("call" if side_filter == "Calls" else "put")]
        ).reset_index(drop=True)

        if filtered.empty:
            st.info(f"No {side_filter.lower()} contracts available for this expiration.")
        else:
            idx = iv_col2.selectbox(
                "Select contract",
                list(range(len(filtered))),
                format_func=lambda i: (
                    f"{filtered.loc[i, 'contractSymbol']} · {str(filtered.loc[i, 'option_type']).upper()} · K={filtered.loc[i, 'strike']:g}"
                ),
            )
            sel = filtered.loc[idx]
            px_val = pd.to_numeric(sel.get("quote_price_used"), errors="coerce")
            default_mkt_px = float(px_val) if pd.notna(px_val) and float(px_val) > 0 else 1.0
            mkt_price_input = iv_col3.number_input(
                "Market option price (input)",
                min_value=0.0001,
                value=default_mkt_px,
                format="%.4f",
                key="iv_solver_mkt_px",
            )

            if st.button("Solve Implied Volatility", type="primary"):
                try:
                    tte_raw = pd.to_numeric(sel.get("time_to_expiry"), errors="coerce")
                    tte = max(float(tte_raw), 1e-6) if pd.notna(tte_raw) and float(tte_raw) > 0 else 1.0 / 365.0
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
            if solved_multi.empty or "implied_volatility" not in solved_multi.columns:
                valid_iv = pd.DataFrame()
            else:
                valid_iv = solved_multi.dropna(subset=["implied_volatility"]).copy()
                valid_iv = valid_iv[(valid_iv["implied_volatility"] > 0.0) & (valid_iv["implied_volatility"] < 5.0)]
                if not valid_iv.empty:
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

            if valid_iv.empty:
                st.info(f"No {side_choice.lower()} IV observations available.")
            else:
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
        if valid.empty:
            st.info("No valid contracts with strike and time-to-expiry found.")
        else:
            c_idx = st.selectbox(
                "Contract",
                list(range(len(valid))),
                format_func=lambda i: f"{valid.loc[i, 'contractSymbol']} · {str(valid.loc[i, 'option_type']).upper()} · K={valid.loc[i, 'strike']:g}",
            )
            row = valid.loc[c_idx]

            s_col, v_col = st.columns(2)
            spot_shock = s_col.slider("Spot shock (%)", -30, 30, 0, step=1) / 100.0
            vol_shock = v_col.slider("Vol shock (percentage points)", -30, 30, 0, step=1) / 100.0

            s_eff = max(spot * (1.0 + spot_shock), 0.0001)
            iv_val = pd.to_numeric(row.get("implied_volatility"), errors="coerce")
            base_vol = float(iv_val) if pd.notna(iv_val) and 0.005 <= float(iv_val) <= 5.0 else 0.25
            vol_eff = max(0.001, base_vol + vol_shock)
            strike_val = max(float(row["strike"]), 0.0001)
            t0 = max(float(row["time_to_expiry"]), 2.0 / 365.0)
            kind = str(row["option_type"]).lower()

            try:
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
            except Exception as exc:
                st.error(f"Could not compute time-decay profile: {exc}")


# ==============================================================================
# PAGE 5: STRATEGY & PAPER BOOK
# ==============================================================================
elif page == "Strategy & Paper Book":
    st.subheader("Multi-Leg Strategy Lab & Session Paper Portfolio")
    if spot is None or snap is None:
        st.info("Load a ticker chain first so strategy legs can seed from current spot and market quotes.")
    else:
        preset_names = ["Custom Legs"] + list(STRATEGY_REGISTRY.keys())
        preset = st.selectbox("Strategy template", preset_names)

        leg_state_key = f"legs_{ticker}_{snap.expiration}"
        if leg_state_key not in st.session_state:
            st.session_state[leg_state_key] = _default_legs_from_chain(chain, spot=spot, rate=rate, q=q)

        if preset != "Custom Legs":
            p_col1, p_col2, p_col3 = st.columns(3)
            tte_series = pd.to_numeric(chain["time_to_expiry"], errors="coerce").dropna() if not chain.empty and "time_to_expiry" in chain.columns else pd.Series(dtype=float)
            default_tte = max(float(tte_series.iloc[0]), 0.05) if not tte_series.empty else 30.0 / 365.0
            p_tte = p_col1.number_input(
                "Template expiry (years)",
                min_value=0.005,
                max_value=5.0,
                value=default_tte,
                step=0.01,
            )
            p_vol = p_col2.number_input("Template vol (decimal)", min_value=0.01, max_value=3.0, value=0.22, step=0.01)
            p_qty = p_col3.number_input("Base quantity", min_value=1.0, max_value=1000.0, value=1.0, step=1.0)

            k_step = max(round(spot * 0.02, 2), 1.0)
            k1 = max(round(spot - k_step, 2), 0.5)
            k2 = max(round(spot, 2), 0.5)
            k3 = max(round(spot + k_step, 2), 0.5)

            if st.button("Apply template to legs table"):
                try:
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
                    st.session_state["leg_editor_ver"] += 1
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not apply template '{preset}': {exc}")

        editor_key = f"editor_{leg_state_key}_v{st.session_state['leg_editor_ver']}"
        edited_legs = st.data_editor(
            st.session_state[leg_state_key],
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            key=editor_key,
        )
        st.session_state[leg_state_key] = edited_legs

        s_col1, s_col2, s_col3 = st.columns(3)
        days_fwd = s_col1.slider("Scenario horizon (days)", 0, 90, 7)
        iv_shift_pct = s_col2.slider("IV shock (pp)", -30, 30, 0)
        mult = s_col3.number_input("Contract multiplier", min_value=1, max_value=10000, value=1, step=1)

        if st.button("Analyze Strategy & Expected Payoff", type="primary"):
            try:
                clean_legs = _clean_strategy_legs(edited_legs)
                res = analyze_strategy(
                    clean_legs,
                    spot=spot,
                    rate=rate,
                    dividend_yield=q,
                    days_forward=days_fwd,
                    iv_shift=iv_shift_pct / 100.0,
                    multiplier=int(mult),
                )
                avg_tte = float(clean_legs["time_to_expiry"].mean())
                avg_vol = float(clean_legs["volatility"].mean())
                summary = summarize_strategy_payoff(
                    res,
                    spot=spot,
                    rate=rate,
                    dividend_yield=q,
                    time_to_expiry=avg_tte,
                    volatility=avg_vol,
                )
                st.session_state["last_strategy_bundle"] = (res, summary, clean_legs.copy())
            except Exception as exc:
                st.error(f"Strategy evaluation failed: {exc}")

        bundle = st.session_state.get("last_strategy_bundle")
        if bundle:
            res, summary, saved_legs = bundle
            exp_val = summary.get("expected_payoff") if isinstance(summary, dict) else None
            max_p = summary.get("max_profit", np.nan) if isinstance(summary, dict) else np.nan
            max_l = summary.get("max_loss", np.nan) if isinstance(summary, dict) else np.nan
            be_list = summary.get("breakeven_spots", []) if isinstance(summary, dict) else []

            p_metrics = st.columns(4)
            p_metrics[0].metric(
                "Expected Payoff (BSM)",
                f"{float(exp_val):,.4f}" if exp_val is not None and np.isfinite(float(exp_val)) else "—",
            )
            p_metrics[1].metric("Max Profit", f"{float(max_p):,.4f}" if np.isfinite(float(max_p)) else "Unbounded")
            p_metrics[2].metric("Max Loss", f"{float(max_l):,.4f}" if np.isfinite(float(max_l)) else "Unbounded")
            be_text = ", ".join(f"{float(x):,.2f}" for x in be_list[:3]) or "None in grid"
            p_metrics[3].metric("Breakeven Spot(s)", be_text)

            g_metrics = st.columns(5)
            g_metrics[0].metric("Net Delta", f"{float(res.get('net_delta', 0.0)):,.4f}")
            g_metrics[1].metric("Net Gamma", f"{float(res.get('net_gamma', 0.0)):,.4f}")
            g_metrics[2].metric("Net Theta / yr", f"{float(res.get('net_theta', 0.0)):,.4f}")
            g_metrics[3].metric("Net Vega", f"{float(res.get('net_vega', 0.0)):,.4f}")
            g_metrics[4].metric("Hedge Shares", f"{float(res.get('delta_hedge_shares', 0.0)):+,.4f}")

            fig_payoff = go.Figure()
            fig_payoff.add_trace(go.Scatter(x=res["spot_grid"], y=res["expiry_pnl"], mode="lines", name="Payoff P&L"))
            fig_payoff.add_hline(y=0.0, line_dash="dash")
            fig_payoff.update_layout(title="Strategy Payoff Profile", xaxis_title="Spot Price", yaxis_title="P&L")
            st.plotly_chart(_style_fig(fig_payoff), width="stretch")

            scen_df = res["scenario"]
            if isinstance(scen_df, pd.DataFrame) and not scen_df.empty:
                scen_y_col = "pnl" if "pnl" in scen_df.columns else "pnl_per_share"
                if scen_y_col in scen_df.columns and "spot_change_pct" in scen_df.columns:
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
                pos_id = max([int(p.get("id", 0)) for p in book], default=0) + 1
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
    ticker_positions = [p for p in book if p.get("ticker") == ticker]
    if spot is not None and ticker_positions:
        now_utc = datetime.now(timezone.utc)
        rows = []
        for pos in ticker_positions:
            try:
                saved_dt = datetime.fromisoformat(pos["saved_at"])
                elapsed_yr = max((now_utc - saved_dt).total_seconds(), 0.0) / (365.0 * 24.0 * 3600.0)
                p_mult = float(pos.get("multiplier", 1))
                mtm = 0.0
                entry_total = 0.0
                for leg in pos.get("legs", []):
                    qty = float(leg["quantity"])
                    strike = float(leg["strike"])
                    tte_rem = max(float(leg["time_to_expiry"]) - elapsed_yr, 0.0)
                    vol = max(float(leg["volatility"]), 0.001)
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
            except Exception:
                continue
        if rows:
            p_df = pd.DataFrame(rows)
            st.dataframe(p_df, width="stretch", hide_index=True)

            pos_ids = p_df["id"].tolist()
            remove_id = st.selectbox("Select position ID to close", pos_ids, key="remove_pos_id")
            if st.button("Close Selected Paper Position"):
                st.session_state["paper_book"] = [p for p in st.session_state["paper_book"] if p.get("id") != remove_id]
                st.rerun()
        else:
            st.caption("No valid open session paper positions for this ticker.")
    else:
        st.caption("No open session paper positions for this ticker.")
