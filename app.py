"""QuantOptions-Engine: interactive options research and paper-trading workbench.

Run: streamlit run app.py
Market-data adapter: Yahoo Finance via yfinance (unofficial, delayed/limited feed).
No broker connection or order execution is implemented.
"""
from __future__ import annotations
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data.fetcher import (MarketDataError, fetch_market_data, get_available_expirations,
                          search_tickers, fetch_multiple_expiries)
from core.pricing import black_scholes_price, calculate_greeks
from core.implied_vol import solve_implied_volatility, implied_volatility_chain
from analytics.strategy_engine import analyze_strategy

st.set_page_config(page_title="QuantOptions Engine | Volatility & Strategy Lab", page_icon="📊", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.25rem;max-width:1700px}
[data-testid="stMetricValue"]{font-variant-numeric:tabular-nums}
.small-note{font-size:.85rem;color:var(--text-color)}
</style>""", unsafe_allow_html=True)
st.title("QuantOptions Engine")
st.caption("Black–Scholes–Merton · Implied volatility · Smile & term structure · Greeks analytics · Strategy lab")
st.warning("Research and paper-analysis only. Yahoo Finance data may be delayed, incomplete or rate-limited. This app does not place orders and does not guarantee executable prices or profitable trades.")

@st.cache_data(ttl=90, show_spinner=False)
def cached_search(q):
    return search_tickers(q)
@st.cache_data(ttl=90, show_spinner=False)
def cached_expiries(sym):
    return get_available_expirations(sym)
@st.cache_data(ttl=60, show_spinner=False)
def cached_chain(sym, exp, r):
    return fetch_market_data(sym, exp, risk_free_rate=r)
@st.cache_data(ttl=120, show_spinner=False)
def cached_multi(sym, exps, r):
    return fetch_multiple_expiries(sym, list(exps), risk_free_rate=r)

def add_mid_iv(frame, spot, rate, q):
    df = frame.copy()
    valid = df.bid.gt(0) & df.ask.gt(0) & df.ask.ge(df.bid)
    df["market_mid"] = ((df.bid + df.ask) / 2).where(valid)
    return implied_volatility_chain(df, spot=spot, rate=rate, dividend_yield=q)

def fmt_pct(x): return f"{100*x:.2f}%" if pd.notna(x) else "—"

with st.sidebar:
    st.header("Market & model settings")
    search_q = st.text_input("Find a ticker", "AAPL", help="Search company/ETF names or symbols through Yahoo Finance.")
    if st.button("Search tickers", use_container_width=True):
        try:
            st.session_state.search_results = cached_search(search_q)
        except Exception as e: st.error(str(e))
    results = st.session_state.get("search_results", pd.DataFrame())
    if isinstance(results, pd.DataFrame) and not results.empty:
        choices = results.symbol.dropna().astype(str).tolist()
        labels = {str(r.symbol): f"{r.symbol} — {r.name} ({r.exchange})" for r in results.itertuples()}
        picked_symbol = st.selectbox("Click to select a ticker", choices, format_func=lambda x: labels.get(x, x))
        default_symbol = picked_symbol if picked_symbol else (choices[0] if choices else "AAPL")
    else: default_symbol = search_q.strip().upper() or "AAPL"
    ticker = st.text_input("Selected underlying ticker", value=st.session_state.get("ticker", default_symbol)).strip().upper()
    st.session_state.ticker = ticker
    rate_pct = st.number_input("Risk-free rate (%)", -5.0, 30.0, 4.0, .25)
    q_pct = st.number_input("Continuous dividend yield (%)", -5.0, 30.0, 0.0, .25)
    rate, q = rate_pct/100, q_pct/100
    st.caption("Rates are annualized decimals inside the model.")
    refresh = st.button("Fetch option expirations", type="primary", use_container_width=True)

if refresh or "expiries" not in st.session_state or st.session_state.get("expiry_ticker") != ticker:
    try:
        with st.spinner(f"Searching listed expirations for {ticker}..."):
            st.session_state.expiries = cached_expiries(ticker)
            st.session_state.expiry_ticker = ticker
    except Exception as e:
        st.session_state.expiries = []
        st.error(f"Could not fetch expirations: {e}")
expiries = st.session_state.get("expiries", [])
if not expiries:
    st.info("Enter a valid listed ticker, then click **Fetch option expirations**. Examples: AAPL, MSFT, SPY, NVDA. The selected data provider currently targets Yahoo Finance-listed US options.")
    st.stop()

with st.sidebar:
    expiry = st.selectbox("Primary option expiry", expiries, index=0)
    max_curve_exp = st.slider("Expiries for volatility curve/surface", 2, min(12, len(expiries)), min(6, len(expiries))) if len(expiries)>=2 else 1
    fetch_btn = st.button("Load selected option chain", use_container_width=True)

try:
    if fetch_btn or st.session_state.get("chain_key") != (ticker, expiry, rate):
        with st.spinner(f"Loading {ticker} chain for {expiry}..."):
            st.session_state.snapshot = cached_chain(ticker, expiry, rate)
            st.session_state.chain_key = (ticker, expiry, rate)
    snap = st.session_state.snapshot
except Exception as e:
    st.error(f"Option chain could not be loaded: {e}")
    st.stop()
spot = float(snap.spot)
chain = snap.combined_options().copy()
chain = add_mid_iv(chain, spot, rate, q)
chain["iv_pct"] = chain.implied_volatility*100
chain["model_price_25"] = [black_scholes_price(spot, float(r.strike), float(r.time_to_expiry), rate, .25, str(r.option_type), q) if r.time_to_expiry>0 else np.nan for r in chain.itertuples()]
chain["spread_pct"] = np.where(chain.market_mid>0, (chain.ask-chain.bid)/chain.market_mid*100, np.nan)

h = st.columns(4)
h[0].metric("Underlying", f"{spot:,.2f}")
h[1].metric("Contracts loaded", f"{len(chain):,}")
h[2].metric("Calls / puts", f"{len(snap.calls)} / {len(snap.puts)}")
h[3].metric("Valid solved IVs", f"{chain.implied_volatility.notna().sum():,}")
st.caption(f"{ticker} · {expiry} · retrieved {snap.retrieved_at:%Y-%m-%d %H:%M:%S UTC} · Yahoo Finance/yfinance")

# Cache multi-expiry analysis independently, and keep user-controlled number of expiries.
vol_data = pd.DataFrame()
if len(expiries) >= 2:
    try:
        selected_exps = tuple(expiries[:max_curve_exp])
        with st.spinner("Building cross-expiry implied-volatility data..."):
            vol_data = cached_multi(ticker, selected_exps, rate)
        vol_data = add_mid_iv(vol_data, spot, rate, q)
        vol_data["moneyness"] = vol_data.strike / spot
        vol_data["days_to_expiry"] = vol_data.time_to_expiry * 365
    except Exception as e:
        st.warning(f"Cross-expiry curve unavailable: {e}")

tabs = st.tabs(["Market chain", "Option calculator", "IV smile & curve", "Greeks through time", "Strategy & hedge lab", "Signals & research notes"])

with tabs[0]:
    st.subheader("Live option chain snapshot")
    side = st.radio("Contract side", ["All", "Calls", "Puts"], horizontal=True, key="chain_side")
    view = chain if side=="All" else chain[chain.option_type==("call" if side=="Calls" else "put")]
    cols = [c for c in ["contractSymbol","option_type","expiration","strike","bid","ask","market_mid","lastPrice","volume","openInterest","implied_volatility","iv_status","iv_price_error"] if c in view]
    st.dataframe(view[cols], use_container_width=True, hide_index=True)
    st.download_button("Download selected chain CSV", view[cols].to_csv(index=False), f"{ticker}_{expiry}_chain.csv", "text/csv")
    st.caption("The provider's own IV is not used for the primary IV analytics: the app solves IV from the bid/ask midpoint using the BSM pricing formula. Invalid or non-identifiable quotes remain flagged.")

with tabs[1]:
    st.subheader("Price an option or solve its implied volatility")
    mode = st.radio("Choose calculation", ["Calculate theoretical option price", "Solve implied volatility from market price"], horizontal=True)
    contract_names = chain.contractSymbol.fillna("unknown contract").astype(str).tolist()
    selected_idx = st.selectbox("Optional: choose a listed contract", list(range(len(chain))), format_func=lambda i: f"{chain.iloc[i].contractSymbol} · {chain.iloc[i].option_type.upper()} · K={chain.iloc[i].strike:g}")
    cr = chain.iloc[selected_idx]
    with st.form("option_calc_form"):
        c1,c2,c3 = st.columns(3)
        calc_ticker = c1.text_input("Underlying ticker", ticker)
        calc_spot = c1.number_input("Underlying spot", min_value=.0001, value=float(spot), format="%.4f")
        calc_strike = c2.number_input("Strike", min_value=.0001, value=float(cr.strike), format="%.4f")
        calc_type = c2.selectbox("Option type", ["call","put"], index=0 if cr.option_type=="call" else 1)
        calc_days = c3.number_input("Days to expiry", min_value=.01, value=max(float(cr.time_to_expiry)*365, 1.0), step=1.0)
        calc_vol_pct = c3.number_input("Volatility (%) for price", min_value=.01, max_value=500.0, value=float(cr.implied_volatility*100) if pd.notna(cr.implied_volatility) else 25.0)
        market_px = st.number_input("Observed market option price (for IV mode)", min_value=0.0, value=float(cr.market_mid) if pd.notna(cr.market_mid) and cr.market_mid>0 else 1.0, format="%.4f")
        submit_calc = st.form_submit_button("Calculate option price / implied volatility", type="primary")
    if submit_calc:
        try:
            t = float(calc_days)/365
            if mode.startswith("Calculate"):
                px_model = black_scholes_price(calc_spot, calc_strike, t, rate, calc_vol_pct/100, calc_type, q)
                greeks = calculate_greeks(calc_spot, calc_strike, t, rate, calc_vol_pct/100, calc_type, q)
                st.success(f"{calc_ticker.upper()} {calc_type.upper()} theoretical BSM price: {px_model:.4f}")
                gc = st.columns(5)
                for c,g in zip(gc,["delta","gamma","theta","vega","rho"]): c.metric(g.title(), f"{greeks[g]:.6f}")
            else:
                solved = solve_implied_volatility(market_px, calc_spot, calc_strike, t, rate, calc_type, dividend_yield=q)
                st.success(f"Solved Black–Scholes–Merton implied volatility: {solved.volatility*100:.4f}%")
                st.write({"status":solved.status,"iterations":solved.iterations,"reconstructed_model_price":solved.model_price,"price_error":solved.price_error})
        except Exception as e: st.error(f"Calculation failed: {e}")
    st.caption("Ticker field labels the calculation; numerical pricing uses the spot and contract inputs shown. For listed-contract calculations, verify the quote, expiry timestamp, exercise style and dividend assumptions.")

with tabs[2]:
    st.subheader("Implied-volatility smile, term structure and surface")
    if not vol_data.empty and vol_data.implied_volatility.notna().any():
        valid_iv = vol_data.dropna(subset=["implied_volatility"]).copy()
        valid_iv = valid_iv[(valid_iv.implied_volatility>0)&(valid_iv.implied_volatility<5)]
        vside = st.radio("Smile side", ["Calls","Puts","Both"], horizontal=True, key="smile_side")
        if vside!="Both": valid_iv=valid_iv[valid_iv.option_type==("call" if vside=="Calls" else "put")]
        fig=px.scatter(valid_iv,x="strike",y=valid_iv.implied_volatility*100,color="expiration",symbol="option_type",hover_data=["contractSymbol","market_mid","volume","openInterest"],title="Solved IV by strike across expiries")
        fig.update_yaxes(title="Implied volatility (%)")
        st.plotly_chart(fig,use_container_width=True)
        valid_iv["abs_atm_distance"]=(valid_iv.moneyness-1).abs()
        atm_rows=valid_iv.sort_values("abs_atm_distance").groupby("expiration",as_index=False).first()
        atm_rows["ATM IV (%)"]=atm_rows.implied_volatility*100
        st.plotly_chart(px.line(atm_rows,x="expiration",y="ATM IV (%)",markers=True,title="Near-ATM implied-volatility term structure"),use_container_width=True)
        # 3D surface from binned moneyness and days-to-expiry to reduce quote noise.
        surf=valid_iv.copy(); surf["moneyness_bin"]=(surf.moneyness*100).round()/100
        surf["expiry_days_bin"]=(surf.days_to_expiry/7).round()*7
        piv=surf.pivot_table(index="expiry_days_bin",columns="moneyness_bin",values="implied_volatility",aggfunc="median")*100
        if piv.shape[0]>=2 and piv.shape[1]>=2:
            fig3=go.Figure(data=[go.Surface(x=piv.columns.to_numpy(),y=piv.index.to_numpy(),z=piv.to_numpy(),colorbar_title="IV %")])
            fig3.update_layout(title="Interactive implied-volatility surface",scene={"xaxis_title":"Strike / spot","yaxis_title":"Days to expiry","zaxis_title":"IV (%)"},height=650)
            st.plotly_chart(fig3,use_container_width=True)
        st.dataframe(valid_iv[["contractSymbol","option_type","expiration","strike","moneyness","market_mid","implied_volatility","iv_status","iv_price_error"]],use_container_width=True,hide_index=True)
    else:
        st.info("No valid cross-expiry IV points. Try another ticker or fetch fewer expiries. The app needs valid two-sided quotes and positive time value for a reliable IV root.")
    st.caption("Smile = IV versus strike for one expiry. Term structure/curve = IV across expiries. Surface = strike/moneyness and expiry together. These are recalculated from market midpoint prices using the BSM inversion, not copied from the provider's IV field.")

with tabs[3]:
    st.subheader("How Greeks change as maturity approaches")
    valid=chain.dropna(subset=["strike","time_to_expiry"])
    if not valid.empty:
        ix=st.selectbox("Contract for time-decay analysis",list(valid.index),format_func=lambda i:f"{valid.loc[i,'contractSymbol']} · {valid.loc[i,'option_type']} · K={valid.loc[i,'strike']:g}",key="greek_contract")
        row=chain.loc[ix]
        gspot=st.slider("Spot shock for this analysis (%)",-30,30,0,1)/100
        gvol=st.slider("Volatility shock (percentage points)",-30,30,0,1)/100
        s0=spot*(1+gspot); vol0=float(row.implied_volatility) if pd.notna(row.implied_volatility) else .25
        vol0=max(.001,vol0+gvol); k=float(row.strike); t0=float(row.time_to_expiry); kind=str(row.option_type)
        tgrid=np.linspace(max(t0,1/365),1/365,80)
        rows=[]
        for tt in tgrid:
            gg=calculate_greeks(s0,k,float(tt),rate,vol0,kind,q)
            rows.append({"days_to_expiry":tt*365,**gg,"model_price":black_scholes_price(s0,k,float(tt),rate,vol0,kind,q)})
        gd=pd.DataFrame(rows).sort_values("days_to_expiry")
        greek=st.multiselect("Greeks / value to plot",["delta","gamma","theta","vega","rho","model_price"],default=["delta","gamma","theta","vega"])
        if greek:
            long=gd.melt(id_vars="days_to_expiry",value_vars=greek,var_name="metric",value_name="value")
            st.plotly_chart(px.line(long,x="days_to_expiry",y="value",color="metric",markers=False,title="BSM Greeks and price versus remaining maturity"),use_container_width=True)
        current=calculate_greeks(s0,k,t0,rate,vol0,kind,q)
        c=st.columns(5)
        for cc,key in zip(c,["delta","gamma","theta","vega","rho"]): cc.metric(f"Current {key}",f"{current[key]:.6f}")
        st.caption("This is a controlled time-decay experiment: spot, rate, dividend yield and volatility are held constant while time to expiry decreases. Real markets change all of these simultaneously. Theta is per year; Vega/Rho are per 1.00 decimal change.")
    else: st.info("No valid contracts available for Greeks analysis.")

with tabs[4]:
    st.subheader("Multi-leg strategy builder and hedge diagnostics")
    st.write("Add each option leg. Quantity is signed: positive = buy/long; negative = sell/short. Entry premium is per underlying share; contract multipliers are not automatically applied.")
    call_rows=chain[chain.option_type=="call"].dropna(subset=["strike"]).copy()
    call_rows=call_rows.sort_values("strike",key=lambda x:(x-spot).abs())
    seed_rows=[]
    for j, (_, cr0) in enumerate(call_rows.head(2).iterrows()):
        entry0=cr0.market_mid if pd.notna(cr0.market_mid) and cr0.market_mid>0 else (cr0.lastPrice if pd.notna(cr0.lastPrice) and cr0.lastPrice>0 else 0.01)
        iv0=cr0.implied_volatility if pd.notna(cr0.implied_volatility) and cr0.implied_volatility>0 else .25
        seed_rows.append({"option_type":"call","strike":float(cr0.strike),"time_to_expiry":max(float(cr0.time_to_expiry),1/365),"volatility":float(iv0),"quantity":1 if j==0 else -1,"entry_price":float(entry0)})
    if not seed_rows:
        seed_rows=[{"option_type":"call","strike":spot,"time_to_expiry":.25,"volatility":.25,"quantity":1,"entry_price":1.0}]
    seed=pd.DataFrame(seed_rows)
    if "strategy_legs" not in st.session_state: st.session_state.strategy_legs=seed
    legs=st.data_editor(st.session_state.strategy_legs,num_rows="dynamic",use_container_width=True,hide_index=True,column_config={"option_type":st.column_config.SelectboxColumn(options=["call","put"]),"strike":st.column_config.NumberColumn(min_value=.0001),"time_to_expiry":st.column_config.NumberColumn(min_value=.0001,help="Years to expiry (ACT/365)"),"volatility":st.column_config.NumberColumn(min_value=.0001,max_value=5.0,help="Decimal IV, e.g. 0.25"),"quantity":st.column_config.NumberColumn(help="Signed quantity"),"entry_price":st.column_config.NumberColumn(min_value=0.0)})
    st.session_state.strategy_legs=legs
    sc1,sc2=st.columns(2)
    days_forward=sc1.slider("Revaluation horizon (days)",0,90,7)
    iv_move=sc2.slider("IV scenario shift (percentage points)",-30,30,0)/100
    if st.button("Analyse strategy & hedge",type="primary"):
        try:
            result=analyze_strategy(legs,spot=spot,rate=rate,dividend_yield=q,days_forward=days_forward,iv_shift=iv_move)
            st.session_state.strategy_result=result
        except Exception as e: st.error(f"Strategy analysis failed: {e}")
    result=st.session_state.get("strategy_result")
    if result:
        mc=st.columns(6)
        for cc,label,val in zip(mc,["Net Delta","Net Gamma","Net Theta / yr","Net Vega / 1%","Net Rho","Delta hedge shares"],[result['net_delta'],result['net_gamma'],result['net_theta'],result['net_vega']*.01,result['net_rho'],result['delta_hedge_shares']]): cc.metric(label,f"{val:,.5f}")
        fig=go.Figure(); fig.add_trace(go.Scatter(x=result["spot_grid"],y=result["expiry_pnl"],mode="lines",name="Expiry P&L per share")); fig.add_hline(y=0,line_dash="dash"); fig.update_layout(title="Strategy expiration payoff (per share; before fees)",xaxis_title="Underlying at expiry",yaxis_title="P&L per share")
        st.plotly_chart(fig,use_container_width=True)
        st.plotly_chart(px.line(result["scenario"],x="spot_change_pct",y="pnl_per_share",markers=True,title=f"Repriced scenario P&L after {days_forward} days (IV shift {iv_move*100:+.1f} pp)"),use_container_width=True)
        st.dataframe(result["legs"],use_container_width=True,hide_index=True)
        hedge=-result["net_delta"]
        st.info(f"Delta-neutral diagnostic: approximately {hedge:+.4f} underlying shares per one strategy unit. This is a model hedge estimate, not an order instruction. Re-hedge frequency, lot sizes, costs and liquidity matter.")
        if st.button("Save strategy to paper portfolio", type="secondary"):
            if "paper_book" not in st.session_state: st.session_state.paper_book=[]
            st.session_state.paper_book.append({"ticker":ticker,"legs":legs.to_dict(orient="records"),"saved_at":datetime.utcnow().isoformat(),"entry_spot":spot})
            st.success("Strategy saved to this browser session's paper portfolio. No order was sent.")

    st.markdown("#### Paper portfolio (session-only simulation)")
    book=st.session_state.get("paper_book",[])
    if book:
        portfolio_rows=[]
        now_utc=datetime.utcnow()
        for pos_no,pos in enumerate(book):
            if pos["ticker"] != ticker: continue
            saved=datetime.fromisoformat(pos["saved_at"])
            elapsed=max((now_utc-saved).total_seconds(),0)/(365*24*3600)
            mtm=0.0; entry_cash=0.0; pdelta=pgamma=pvega=ptheta=0.0
            for leg in pos["legs"]:
                qty=float(leg["quantity"]); strike=float(leg["strike"]); t=max(float(leg["time_to_expiry"])-elapsed,0.0)
                vol=float(leg["volatility"]); kind=str(leg["option_type"]).lower(); entry=float(leg["entry_price"])
                if t>0: mark=float(black_scholes_price(spot,strike,t,rate,vol,kind,q)); gg=calculate_greeks(spot,strike,t,rate,vol,kind,q)
                else:
                    mark=max(spot-strike,0) if kind=="call" else max(strike-spot,0)
                    gg={"delta":(1.0 if spot>strike else 0.0) if kind=="call" else (-1.0 if spot<strike else 0.0),"gamma":0.0,"theta":0.0,"vega":0.0}
                mtm += qty*mark; entry_cash += qty*entry
                pdelta += qty*float(gg["delta"]); pgamma += qty*float(gg["gamma"]); ptheta += qty*float(gg["theta"]); pvega += qty*float(gg["vega"])*.01
            portfolio_rows.append({"position_id":pos_no+1,"ticker":ticker,"saved_at":pos["saved_at"],"entry_spot":pos["entry_spot"],"current_spot":spot,"model_PnL_per_share":mtm-entry_cash,"net_delta":pdelta,"net_gamma":pgamma,"net_theta_per_year":ptheta,"net_vega_per_1pct":pvega})
        if portfolio_rows:
            pframe=pd.DataFrame(portfolio_rows)
            st.dataframe(pframe,use_container_width=True,hide_index=True)
            st.metric("Paper book model P&L (per strategy unit, before costs)",f"{pframe.model_PnL_per_share.sum():+.4f}")
            close_id=st.selectbox("Remove a paper position",pframe.position_id.tolist(),key="paper_remove_id")
            if st.button("Remove selected paper position"):
                target=book[int(close_id)-1]
                st.session_state.paper_book.remove(target)
                st.rerun()
        else: st.caption("No saved positions for this ticker. Paper positions are session-only and disappear when the session ends.")
    else:
        st.caption("Analyse a strategy and select 'Save strategy to paper portfolio' to start a simulated position ledger.")

with tabs[5]:
    st.subheader("Research signals and strategy education")
    iv_ok=chain.dropna(subset=["implied_volatility"]).copy()
    if not iv_ok.empty:
        atm=iv_ok.iloc[(iv_ok.strike-spot).abs().argsort()[:min(6,len(iv_ok))]]
        atm_iv=float(atm.implied_volatility.median())
        st.metric("Near-ATM solved IV",fmt_pct(atm_iv))
        st.write("**Dynamic observations**")
        if atm_iv>.60: st.write("- Near-ATM IV is elevated (>60%). Option premiums may be expensive in absolute terms; compare IV with its own history and expected realized volatility before considering short-volatility structures.")
        elif atm_iv<.20: st.write("- Near-ATM IV is relatively low (<20%). Long-volatility structures may be worth researching, but low IV alone is not a buy signal.")
        else: st.write("- Near-ATM IV is between 20% and 60%. Evaluate skew, term structure, event risk, liquidity and transaction costs rather than relying on the level alone.")
        calls=iv_ok[iv_ok.option_type=="call"]; puts=iv_ok[iv_ok.option_type=="put"]
        if len(calls) and len(puts):
            c_atm=calls.iloc[(calls.strike-spot).abs().argsort()[:min(5,len(calls))]].implied_volatility.median()
            p_atm=puts.iloc[(puts.strike-spot).abs().argsort()[:min(5,len(puts))]].implied_volatility.median()
            st.write(f"- Near-ATM put IV minus call IV: **{(p_atm-c_atm)*100:+.2f} volatility points**. A positive value indicates put-side richness in this selected snapshot; verify matched strikes and expiries.")
        spread=chain.spread_pct.replace([np.inf,-np.inf],np.nan).median()
        st.write(f"- Median quoted spread / midpoint: **{spread:.2f}%**." if pd.notna(spread) else "- Bid/ask spread statistic unavailable.")
    else: st.info("Load valid two-sided quotes to generate IV observations.")
    st.markdown("#### Strategy playbook (educational comparison)")
    strategies=pd.DataFrame([
        ["Covered call","Long underlying + short call","Mildly bullish / neutral","Premium income; capped upside; downside remains in shares"],
        ["Protective put","Long underlying + long put","Bullish with downside floor","Insurance-like hedge; put premium is the cost"],
        ["Bull call spread","Long lower-strike call + short higher-strike call","Moderately bullish","Lower debit than a naked call; capped upside"],
        ["Bear put spread","Long higher-strike put + short lower-strike put","Moderately bearish","Defined-risk bearish exposure; capped payoff"],
        ["Long straddle","Long ATM call + long ATM put","Large move expected; direction uncertain","Long gamma/vega; needs movement to overcome premium and decay"],
        ["Iron condor","Short OTM call/put spreads","Range-bound; short volatility","Defined risk if wings are present; short gamma/vega and gap risk"],
        ["Calendar spread","Short near expiry + long later expiry at similar strike","Term-structure / near-term decay view","Expiry and volatility-surface risk; not a pure IV trade"],
        ["Delta hedge","Offset option delta with underlying or other options","Reduce first-order directional exposure","Gamma causes delta to change; hedging has costs and residual risks"],
    ],columns=["Strategy","Typical legs","Market thesis","Main trade-off / risk"])
    st.dataframe(strategies,use_container_width=True,hide_index=True)
    st.markdown("#### Interpretation guardrails")
    st.write("- A high IV is not automatically overpriced: compare it with realized volatility, event risk, skew, term structure and your forecast.")
    st.write("- Greeks are local sensitivities, not fixed exposures. Gamma changes delta; Vega changes with spot, strike and maturity; theta accelerates near expiry in many ATM cases.")
    st.write("- For multi-leg portfolios, monitor net Greeks, payoff bounds, assignment/exercise style, liquidity, bid/ask costs, margin and stress P&L.")
    st.write("- Signals here are descriptive rules, not trained forecasts. No strategy is recommended as suitable without portfolio, risk tolerance and execution assumptions.")

st.divider()
st.caption("QuantOptions Engine · BSM European-option baseline · ACT/365 maturity · Continuous dividend yield · Research only; no broker connectivity, live execution, or personalized financial advice.")
