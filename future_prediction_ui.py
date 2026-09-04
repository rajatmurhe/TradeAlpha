import streamlit as st
import pandas as pd
import numpy as np
from future_prediction_engine import generate_prediction

def render_future_prediction_terminal():
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Header area
    st.markdown(
        """
        <div style="margin-bottom:20px;">
            <div style="font-size:18px; font-weight:600; color:#e6edf3; margin-bottom:6px;">Investment Forecast</div>
            <div style="font-size:13px; color:#8b949e;">Simulate future portfolio projections based on quantitative market data.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Control Panel Layout
    c1, c2, c3 = st.columns([1.5, 1.5, 1])

    with c1:
        horizon = st.selectbox(
            "Investment Horizon",
            ["1 Day", "1 Week", "1 Month", "1 Year"],
            key="future_horizon"
        )
    with c2:
        investment = st.number_input(
            "Planned Investment (₹)",
            min_value=1000.0,
            max_value=100000000.0,
            value=100000.0,
            step=10000.0,
            key="future_investment"
        )
    with c3:
        st.write("") # spacing to align button
        st.write("")
        generate = st.button("Generate Projections", use_container_width=True, type="primary")

    # Disclaimer
    st.markdown(
        """
        <div style="
            background:rgba(210,153,34,0.08);
            border:1px solid rgba(210,153,34,0.2);
            padding:10px 12px;
            border-radius:6px;
            margin-top:8px;
            margin-bottom:24px;
        ">
            <span style="
                color:#d29922;
                font-size:11px;
                font-weight:500;
            ">
                DISCLAIMER: Projections are model-based estimates using
                current market signals. Longer-term results assume the
                influence of today's signals gradually decreases and
                are not guaranteed returns.
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    if generate or "future_prediction_result" not in st.session_state:
        try:
            with st.spinner("Analyzing market conditions and generating projections..."):
                st.session_state.future_prediction_result = generate_prediction(
                    horizon=horizon,
                    investment=float(investment),
                )
        except Exception as e:
            st.error("Forecast generation failed.")
            st.exception(e)
            return

    # Load Result Data
    result = st.session_state.future_prediction_result
    active_horizon = result.get("horizon", horizon)
    invested = float(result.get("investment", investment))
    portfolio_return = float(result.get("portfolio_return", 0.0))
    projected_value = float(result.get("projected_value", invested))
    expected_profit = float(result.get("expected_profit", projected_value - invested))
    stocks = result.get("stocks", pd.DataFrame())
    
    if not isinstance(stocks, pd.DataFrame): stocks = pd.DataFrame(stocks)

    # --------------------------------------------------------
    # FORECAST SUMMARY CARDS
    # --------------------------------------------------------
    is_positive = portfolio_return >= 0
    sentiment_text = "Moderately Bullish" if is_positive else "Bearish"
    
    st.markdown(f"""<div style="font-size:14px; font-weight:600; color:#e6edf3; margin-top:10px; margin-bottom:12px;">{str(active_horizon).upper()} Portfolio Outlook</div>""", unsafe_allow_html=True)
    
    metric1, metric2, metric3, metric4 = st.columns(4)
    with metric1: 
        # Custom HTML to prevent text truncation
        st.markdown(
            f"""
            <div style="background:#0d1117; border:1px solid #21262d; border-radius:8px; padding:16px;">
                <div style="font-size:12px; color:#8b949e; margin-bottom:4px;">Overall Market Outlook</div>
                <div style="font-family:'IBM Plex Mono', monospace; font-size:18px; color:#e6edf3;">{sentiment_text}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    with metric2: st.metric("Expected Return", f"{portfolio_return * 100:+.2f}%")
    with metric3: st.metric("Projected Value", f"₹{projected_value:,.0f}")
    with metric4: st.metric("Potential P/L", f"₹{expected_profit:+,.0f}")
    
    # Inject Custom CSS for these specific metrics
    st.markdown(
        f"""
        <style>
        [data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace; font-size: 24px; color: #e6edf3; }}
        div[data-testid="stMetric"] {{ background: #0d1117; border: 1px solid #21262d; border-radius: 8px; padding: 16px; }}
        </style>
        """,
        unsafe_allow_html=True
    )

    if not stocks.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div style="font-size:14px; font-weight:600; color:#e6edf3; margin-bottom:12px;">Suggested Asset Distribution (₹)</div>""", unsafe_allow_html=True)
        
        display = stocks.copy()
        display["Ticker"] = display["Ticker"].astype(str).str.replace(".NS", "")
        display = display.sort_values("recommended_weight", ascending=False)
        
        # Filter out zero-weight assets to keep UI clean
        display = display[display["recommended_weight"] > 0].copy()
        
        # 1. Calculate raw capital allocation
        display["Raw Capital"] = display["recommended_weight"] * invested
        
        # 2. Calculate executable whole shares (floored)
        display["Executable Shares"] = np.floor(display["Raw Capital"] / display["Close"]).astype(int)
        
        # 3. Calculate actual capital deployed
        display["Actual Capital Deployed (₹)"] = (display["Executable Shares"] * display["Close"]).round(2)
        
        # Format table
        display = display[["Ticker", "Close", "projected_return", "recommended_weight", "Executable Shares", "Actual Capital Deployed (₹)"]]
        display.columns = ["Asset", "Current Price", "Projected Move", "Target Weight", "Executable Shares", "Capital Required"]
        
        display["Current Price"] = display["Current Price"].apply(lambda x: f"₹{float(x):,.2f}")
        display["Projected Move"] = display["Projected Move"].apply(lambda x: f"{float(x) * 100:+.2f}%")
        display["Target Weight"] = display["Target Weight"].apply(lambda x: f"{float(x) * 100:.1f}%")
        display["Capital Required"] = display["Capital Required"].apply(lambda x: f"₹{float(x):,.2f}")
        
        st.dataframe(display, use_container_width=True, hide_index=True)