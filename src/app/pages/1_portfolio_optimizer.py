import streamlit as st
import pandas as pd, numpy as np
import altair as alt
from app.core.portfolioOptimizer.PortfolioOptimizer import (
    PortfolioOptimizer,
    TIMEFRAME_NORM
)
import matplotlib.pyplot as plt


st.title("Portfolio Optimizer")
st.header("This module computes the optimal portfolio of selected stocks by a selected metric")

st.divider()

tickers_list = st.text_input("Input the tickers separated with a comma to construct the portfolio with")

methods = ['return', 'risk', 'sharpe']
timeframes = ['daily', 'monthly', 'quarterly', 'annually']

method = st.selectbox("Choose a method:", methods)
st.divider()

if tickers_list and method:
    _po = PortfolioOptimizer(
        tickers=tickers_list.split(","),
    )
    _po.download_data()
    _po.enrich_data()
    portfolio = _po.optimize(
        method=method,
    )

    timeframe = st.selectbox("Choose a timeframe:", timeframes)
    timeframe_norm = TIMEFRAME_NORM[timeframe]


    col1, col2, col3 = st.columns(3)

    col1.metric("Expected Return", f"{portfolio['expected_return']*100*timeframe_norm:.2f}%")
    col2.metric("Volatility", f"{portfolio['volatility']*100*np.sqrt(timeframe_norm):.2f}%")
    col3.metric("Sharpe Ratio", f"{portfolio['sharpe_ratio']*np.sqrt(timeframe_norm):.2f}")

    # --- Display Portfolio Allocations as a Histogram ---
    st.subheader("Portfolio Allocation")

    # Convert weights to DataFrame
    weights_df = pd.DataFrame(list(portfolio["weights"].items()), columns=["Asset", "Weight"])
    weights_df['Assets'] = weights_df.apply(lambda row: f"{row['Asset']} : {row['Weight']:.2f}", axis=1)
    st.bar_chart(data=weights_df, x='Asset', y='Weight')

    fig = alt.Chart(weights_df).mark_arc().encode(
        theta='Weight',
        color='Assets',
    )

    st.altair_chart(fig, width='stretch')