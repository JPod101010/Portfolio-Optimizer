import streamlit as st
import pandas as pd, numpy as np
import altair as alt
from app.database_engine import ENGINE
from app.core.portfolioOptimizer.PortfolioOptimizer import (
    PortfolioOptimizer,
    TIMEFRAME_NORM
)

st.set_page_config(page_title="Portfolio Optimizer",layout="wide",page_icon="X")
st.title("Portfolio Optimizer")
st.header("This module computes the optimal portfolio of selected stocks by a selected metric")

st.divider()

tickers_list = st.text_input("Input the tickers separated with a comma to construct the portfolio with")

methods = ['return', 'risk', 'sharpe']
timeframes = ['daily', 'monthly', 'quarterly', 'annually']

method = st.selectbox("Choose a method:", methods)
st.divider()

@st.cache_data(show_spinner="Loading market data...")
def load_prices_from_db():
    query = "SELECT * FROM prices"
    return pd.read_sql(query, ENGINE, parse_dates=["date_"])

@st.cache_data(show_spinner="Optimizing portfolio...")
def compute_portfolio(tickers, method, prices_df):
    po = PortfolioOptimizer(tickers=tickers)
    po.load_data_from_dataframe(prices_df)
    return po.optimize(method=method)

if tickers_list and method:
    
    prices_df = load_prices_from_db()
       
    portfolio = compute_portfolio(
        tickers=list(t.strip() for t in tickers_list.split(",")),
        method=method,
        prices_df=prices_df,
    )

    
    timeframe = st.selectbox("Choose a display timeframe:", timeframes)
    timeframe_norm = TIMEFRAME_NORM[timeframe]


    col1, col2, col3 = st.columns(3)

    col1.metric("Expected Return", f"{portfolio['expected_return']*100*timeframe_norm:.2f}%")
    col2.metric("Volatility", f"{portfolio['volatility']*100*np.sqrt(timeframe_norm):.2f}%")
    col3.metric("Sharpe Ratio", f"{portfolio['sharpe_ratio']*np.sqrt(timeframe_norm):.2f}")

    st.subheader("Portfolio Allocation")

    weights_df = pd.DataFrame(list(portfolio["weights"].items()), columns=["Asset", "Weight"])
    weights_df['Assets'] = weights_df.apply(lambda row: f"{row['Asset']} : {row['Weight']:.4f}", axis=1)
    st.bar_chart(data=weights_df, x='Asset', y='Weight')

    fig = alt.Chart(weights_df).mark_arc().encode(
        theta='Weight',
        color='Assets',
    )

    st.altair_chart(fig, width='stretch')