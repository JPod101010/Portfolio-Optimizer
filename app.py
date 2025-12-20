import streamlit as st
import numpy as np
from PortfolioOptimizer import PortfolioOptimizer

st.title("🎯 Portfolio Optimizer Pro")

with st.sidebar:
    tickers = st.text_input("Enter Tickers (comma separated)", "AAPL,MSFT,TSLA,GOOG")
    method = st.selectbox("Optimization Goal", ["sharpe", "risk", "return"])
    allow_short = st.checkbox("Allow Short Selling")
    run_btn = st.button("Optimize Portfolio")

if run_btn:
    ticker_list = [t.strip() for t in tickers.split(",")]
    optimizer = PortfolioOptimizer(tickers=ticker_list)
    
    with st.spinner("Fetching data and crunching numbers..."):
        optimizer.download_data()
        optimizer.enrich_data()
        results = optimizer.optimize(method=method, allow_short=allow_short)
    
    # Display Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Return", f"{results['expected_return']*252:.2%}")
    col2.metric("Volatility", f"{results['volatility']*np.sqrt(252):.2%}")
    col3.metric("Sharpe Ratio", f"{results['sharpe_ratio']*np.sqrt(252):.2f}")

    # Display Weights as a Bar Chart
    st.bar_chart(results['weights'])