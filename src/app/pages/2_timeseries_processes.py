import streamlit as st
import pandas as pd
from app.core.Processes import RandomWalk
from app.core.timeseries import TimeSeriesAnalyser as TSA
from app._utils import get_timeseries_by_ticker, compute_returs
from app.core.visualizers.Visualizer import Visualizer

st.set_page_config(page_title="Timeseries",layout="wide",page_icon="X")
st.title("Timeseries analyzer")
st.header("This module shows some basic timeseries data, and is also able to show their properties on demand")

st.divider()


timeseries_ticker = st.text_input("Provide a ticker to visualize the corresponding timeseries or type RW for random walk")
st.divider()

TO_VISUALIZE = ['Close', 'RawChange', 'LogChange']

@st.cache_data
def load_data(ticker):
    if ticker == "RW":
        rw = RandomWalk()
        timeseries = rw.generate(1000)
        #we need to convert the ts to pandas sadly
        df = pd.DataFrame(timeseries, columns=['Close'])
    else:
        df = get_timeseries_by_ticker(ticker)
    df = compute_returs(df)
    return df

if timeseries_ticker:
    df = load_data(timeseries_ticker)
    st.write(f"Index Type: {type(df.index)}")
    col1, col2 = st.columns(2)

    with col1: show_mean = st.checkbox("-Mean")
    with col2: show_stddev = st.checkbox("-Standard dev")

    fig = Visualizer.visualize(
        df=df,
        values_to_visualize=TO_VISUALIZE,
        ticker=timeseries_ticker,
        show_mean=show_mean,
        show_stddev=show_stddev,
        show_grid=True,
        get_fig=True,
    )

    st.pyplot(fig, width='stretch')

    st.divider()

    col1_d, col2_d = st.columns(2)

    with col1_d: show_mean_d = st.checkbox("--Mean")
    with col2_d: show_stddev_d = st.checkbox("--Standard dev")

    fig_d = Visualizer.visualize_distribution(
        df=df,
        values_to_visualize=TO_VISUALIZE,
        ticker=timeseries_ticker,
        show_mean=show_mean_d,
        show_stddev=show_stddev_d,
        get_fig=True,
    )

    st.pyplot(fig_d, width='stretch')
    