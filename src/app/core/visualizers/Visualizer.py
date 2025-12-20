import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from typing import List
import seaborn as sns
from app.core.timeseries.TimeSeriesAnalyser import TimeSeriesAnalyser as TSA
import math

class Visualizer:
    @staticmethod
    def visualize(
        df: pd.DataFrame, 
        values_to_visualize: List[str] = [], 
        ticker : str = '',
        show_mean : bool = False,
        show_stddev : bool = False,
        print_analysis : bool = False,
        show_grid : bool = True,
        get_fig  : bool = False,
        ):
        if not values_to_visualize:
            raise ValueError("No columns provided to visualize.")

        # Filter columns that actually exist in the dataframe
        columns = [col for col in values_to_visualize if col in df.columns]
        if not columns:
            raise ValueError("None of the specified columns exist in the DataFrame.")

        n_cols = 3  # max plots per row
        n_plots = len(columns)
        n_rows = math.ceil(n_plots / n_cols)

        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        axes = axes.flatten()  # flatten in case we have multiple rows

        for i, col in enumerate(columns):
            mean,_,stddev,_,_,_ = TSA.BasicAnalysis(df[col], verbose=print_analysis)
            axes[i].plot(df.index, df[col])
            axes[i].set_title(f"{ticker}-{col}")
            
            
            if show_mean:
                axes[i].axhline(y=mean, color='red', linestyle='--', linewidth=2, label=f'Mean ({mean:.2f})')
            
            if show_stddev:
                x = df.index 
                axes[i].axhline(y=mean + stddev, color='green', linestyle=':', alpha=0.5, label=f'+1σ ({mean + stddev:.2f})')
                axes[i].axhline(y=mean - stddev, color='green', linestyle=':', alpha=0.5, label=f'-1σ ({mean - stddev:.2f})')
                axes[i].fill_between(x, mean - stddev, mean + stddev, color='green', alpha=0.2, label='±1σ interval')


            axes[i].grid(show_grid)
            axes[i].legend()

        # Turn off any unused axes
        for j in range(len(columns), len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        if get_fig:
            return fig
        plt.show()

    @staticmethod
    def visualize_distribution(
        df: pd.DataFrame,
        values_to_visualize: List[str] = [],
        ticker: str = '',
        bins: int = 30,
        show_mean: bool = False,
        show_stddev: bool = False,
        show_grid: bool = True,
        print_analysis: bool = False,
        get_fig: bool = False,
    ):
        if not values_to_visualize:
            raise ValueError("No columns provided to visualize.")

        # Filter columns that actually exist
        columns = [col for col in values_to_visualize if col in df.columns]
        if not columns:
            raise ValueError("None of the specified columns exist in the DataFrame.")

        n_cols = 3  # max plots per row
        n_plots = len(columns)
        n_rows = math.ceil(n_plots / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(columns):
            data = df[col].dropna()  # remove NaNs for plotting

            # Histogram + KDE
            sns.histplot(data, bins=bins, kde=True, ax=axes[i], color='skyblue')
            axes[i].set_title(f"{ticker}-{col} Distribution")
            axes[i].grid(show_grid)

            mean,_,stddev,_,_,_ = TSA.BasicAnalysis(df[col], verbose=print_analysis)

            if show_mean:
                axes[i].axvline(mean, color='red', linestyle='--', label=f"Mean ({mean:.2f})")

            if show_stddev:
                #x = df.index 
                axes[i].axvline(mean + stddev, color='green', linestyle=':', label=f'+1σ ({mean + stddev:.2f})')
                axes[i].axvline(mean - stddev, color='green', linestyle=':', label=f'-1σ ({mean - stddev:.2f})')
                #axes[i].fill_between(x, mean - stddev, mean + stddev, color='green', alpha=0.2, label='±1σ interval')
            axes[i].legend()

        # Turn off any unused axes
        for j in range(len(columns), len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        if get_fig:
            return fig
        plt.show()


    
