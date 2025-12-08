from typing import List, Dict, Literal
import numpy as np
import yfinance as yf
import pandas as pd
import os
from scipy.optimize import minimize

class PortfolioOptimizer():
    def __init__(
            self, 
            tickers : List[str],
            period : str = '5y',
            interval : str = '1d'
        ):
        self._tickers = tickers
        self._period = period
        self._interval = interval
        self._data : Dict[str, pd.DataFrame] = {}
        self._DATA_PATH = f'data/portfolio/{self._tickers}p={self._period},i={self._interval}/'
        os.makedirs(self._DATA_PATH, exist_ok=True)

        # Cached matrices
        self._log_returns : pd.DataFrame = pd.DataFrame()
        self._cov_matrix : pd.DataFrame = pd.DataFrame()
        self._mean_returns : pd.Series = pd.Series()

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Flatten MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    def download_data(self):
        for ticker in self._tickers:
            df = yf.download(
                ticker, 
                period=self._period, 
                interval=self._interval, 
                auto_adjust=False
            )
            df = self._clean_data(df)
            self._data[ticker] = df

    def enrich_data(self):
        # Compute log returns for each asset
        log_returns_dict = {}
        for ticker, df in self._data.items():
            df['RawChange'] = df['Close'].diff()
            df['PctChange'] = df['Close'].pct_change()
            df['LogChange'] = np.log1p(df['PctChange'])
            log_returns_dict[ticker] = df['LogChange']
        # Construct matrix
        self._log_returns = pd.DataFrame(log_returns_dict)
        self._mean_returns = self._log_returns.mean()
        self._cov_matrix = self._log_returns.cov()

    def save_data(self):
        for ticker, dataframe in self._data.items():
            dataframe.to_csv(f"{self._DATA_PATH}{ticker}.csv")

    def _portfolio_metrics(self, weights: np.ndarray):
        """Compute portfolio return, volatility, and Sharpe ratio."""
        port_return = np.dot(weights, self._mean_returns)
        port_vol = np.sqrt(weights.T @ self._cov_matrix.values @ weights)
        sharpe_ratio = port_return / port_vol if port_vol != 0 else 0
        return port_return, port_vol, sharpe_ratio

    def _objective_neg_sharpe(self, weights: np.ndarray):
        """Objective function for maximizing Sharpe: minimize negative Sharpe."""
        _, _, sharpe = self._portfolio_metrics(weights)
        return -sharpe

    def optimize(
            self,
            method: Literal['sharpe'] = 'sharpe',
            bounds: List[tuple] = None,
            allow_short: bool = False
        ) -> Dict[str, float]:
        """Optimize portfolio weights."""
        n_assets = len(self._tickers)
        if bounds is None:
            # default: no shorting
            if allow_short:
                bounds = [(-1.0, 1.0) for _ in range(n_assets)]
            else:
                bounds = [(0.0, 1.0) for _ in range(n_assets)]

        # constraint: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

        # initial guess: equal weights
        init_guess = np.array([1/n_assets]*n_assets)

        if method == 'sharpe':
            res = minimize(
                self._objective_neg_sharpe,
                init_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        else:
            raise NotImplementedError(f"Optimization method {method} not implemented.")

        optimal_weights = res.x
        port_return, port_vol, sharpe_ratio = self._portfolio_metrics(optimal_weights)

        # return dict with weights + metrics
        return {
            'weights': dict(zip(self._tickers, optimal_weights)),
            'expected_return': port_return,
            'volatility': port_vol,
            'sharpe_ratio': sharpe_ratio
        }
