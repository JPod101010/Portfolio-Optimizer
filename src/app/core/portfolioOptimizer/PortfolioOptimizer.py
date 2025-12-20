from typing import List, Dict, Literal
import numpy as np
import yfinance as yf
import pandas as pd
from pathlib import Path
import os
from .sp500_ticks import sp500_ticks
from .blacklist import BLACKLIST
from scipy.optimize import minimize

PATH_TO_DATA_PORTFOLIO = Path(__file__).parent.parent.parent.parent.parent / "data" / "portfolio"

OPTIMIZITATION_METHODS = Literal[
    'return',
    'risk',
    'sharpe'
]

TIMEFRAME = Literal[
    'daily',
    'monthly',
    'quarterly',
    'annually',
]

#expect daily data !
TIMEFRAME_NORM : Dict[TIMEFRAME, float]= {
    'daily' : 1.0,
    'monthly' : 21.0,
    'quarterly' : 63.0,
    'annually' : 252.0,
}

class PortfolioOptimizer():
    def __init__(
            self, 
            tickers : List[str],
            period : str = '5y',
            interval : str = '1d',
            dir_name : str = 'perfect-portfolio'
        ):
        self._tickers = tickers
        sp500_flag = False
        for ticker in self._tickers:
            if ticker in ['sp500','SP500']:
                sp500_flag = True
        if sp500_flag:
            self._tickers.extend(sp500_ticks)

        self._period = period
        self._interval = interval
        self._data : Dict[str, pd.DataFrame] = {}

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
        valid = {}

        for ticker in self._tickers:
            print(os.listdir(PATH_TO_DATA_PORTFOLIO))
            if ticker in BLACKLIST:
                continue
            if f"{ticker}.csv" in os.listdir(PATH_TO_DATA_PORTFOLIO):
                valid[ticker] = pd.read_csv(f"{PATH_TO_DATA_PORTFOLIO}/{ticker}.csv")
                continue

            try:
                df = yf.download(
                    ticker,
                    period=self._period,
                    interval=self._interval,
                    auto_adjust=False,
                    progress=False
                )
    
                if df is None or df.empty:
                    print(f"[WARN] Skipping {ticker}: no data returned")

                    continue

                df = self._clean_data(df)
                valid[ticker] = df
                df.to_csv(f"{PATH_TO_DATA_PORTFOLIO}/{ticker}.csv")

            except Exception as e:
                print(f"[ERROR] Failed downloading {ticker}: {e}")
                continue

        self._data = valid
        self._tickers = list(valid.keys())


    def enrich_data(self):
        # Compute log returns for each asset
        log_returns_dict = {}
        for ticker, df in self._data.items():
            df['RawChange'] = df['Close'].diff()
            df['PctChange'] = df['Close'].pct_change()
            df['LogChange'] = np.log1p(df['PctChange'])
            log_returns_dict[ticker] = df['LogChange']
        # Construct matrix
        self._log_returns = self._log_returns = pd.DataFrame(log_returns_dict).dropna()
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
    
    def _objective_risk(self, weights: np.ndarray):
        """Objective: minimize portfolio volatility."""
        _, vol, _ = self._portfolio_metrics(weights)
        return vol

    def _objective_neg_return(self, weights: np.ndarray):
        """Objective: maximize return by minimizing negative return."""
        ret, _, _ = self._portfolio_metrics(weights)
        return -ret

    def _get_constraints(self, method: OPTIMIZITATION_METHODS, target : tuple[TIMEFRAME, float] | None):
        constrains = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        if target is None: return constrains

        _timeframe, _target = target
        _norm = TIMEFRAME_NORM[_timeframe]

        daily_var_target = (_target ** 2) / TIMEFRAME_NORM[_timeframe]
        daily_ret_target = _target / _norm

        match method:
            case 'return':
                constrains.append({"type": "eq", "fun": lambda w: np.dot(w, self._mean_returns) - daily_ret_target})
            case 'risk':
                constrains.append({"type": "eq", "fun": lambda w: w.T @ self._cov_matrix.values @ w - daily_var_target})
        return constrains

    def optimize(
            self,
            method: OPTIMIZITATION_METHODS = 'sharpe',
            target: tuple[TIMEFRAME, float] | None = None,
            bounds: List[tuple] = None,
            allow_short: bool = False
        ) -> Dict[str, float]:
        """Optimize portfolio weights."""
        n_assets = len(self._tickers)

        # Default bounds (no shorting unless specified)
        if bounds is None:
            bounds = [(-1.0, 1.0) for _ in range(n_assets)] if allow_short else [(0.0, 1.0)] * n_assets

        # Constraint: weights sum to 1
        constraints = self._get_constraints(
            method=method,
            target=target
        )

        # Initial guess: equal weighting
        init_guess = np.array([1.0 / n_assets] * n_assets)

        if method == 'sharpe':
            objective = self._objective_neg_sharpe
        elif method == 'risk':
            objective = self._objective_risk
        elif method == 'return':
            objective = self._objective_neg_return
        else:
            raise NotImplementedError(f"Optimization method {method} is not implemented.")

        res = minimize(
            objective,
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimal_weights = res.x
        port_return, port_vol, sharpe_ratio = self._portfolio_metrics(optimal_weights)
        result_portfolio = {k: v for k, v in zip(self._tickers, optimal_weights) if v > 0.00000000001}
        return {
            'weights': result_portfolio,
            'expected_return': port_return,
            'volatility': port_vol,
            'sharpe_ratio': sharpe_ratio
        }

    def interpret_result(self, result: Dict[str, float], timeframe: TIMEFRAME):
        norm = TIMEFRAME_NORM[timeframe]

        # Clean floats
        weights = {k: float(v) for k, v in result["weights"].items()}
        ret = float(result["expected_return"]) * norm
        vol = float(result["volatility"]) * np.sqrt(norm)
        sharpe = float(result["sharpe_ratio"]) * np.sqrt(norm)

        print("=== Portfolio Results ===")
        print(f"Timeframe         : {timeframe}")
        print(f"Expected Return   : {ret:.4%}") 
        print(f"Risk (Volatility) : {vol:.4%}")
        print(f"Sharpe Ratio      : {sharpe:.4f}")
        print("\n--- Allocations ---")
        for asset, w in weights.items():
            if w < 0.0001: continue
            print(f"{asset:<10} : {w:.2%}")
   

