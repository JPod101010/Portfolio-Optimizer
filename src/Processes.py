import random
import numpy as np
from src.TimeSeriesAnalyser import time_series_t
from typing import List, Optional

class RandomWalk():
    def __init__(self, mean : float = 0, var : float = 1):
        self.mean = mean
        self.std_div = np.sqrt(var)

    def generate(self, n : int, start : int = 0) -> time_series_t:
        steps = np.array([
            random.normalvariate(self.mean, self.std_div)
            for _ in range(n)
        ])
        return np.cumsum(steps) + start
    
class TimeSeriesProcess:
    def __init__(self, mean: float = 0.0, var: float = 1.0):
        self.mean = mean
        self.std = np.sqrt(var)

    def generate(self, n: int) -> time_series_t:
        raise NotImplementedError("Subclasses must implement generate()")


class MovingAverageProcess(TimeSeriesProcess):
    def __init__(
            self, 
            weights: Optional[List[float]] = None,  #if this is set to None we end up with RW
            mean: float = 0.0,
            var: float = 1.0

        ) -> time_series_t:

        super().__init__(mean, var)
        self.steps = len(weights)
        self.weights = np.array(weights if weights is not None else np.zeros(self.steps))

    def generate(self, n: int) -> time_series_t:
        # generate a field of random terms of len N + steps (for calculating the upfront values) 
        random_term_ts = np.random.normal(self.mean, self.std, n + self.steps)
        final_ts = np.zeros(n)

        for t in range(n):
            #get a window of values from the index we at until the steps to account the average
            window = random_term_ts[t:t + self.steps + 1]
            #notice that the last random term is unweighted ... then we sumproduct all the other terms
            final_ts[t] = window[0] + np.dot(self.weights, window[1:])

        return final_ts
    
    def __str__(self):
        return (
            f"MA{self.steps} {self.weights} Process"
        )
    

class AutoregressiveProcess(TimeSeriesProcess):
    def __init__(
            self, 
            weights: List[float], 
            mean: float = 0.0,
            var: float = 1.0

        ) -> time_series_t:
        super().__init__(mean, var)
        self.weights = np.array(weights)
        self.steps = len(weights)

    def generate(self, n: int) -> np.ndarray:
        # generate a field of random terms of len N
        random_term_ts = np.random.normal(self.mean, self.std, n)
        final_ts = np.zeros(n)

        # get the first steps of the process 
        final_ts[:self.steps] = random_term_ts[:self.steps]

        # start at the next index and just calculate the AR process
        for t in range(self.steps, n):
            # we multiply the weight with the indexes reversed to ensure that the first weight corresponds
            # to the last term generated and then we get another random term to continue unweighted
            final_ts[t] = np.dot(self.weights, final_ts[t-self.steps:t][::-1]) + random_term_ts[t]

        return final_ts
    
    def __str__(self):
        return (
            f"AR{self.steps} {self.weights} Process"
        )


class AutoRegressiveMovingAverageProcess(TimeSeriesProcess):
    def __init__(
        self,
        AR_weights: Optional[List[float]] = None,
        MA_weights: Optional[List[float]] = None,
        mean: float = 0.0,
        var: float = 1.0,
    ):
        super().__init__(mean, var)

        self.AR_weights = np.array(AR_weights or [])
        self.MA_weights = np.array(MA_weights or [])
        self.AR_steps = len(self.AR_weights)
        self.MA_steps = len(self.MA_weights)

    def generate(self, n: int) -> time_series_t:
        # generate a field of random terms of len N + MA_steps (see MA process)
        random_term_ts = np.random.normal(self.mean, self.std, n + self.MA_steps)

        X = np.zeros(n)

        # initalize the final ts (X) with the more terms AR or MA
        init = max(self.AR_steps, self.MA_steps)
        X[:init] = random_term_ts[:init]

        # calculate each part if they are non-null and then sum together
        for t in range(init, n):
            AR_part = np.dot(self.AR_weights, X[t-self.AR_steps:t][::-1]) if self.p > 0 else 0.0
            MA_part = np.dot(self.MA_weights, random_term_ts[t-self.MA_steps:t][::-1]) if self.q > 0 else 0.0

            X[t] = AR_part + MA_part + random_term_ts[t]

        return X
    
    def __str__(self):
        return (
            f"ARMA: AR{self.AR_steps} {self.AR_weights} MA{self.MA_steps} {self.MA_weights} Process"
        )


from enum import Enum

class processEnum(Enum):
    MA = MovingAverageProcess,
    AR = AutoregressiveProcess,
    ARMA = AutoregressiveProcess,
    ARIMA = None,

# For simplicity i will only fit by hand MA1 and AR1 other for everything else
# Use the ARIMA model from the statsmodels
class MA1Model():
    def __init__(
        self,
        time_series_to_fit : time_series_t,
    ):
        self.time_series = time_series_to_fit

    def fit(

    ) -> TimeSeriesProcess:
        pass