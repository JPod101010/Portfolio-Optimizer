import random
import numpy as np
from src.TimeSeriesAnalyser import time_series_t

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
    

class MovingAverageProcess:
    def __init__(self, q=1, theta=None, mean=0, var=1):
        self.q = q
        self.theta = np.array(theta if theta is not None else np.zeros(q))
        self.mean = mean
        self.std = np.sqrt(var)

    def generate(self, n: int) -> np.ndarray:
        eps = np.random.normal(0, self.std, n + self.q)
        X = np.zeros(n)

        for t in range(n):
            window = eps[t : t + self.q + 1]
            X[t] = self.mean + window[0] + np.dot(self.theta, window[1:])

        return X
    

class AutoregressiveProcess:
    def __init__(self, phi: list[float], mean=0, var=1):
        self.phi = phi
        self.mean = mean
        self.std = np.sqrt(var)
        self.p = len(phi)

    def generate(self, n: int) -> np.ndarray:
        ts = np.zeros(n)

        for i in range(self.p):
            ts[i] = random.normalvariate(self.mean, self.std)

        for t in range(self.p, n):
            noise = random.normalvariate(self.mean, self.std)
            ts[t] = sum(self.phi[i] * ts[t - 1 - i] for i in range(self.p)) + noise
        
        return ts
