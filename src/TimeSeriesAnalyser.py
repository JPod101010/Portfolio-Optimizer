import numpy as np
from math import sqrt

time_series_t = np.ndarray[1,float]

class Filters:
    @staticmethod
    def LinearFilter(weights : time_series_t, input_array : time_series_t) -> time_series_t:
        Y = np.zeros_like(input_array)

        for i in range(len(Y)):
            Y[i] = weights[i] * input_array[i] + (Y[i-1] if i > 0 else 0)

        return Y
    
    @staticmethod 
    def Convolution(input_array : time_series_t, filter_ : time_series_t) -> time_series_t:
        K = len(filter_)
        N = len(input_array)
        result = np.zeros_like(input_array)
        for n in range(N):
            val = 0.0
            for k in range(K):
                if n - k >= 0:
                    val += filter_[k] * input_array[n - k]
                else:
                    val = np.nan
            result[n] = val

        return result

class TimeSeriesAnalyser(Filters):
    @staticmethod
    def SMA(input_array : time_series_t, window : int) -> time_series_t:
        if window <= 0:
            raise ValueError("Window size must be positive.")
        if window > len(input_array):
            raise ValueError("Window size cannot exceed input length.")
        
        weights = np.full(window, 1) / window

        return Filters.Convolution(
            input_array=input_array,
            filter_=weights, 
        )

    @staticmethod
    def Mean(input_array : time_series_t) -> float:
        N = len(input_array)
        return sum(input_array) / N

    @staticmethod
    def Variance(input_array : time_series_t) -> float:
        return TimeSeriesAnalyser.Mean(
            (input_array - TimeSeriesAnalyser.Mean(
                input_array
            ))**2
        )

    @staticmethod
    def StandardDeviation(input_array : time_series_t) -> float:
        return sqrt(
            TimeSeriesAnalyser.Variance(input_array)
        )

    @staticmethod
    def CoefficientOfVariance(input_array : time_series_t) -> float:
        return (TimeSeriesAnalyser.StandardDeviation(
            input_array
        ) / TimeSeriesAnalyser.Mean(
            input_array
        ))