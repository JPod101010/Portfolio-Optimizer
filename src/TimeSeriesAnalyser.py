import numpy as np
from math import sqrt, prod
from typing import Tuple, List

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
    
    @staticmethod
    def GeometricMean(input_array : time_series_t) -> float:
        return prod(input_array) ** (1/len(input_array))


    @staticmethod
    def Covariance(input_arrayA : time_series_t, 
                   input_arrayB : time_series_t) -> float:
        """
        Tells how are the two time series correlated:
        >0 : positive correlation
        0  : no correlation
        <0 : negative correlation 
        """
        return TimeSeriesAnalyser.Mean(
            (input_arrayA - TimeSeriesAnalyser.Mean(input_arrayA)) * 
            (input_arrayB - TimeSeriesAnalyser.Mean(input_arrayB))
        )


    @staticmethod
    def Correlation(input_arrayA : time_series_t, 
                    input_arrayB : time_series_t) -> float:
        """
        Pearsons correlation coefficient TODO:
        """
        MEAN_A = TimeSeriesAnalyser.Mean(input_arrayA)
        MEAN_B = TimeSeriesAnalyser.Mean(input_arrayB)

        numerator = sum((input_arrayA - MEAN_A) * (input_arrayB - MEAN_B))
        denominator = sqrt(
            sum((input_arrayA - MEAN_A**2)) *
            sum((input_arrayB - MEAN_B**2))
        )

        return numerator / denominator

    @staticmethod
    def AutoCorrelation(input_array : time_series_t, lag : int = 1) -> float:
        return TimeSeriesAnalyser.Correlation(
            input_arrayA=input_array[:-lag],
            input_arrayB=input_array[lag:]
        )

    
    @staticmethod
    def AutoCorrelationAnalysis(input_array : time_series_t, MAX_LAG = 30) -> List[float]:
        MAX_LAG = min(MAX_LAG, len(input_array) - 1)
        ret_l = []
        for lag in range(1, MAX_LAG):
            ret_l.append(
                TimeSeriesAnalyser.AutoCorrelation(
                    input_array, lag
                )
            )
        return ret_l

    
    @staticmethod
    def BasicAnalysis(input_array : time_series_t) -> Tuple[float]:
        EX = TimeSeriesAnalyser.Mean(input_array)
        VX = TimeSeriesAnalyser.Variance(input_array)
        DX = TimeSeriesAnalyser.StandardDeviation(input_array)
        CV = TimeSeriesAnalyser.CoefficientOfVariance(input_array)
        EDM = EX * CV 

        print(f"Mean : {EX}\nVariance : {VX}\nStandard-Deviation : {DX}\nCoefficient of variance : {CV}\nExpected daily move : {EDM}")

        return (
            EX, VX, DX, CV
        )