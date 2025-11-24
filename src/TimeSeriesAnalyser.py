import numpy as np
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
    
    @staticmethod
    def DiffArithmetic(input_array : time_series_t, lag : int = 1) -> time_series_t:
        return (
            input_array[:-lag] - input_array[lag:]
        )
    
    @staticmethod
    def DiffGeometric(input_array : time_series_t, lag : int = 1) -> time_series_t:
        return (
            (input_array[:-lag] - input_array[lag:]) / input_array[:-lag]
        )
    
class TimeSeriesAnalyser(Filters):
    @staticmethod 
    def RawMoment(input_array : time_series_t, order : int = 0) -> float:
        indexes = np.arange(1, len(input_array) + 1)   #assuming we index the ts from 0
        return np.sum(input_array * (indexes ** order))
    
    @staticmethod
    def StandardMoment(input_array : time_series_t, order : int = 1) -> float:
        mean = TimeSeriesAnalyser.Mean(input_array)
        return TimeSeriesAnalyser.Mean(
            (input_array - mean) ** order
        ) / (TimeSeriesAnalyser.StandardDeviation(
            input_array
        ) ** order)

    @staticmethod
    def Skewness(input_array : time_series_t) -> float:
        return TimeSeriesAnalyser.StandardMoment(
            input_array,
            order=3
        )

    @staticmethod
    def Kurtosis(input_array : time_series_t) -> float:
        return TimeSeriesAnalyser.StandardMoment(
            input_array,
            order=4
        )

    @staticmethod
    def Mean(input_array : time_series_t) -> float:
        return np.sum(input_array) / len(input_array)

    @staticmethod
    def GeometricMean(input_array : time_series_t) -> float:
        return np.prod(input_array) ** (1/len(input_array))
    
    @staticmethod
    def HarmonicMean(input_array : time_series_t) -> float:
        return len(input_array) / sum(1/input_array)
    
    @staticmethod
    def Covariance(input_array : time_series_t,
                   input_array_other : time_series_t) -> float:
        mean = TimeSeriesAnalyser.Mean(input_array)
        mean_other = TimeSeriesAnalyser.Mean(input_array_other)

        return TimeSeriesAnalyser.Mean(
            (input_array - mean) * (input_array_other - mean_other)
        )

    @staticmethod
    def AutoCovariance(input_array : time_series_t, lag : int = 1) -> float:
        return TimeSeriesAnalyser.Covariance(
            input_array[:-lag] if lag != 0 else input_array,
            input_array[lag:] if lag != 0 else input_array
        )

    @staticmethod
    def Variance(input_array : time_series_t) -> float:
        return TimeSeriesAnalyser.AutoCovariance(
            input_array,
            lag=0
        )
    
    @staticmethod
    def StandardDeviation(input_array : time_series_t) -> float:
        return np.sqrt(
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
    def CorrelationCoefficient(input_array : time_series_t, 
                               input_array_other : time_series_t) -> float:
        cov = TimeSeriesAnalyser.Covariance(input_array, input_array_other)
        stddev = TimeSeriesAnalyser.StandardDeviation(input_array)
        stddev_other = TimeSeriesAnalyser.StandardDeviation(input_array_other)
        return cov / (stddev * stddev_other)


    @staticmethod
    def AutoCorrelation(input_array : time_series_t, lag : int = 1) -> float:
        return TimeSeriesAnalyser.CorrelationCoefficient(
            input_array[:-lag],
            input_array[lag:]
        )

    """
    Complex analysis
    """
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
    def BasicAnalysis(input_array : time_series_t, verbose : bool = True) -> Tuple[float]:
        """
        Returns order:
        EX, VX, DX, CV, SKW, KRR
        """
        EX = TimeSeriesAnalyser.Mean(input_array)
        VX = TimeSeriesAnalyser.Variance(input_array)
        DX = TimeSeriesAnalyser.StandardDeviation(input_array)
        CV = TimeSeriesAnalyser.CoefficientOfVariance(input_array)
        SKW = TimeSeriesAnalyser.Skewness(input_array)
        KRR = TimeSeriesAnalyser.Kurtosis(input_array)

        if verbose:
            print(f"Mean : {EX}\nVariance : {VX}\nStandard-Deviation : {DX}\nCoefficient of variance : {CV}\n")
            print(f"Skewness : {SKW}\nKurtosis : {KRR}")

        return (
            EX, VX, DX, CV, SKW, KRR
        )