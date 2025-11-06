from src.strategy.Strategy import Strategy, P
import pandas as pd
from src.TimeSeriesAnalyser import TimeSeriesAnalyser as TSA
import random

class MA_Strategy(Strategy):
    def __init__(self, data : pd.DataFrame, _verbose=True):
        super().__init__(data, _verbose)
        self.WINDOWS = [200]

    def prepare(self) -> None:
        """
        Precompute necessary 
        """
        self.SMAs = [
            TSA.SMA(
                self._data['Close'],
                window=window_,
            ) for window_ in self.WINDOWS
        ]

    def execute(self):
        """
        We will return an array of commands to open a short, long or stay
        """
        for price, i in enumerate(self._data['Close']):
            indicator_c = 0
            for sma in self.SMAs:
                if price == sma[i]:
                    indicator_c += 1
                

class DUMMY_Strategy(Strategy):
    def __init__(self, data, _verbose = True):
        super().__init__(data, _verbose)

    def prepare(self) -> None:
        pass
    def execute(self):
        FREQ = 10
        action = 1
        for index,price in enumerate(self._data['Close'], start=1):
            if index % FREQ == 0:
                action = -1

            if action == 1:
                self.open_position(
                    pos_type=P.LONG if random.random() > 0.5 else P.SHORT,
                    entry_price=price
                )
            elif action == -1:
                self.close_position(
                    pos_id=list(self._positions.keys())[0],
                    exit_price=price
                )
                action = 1
                continue

            action = 0