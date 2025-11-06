from typing import List
import pandas as pd
from src.strategy.Strategy import Strategy

class BackTestEngine():
    def __init__(self, assets : List[pd.DataFrame]):
        self.__assets = assets

    def test(self, strategy : Strategy):
        strategy.prepare()
        strategy.execute()

        print(f"[INFO] Overall PnL: {strategy.calculate_profit()}")