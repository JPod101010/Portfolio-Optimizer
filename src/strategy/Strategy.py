from enum import StrEnum, auto, Enum
from typing import List, Callable, Dict, Literal
from src.position import Position as Pos, P
from src.logger import Logger
import numpy as np, pandas as pd

class complexity(StrEnum):
    SIMPLE = 'simple'
    MEDIOCRE = 'mediocre'
    COMPLEX = 'complex'

class tools(StrEnum):
    MA = 'moving average'
    EMA = 'exponential moving average'

class KEY(Enum):
    C = auto()
    T = auto()

C = complexity
T = tools

execute_orders = np.ndarray[1, Literal[-1,0,1]]

COMP_DICT : Dict[Callable, Dict[KEY, C | List[T]]] = {}


def map_complexity(complexity_ : C, tools_used_ : List[T]):
    def decorator(fn: callable):
        COMP_DICT[fn][KEY.C] = complexity_
        COMP_DICT[fn][KEY.T] = tools_used_
        return fn
    return decorator

class Strategy:
    def __init__(self, data : pd.DataFrame, verbose : bool = True):
        self._data = data
        self._positions : Dict[int, Pos] = {}
        self._closed_pos : List[Pos] = []
        self._verbose = verbose

    @Logger.log_position
    def open_position(self, pos_type : P, entry_price : float, size : float = 1.0):
        pos = Pos(pos_type, entry_price, size)
        self._positions[pos.pos_id] = pos
        return pos

    @Logger.log_position
    def close_position(self, pos_id : int, exit_price : float):
        pos = self._positions.pop(pos_id)
        pos.close(exit_price)
        self._closed_pos.append(
            pos
        )
        return pos
    
    def calculate_profit(self):
        return sum(pos.profit for pos in self._closed_pos if pos is not None)

    def prepare(self):
        raise NotImplementedError

    def execute(self) -> execute_orders:
        raise NotImplementedError