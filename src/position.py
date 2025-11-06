from enum import Enum, auto

class position_t(Enum):
    LONG = auto()
    SHORT = auto

class Action(Enum):
    OPEN_SHORT = -1
    NO_ACTION = 0
    OPEN_LONG = 1
    CLOSE_SHORT = 2
    CLOSE_LONG = 3

P = position_t
A = Action

class Position():
    pos_id = 0

    def __init__(self, pos_t : P, entry_price : float, size : float = 1.0):
        Position.pos_id += 1
        self.type = pos_t
        self.size = size
        self.entry_price = entry_price
        self.exit_price = None
        self.closed = False
        self.profit = None

    def __repr__(self):
        return f"<Pos#{self.id} {self.type.__repr__()} @ {self.entry_price}>"

    def __calculate_profit(self) -> float:
        return (self.entry_price - self.exit_price if self.type == P.LONG 
                else self.exit_price - self.entry_price)

    def close(self, exit_price : float):
        if self.closed:
            raise ValueError("Cannot close a closed position")
        self.exit_price = exit_price
        self.closed = True
        self.profit = self.__calculate_profit()