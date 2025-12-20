import functools, logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

class Logger:
    @staticmethod
    def log_action(fn):
        """Decorator for logging method calls and results."""
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            logging.info(f"→ {type(self).__name__}.{fn.__name__} called with args={args}, kwargs={kwargs}")
            result = fn(self, *args, **kwargs)
            logging.info(f"← {type(self).__name__}.{fn.__name__} returned {result}")
            return result
        return wrapper
    
    @staticmethod
    def log_position(fn):
        """Decorator for logging opening and closing positions"""
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            result = fn(self, *args, **kwargs)

            if not getattr(self, "_verbose", False):
                return result

            fn_name = fn.__name__.lower()
            is_open = "open" in fn_name
            is_close = "close" in fn_name

            pos_id = getattr(result, "id", None)
            pos_type = getattr(result, "pos_type", None)
            entry = getattr(result, "entry_price", None)
            exit_ = getattr(result, "exit_price", None)
            profit = getattr(result, "profit", None)

            if is_open:
                msg = f"[{type(self).__name__}] 🟢 OPENED Position ID={pos_id}, TYPE={pos_type.__repr__() if pos_type else None}, ENTRY={entry}"
            elif is_close:
                msg = f"[{type(self).__name__}] 🔴 CLOSED Position ID={pos_id}, EXIT={exit_}, P/L={profit}"
            else:
                msg = f"[{type(self).__name__}] ⚪ Position action '{fn.__name__}' executed."

            logging.info(msg)
            return result

        return wrapper

    @staticmethod
    def log_net_profit(fn):
        """Decorator for logging the overall net profit"""
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            result = fn(self, *args, **kwargs)

            if not getattr(self, "_verbose", False):
                return result
            
            msg = f"[{type(self).__name__}] total profit: {result}"
            logging.info(msg)
            return result
        return wrapper