import time
import logging


def timeit(func):
    """logs time of execution of a function or method"""

    def timed(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        if args and getattr(args[0], func.__name__, None):  # in other words: first argument is "self"
            func_spec = f"{args[0].__class__.__name__}.{func.__name__}"
        else:
            func_spec = func.__name__
        logging.debug(f"{func_spec}: {time.time() - start_time:.2f} seconds")
        return result

    return timed


def pluralize(string: str) -> str:
    """adds "s" to the end of a string if it is not already there"""
    return string if string[-1] == "s" else string + "s"
