"""
Util functions for timing code.
"""

import cProfile
import pstats
import io


def time_function(func):
    """
    Decorator to time a function using cProfile.

    Args:
        func (callable): The function to time.

    Returns:
        callable: The wrapped function
    """

    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        s = io.StringIO()
        sortby = "tottime"
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats(20)
        print(s.getvalue())
        return result

    return wrapper
