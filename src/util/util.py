import re
import cProfile
import time

def parse_filename(filename: str) -> str:
    match = re.match(r'file_(\w+)_AX(\w+)_([0-9]+)_([0-9]+)', filename)
    if match:
        mri_type = match.group(2).lower()
        hex1 = hex(int(match.group(3)))[2:]
        hex2 = hex(int(match.group(4)))[2:]
        return f"{match.group(1)}_{mri_type}_{hex1}_{hex2}"
    else:
        return None

import cProfile
import pstats
import io

def time_function(func):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        s = io.StringIO()
        sortby = 'tottime'
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats(20)
        print(s.getvalue())
        return result
    return wrapper