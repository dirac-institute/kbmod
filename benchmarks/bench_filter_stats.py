import timeit
import numpy as np

from kbmod.filters.stats_filters import *
from kbmod.result_list import ResultRow
from kbmod.search import Trajectory

def run_row_benchmark(create_filter=""):
    row = ResultRow(Trajectory(), 1000)

    filt = eval(create_filter)

    # Do the timing runs.
    full_cmd = "filt.keep_row(row)"
    tmr = timeit.Timer(stmt="filt.keep_row(row)", globals=locals())
    res_time = np.mean(tmr.repeat(repeat=10, number=20))
    return res_time


def run_all_benchmarks():
    print("Method       |    Time")
    print("-" * 40)

    res_time = run_row_benchmark(f"LHFilter(1, 19)")
    print(f"LHFilter     | {res_time:10.7f}")

    res_time = run_row_benchmark(f"NumObsFilter(1)")
    print(f"NumObsFilter | {res_time:10.7f}")

if __name__ == "__main__":
    run_all_benchmarks()