import timeit
import numpy as np

from kbmod.filters.clustering_filters import *
from kbmod.result_list import ResultList, ResultRow
from kbmod.search import *

def _make_data(objs, times):
    """Create a ResultList for the given objects.

    Parameters
    ----------
    obj : list of lists
        A list where each element specifies a Trajectory
        as [x, y, xv, yv].

    Returns
    -------
    ResultList
    """
    num_times = len(times)
    rs = ResultList(times, track_filtered=True)
    for x in objs:
        t = Trajectory()
        t.x = x[0]
        t.y = x[1]
        t.vx = x[2]
        t.vy = x[3]
        t.lh = 100.0

        row = ResultRow(t, num_times)
        rs.append_result(row)
    return rs

def run_index_benchmark(filter, rs):
    tmr = timeit.Timer(stmt="filter.keep_indices(rs)", globals=locals())
    res_time = np.mean(tmr.repeat(repeat=10, number=20))
    return res_time


def run_all_benchmarks():
    times = [(10.0 + 0.1 * float(i)) for i in range(20)]
    rs1 = _make_data(
        [
            [10, 11, 1, 2],
            [10, 11, 1000, -1000],
            [10, 11, 0.0, 0.0],
            [25, 24, 1.0, 1.0],
            [25, 26, 10.0, 10.0],
            [10, 12, 5, 5],
        ],
        times
    )


    print("Method                    |    Time")
    print("-" * 40)

    f1 = DBSCANFilter("position", 0.025, 100, 100, [0, 50], [0, 1.5], times)

    res_time = run_index_benchmark(f1, rs1)
    print(f"position - 2 clusters     | {res_time:10.7f}")

    f2 = DBSCANFilter("position", 0.025, 100, 100, [0, 50], [0, 1.5], times)

    res_time = run_index_benchmark(f2, rs1)
    print(f"position - 4 clusters     | {res_time:10.7f}")

    f3 = DBSCANFilter("position", 0.025, 1000, 1000, [0, 50], [0, 1.5], times)

    res_time = run_index_benchmark(f3, rs1)
    print(f"position - 1 cluster      | {res_time:10.7f}")

    rs2 = _make_data(
        [
            [10, 11, 1, 2],
            [10, 11, 1000, -1000],
            [10, 11, 1.0, 2.1],
            [55, 54, 1.0, 1.0],
            [55, 56, 10.0, 10.0],
            [10, 12, 4.1, 8],
        ],
        times
    )

    f4 = DBSCANFilter("all", 0.025, 100, 100, [0, 100], [0, 1.5], times)

    res_time = run_index_benchmark(f4, rs2)
    print(f"all - 5 clusters          | {res_time:10.7f}")

    # Larger eps is 3 clusters.
    f5 = DBSCANFilter("all", 0.25, 100, 100, [0, 100], [0, 1.5], times)

    res_time = run_index_benchmark(f5, rs2)
    print(f"all - 3 clusters          | {res_time:10.7f}")

    # Larger scale is 3 clusters.
    f6 = DBSCANFilter("all", 0.025, 100, 100, [0, 5000], [0, 1.5], times)

    res_time = run_index_benchmark(f6, rs2)
    print(f"all (larger) - 3 clusters | {res_time:10.7f}")

    rs3 = _make_data(
        [
            [10, 11, 1, 2],
            [10, 11, 2, 5],
            [10, 11, 1.01, 1.99],
            [21, 23, 1, 2],
            [21, 23, -10, -10],
            [5, 10, 6, 1],
            [5, 10, 1, 2],
        ],
        times
    )

    f7 = DBSCANFilter("mid_position", 0.1, 20, 20, [0, 100], [0, 1.5], times)
    res_time = run_index_benchmark(f7, rs3)
    print(f"mid_position              | {res_time:10.7f}")


if __name__ == "__main__":
    run_all_benchmarks()
