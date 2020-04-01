import argparse
from collections import defaultdict as dd
from time import perf_counter as timer

import numpy as np
import gc


def functions(nruns):
    size = 30_000_000

    funcs = dd(list)
    name0 = "Allocation, writing and deallocation of a [{}] array ".format(size)
    for _ in range(nruns):
        start = timer()

        for j in range(5):
            float_arrayA = np.random.rand(size)
            float_arrayA[j] = 100.0
            del float_arrayA
            gc.collect()

        end = timer()
        funcs[name0].append(end - start)

    name1 = "Allocation, writing and deallocation of a several arrays "
    mods = [10, 3, 5, 60, 1]
    for _ in range(nruns):
        start = timer()
        for mod in mods:
            float_arrayA = np.random.rand(size // mod)
            float_arrayA[mod] = 100.0
            del float_arrayA
            gc.collect()

        end = timer()
        funcs[name1].append(end - start)

    name2 = "Reallocation of one [{}] array into three arrays".format(size / 60000)
    float_arrayA = np.random.rand(size)
    size0 = size / 60000
    for _ in range(nruns):
        start = timer()
        for i in range(size0):
            a = float_arrayA[i]
            b = float_arrayA[i:-1]
            c = float_arrayA[-i:-1]

        end = timer()
        funcs[name2].append(end - start)
        del a, b, c
        gc.collect()

    return funcs


def benchmark():
    results = functions(args.nruns)
    for name, runs in results.items():
        print("| {} | {} |".format(name, sum(runs) / len(runs)))


if __name__ == "__main__":
    intro = "Run NumPy Python memory benchmarks."
    formatter = argparse.ArgumentDefaultsHelpFormatter
    prs = argparse.ArgumentParser(formatter_class=formatter, description=intro)
    prs.add_argument(
        "-n",
        "--nruns",
        required=False,
        type=int,
        default=20,
        help="number of runs, the  time will be averaged",
    )
    args = prs.parse_args()
    benchmark()
