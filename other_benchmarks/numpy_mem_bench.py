"""
NumPy docs:

The NumPy linear algebra functions rely on BLAS and LAPACK to provide efficient low level
implementations of standard linear algebra algorithms.
Those libraries may be provided by NumPy itself using C versions of a subset of their
reference implementations but, when possible, highly optimized libraries that take advantage
of specialized processor functionality are preferred. Examples of such libraries are
OpenBLAS, MKL (TM), and ATLAS.
Because those libraries are multithreaded and processor dependent, environmental variables
and external packages such as threadpoolctl may be needed to control the number of threads
or specify the processor architecture.

env variables to control NumPy threads:

    export OPENBLAS_NUM_THREADS = 1
    export MKL_NUM_THREADS = 1
    export NUMEXPR_NUM_THREADS = 1
    export VECLIB_MAXIMUM_THREADS = 1
    export OMP_NUM_THREADS = 1
"""

import argparse
import gc
from collections import defaultdict as dd
from time import perf_counter as timer

import numpy as np


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

    name1 = "Allocation, writing and deallocation of a several big arrays of different sizes"
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

    size1 = size // 1000
    name2 = "Slicing [{}] array into another array ({} loops)".format(size1, size1)
    float_arrayA = np.random.rand(size1)
    for _ in range(nruns):
        start = timer()
        a = []
        for i in np.arange(size1):
            a.append(float_arrayA[i:-1])

        a = np.concatenate(np.array(a))
        end = timer()
        funcs[name2].append(end - start)
        del a
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

