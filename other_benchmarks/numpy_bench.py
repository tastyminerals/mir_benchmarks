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
from collections import defaultdict as dd
from time import perf_counter as timer

import numpy as np


def functions(nruns=1):
    rows, cols = 500, 600
    reduceRows, reduceCols = rows / 5, cols / 6

    small_int_matrixA = np.random.randint(1, 10, [int(reduceRows), int(reduceCols)])
    small_int_matrixB = np.random.randint(1, 10, [int(reduceRows), int(reduceCols)])
    float_matrixA = np.random.rand(rows, cols)
    small_float_matrixA = np.random.rand(int(reduceRows), int(reduceCols))
    small_float_matrixB = np.random.rand(int(reduceRows), int(reduceCols))
    float_matrixC = np.random.rand(cols, rows)
    float_arrayA = np.random.rand(rows * cols)
    float_arrayB = np.random.rand(rows * cols)

    funcs = dd(list)
    name = "Element-wise sum of two {}x{} matrices (int), (1000 loops)".format(
        int(reduceRows), int(reduceCols)
    )
    for _ in range(nruns):
        start = timer()
        for _ in range(1000):
            _ = small_int_matrixA + small_int_matrixB
        end = timer()
        funcs[name].append(end - start)

    name = "Element-wise multiplication of two {}x{} matrices (float64), (1000 loops)".format(
        int(reduceRows), int(reduceCols)
    )
    for _ in range(nruns):
        start = timer()
        for _ in range(1000):
            _ = small_float_matrixA * small_float_matrixB
        end = timer()
        funcs[name].append(end - start)

    name = "Dot (scalar) product of two {} arrays (float64), (1000 loops)".format(
        rows * cols
    )
    for _ in range(nruns):
        start = timer()
        for _ in range(1000):
            _ = float_arrayA @ float_arrayB
        end = timer()
        funcs[name].append(end - start)

    name = "Matrix product of {}x{} and {}x{} matrices (float64)".format(
        rows, cols, cols, rows
    )
    for _ in range(nruns):
        start = timer()
        _ = float_matrixA @ float_matrixC
        end = timer()
        funcs[name].append(end - start)

    name = "L2 norm of {}x{} matrix (float64), (1000 loops)".format(rows, cols)
    for _ in range(nruns):
        start = timer()
        for _ in range(1000):
            _ = np.linalg.norm(float_matrixA)
        end = timer()
        funcs[name].append(end - start)

    name = "Sort of {}x{} matrix (float64)".format(rows, cols)
    for _ in range(nruns):
        np.random.shuffle(float_matrixA.reshape(rows * cols))
        start = timer()
        _ = np.sort(float_matrixA)  # dispathes to C
        end = timer()
        funcs[name].append(end - start)

    return funcs


def benchmark():
    results = functions(args.nruns)
    for name, runs in results.items():
        print("| {} | {} |".format(name, sum(runs) / len(runs)))


if __name__ == "__main__":
    intro = "Run NumPy Python benchmarks."
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
