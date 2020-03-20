import argparse
from collections import defaultdict as dd
from time import perf_counter as timer

import numpy as np


def allocation_and_functions():
    pass


def functions(nruns=1):
    rows, cols = 2000, 2400
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
    # TODO allocation_and_functions()
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
