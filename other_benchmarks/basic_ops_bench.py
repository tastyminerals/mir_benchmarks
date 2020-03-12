import time
from collections import defaultdict as dd
from timeit import default_timer as timer

import numpy as np


def allocation_and_functions():
    pass


def functions(nruns=1):
    rows, cols = 500, 600
    reduceRowsBy, reduceColsBy = 5, 6

    small_int_matrixA = np.random.randint(
        1, 10, [int(rows / reduceRowsBy), int(cols / reduceColsBy)]
    )
    small_int_matrixB = np.random.randint(
        1, 10, [int(rows / reduceRowsBy), int(cols / reduceColsBy)]
    )
    float_matrixA = np.random.rand(rows, cols)
    small_float_matrixA = np.random.rand(
        int(rows / reduceRowsBy), int(cols / reduceColsBy)
    )
    small_float_matrixB = np.random.rand(
        int(rows / reduceRowsBy), int(cols / reduceColsBy)
    )
    float_matrixC = np.random.rand(cols, rows)
    float_arrayA = np.random.rand(rows * cols)
    float_arrayB = np.random.rand(rows * cols)

    funcs = dd(list)
    name = "Element-wise sum of two {}x{} matrices (int), (500 loops)".format(
        int(rows / reduceRowsBy), int(cols / reduceColsBy)
    )
    for _ in range(nruns):
        start = timer()
        for _ in range(500):
            _ = small_int_matrixA + small_int_matrixB
        end = timer()
        funcs[name].append(end - start)

    name = "Element-wise multiplication of two {}x{} matrices (float64), (500 loops)".format(
        int(rows / reduceRowsBy), int(cols / reduceColsBy)
    )
    for _ in range(nruns):
        start = timer()
        for _ in range(500):
            _ = small_float_matrixA * small_float_matrixB
        end = timer()
        funcs[name].append(end - start)

    name = "Scalar product of two {} arrays (float64)".format(rows * cols)
    for _ in range(nruns):
        start = timer()
        _ = float_arrayA @ float_arrayB
        end = timer()
        funcs[name].append(end - start)

    name = "Dot product of {}x{} and {}x{} matrices (float64)".format(
        rows, cols, cols, rows
    )
    for _ in range(nruns):
        start = timer()
        _ = float_matrixA @ float_matrixC
        end = timer()
        funcs[name].append(end - start)

    name = "L2 norm of {}x{} matrix (float64)".format(rows, cols)
    for _ in range(nruns):
        start = timer()
        _ = np.linalg.norm(float_matrixA) ** 2
        end = timer()
        funcs[name].append(end - start)

    name = "Sort of {}x{} matrix (float64)".format(rows, cols)
    for _ in range(nruns):
        start = timer()
        _ = np.sort(float_matrixA, axis=None)
        end = timer()
        funcs[name].append(end - start)

    return funcs


def benchmark():
    # TODO allocation_and_functions()
    results = functions(20)
    for name, runs in results.items():
        print("| {} | {} |".format(name, sum(runs) / len(runs)))


if __name__ == "__main__":
    benchmark()
