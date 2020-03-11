from collections import defaultdict as dd
import numpy as np
from timeit import default_timer as timer


def allocation_and_functions():
    pass


def functions(nruns=1):
    rows, cols = 5000, 6000
    int_matrixA = np.random.randint(1, 10, [rows, cols])
    int_matrixB = np.random.randint(1, 10, [rows, cols])
    float_matrixA = np.random.rand(rows, cols)
    float_matrixB = np.random.rand(rows, cols)
    float_matrixC = np.random.rand(cols, rows)
    float_arrayA = np.random.rand(rows * cols)
    float_arrayB = np.random.rand(rows * cols)

    funcs = dd(list)
    name = "2d sum of two {}x{} matrices (float64)".format(rows, cols)
    for _ in range(nruns):
        start = timer()
        _ = int_matrixA + int_matrixB
        end = timer()
        funcs[name].append(end - start)

    name = "2d multiplication of two {}x{} matrices (float64)".format(rows, cols)
    for _ in range(nruns):
        start = timer()
        _ = float_matrixB * float_matrixB
        end = timer()
        funcs[name].append(end - start)

    name = "scalar dot product of two {} arrrays (float64)".format(rows * cols)
    for _ in range(nruns):
        start = timer()
        _ = float_arrayA @ float_arrayB
        end = timer()
        funcs[name].append(end - start)

    name = "dot product of {}x{} and {}x{} matrices (float64)".format(
        rows, cols, cols, rows
    )
    for _ in range(nruns):
        start = timer()
        _ = float_matrixA @ float_matrixC
        end = timer()
        funcs[name].append(end - start)

    name = "L2 norm for {}x{} matrix (float64)".format(rows, cols)
    for _ in range(nruns):
        start = timer()
        _ = np.linalg.norm(float_matrixA) ** 2
        end = timer()
        funcs[name].append(end - start)

    name = "Quicksort of {}x{} matrix (float64)".format(rows, cols)
    for _ in range(nruns):
        start = timer()
        _ = np.sort(float_matrixA, axis=None)
        end = timer()
        funcs[name].append(end - start)

    return funcs


def benchmark():
    # allocation_and_functions()
    results = functions(10)
    for name, runs in results.items():
        print("{}, {} sec.".format(name, sum(runs) / len(runs)))


if __name__ == "__main__":
    benchmark()
