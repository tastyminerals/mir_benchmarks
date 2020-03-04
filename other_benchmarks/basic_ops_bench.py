import numpy as np
from timeit import default_timer as timer


def simple_2d_sum(rows, cols):
    a = np.random.randint(1, 10, [rows, cols])
    b = np.random.randint(1, 10, [rows, cols])
    start = timer()
    c = a + b
    end = timer()
    print("Sum of two {}x{} int matrices: {} sec.".format(
        rows, cols, end - start))


def simple_2d_mul(rows, cols):
    a = np.random.rand(rows, cols)
    b = np.random.rand(rows, cols)
    start = timer()
    c = a * b
    end = timer()
    print("Multiplication of two {}x{} float matrices: {} sec.".format(
        rows, cols, end - start))


def scalar_product(elems):
    a = np.random.rand(elems)
    b = np.random.rand(elems)
    start = timer()
    c = a @ b
    end = timer()
    print("Scalar product between 2x{} arrays: {} sec.".format(elems, end - start))


def dot_product(rows, cols):
    a = np.random.rand(rows, cols)
    b = np.random.rand(cols, rows)
    start = timer()
    # c = np.dot(a, b)
    c = a @ b
    end = timer()
    print("Dot product between {}x{} and {}x{} float matrices: {} sec.".format(
        rows, cols, cols, rows, end - start))


def square_l2_norm(rows, cols):
    a = np.random.rand(rows, cols)
    start = timer()
    c = np.linalg.norm(a) ** 2
    end = timer()
    print("L2 norm of {}x{} float matrix: {} sec.".format(rows, cols, end - start))


def simple_sort(rows, cols):
    a = np.random.rand(rows, cols)
    start = timer()
    c = np.sort(a, axis=None)
    end = timer()
    print("Quicksort of {}x{} float matrix: {} sec.".format(
        rows, cols, end - start))


def benchmark():
    simple_2d_sum(5000, 6000)
    simple_2d_mul(5000, 6000)
    scalar_product(5000 * 6000)
    dot_product(5000, 6000)
    square_l2_norm(5000, 6000)
    simple_sort(5000, 6000)


if __name__ == "__main__":
    benchmark()
