# Benchmarks for mir.algorithm D Library

This repo contains various benchmarks for multidimensional D arrays:

    - multidimensional array element-wise sum
    - multidimensional array element-wise multiplication
    - single array struct (Matrix) element-wise sum
    - single array struct (Matrix) element-wise multiplication
    - scalar product of two arrays
    - dot product of two single array struct (Matrix)
    - L2 norm of single array struct (Matrix)
    - Standard sort of single array struct (Matrix)

### Numpy (BLAS)

| benchmark                                                     | time   |
| ------------------------------------------------------------- | ------ |
| Sum of two [5000, 6000] int matrices                          | ~0.062 |
| Multiplication of two [5000, 6000] float64 matrices           | ~0.116 |
| Scalar product of two [30_000_000] float64 arrays             | ~0.022 |
| Dot product of [5000, 6000] and [6000, 5000] float64 matrices | ~2.225 |
| L2 norm of [5000, 6000] float64 matrix                        | ~0.001 |
| Sort of [5000, 6000] float64 matrix (axis=-1)                 | ~1.315 |

### Standard D

| benchmark                                                                                     | time   | speed vs Numpy |
| --------------------------------------------------------------------------------------------- | ------ | :------------: |
| Sum of two [5000, 6000] int array of arrays                                                   | ~0.28  |      x4.5      |
| Multiplication of two [5000, 6000] double array of arrays matrices                            | ~0.3   |      x2.6      |
| Sum of two [5000, 6000] int struct matrices                                                   | ~0.039 |      x0.6      |
| Multiplication of two [5000, 6000] double struct matrices                                     | ~0.135 |      x1.2      |
| Scalar product of two [30_000_000] double arrays                                              | ~0.025 |      x1.1      |
| [Naive dot product](#naive-dot-product) of [500, 1000] and [1000, 500] double struct matrices | ~0.261 |       --       |
| L2 norm of [5000, 6000] double struct matrix                                                  | ~0.014 |      x14       |
| Sort of [5000, 6000] double struct matrix (axis=-1)                                           | ~2.435 |      x1.9      |

### Mir D
// TODO

##### Naive Dot Product
Although our naive dot product implementation is pretty fast with small to medium sized matrices it becomes prohibitively slow with big matrices.
Numpy uses well-known BLAS [general matrix multiplication `gemm`](https://software.intel.com/en-us/mkl-developer-reference-fortran-gemm) routine which does not have this issue.

| naive matrixDotProduct | time (sec.) |
| ---------------------- | ----------: |
| 2x[100 x 100]          |        0.01 |
| 2x[1000 x 1000]        |        2.21 |
| 2x[1500 x 1000]        |         5.6 |
| 2x[1500 x 1500]        |        9.28 |
| 2x[2000 x 2000]        |       44.59 |
| 2x[2100 x 2100]        |       55.13 |
