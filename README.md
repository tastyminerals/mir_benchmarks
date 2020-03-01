# Benchmarks for mir.algorithm D Library

This repo contains (so far) various benchmarks for multidimensional D arrays:

    - multidimensional array element-wise sum
    - multidimensional array element-wise multiplication
    - single array struct (Matrix) element-wise sum
    - single array struct (Matrix) element-wise multiplication
    - dot product of two single array struct (Matrix)
    - L2 norm of single array struct (Matrix)
    - Standard sort of single array struct (Matrix)

The implementation is naive, so no black magic optimization done.

### Numpy

| benchmark                                                   | time   |
| ----------------------------------------------------------- | ------ |
| Sum of two [5000, 6000] int matrices                        | ~0.062 |
| Multiplication of two [5000, 6000] float matrices           | ~0.116 |
| Dot product of [5000, 6000] and [6000, 5000] float matrices | ~2.225 |
| L2 norm of [5000, 6000] float matrix                        | ~0.001 |
| Sort of [5000, 6000] float matrix (axis=-1)                 | ~1.315 |

### Standard D

| benchmark                                                          | time   | speed vs Numpy |
| ------------------------------------------------------------------ | ------ | :------------: |
| Sum of two [5000, 6000] int array of arrays                        | ~0.28  |      x4.5      |
| Multiplication of two [5000, 6000] double array of arrays matrices | ~0.3   |      x2.6      |
| Sum of two [5000, 6000] int struct matrices                        | ~0.039 |      x0.6      |
| Multiplication of two [5000, 6000] double struct matrices          | ~0.135 |      x1.2      |
| Dot product of [500, 600] and [600, 500] double struct matrices    | ~0.172 |       --       |
| L2 norm of [5000, 6000] double struct matrix                       | ~0.015 |      x15       |
| Sort of [5000, 6000] double struct matrix (axis=-1)                | ~2.435 |      x1.9      |
