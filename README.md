# D Benchmarks against NumPy
Each benchmark was run 20 times with 0 sec. timeout, the timings were then collected and averaged.

Keep in mind that NumPy scalar, dot product operations are **multithreaded** while D benchmarks use **single-threaded** openBLAS version.

### Compile and Run

- D
```
dub run --compiler=ldc2 --build=release --force
```

- Numpy
```
python3 other_benchmarks/basic_ops_bench.py 
```

### Hardware

 * CPU: Quad Core Intel Core i7-7700HQ (-MT MCP-) speed/min/max: 919/800/3800 MHz Kernel: 5.5.7-1-MANJARO x86_64 Up: 4h 36m 
 * Mem: 2814.2/48147.6 MiB (5.8%) Storage: 489.05 GiB (6.6% used) Procs: 271 Shell: fish 3.1.0 inxi: 3.0.37 

### Benchmarks Yay!

| Description                                                                 | Numpy (BLAS) (sec.)   | Standard D (sec.)                           | Mir D (sec.)        |
| --------------------------------------------------------------------------- | --------------------- | ------------------------------------------- | ------------------- |
| Element-wise sum of two 100x100 matrices (int), (1000 loops)                | 0.003599030050145302  | 0.0036699 (x1)                              | 0.00133468 (x1/2.7) |
| Element-wise multiplication of two 100x100 matrices (float64), (1000 loops) | 0.004327521699860881  | 0.0379033 (x8.8)                            | 0.00298525 (x1/1.5) |
| Scalar product of two 300000 arrays (float64), (1000 loops)                 | 0.034868766399995366  | 0.169866 (x4.9)                             | 0.0297874 (x1/1.2)  |
| Dot product of 500x600 and 600x500 matrices (float64)                       | 0.0018661658000382886 | 0.158588 (x85)[*](#unoptimized-dot-product) | 0.00224537 (x1.2)   |
| L2 norm of 500x600 matrix (float64), (1000 loops)                           | 0.02373401529989678   | 0.11789 (x5)                                | 0.0394499 (x1.7)    |
| Sort of 500x600 matrix (float64)                                            | 0.010416021049923074  | 0.00964899 (x1/1.1)                         | 0.00951812 (x1/1.1) |

##### Numpy (BLAS)

| Description                                                                 | Time (sec.)           |
| --------------------------------------------------------------------------- | --------------------- |
| Element-wise sum of two 100x100 matrices (int), (1000 loops)                | 0.003599030050145302  |
| Element-wise multiplication of two 100x100 matrices (float64), (1000 loops) | 0.004327521699860881  |
| Scalar product of two 300000 arrays (float64), (1000 loops)                 | 0.034868766399995366  |
| Dot product of 500x600 and 600x500 matrices (float64)                       | 0.0018661658000382886 |
| L2 norm of 500x600 matrix (float64), (1000 loops)                           | 0.02373401529989678   |
| Sort of 500x600 matrix (float64)                                            | 0.010416021049923074  |

##### Standard D

| Description                                                                          | Time (sec.) |
| ------------------------------------------------------------------------------------ | ----------- |
| (Reference only) unoptimized dot product of two [500x600] struct matrices (double)   | 0.158588    |
| Element-wise multiplication of two [100x100] arrays of arrays (double), (1000 loops) | 0.0687838   |
| Element-wise multiplication of two [100x100] struct matrices (double), (1000 loops)  | 0.0379033   |
| Element-wise sum of two [100x100] arrays of arrays (int), (1000 loops)               | 0.0735009   |
| Element-wise sum of two [100x100] struct matrices (int), (1000 loops)                | 0.0036699   |
| L2 norm of [500x600] struct matrix (double), (1000 loops)                            | 0.11789     |
| Scalar product of two [300000] arrays (double), (1000 loops)                         | 0.169866    |
| Sort of [500x600] struct matrix (double)                                             | 0.00964899  |

##### Mir D

| Description                                                                | Time (sec.) |
| -------------------------------------------------------------------------- | ----------- |
| Dot product of two [500x600] and [600x500] slices (double)                 | 0.00215475  |
| Element-wise multiplication of two [100x100] slices (double), (1000 loops) | 0.00298525  |
| Element-wise sum of two [100x100] slices (int), (1000 loops)               | 0.00133468  |
| L2 norm of [500x600] slice (double), (1000 loops)                          | 0.0394499   |
| Scalar product of two [300000] slices (double), (1000 loops)               | 0.0911593   |
| Scalar product of two [300000] slices (double), (openBLAS), (1000 loops)       | 0.0297874   |
| Scalar product of two [300000] slices (double), (plain loop), (1000 loops) | 0.0902336   |
| Sort of [500x600] slice (double)                                           | 0.00951812  |

##### Unoptimized Dot Product
Standard D library does not have a function for dot product therefore we are using plain loop implementation.
Although looped function is pretty fast with small to medium sized matrices, it becomes prohibitively slow with bigger matrices (efficient matrix multiplication is a field on its own).
Numpy uses heavily optimized BLAS [general matrix multiplication `gemm`](https://software.intel.com/en-us/mkl-developer-reference-fortran-gemm) routine.
Nothing really stops you from using the same via [D CBLAS package](https://code.dlang.org/packages/cblas) directly in your code.

Unoptimized `matrixDotProduct` function timings:

| Matrix Sizes      | Time (sec.) |
| ----------------- | ----------: |
| 2 x [100 x 100]   |        0.01 |
| 2 x [1000 x 1000] |        2.21 |
| 2 x [1500 x 1000] |         5.6 |
| 2 x [1500 x 1500] |        9.28 |
| 2 x [2000 x 2000] |       44.59 |
| 2 x [2100 x 2100] |       55.13 |
