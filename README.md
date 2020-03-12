# D Benchmarks against Numpy
Each benchmark was run 20 times with 0 sec. timeout, the timings were then collected and averaged.

### Hardware

 * CPU: Quad Core Intel Core i7-7700HQ (-MT MCP-) speed/min/max: 919/800/3800 MHz Kernel: 5.5.7-1-MANJARO x86_64 Up: 4h 36m 
 * Mem: 2814.2/48147.6 MiB (5.8%) Storage: 489.05 GiB (6.6% used) Procs: 271 Shell: fish 3.1.0 inxi: 3.0.37 

### Benchmarks Yay!

| Description                                                                          | Numpy (BLAS) (sec.) | Standard D (sec.) | Mir D (sec.)         |
| ------------------------------------------------------------------------------------ | ------------------- | ----------------- | -------------------- |
| Element-wise sum of two 250x200 matrices (int), (50 loops)                           | 0.00115             | 0.00400213 (x3.5) | 0.00014372 (x1/8)    |
| Element-wise multiplication of two 250x200 matrices (float64/double), (50 loops)     | 0.0011578           | 0.0132323 (x11.4) | 0.00013852 (x1/8.3)  |
| Element-wise sum of two 500x600 matrices (int), (50 loops)                           | 0.0101275           | 0.016496 (x1.6)   | 0.00021556 (x1/47)   |
| Element-wise multiplication of two 500x600 matrices (float64/double), (50 loops)     | 0.010182            | 0.06857 (x6.7)    | 0.00021717 (x1/47)   |
| Element-wise sum of two 1k x 1k matrices (int), (50 loops)                           | 0.0493201           | 0.0614544 (x1.3)  | 0.000422135 (x1/117) |
| Element-wise multiplication of two 1k x 1k matrices (float64/double), (50 loops)     | 0.0493693           | 0.233827 (x4.7)   | 0.000453535 (x1/109) |
| Scalar product of two 30000000 arrays (float64/double)                               | 0.0152186           | 0.0227465 (x1.5)  | 0.0198812 (x1.3)     |
| Dot product of 5000x6000 and 6000x5000 matrices (float64/double), (BLAS vs OpenBLAS) | 1.6084685           | ---               | 2.03398 (x1.2)       |
| L2 norm of 5000x6000 matrix (float64/double)                                         | 0.0072423           | 0.0160546 (x2.2)  | 0.0110136 (x1.6)     |
| Quicksort of 5000x6000 matrix (float64/double)                                       | 2.6516816           | 0.178071 (x14.8)  | 1.52406 (x1/0.6)     |

##### Numpy (BLAS)

| Description                                                                 | Time (sec.)           |
| --------------------------------------------------------------------------- | --------------------- |
| Element-wise sum of two 250x200 matrices (int), (50 loops)                  | 0.0011499844000354641 |
| Element-wise multiplication of two 250x200 matrices (float64), (50 loops)   | 0.0011577772499094864 |
| Element-wise sum of two 500x600 matrices (int), (50 loops)                  | 0.010127495700044165  |
| Element-wise multiplication of two 500x600 matrices (float64), (50 loops)   | 0.010181960850013638  |
| Element-wise sum of two 1000x1000 matrices (int), (50 loops)                | 0.04932012745002794   |
| Element-wise multiplication of two 1000x1000 matrices (float64), (50 loops) | 0.04936937039983604   |
| Scalar product of two 30000000 arrays (float64)                             | 0.015218639799968514  |
| Dot product of 5000x6000 and 6000x5000 matrices (float64)                   | 1.6084685370500664    |
| L2 norm of 5000x6000 matrix (float64)                                       | 0.007242295700052637  |
| Sort of 5000x6000 matrix (float64)                                          | 2.6516815754499476    |

##### Standard D

| Description                                                                                                     | Time (sec.) |
| --------------------------------------------------------------------------------------------------------------- | ----------- |
| Element-wise sum of two [250x200] arrays of arrays (int), (50 loops)                                            | 0.0164181   |
| Element-wise multiplication of two [250x200] arrays of arrays (double), (50 loops)                              | 0.0173038   |
| Element-wise sum of two [500x600] arrays of arrays (int), (50 loops)                                            | 0.0975254   |
| Element-wise multiplication of two [500x600] arrays of arrays (double), (50 loops)                              | 0.0963054   |
| Element-wise sum of two [250x200] struct matrices (int), (50 loops)                                             | 0.00400213  |
| Element-wise multiplication of two [250x200] struct matrices (double), (50 loops)                               | 0.0132323   |
| Element-wise sum of two [500x600] struct matrices (int), (50 loops)                                             | 0.016496    |
| Element-wise multiplication of two [500x600] struct matrices (double), (50 loops)                               | 0.06857     |
| Element-wise sum of two [1000x1000] arrays of arrays (int), (50 loops)                                          | 0.386411    |
| Element-wise multiplication of two [1000x1000] arrays of arrays (double), (50 loops)                            | 0.428913    |
| Element-wise sum of two [1000x1000] struct matrices (int), (50 loops)                                           | 0.0614544   |
| Element-wise multiplication of two [1000x1000] struct matrices (double), (50 loops)                             | 0.233827    |
| Scalar product of two [30000000] arrays (double)                                                                | 0.0227465   |
| (Reference only) [unoptimized](#unoptimized-dot-product) dot product of two [1000x500] struct matrices (double) | 0.567586    |
| L2 norm of [5000x6000] struct matrix (double)                                                                   | 0.0160546   |
| Sort of [5000x6000] struct matrix (double)                                                                      | 0.178071    |

##### Mir D

| Description                                                                | Time (sec.) |
| -------------------------------------------------------------------------- | ----------- |
| Element-wise sum of two [250x200] slices (int), (50 loops)                 | 0.00014372  |
| Element-wise multiplication of two [250x200] slices (double), (50 loops)   | 0.00013852  |
| Element-wise sum of two [500x600] slices (int), (50 loops)                 | 0.00021556  |
| Element-wise multiplication of two [500x600] slices (double), (50 loops)   | 0.00021717  |
| Element-wise sum of two [1000x1000] slices (int), (50 loops)               | 0.000422135 |
| Element-wise multiplication of two [1000x1000] slices (double), (50 loops) | 0.000453535 |
| Scalar product of two [30000000] slices (double)                           | 0.0198812   |
| Scalar product of two [30000000] slices (double), (plain loop)             | 0.0199915   |
| L2 norm of [5000x6000] slice (double)                                      | 0.0110136   |
| Dot product of two [5000x6000] and [6000x5000] slices (double)             | 2.03398     |
| Sort of [5000x6000] slice (double)                                         | 1.52406     |

##### Unoptimized Dot Product
Although our dot product implementation is pretty fast with small to medium sized matrices it becomes prohibitively slow with big matrices.
Efficient matrix multiplication is a field of its own.
Numpy uses heavily optimized well-known BLAS [general matrix multiplication `gemm`](https://software.intel.com/en-us/mkl-developer-reference-fortran-gemm) routine.

| Unoptimized dot product of struct matrices | Time (sec.) |
| ------------------------------------------ | ----------: |
| 2 x [100 x 100]                            |        0.01 |
| 2 x [1000 x 1000]                          |        2.21 |
| 2 x [1500 x 1000]                          |         5.6 |
| 2 x [1500 x 1500]                          |        9.28 |
| 2 x [2000 x 2000]                          |       44.59 |
| 2 x [2100 x 2100]                          |       55.13 |