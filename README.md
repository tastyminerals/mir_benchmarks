# D Benchmarks against Numpy
Each benchmark was run 20 times with 0.5 sec. timeout, the timings were then collected and averaged.

### Hardware

 * CPU: Quad Core Intel Core i7-7700HQ (-MT MCP-) speed/min/max: 919/800/3800 MHz Kernel: 5.5.7-1-MANJARO x86_64 Up: 4h 36m 
 * Mem: 2814.2/48147.6 MiB (5.8%) Storage: 489.05 GiB (6.6% used) Procs: 271 Shell: fish 3.1.0 inxi: 3.0.37 

### Benchmarks Yay!

| Description                                                                            | Numpy (BLAS) (sec.) | Standard D (sec.) | Mir D (sec.)        |
| ------------------------------------------------------------------------------------ | ------------------- | ----------------- | ------------------- |
| Element-wise sum of two 250x200 matrices (int), (200 loops)                          | 0.0105144           | 0.0207412 (x2)    | 0.00239246 (x1/4.4) |
| Element-wise multiplication of two 250x200 matrices (float64/double), (200 loops)    | 0.0079411           | 0.0430237 (x5.4)  | 0.00257717 (x1/3.1) |
| Scalar product of two 30000000 arrays (float64/double)                               | 0.0179827           | 0.0273022 (x1.5)  | 0.0239525 (x1.3)    |
| Dot product of 5000x6000 and 6000x5000 matrices (float64/double), (BLAS vs OpenBLAS) | 1.6588486           | ---               | 2.05365 (x1.2)      |
| L2 norm of 5000x6000 matrix (float64/double)                                         | 0.0098255           | 0.023526 (x2.4)   | 0.0161355 (x1.6)    |
| Quicksort of 5000x6000 matrix (float64/double)                                       | 2.6813289           | ---               | 1.52261 (x1/0.6)    |

### Numpy (BLAS)

| Benchmark                                                                  | Time (sec.)          |
| -------------------------------------------------------------------------- | -------------------- |
| Element-wise sum of two 250x200 matrices (int), (200 loops)                | 0.01051435709987345  |
| Element-wise multiplication of two 250x200 matrices (float64), (200 loops) | 0.007941059350105206 |
| Scalar product of two 30000000 arrays (float64)                            | 0.0179826627500006   |
| Dot product of 5000x6000 and 6000x5000 matrices (float64)                  | 1.6588485575500271   |
| L2 norm of 5000x6000 matrix (float64)                                      | 0.009825466249913007 |
| Quicksort of 5000x6000 matrix (float64)                                    | 2.681328929350184    |


### Standard D

| Benchmark                                                                           | Time (sec.) |
| ----------------------------------------------------------------------------------- | ----------- |
| Element-wise sum of two [250x200] arrays of arrays (int), (200 loops)               | 0.0747522   |
| Element-wise multiplication of two [250x200] arrays of arrays (double), (200 loops) | 0.0735269   |
| Element-wise sum of two [250x200] struct matrices (int), (200 loops)                | 0.0207412   |
| Element-wise multiplication of two [250x200] struct matrices (double), (200 loops)  | 0.0430237   |
| Scalar product of two [30000000] arrays (double)                                    | 0.0273022   |
| (Reference only) unoptimized dot product of two [1000x500] struct matrices (double) | 0.587706    |
| L2 norm of [5000x6000] struct matrix (double)                                       | 0.023526    |
| (Reference only) destructive sort of [5000x6000] struct matrix (double)             | 0.528105    |


### Mir D

| Benchmark                                                                 | Time (sec.) |
| ------------------------------------------------------------------------- | ----------- |
| Element-wise sum of two [250x200] slices (int), (200 loops)               | 0.00239246  |
| Element-wise multiplication of two [250x200] slices (double), (200 loops) | 0.00257717  |
| Scalar product of two [30000000] slices (double)                          | 0.0456421   |
| Scalar product of two [30000000] slices (double), (plain loop)            | 0.0239525   |
| Dot product of two [5000x6000] and [6000x5000] slices (double)            | 2.05365     |
| L2 norm of [5000x6000] slice (double)                                     | 0.0161355   |
| Sort of [5000x6000] slice (double)                                        | 1.52261     |


##### Unoptimized Dot Product
Although our unoptimized dot product implementation is pretty fast with small to medium sized matrices it becomes prohibitively slow with big matrices.
Numpy uses well-known BLAS [general matrix multiplication `gemm`](https://software.intel.com/en-us/mkl-developer-reference-fortran-gemm) routine which does not have this issue.

| unoptimized dot product of struct matrices | time (sec.) |
| ------------------------------------------ | ----------: |
| 2 x [100 x 100]                            |        0.01 |
| 2 x [1000 x 1000]                          |        2.21 |
| 2 x [1500 x 1000]                          |         5.6 |
| 2 x [1500 x 1500]                          |        9.28 |
| 2 x [2000 x 2000]                          |       44.59 |
| 2 x [2100 x 2100]                          |       55.13 |

##### Destructive Sort
TODO
