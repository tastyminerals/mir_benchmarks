# D Benchmarks against Numpy

### Hardware

 * CPU: Quad Core Intel Core i7-7700HQ (-MT MCP-) speed/min/max: 919/800/3800 MHz Kernel: 5.5.7-1-MANJARO x86_64 Up: 4h 36m 
 * Mem: 2814.2/48147.6 MiB (5.8%) Storage: 489.05 GiB (6.6% used) Procs: 271 Shell: fish 3.1.0 inxi: 3.0.37 

| Benchmark                                                                            | Numpy (BLAS)   | Standard D            | Mir D                    |
| ------------------------------------------------------------------------------------ | -------------- | --------------------- | ------------------------ |
| Element-wise sum of two 250x200 matrices (int), (200 loops)                          | 0.0105144 sec. | 0.0207412 sec. (x2)   | 0.00239246 sec. (x1/4.4) |
| Element-wise multiplication of two 250x200 matrices (float64/double), (200 loops)    | 0.0079411 sec. | 0.0430237 sec. (x5.4) | 0.00257717 sec. (x1/3.1) |
| Scalar product of two 30000000 arrays (float64/double)                               | 0.0179827 sec. | 0.0273022 sec. (x1.5) | 0.0239525  sec. (x1.3)   |
| Dot product of 5000x6000 and 6000x5000 matrices (float64/double), (BLAS vs OpenBLAS) | 1.6588486 sec. | ---                   | 2.05365 sec. (x1.2)      |
| L2 norm of 5000x6000 matrix (float64/double)                                         | 0.0098255 sec. | 0.023526 sec. (x2.4)  | 0.0161355 sec. (x1.6)    |
| Quicksort of 5000x6000 matrix (float64/double)                                       | 2.6813289 sec. | ---                   | 1.52261 sec. (x1/0.6)    |

### Numpy (BLAS)

| Benchmark                                                                  | Time                      |
| -------------------------------------------------------------------------- | ------------------------- |
| Element-wise sum of two 250x200 matrices (int), (200 loops)                | 0.01051435709987345 sec.  |
| Element-wise multiplication of two 250x200 matrices (float64), (200 loops) | 0.007941059350105206 sec. |
| Scalar product of two 30000000 arrays (float64)                            | 0.0179826627500006 sec.   |
| Dot product of 5000x6000 and 6000x5000 matrices (float64)                  | 1.6588485575500271 sec.   |
| L2 norm of 5000x6000 matrix (float64)                                      | 0.009825466249913007 sec. |
| Quicksort of 5000x6000 matrix (float64)                                    | 2.681328929350184 sec.    |


### Standard D

| Benchmark                                                                           | Time           |
| ----------------------------------------------------------------------------------- | -------------- |
| Element-wise sum of two [250x200] arrays of arrays (int), (200 loops)               | 0.0747522 sec. |
| Element-wise multiplication of two [250x200] arrays of arrays (double), (200 loops) | 0.0735269 sec. |
| Element-wise sum of two [250x200] struct matrices (int), (200 loops)                | 0.0207412 sec. |
| Element-wise multiplication of two [250x200] struct matrices (double), (200 loops)  | 0.0430237 sec. |
| Scalar product of two [30000000] arrays (double)                                    | 0.0273022 sec. |
| (Reference only) unoptimized dot product of two [1000x500] struct matrices (double) | 0.587706 sec.  |
| L2 norm of [5000x6000] struct matrix (double)                                       | 0.023526 sec.  |
| (Reference only) destructive sort of [5000x6000] struct matrix (double)             | 0.528105 sec.  |


### Mir D

| Benchmark                                                                 | Time            |
| ------------------------------------------------------------------------- | --------------- |
| Element-wise sum of two [250x200] slices (int), (200 loops)               | 0.00239246 sec. |
| Element-wise multiplication of two [250x200] slices (double), (200 loops) | 0.00257717 sec. |
| Scalar product of two [30000000] slices (double)                          | 0.0456421 sec.  |
| Scalar product of two [30000000] slices (double), (plain loop)            | 0.0239525 sec.  |
| Dot product of two [5000x6000] and [6000x5000] slices (double)            | 2.05365 sec.    |
| L2 norm of [5000x6000] slice (double)                                     | 0.0161355 sec.  |
| Sort of [5000x6000] slice (double)                                        | 1.52261 sec.    |


##### Unoptimized Dot Product
Although our unoptimized dot product implementation is pretty fast with small to medium sized matrices it becomes prohibitively slow with big matrices.
Numpy uses well-known BLAS [general matrix multiplication `gemm`](https://software.intel.com/en-us/mkl-developer-reference-fortran-gemm) routine which does not have this issue.

| unoptimized dot product of struct matrices | time (sec.) |
| ------------------------------------------ | ----------: |
| 2x[100 x 100]                              |        0.01 |
| 2x[1000 x 1000]                            |        2.21 |
| 2x[1500 x 1000]                            |         5.6 |
| 2x[1500 x 1500]                            |        9.28 |
| 2x[2000 x 2000]                            |       44.59 |
| 2x[2100 x 2100]                            |       55.13 |

##### Destructive Sort
TODO