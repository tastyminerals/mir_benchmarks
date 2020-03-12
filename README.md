# D Benchmarks against Numpy
Each benchmark was run 20 times with 0 sec. timeout, the timings were then collected and averaged.

### Compile and Run

- D
```
dub run --compiler=ldc --build=release
```

- Numpy
```
python other_benchmarks/basic_ops_bench.py 
```

### Hardware

 * CPU: Quad Core Intel Core i7-7700HQ (-MT MCP-) speed/min/max: 919/800/3800 MHz Kernel: 5.5.7-1-MANJARO x86_64 Up: 4h 36m 
 * Mem: 2814.2/48147.6 MiB (5.8%) Storage: 489.05 GiB (6.6% used) Procs: 271 Shell: fish 3.1.0 inxi: 3.0.37 

### Benchmarks Yay!

| Description                                                                 | Numpy (BLAS) (sec.)   | Standard D (sec.)    | Mir D (sec.)         |
| --------------------------------------------------------------------------- | --------------------- | -------------------- | -------------------- |
| Element-wise sum of two 100x100 matrices (int), (1000 loops)                | 0.0037406866998935585 | 0.00391301 (x1.1)    | 0.00135104 (x1/2.8)  |
| Element-wise multiplication of two 100x100 matrices (float64), (1000 loops) | 0.004177645951131126  | 0.0386691 (x9.3)     | 0.00310666 (x1/0.7)  |
| Scalar product of two 300000 arrays (float64)                               | 0.0001648812492931029 | 0.00018331 (x1.1)    | 0.000112205 (x1/1.5) |
| Dot product of 500x600 and 600x500 matrices (float64)                       | 0.0018967882002471014 | 0.171296 (x90.3)     | 0.00224537 (x1.2)    |
| L2 norm of 500x600 matrix (float64)                                         | 3.248330031055957e-05 | 0.00013854 (x4.3)    | 6.749e-05 (x1/0.5)   |
| Sort of 500x600 matrix (float64)                                            | 0.020701468300103443  | 0.00121251 (x1/17.1) | 0.00981535 (x1/2.1)  |

##### Numpy (BLAS)

| Description                                                                 | Time (sec.)           |
| --------------------------------------------------------------------------- | --------------------- |
| Element-wise sum of two 100x100 matrices (int), (1000 loops)                | 0.0037406866998935585 |
| Element-wise multiplication of two 100x100 matrices (float64), (1000 loops) | 0.004177645951131126  |
| Scalar product of two 300000 arrays (float64)                               | 0.0001648812492931029 |
| Dot product of 500x600 and 600x500 matrices (float64)                       | 0.0018967882002471014 |
| L2 norm of 500x600 matrix (float64)                                         | 3.248330031055957e-05 |
| Sort of 500x600 matrix (float64)                                            | 0.020701468300103443  |

##### Standard D

| Description                                                                          | Time (sec.) |
| ------------------------------------------------------------------------------------ | ----------- |
| (Reference only) unoptimized dot product of two [500x600] struct matrices (double)   | 0.171296    |
| Element-wise multiplication of two [100x100] arrays of arrays (double), (1000 loops) | 0.0739148   |
| Element-wise multiplication of two [100x100] struct matrices (double), (1000 loops)  | 0.0386691   |
| Element-wise sum of two [100x100] arrays of arrays (int), (1000 loops)               | 0.0743575   |
| Element-wise sum of two [100x100] struct matrices (int), (1000 loops)                | 0.00391301  |
| L2 norm of [500x600] struct matrix (double)                                          | 0.00013854  |
| Scalar product of two [300000] arrays (double)                                       | 0.00018331  |
| Sort of [500x600] struct matrix (double)                                             | 0.00121251  |


##### Mir D

| Description                                                                | Time (sec.) |
| -------------------------------------------------------------------------- | ----------- |
| Element-wise sum of two [100x100] slices (int), (1000 loops)               | 0.00135104  |
| Element-wise multiplication of two [100x100] slices (double), (1000 loops) | 0.00310666  |
| Dot product of two [500x600] and [600x500] slices (double)                 | 0.00224537  |
| L2 norm of [500x600] slice (double)                                        | 6.749e-05   |
| Scalar product of two [300000] slices (double)                             | 0.000112205 |
| Scalar product of two [300000] slices (double), (plain loop)               | 9.573e-05   |
| Sort of [500x600] slice (double)                                           | 0.00981535  |

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
