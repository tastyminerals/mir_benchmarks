# Benchmarks against NumPy

Here is a series of various D and Julia benchmarks timed against popular NumPy functions.

We test standard D functions as well as Mir numerical library across different tasks such as multiplication, dot product, sorting and one general [neural network data preprocessing task](#neural-network-data-preprocessing).
Added Julia benchmarks out of curiosity.

Each benchmark was run 20 times with 0 sec. timeout, the timings were then collected and averaged (NumPy, D).
Julia code was benchmarked with `@btime` macro from `BenchmarkTools` package.

## Versions

* D compiler LDC 1.19.0, mir-algorithm 3.7.19, mir-random 2.2.11, mir-blas 1.1.10, (OpenBLAS)
* (Anaconda) Python 3.7.6, NumPy 1.18.1 (MKL)
* Julia 1.4.0, (OpenBLAS)

## Hardware

 * CPU: Quad Core Intel Core i7-7700HQ (-MT MCP-) speed/min/max: 919/800/3800 MHz Kernel: 5.5.7-1-MANJARO x86_64
 * Mem: 2814.2/48147.6 MiB (5.8%) Storage: 489.05 GiB (6.6% used) Procs: 271 Shell: fish 3.1.0

## Compile and Run

- D
```
dub run --compiler=ldc2 --build=release --force
```

- Julia
```
julia julia_bench.jl
```

- NumPy
```
python3 other_benchmarks/basic_ops_bench.py
```

### Cumulative Table Benchmarks (single-thread)

#### General Purpose (single-thread)

| Description                                                                 | NumPy (MKL) (sec.)       | Standard D (sec.)                                 | Mir D (sec.)            |
| --------------------------------------------------------------------------- | ------------------------ | ------------------------------------------------- | ----------------------- |
| Dot (scalar) product of two 300000 arrays (float64), (1000 loops)           | 0.10080685440025264      | 0.170738 (x1.7)                                   | **0.0892025** (x1/1.1)  |
| Element-wise sum of two 100x100 matrices (int), (1000 loops)                | 0.0038669097997626523    | 0.00368572 (x1/1.1)                               | **0.00133979** (x1/2.9) |
| Element-wise multiplication of two 100x100 matrices (float64), (1000 loops) | 0.004186091849987861     | 0.037623 (x9)                                     | **0.00301221** (x1/1.4) |
| L2 norm of 500x600 matrix (float64), (1000 loops)                           | 0.06301813390018651      | 0.117289 (x1.9)                                   | **0.0390259** (x1/1.6)  |
| Matrix product of 500x600 and 600x500 matrices (float64)                    | **0.005560713250088156** | 0.157694 (x28) [*](#not-optimized-matrix-product) | 0.00591477 (x1.1)       |
| Sort of 500x600 matrix (float64)                                            | **0.009630701900277927** | 0.0110437 (x1.2)                                  | 0.011357 (x1.2)         |

### Cumulative Table Benchmarks (multi-thread)

#### General Purpose (multi-thread)

| Description                                                                 | NumPy (MKL) (sec.)        | Mir D (sec.)            | Julia (sec.)      |
| --------------------------------------------------------------------------- | ------------------------- | ----------------------- | ----------------- |
| Dot (scalar) product of two 300000 arrays (float64), (1000 loops)           | 0.03528142820068751       | 0.0309097 (x1/1.1)      | **0.03** (x1/1.1) |
| Element-wise sum of two 100x100 matrices (int), (1000 loops)                | 0.0037877704002312385     | **0.0015176** (x1/2.5)  | 0.0063 (x1.6)     |
| Element-wise multiplication of two 100x100 matrices (float64), (1000 loops) | 0.004193491550176986      | **0.00293436** (x1/1.4) | 0.00617 (x1.5)    |
| L2 norm of 500x600 matrix (float64), (1000 loops)                           | **0.023907507749936486**  | 0.0398216 (x1.7)        | 0.097 (x4.1)      |
| Matrix product of 500x600 and 600x500 matrices (float64)                    | **0.0018566828504845035** | 0.00206505 (x1.1)       | 0.01988 (x10.7)   |
| Sort of 500x600 matrix (float64)                                            | **0.010326230399914493**  | 0.0112988 (x1.1)        | 0.0161 (x1.6)     |

#### [Domain Specific](#neural-network-data-preprocessing)

| Description                                         | Python + NumPy (sec.) | Standard D + Mir (sec.) |
| --------------------------------------------------- | --------------------- | ----------------------- |
| Neural network training data preprocessing (1.5 MB) | 0.15562605835148133   | **0.0460182** (x1/3.4)  |
| Neural network training data preprocessing (16 MB)  | 1.8649751371500316    | **0.454537** (x1/4.1)   |

#### NumPy (MKL) (single-thread)

In order to limit the number of threads, set the environment variable prior to running the benchmarks.
For example, anaconda NumPy uses **intel-mkl**, therefore the number of threads is controlled with `MKL_NUM_THREADS` variable.

Check which backend is used:

```python
In [1]: import numpy as np

In [2]: np.show_config()
blas_mkl_info:
    libraries = ['mkl_rt', 'pthread']
    library_dirs = ['/home/pavel/miniconda3/envs/torch/lib']
    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
    include_dirs = ['/home/pavel/miniconda3/envs/torch/include']
blas_opt_info:
    libraries = ['mkl_rt', 'pthread']
    library_dirs = ['/home/pavel/miniconda3/envs/torch/lib']
    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
    include_dirs = ['/home/pavel/miniconda3/envs/torch/include']
lapack_mkl_info:
    libraries = ['mkl_rt', 'pthread']
    library_dirs = ['/home/pavel/miniconda3/envs/torch/lib']
    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
    include_dirs = ['/home/pavel/miniconda3/envs/torch/include']
lapack_opt_info:
    libraries = ['mkl_rt', 'pthread']
    library_dirs = ['/home/pavel/miniconda3/envs/torch/lib']
    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
    include_dirs = ['/home/pavel/miniconda3/envs/torch/include']
```

**Bash**:

Your NumPy threads are controlled by one or several of the variables below:

```shell
export OPENBLAS_NUM_THREADS = 1
export MKL_NUM_THREADS = 1
export NUMEXPR_NUM_THREADS = 1
export VECLIB_MAXIMUM_THREADS = 1
export OMP_NUM_THREADS = 1
```

**Fish**: use `set -x ENV_VAR val`

| Description                                                                 | Time (sec.)           |
| --------------------------------------------------------------------------- | --------------------- |
| Dot (scalar) product of two 300000 arrays (float64), (1000 loops)           | 0.10080685440025264   |
| Element-wise sum of two 100x100 matrices (int), (1000 loops)                | 0.0038669097997626523 |
| Element-wise multiplication of two 100x100 matrices (float64), (1000 loops) | 0.004186091849987861  |
| Matrix product of 500x600 and 600x500 matrices (float64)                    | 0.005560713250088156  |
| L2 norm of 500x600 matrix (float64), (1000 loops)                           | 0.06301813390018651   |
| Sort of 500x600 matrix (float64)                                            | 0.009630701900277927  |

#### Standard D (single-thread)

| Description                                                                             | Time (sec.) |
| --------------------------------------------------------------------------------------- | ----------- |
| (Reference only) not optimized matrix product of two [500x600] struct matrices (double) | 0.157694    |
| Dot (scalar) product of two [300000] arrays (double), (1000 loops)                      | 0.170738    |
| Element-wise multiplication of two [100x100] arrays of arrays (double), (1000 loops)    | 0.0675184   |
| Element-wise multiplication of two [100x100] struct matrices (double), (1000 loops)     | 0.037623    |
| Element-wise sum of two [100x100] arrays of arrays (int), (1000 loops)                  | 0.0728854   |
| Element-wise sum of two [100x100] struct matrices (int), (1000 loops)                   | 0.00368572  |
| L2 norm of [500x600] struct matrix (double), (1000 loops)                               | 0.117289    |
| Sort of [500x600] struct matrix (double)                                                | 0.0110437   |

#### Mir D (single-thread)

Set environment variables.

**Bash**: `export OPENBLAS_NUM_THREADS=1`
**Fish**: `set -x OPENBLAS_NUM_THREADS 1`

| Description                                                                      | Time (sec.) |
| -------------------------------------------------------------------------------- | ----------- |
| Dot (scalar) product of two [300000] slices (double), (1000 loops)               | 0.0892025   |
| Dot (scalar) product of two [300000] slices (double), (OpenBLAS), (1000 loops)   | 0.0900235   |
| Dot (scalar) product of two [300000] slices (double), (plain loop), (1000 loops) | 0.0893657   |
| Element-wise multiplication of two [100x100] slices (double), (1000 loops)       | 0.00301221  |
| Element-wise sum of two [100x100] slices (int), (1000 loops)                     | 0.00133979  |
| L2 norm of [500x600] slice (double), (1000 loops)                                | 0.0390259   |
| Matrix product of two [500x600] and [600x500] slices (double), (OpenBLAS)        | 0.00591477  |
| Sort of [500x600] slice (double)                                                 | 0.011357    |

#### NumPy (MKL) (multi-thread)

**Bash**:

Your NumPy threads are controlled by one or several of the variables below:

```shell
export OPENBLAS_NUM_THREADS = 4
export MKL_NUM_THREADS = 4
export NUMEXPR_NUM_THREADS = 4
export VECLIB_MAXIMUM_THREADS = 4
export OMP_NUM_THREADS = 4
```

**Fish**: use `set -x ENV_VAR val`

| Description                                                                 | Time (sec.)           |
| --------------------------------------------------------------------------- | --------------------- |
| Element-wise sum of two 100x100 matrices (int), (1000 loops)                | 0.0037877704002312385 |
| Element-wise multiplication of two 100x100 matrices (float64), (1000 loops) | 0.004193491550176986  |
| Dot (scalar) product of two 300000 arrays (float64), (1000 loops)           | 0.03528142820068751   |
| Matrix product of 500x600 and 600x500 matrices (float64)                    | 0.0018566828504845035 |
| L2 norm of 500x600 matrix (float64), (1000 loops)                           | 0.023907507749936486  |
| Sort of 500x600 matrix (float64)                                            | 0.010326230399914493  |

#### Standard D (multi-thread)

Not implemented for benchmarks.

See how to use [multi-threading](https://tour.dlang.org/tour/en/multithreading/synchronization-sharing) in standard D.

#### Mir D (multi-thread)

Set environment variables.

**Bash**: `export OPENBLAS_NUM_THREADS=4`
**Fish**: `set -x OPENBLAS_NUM_THREADS 4`

| Description                                                                      | Time (sec.) |
| -------------------------------------------------------------------------------- | ----------- |
| Dot (scalar) product of two [300000] slices (double), (1000 loops)               | 0.0863238   |
| Dot (scalar) product of two [300000] slices (double), (OpenBLAS), (1000 loops)   | 0.0309097   |
| Dot (scalar) product of two [300000] slices (double), (plain loop), (1000 loops) | 0.0860322   |
| Element-wise multiplication of two [100x100] slices (double), (1000 loops)       | 0.00293436  |
| Element-wise sum of two [100x100] slices (int), (1000 loops)                     | 0.0015176   |
| L2 norm of [500x600] slice (double), (1000 loops)                                | 0.0398216   |
| Matrix product of two [500x600] and [600x500] slices (double) (OpenBLAS)         | 0.00206505  |
| Sort of [500x600] slice (double)                                                 | 0.0112988   |

#### Julia (multi-thread)

Julia by default uses single thread according to the docs but `htop` was reporting all cores busy.

Set environment variables.

**Bash**: `export JULIA_NUM_THREADS=4`
**Fish**: `set -x JULIA_NUM_THREADS 4`

Test it with `julia -e "println(Threads.nthreads())`.
| Description                                                                 | Time (sec.) |
| --------------------------------------------------------------------------- | ----------- |
| Element-wise sum of two 100x100 matrices (int), (1000 loops)                | 0.0063      |
| Element-wise multiplication of two 100x100 matrices (float64), (1000 loops) | 0.00617     |
| Dot (scalar) product of two 300000 arrays (float64), (1000 loops)           | 0.03        |
| Matrix product of 500x600 and 600x500 matrices (float64)                    | 0.01988     |
| L2 norm of 500x600 matrix (float64), (1000 loops)                           | 0.097       |
| Sort of 500x600 matrix (float64)                                            | 0.0161      |

#### Not optimized Matrix Product

Standard D library does not have a function for matrix product therefore we are using plain loop implementation.
Although looped function is pretty fast with small to medium sized matrices, it becomes prohibitively slow with bigger matrices (efficient matrix multiplication is a field on its own).
NumPy uses heavily optimized BLAS [general matrix multiplication `gemm`](https://software.intel.com/en-us/mkl-developer-reference-fortran-gemm) routine.
Nothing really stops you from using the same via [D CBLAS package](https://code.dlang.org/packages/cblas) directly in your code.

Not optimized `matrixDotProduct` function timings:

| Matrix Sizes      | Time (sec.) |
| ----------------- | ----------: |
| 2 x [100 x 100]   |        0.01 |
| 2 x [1000 x 1000] |        2.21 |
| 2 x [1500 x 1000] |         5.6 |
| 2 x [1500 x 1500] |        9.28 |
| 2 x [2000 x 2000] |       44.59 |
| 2 x [2100 x 2100] |       55.13 |

#### Neural Network Data Preprocessing

General purpose function benchmarks are great but they tell us little about language efficiency in real world tasks.
Therefore, we implemented a small neural network data preprocessing benchmark.

For this benchmark we used actual code used thousands of times for preprocessing training data for offline BiLSTM model training.
The BiLSTM model is a word classifier, trained to do named entity recognition for just one specific class.
In order to train the model, we need to read our training data, preprocess and convert it into multidimensional tensors.
The multidimensional tensors represent input data and should be sliceable into shapes [batch_size x seq_length x feature_dim], e.g. [32 x 25 x 4].

Our original implementation was written in Python and NumPy for converting to tensors and slicing.
The D version follows the same pattern and uses Mir numeric library for representing multidimensional arrays and do the slicing.

Two test datasets (1.5 MB and 16 MB) used in benchmarks contain private information and can not be provided.
However, the original data looks like the following example table below.

| document_id | masked_token | original_token | feature1 | feature2 | feature3 | feature4 | target_label |
| ----------- | ------------ | -------------- | -------- | -------- | -------- | -------- | ------------ |
| 2314        | ich          | Ich            | 231.23   | 20.10    | 1        | 1        | 0            |
| 2314        | bin          | bin            | 235.1    | 20.10    | 0        | 0        | 0            |
| 2314        | kartoffel    | Kartoffel      | 240.5    | 20.10    | 1        | 0        | 0            |
| 2314        | \<dig>       | 2              | 244.2    | 20.10    | 0        | 0        | 0            |
| 2314        | \<prep>      | für            | 250      | 20.10    | 0        | 0        | 0            |
| 2314        | \<dig>       | 3              | 255      | 20.10    | 0        | 0        | 0            |
| 2314        | \<punct>     | !              | 240.5    | 20.10    | 0        | 0        | 0            |
| 2314        | münchen      | München        | 340.32   | 130.23   | 1        | 0        | 1            |
| 2314        | \<func>      | ist            | 355.21   | 130.23   | 0        | 1        | 0            |
| 2314        | grün         | grün           | 364.78   | 130.23   | 0        | 0        | 0            |
