module mir_ops_bench;
/*
This file contains several benchmarks for multidimensional Mir D slices:

    - multidimensional slice element-wise sum
    - multidimensional slice element-wise multiplication
    - Dot (scalar) product of slices
    - Matrix product of multidimensional slices
    - L2 norm of slices
    - Multidimensional slice sort

RUN: dub run --compiler=ldc2 --build=release
TEST: dub run --compiler=ldc2 --build=tests
*/

import mir.blas : dot, gemm;
import mir.math.common : fastmath, optmath;
import mir.math.common : pow, sqrt;
import mir.math.sum;
import mir.ndslice;
import mir.ndslice.sorting : sort;
import mir.random.algorithm : randomSlice, shuffle;
import mir.random.variable : uniformVar;
import std.datetime.stopwatch : StopWatch;
import std.format : format;
import std.math : approxEqual;
import std.stdio;

alias SliceArr = Slice!(double*);
alias SliceMatrix = Slice!(double*, 2);
alias SliceMatrixArr = Slice!(double*, 2)[];

@fastmath double scalarProduct(SliceArr sliceA1D, SliceArr sliceB1D)
{
    pragma(inline, false);
    return sum!"fast"(sliceA1D * sliceB1D);
}

@fastmath double loopedScalarProduct(in SliceArr s0, in SliceArr s1)
{
    pragma(inline, false);
    double accu = 0;
    assert(s0.length == s1.length);
    foreach (size_t i; 0 .. s0.length)
    {
        // will result in vectorized fused-multiply-add instructions
        accu += s0._iterator[i] * s1._iterator[i];
        // NOTE: No vectorized fused-multiply-add instructions
        // for direct call s[i] for now.
        // Will be fixed in the next mir-algorithm version.
        // accu += s0[i] * s1[i];
    }
    return accu;
}

@fastmath double squareL2Norm(SliceMatrix m)
{
    pragma(inline, false);
    double accu = 0;
    // NOTE: that BLAS nrm2
    // has more robust but slower algorithm

    // NOTE: .field returns a common D array
    // to workaround the same vectorization bug as for loopedScalarProduct
    // Note: ^^ 2 is faster then .pow(2)
    // Note: e * e is better for optimizer then ^^ 2
    foreach (e; m.field)
        accu += e * e;
    return accu.sqrt;
}

/// Measure only op execution excluding the time for matrix/slice allocations.
long[][string] functions(in int nruns = 10)
{
    /*
    __gshared is a hack that makes value equivalent to a raw C global (module-level) variables.
    It is used here to trigger eager evaluation of the scalar product function which returns single double value.
    */
    __gshared double gsharedRes;

    StopWatch sw;
    const rows = 500;
    const cols = 600;
    const reduceRows = rows / 5;
    const reduceCols = cols / 6;

    auto smallIntMatrixA = uniformVar!int(1, 10).randomSlice(reduceRows, reduceCols);
    auto smallIntMatrixB = uniformVar!int(1, 10).randomSlice(reduceRows, reduceCols);
    auto smallMatrixA = uniformVar!double(0.0, 1.0).randomSlice(reduceRows, reduceCols);
    auto smallMatrixB = uniformVar!double(0.0, 1.0).randomSlice(reduceRows, reduceCols);
    auto matrixA = uniformVar!double(0.0, 1.0).randomSlice(rows, cols);
    auto matrixB = uniformVar!double(0.0, 1.0).randomSlice(rows, cols);
    auto matrixC = uniformVar!double(0.0, 1.0).randomSlice(cols, rows);
    auto sliceA = uniformVar!double(0.0, 1.0).randomSlice(rows * cols);
    auto sliceB = uniformVar!double(0.0, 1.0).randomSlice(rows * cols);

    long[][string] funcs;

    /// Element-wise sum of two int Slices.
    string name0 = format("Element-wise sum of two [%sx%s] slices (int), (1000 loops)",
            reduceRows, reduceCols);
    auto measurements0 = new long[nruns];
    funcs[name0] = measurements0;
    foreach (i; 0 .. nruns)
    {
        sw.reset;
        sw.start;
        // The benefit of D/Mir over numpy is that you can
        // reuse memory and use native operations in the same time.
        // In numpy you can either use builtin C funcitons
        // or reuse memory by runing slow python loops.
        auto res = smallIntMatrixA.shape.slice!int;
        for (int j; j < 1000; ++j)
        {
            // zero memory allocation
            res[] = smallIntMatrixA * smallIntMatrixB; // can be any element-wise math expression
        }
        sw.stop;
        measurements0[i] = sw.peek.total!"nsecs"; // div by 1000^3 to get sec.
    }

    /// Element-wise multiplication of two double Slices.
    string name1 = format("Element-wise multiplication of two [%sx%s] slices (double), (1000 loops)",
            reduceRows, reduceCols);
    auto measurements1 = new long[nruns];
    funcs[name1] = measurements1;
    foreach (i; 0 .. nruns)
    {
        sw.reset;
        sw.start;
        // ditto, see comment for Element-wise sum of two int Slices.
        auto res = smallMatrixA.shape.slice!double;
        for (int j; j < 1000; ++j)
        {
            // zero memory allocation
            res[] = smallMatrixA * smallMatrixB; // can be any element-wise math expression
        }
        sw.stop;
        measurements1[i] = sw.peek.total!"nsecs";
    }

    /// Dot (scalar) product of two double Slices.
    string name2 = format("Dot (scalar) product of two [%s] slices (double), (1000 loops)",
            rows * cols);
    auto measurements2 = new long[nruns];
    funcs[name2] = measurements2;
    foreach (i; 0 .. nruns)
    {
        sw.reset;
        sw.start;
        for (int j; j < 1000; ++j)
            gsharedRes = scalarProduct(sliceA, sliceB);
        sw.stop;
        measurements2[i] = sw.peek.total!"nsecs";
    }

    /// Dot (scalar) product of two double arrays (plain loop).
    string name3 = format("Dot (scalar) product of two [%s] slices (double), (plain loop), (1000 loops)",
            rows * cols);
    auto measurements3 = new long[nruns];
    funcs[name3] = measurements3;
    foreach (i; 0 .. nruns)
    {
        sw.reset;
        sw.start;
        for (int j; j < 1000; ++j)
            gsharedRes = loopedScalarProduct(sliceA, sliceB);
        sw.stop;
        measurements3[i] = sw.peek.total!"nsecs";
    }

    /// Dot (scalar) product of two double arrays (OpenBLAS).
    string nameDotBLAS = format(
            "Dot (scalar) product of two [%s] slices (double), (OpenBLAS), (1000 loops)",
            rows * cols);
    auto measurementsDotBLAS = new long[nruns];
    funcs[nameDotBLAS] = measurementsDotBLAS;
    foreach (i; 0 .. nruns)
    {
        sw.reset;
        sw.start;
        for (int j; j < 1000; ++j)
            gsharedRes = dot(sliceA, sliceB);
        sw.stop;
        measurementsDotBLAS[i] = sw.peek.total!"nsecs";
    }

    /// Matrix product of two double 2D slices (OpenBLAS).
    string name4 = format("Matrix product of two [%sx%s] and [%sx%s] slices (double), (OpenBLAS)",
            rows, cols, cols, rows);
    auto measurements4 = new long[nruns];
    funcs[name4] = measurements4;
    // The benefit of BLAS API is that it is more cache friendly,
    // don't do allocations in the loop if possible.
    auto matrixD = slice!double([rows, rows]);
    for (int i; i < nruns; ++i)
    {
        sw.reset;
        sw.start;
        // mir-blas can be linked with Intel-MKL (including multithreaded version)
        // please try it instead of OpenBLAS (check its subConfiguration options).
        // Or, at least it compile OpenBLAS with for the native CPU.
        gemm(1.0, matrixA, matrixC, 0, matrixD);
        sw.stop;
        measurements4[i] = sw.peek.total!"nsecs";
    }

    /// L2 norm of double Slice.
    string name5 = format("L2 norm of [%sx%s] slice (double), (1000 loops)", rows, cols);
    auto measurements5 = new long[nruns];
    funcs[name5] = measurements5;
    foreach (i; 0 .. nruns)
    {
        sw.reset;
        sw.start;
        for (int j; j < 1000; ++j)
            gsharedRes = squareL2Norm(matrixB);
        sw.stop;
        measurements5[i] = sw.peek.total!"nsecs";
    }

    /// Sort of double Slice along axis=0.
    string name6 = format("Sort of [%sx%s] slice (double)", rows, cols);
    auto measurements6 = new long[nruns];
    funcs[name6] = measurements6;
    // Don't do allocations in the loop if possible.
    auto matrix = uniformVar!double(-1.0, 1.0).randomSlice([rows, cols]);
    foreach (i; 0 .. nruns)
    {
        // shuffle the matrix
        matrix.flattened.shuffle;
        sw.reset;
        sw.start;
        matrix.byDim!0
            .each!sort;
        sw.stop;
        measurements6[i] = sw.peek.total!"nsecs";
    }

    return funcs;
}

void runMirBenchmarks(int nruns)
{
    pragma(inline, false);
    writeln("---[Mir D]---");
    auto timings = functions(nruns);
    import mir.series; // for sorted output
    foreach (pair; timings.series)
    {
        // convert nsec. to sec. and compute the average
        const double secs = pair.value.sum * 10.0.pow(-9) / pair.value.length;
        writeln(format("| %s | %s |", pair.key, secs));
    }
}

unittest
{
    auto s0 = [1, 2, 0.5, 2, 4, -1, 3].sliced;
    auto s1 = [2, 4, 2.5, 0, 3, 1, 5].sliced;
    assert(scalarProduct(s0, s1) == 37.25);
    assert(loopedScalarProduct(s0, s1) == 37.25);
    auto s2 = [2, 3, 1, -2.3, -1, 5].sliced(2, 3);
    assert(approxEqual(squareL2Norm(s2), 6.72978));
}
