module mir_ops_bench;

import mir.ndslice;
import mir.math.sum;
import mir.math.common : pow, sqrt;
import mir.ndslice.sorting : sort;
import mir.random : threadLocalPtr, Random;
import mir.random.variable : uniformVar, normalVar;
import mir.random.algorithm : randomSlice;

import mir.blas : gemm;
import std.array : array;
import std.datetime.stopwatch : AutoStart, StopWatch;
import std.format : format;
import std.stdio;

import mir.math.common : fastmath, optmath;

alias SliceArr = mir_slice!(double*, 1LU, cast(mir_slice_kind) 2);
alias SliceMatrix = Slice!(double*, 2LU, cast(mir_slice_kind) 2);

pragma(inline) static @optmath fmuladd(T, Z)(const T a, Z z)
{
    return a + z.a * z.b;
}

double dotProduct(SliceArr sliceA1D, SliceArr sliceB1D)
{
    auto zipped = zip!true(sliceA1D, sliceB1D);
    return reduce!fmuladd(0.0, zipped);
}

@fastmath double loopedDotProduct(in SliceArr s0, in SliceArr s1)
{
    pragma(inline, false);
    double accu = 0;
    foreach (size_t i; 0 .. s0.length)
    {
        // will result in vectorized fused-multiply-add instructions
        accu += s0[i] * s1[i];
    }
    return accu;
}

@fastmath double squareL2Norm(SliceMatrix m)
{
    pragma(inline, false);
    auto matrixA = m.flattened;
    double accu = 0;
    foreach (size_t i; 0 .. matrixA.length)
    {
        accu += matrixA[i].pow(2);
    }
    return accu.sqrt;
}

/*
__gshared is a hack that makes value equivalent to a raw C global (module-level) variables.
It is used here to trigger eager evaluation of the scalar product function which returns single double value.
*/
__gshared double res2;
__gshared double res3;
__gshared double res4;

void reportTime(StopWatch sw, string msg)
{
    auto msecs = sw.peek.total!"msecs";
    auto usecs = sw.peek.total!"usecs";
    auto hnsecs = sw.peek.total!"hnsecs";
    auto nsecs = sw.peek.total!"nsecs";
    writeln(format(msg ~ ": %s sec. %s ms. %s μs. %s ns.", msecs / 1000.0,
            usecs / 1000.0, hnsecs / 1000.0, nsecs / 1000.0));
}

void runMirBenchmarks()
{
    auto sw = StopWatch(AutoStart.no);
    const rows = 5000;
    const cols = 6000;

    /// Element-wise sum of two int Slices.
    auto matrixA = threadLocalPtr!Random.randomSlice(uniformVar!double(-1.0, 1.0), [
            rows, cols
            ]);
    auto matrixB = threadLocalPtr!Random.randomSlice(uniformVar!double(-1.0, 1.0), [
            rows, cols
            ]);
    sw.start;
    auto res0 = (matrixA + matrixB).array;
    sw.stop;
    reportTime(sw, format("Element-wise sum of two %sx%s 2D slices", rows, cols));

    /// Element-wise multiplication of two double Slices.
    sw.reset;
    sw.start;
    auto res1 = (matrixA * matrixB).array;
    sw.stop;
    reportTime(sw, format("Element-wise multiplication of two %sx%s 2D slices", rows, cols));

    /// Scalar product of two double arrays.
    auto sliceA1D = threadLocalPtr!Random.randomSlice(uniformVar!double(-1.0, 1.0), 5000);
    auto sliceB1D = threadLocalPtr!Random.randomSlice(uniformVar!double(-1.0, 1.0), 5000);

    /// Scalar product of two double Slices.
    sw.reset;
    sw.start;
    for (int i; i < 50; ++i)
    {
        res3 = loopedDotProduct(sliceA1D, sliceB1D);
    }
    sw.stop;
    reportTime(sw, format("Scalar product of 2x%s double slices (plain loop)", rows * cols));

    sw.reset;
    sw.start;
    for (int i; i < 50; ++i)
    {
        res2 = dotProduct(sliceA1D, sliceB1D);
    }
    sw.stop;
    reportTime(sw, format("Scalar product of 2x%s double slices", rows * cols));

    /// Dot product of two double 2D slices.
    auto matrixC = threadLocalPtr!Random.randomSlice(uniformVar!double(-1.0, 1.0), [
            cols, rows
            ]);
    auto matrixD = slice!double([rows, rows]);
    sw.reset;
    sw.start;
    gemm(1.0, matrixA, matrixC, 0, matrixD);
    sw.stop;
    reportTime(sw, format("Dot product of two 2D double slices"));

    /// L2 norm of double Slice.
    sw.reset;
    sw.start;
    auto res4 = squareL2Norm(matrixB);
    sw.stop;
    reportTime(sw, format("L2 norm of %sx%s double slice = %s", rows, cols, res4));

    /// Sort of double Slice along axis=0
    sw.reset;
    sw.start;
    matrixA.byDim!0
        .each!sort;
    sw.stop;
    reportTime(sw, format("Sorting %sx%s double slice", rows, cols));
}
