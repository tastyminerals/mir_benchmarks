module mir_ops_bench;

import core.thread : Thread;
import mir.blas : gemm;
import mir.math.common : pow, sqrt;
import mir.math.common : fastmath, optmath;
import mir.math.sum;
import mir.ndslice;
import mir.ndslice.sorting : sort;
import mir.random : Random, threadLocalPtr;
import mir.random.algorithm : randomSlice;
import mir.random.variable : normalVar, uniformVar;
import std.array : array;
import std.datetime : dur, Duration;
import std.datetime.stopwatch : AutoStart, StopWatch;
import std.format : format;
import std.stdio;

alias SliceArr = mir_slice!(double*, 1LU, cast(mir_slice_kind) 2);
alias SliceMatrix = Slice!(double*, 2LU, cast(mir_slice_kind) 2);

pragma(inline) static @optmath fmuladd(T, Z)(const T a, Z z)
{
    return a + z.a * z.b;
}

double scalarProduct(SliceArr sliceA1D, SliceArr sliceB1D)
{
    auto zipped = zip!true(sliceA1D, sliceB1D);
    return reduce!fmuladd(0.0, zipped);
}

@fastmath double loopedScalarProduct(in SliceArr s0, in SliceArr s1)
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

/// Measure only op execution excluding the time for matrix/slice allocations.
long[][string] functions(in int nruns = 10)
{
    /*
    __gshared is a hack that makes value equivalent to a raw C global (module-level) variables.
    It is used here to trigger eager evaluation of the scalar product function which returns single double value.
    */
    __gshared double gsharedRes0;
    __gshared double gsharedRes1;
    __gshared double gsharedRes2;

    auto sw = StopWatch(AutoStart.no);
    Duration sleepTime = dur!"msecs"(500);
    const rows = 5000;
    const cols = 6000;
    auto smallIntMatrixA = threadLocalPtr!Random.randomSlice(uniformVar!int(1,
            10), [rows / 20, cols / 30]);
    auto smallIntMatrixB = threadLocalPtr!Random.randomSlice(uniformVar!int(1,
            10), [rows / 20, cols / 30]);
    auto smallMatrixA = threadLocalPtr!Random.randomSlice(uniformVar!double(0.0,
            1.0), [rows / 20, cols / 30]);
    auto smallMatrixB = threadLocalPtr!Random.randomSlice(uniformVar!double(0.0,
            1.0), [rows / 20, cols / 30]);
    auto matrixA = threadLocalPtr!Random.randomSlice(uniformVar!double(0.0, 1.0), [
            rows, cols
            ]);
    auto matrixB = threadLocalPtr!Random.randomSlice(uniformVar!double(0.0, 1.0), [
            rows, cols
            ]);
    auto matrixC = threadLocalPtr!Random.randomSlice(uniformVar!double(0.0, 1.0), [
            cols, rows
            ]);
    auto sliceA = threadLocalPtr!Random.randomSlice(uniformVar!double(0.0, 1.0), rows * cols);
    auto sliceB = threadLocalPtr!Random.randomSlice(uniformVar!double(0.0, 1.0), rows * cols);

    long[][string] funcs;

    /// Element-wise sum of two int Slices.
    string name0 = format("Element-wise sum of two [%sx%s] slices (int), (200 loops)",
            rows / 20, cols / 30);
    for (int i; i < nruns; ++i)
    {
        sw.reset;
        sw.start;
        for (int j; j < 200; ++j)
        {
            auto res = (smallIntMatrixA + smallIntMatrixB).array;
        }
        sw.stop;
        funcs[name0] ~= sw.peek.total!"nsecs"; // div by 1000^3 to get sec.
        Thread.sleep(sleepTime);
    }

    /// Element-wise multiplication of two double Slices.
    string name1 = format("Element-wise multiplication of two [%sx%s] slices (double), (200 loops)",
            rows / 20, cols / 30);
    for (int i; i < nruns; ++i)
    {
        sw.reset;
        sw.start;
        for (int j; j < 200; ++j)
        {
            auto res = (smallMatrixA * smallMatrixB).array;
        }
        sw.stop;
        funcs[name1] ~= sw.peek.total!"nsecs";
        Thread.sleep(sleepTime);
    }

    /// Scalar product of two double Slices.
    string name2 = format("Scalar product of two [%s] slices (double)", rows * cols);
    for (int i; i < nruns; ++i)
    {
        sw.reset;
        sw.start;
        gsharedRes0 = scalarProduct(sliceA, sliceB);
        sw.stop;
        funcs[name2] ~= sw.peek.total!"nsecs";
        Thread.sleep(sleepTime);
    }

    /// Scalar product of two double arrays (plain loop).
    string name3 = format("Scalar product of two [%s] slices (double), (plain loop)", rows * cols);
    for (int i; i < nruns; ++i)
    {
        sw.reset;
        sw.start;
        gsharedRes1 = loopedScalarProduct(sliceA, sliceB);
        sw.stop;
        funcs[name3] ~= sw.peek.total!"nsecs";
        Thread.sleep(sleepTime);
    }

    /// Dot product of two double 2D slices.
    string name4 = format("Dot product of two [%sx%s] and [%sx%s] slices (double)",
            rows, cols, cols, rows);
    for (int i; i < nruns; ++i)
    {
        auto matrixD = slice!double([rows, rows]);
        sw.reset;
        sw.start;
        gemm(1.0, matrixA, matrixC, 0, matrixD);
        sw.stop;
        funcs[name4] ~= sw.peek.total!"nsecs";
        Thread.sleep(sleepTime);
    }

    /// L2 norm of double Slice.
    string name5 = format("L2 norm of [%sx%s] slice (double)", rows, cols);
    for (int i; i < nruns; ++i)
    {
        sw.reset;
        sw.start;
        gsharedRes2 = squareL2Norm(matrixB);
        sw.stop;
        funcs[name5] ~= sw.peek.total!"nsecs";
        Thread.sleep(sleepTime);
    }

    /// Sort of double Slice along axis=0.
    string name6 = format("Sort of [%sx%s] slice (double)", rows, cols);
    for (int i; i < nruns; ++i)
    {
        auto matrix = threadLocalPtr!Random.randomSlice(uniformVar!double(-1.0,
                1.0), [rows, cols]);
        sw.reset;
        sw.start;
        matrix.byDim!0
            .each!sort;
        sw.stop;
        funcs[name6] ~= sw.peek.total!"nsecs";
        Thread.sleep(sleepTime);
    }

    return funcs;
}

void runMirBenchmarks()
{
    auto timings = functions(20);
    foreach (pair; timings.byKeyValue)
    {
        // convert nsec. to sec. and compute the average
        const double secs = pair.value.map!(a => a / pow(1000.0, 3)).sum / pair.value.length;
        writeln(format("%s --> %s sec.", pair.key, secs));
    }
}
