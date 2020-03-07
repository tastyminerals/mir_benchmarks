module mir_ops_bench;

import mir.ndslice;
import mir.random : threadLocalPtr, Random;
import mir.random.variable : uniformVar, normalVar;
import mir.random.algorithm : randomSlice;

import mir.blas : gemm;
import std.array : array;
import std.datetime.stopwatch : AutoStart, StopWatch;
import std.format : format;
import std.stdio;

import mir.math.common : fastmath, optmath;

alias Slice = mir_slice!(double*, 1LU, cast(mir_slice_kind) 2);

pragma(inline) static @optmath fmuladd(T, Z)(const T a, Z z)
{
    return a + z.a * z.b;
}

double dotProduct(Slice arr0, Slice arr1)
{
    auto zipped = zip!true(arr0, arr1);
    return reduce!fmuladd(0.0, zipped);
}

@fastmath double loopedDotProduct(in Slice s0, in Slice s1)
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

/*
__gshared is a hack that makes value equivalent to a raw C global (module-level) variables.
It is used here to trigger eager evaluation of the scalar product function which returns single double value.
*/
__gshared double res2;
__gshared double res3;

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
    auto m0 = threadLocalPtr!Random.randomSlice(uniformVar!double(-1.0, 1.0), [
            rows, cols
            ]);
    auto m1 = threadLocalPtr!Random.randomSlice(uniformVar!double(-1.0, 1.0), [
            rows, cols
            ]);
    sw.start;
    auto res0 = (m0 + m1).array;
    sw.stop;
    reportTime(sw, format("Element-wise sum of two %sx%s 2D slices", rows, cols));

    /// Element-wise multiplication of two double Slices.
    sw.reset;
    sw.start;
    auto res1 = (m0 * m1).array;
    sw.stop;
    reportTime(sw, format("Element-wise multiplication of two %sx%s 2D slices", rows, cols));

    /// Scalar product of two double arrays.
    auto arr0 = threadLocalPtr!Random.randomSlice(uniformVar!double(-1.0, 1.0), 5000);
    auto arr1 = threadLocalPtr!Random.randomSlice(uniformVar!double(-1.0, 1.0), 5000);

    /// Scalar product of two double Slices.
    sw.reset;
    sw.start;
    for (int i; i < 50; ++i)
    {
        res3 = loopedDotProduct(arr0, arr1);
    }
    sw.stop;
    reportTime(sw, format("Scalar product of 2x%s double slices (plain loop)", rows * cols));

    sw.reset;
    sw.start;
    for (int i; i < 50; ++i)
    {
        res2 = dotProduct(arr0, arr1);
    }
    sw.stop;
    reportTime(sw, format("Scalar product of 2x%s double slices", rows * cols));

    /// Dot product of two double 2D slices.
    auto m3 = slice!double([rows, cols]);
    sw.reset;
    sw.start;
    gemm(1.0, m0, m1, 0, m3);
    sw.stop;
    reportTime(sw, format("Dot product of two 2D double slices"));

    /// L2 norm of double Slice.
    /// Sort of double Slice.
}
