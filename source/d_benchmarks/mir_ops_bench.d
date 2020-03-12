module mir_ops_bench;

import mir.blas : gemm;
import mir.math.common : pow, sqrt;
import mir.math.common : fastmath, optmath;
import mir.math.sum;
import mir.ndslice;
import mir.ndslice.sorting : sort;
import mir.random : Random, threadLocalPtr;
import mir.random.algorithm : randomSlice, shuffle;
import mir.random.variable : normalVar, uniformVar;
import std.datetime.stopwatch : AutoStart, StopWatch;
import std.format : format;
import std.stdio;

alias SliceArr = Slice!(double*);
alias SliceMatrix = Slice!(double*, 2);
alias SliceMatrixArr = Slice!(double*, 2)[];

pragma(inline) static @optmath fmuladd(T, Z)(const T a, Z z)
{
    return a + z.a * z.b;
}

double scalarProduct(SliceArr sliceA1D, SliceArr sliceB1D)
{
    //zip is the same as zip!true for Contiguous kind of Slices
    auto zipped = zip(sliceA1D, sliceB1D);
    return reduce!fmuladd(0.0, zipped);
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
    auto matrixA = m.flattened;
    double accu = 0;
    // NOTE: that BLAS nrm2
    // has more robust but slower algorithm

    // NOTE: .field returns a common D array
    // to workaround the same vectorisation bug as for loopedScalarProduct
    // Note: ^^ 2 is faster then .pow(2)
    // Note: e * e is better for optimiser then ^^ 2
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
    __gshared double gsharedRes0;
    __gshared double gsharedRes1;
    __gshared double gsharedRes2;

    auto sw = StopWatch(AutoStart.no);
    const rows = 500;
    const cols = 600;
    const reduceRowsBy = 5;
    const reduceColsBy = 6;

    auto smallIntMatrixA = threadLocalPtr!Random.randomSlice(uniformVar!int(1,
            10), [rows / reduceRowsBy, cols / reduceColsBy]);
    auto smallIntMatrixB = threadLocalPtr!Random.randomSlice(uniformVar!int(1,
            10), [rows / reduceRowsBy, cols / reduceColsBy]);
    auto smallMatrixA = threadLocalPtr!Random.randomSlice(uniformVar!double(0.0,
            1.0), [rows / reduceRowsBy, cols / reduceColsBy]);
    auto smallMatrixB = threadLocalPtr!Random.randomSlice(uniformVar!double(0.0,
            1.0), [rows / reduceRowsBy, cols / reduceColsBy]);
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
    string name0 = format("Element-wise sum of two [%sx%s] slices (int), (1000 loops)",
            rows / reduceRowsBy, cols / reduceColsBy);
    for (int i; i < nruns; ++i)
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
        funcs[name0] ~= sw.peek.total!"nsecs"; // div by 1000^3 to get sec.
    }

    /// Element-wise multiplication of two double Slices.
    string name1 = format("Element-wise multiplication of two [%sx%s] slices (double), (1000 loops)",
            rows / reduceRowsBy, cols / reduceColsBy);
    for (int i; i < nruns; ++i)
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
        funcs[name1] ~= sw.peek.total!"nsecs";
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
    }

    /// Dot product of two double 2D slices.
    string name4 = format("Dot product of two [%sx%s] and [%sx%s] slices (double)",
            rows, cols, cols, rows);
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
        funcs[name4] ~= sw.peek.total!"nsecs";
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
    }

    /// Sort of double Slice along axis=0.
    string name6 = format("Sort of [%sx%s] slice (double)", rows, cols);
    // Don't do allocations in the loop if possible.
    auto matrix = uniformVar!double(-1.0, 1.0).randomSlice([rows, cols]);
    for (int i; i < nruns; ++i)
    {
        // shufle the matrix
        matrix.flattened.shuffle;
        sw.reset;
        sw.start;
        matrix.byDim!0
            .each!sort;
        sw.stop;
        funcs[name6] ~= sw.peek.total!"nsecs";
    }

    return funcs;
}

void runMirBenchmarks()
{
    writeln("---[Mir D]---");
    auto timings = functions(20);
    import mir.series; // for sorted output
    foreach (pair; timings.series)
    {
        // convert nsec. to sec. and compute the average
        const double secs = pair.value.map!(a => a / pow(1000.0, 3)).sum / pair.value.length;
        writeln(format("| %s | %s |", pair.key, secs));
    }
}
