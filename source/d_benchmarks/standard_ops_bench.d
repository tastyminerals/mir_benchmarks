module standard_ops_bench;
/*
This file contains several benchmarks for multidimensional D arrays:

    - multidimensional array element-wise sum
    - multidimensional array element-wise multiplication
    - single array struct (Matrix) element-wise sum
    - single array struct (Matrix) element-wise multiplication
    - Dot (scalar) product of two arrays
    - Matrix product of two single array struct (Matrix)
    - L2 norm of single array struct (Matrix)
    - Standard sort of single array struct (Matrix)

There are two multidimensional array representations: array of arrays and a struct using a single array.
First two benchmarks show how slow array of arrays are in comparison to single array struct.
The remaining benchmarks use single array struct for multidimensional array representation.

RUN: dub run --compiler=ldc2 --build=release
TEST: dub run --compiler=ldc2 --build=tests
*/

import core.memory : GC;
import mir.math.common : fastmath, optmath;
import std.algorithm : each, fill, joiner, map, sort, sum, SwapStrategy;
import std.array : array;
import std.datetime.stopwatch : AutoStart, StopWatch;
import std.format : format;
import std.math : pow, sqrt;
import std.numeric : dotProduct;
import std.random : randomShuffle, uniform, unpredictableSeed, Xorshift;
import std.range : chunks, generate, take, zip;
import std.stdio;

enum OPS
{
    sum = "+",
    mul = "*",
    sub = "-",
    div = "/"
}

struct Matrix(T)
{
    T[] data;
    int rows;
    int cols;
    // allow Matrix[] instead of Matrix.data[]
    alias data this;

    this(int rows, int cols)
    {
        this.data = new T[rows * cols];
        this.data.fill(0); // because double/float initialize to nan
        this.rows = rows;
        this.cols = cols;
    }

    this(int rows, int cols, T[] data)
    {
        assert(data.length == rows * cols);
        this.data = data;
        this.rows = rows;
        this.cols = cols;
    }

    T[][] to2D()
    {
        return to2DLazy.array;
    }

    auto to2DLazy()
    {
        return this.data.chunks(this.cols);
    }

    size_t length()
    {
        return this.data.length;
    }

    /// Allow element 2D indexing, e.g. Matrix[row, col]
    @fastmath T opIndex(in int r, in int c)
    {
        return this.data[this.cols * r + c];
    }

}

static T[] getRandomArray(T)(in T max, in int elems)
{
    Xorshift rnd;
    rnd.seed(unpredictableSeed);
    return generate(() => uniform(0, max, rnd)).take(elems).array;
}

static T[][] getRandomAArray(T)(in T max, in int rows, in int cols)
{
    Xorshift rnd;
    rnd.seed(unpredictableSeed);
    const amount = rows * cols;
    return generate(() => uniform(0, max, rnd)).take(amount).array.chunks(cols).array;
}

@fastmath T[][] elementWiseOP(T)(string op, T[][] arr1, T[][] arr2)
{
    T[][] arr3;
    switch (op)
    {
    case OPS.sum:
        arr3 = zip(arr1.joiner, arr2.joiner).map!(t => t[0] + t[1])
            .array.chunks(arr1[1].length).array;
        break;

    case OPS.mul:
        arr3 = zip(arr1.joiner, arr2.joiner).map!(t => t[0] * t[1])
            .array.chunks(arr1[1].length).array;
        break;

    default:
        break;

    }
    return arr3;
}

@fastmath auto matrixElementWiseOp(T)(string op, Matrix!T m1, Matrix!T m2)
in
{
    assert(m1.rows == m2.rows);
    assert(m1.cols == m2.cols);
}
do
{
    T[] data;
    data.length = m1.data.length;
    switch (op)
    {
    case OPS.sum:
        data[] = m1.data[] + m2.data[];
        break;
    case OPS.mul:
        data[] = m1.data[] * m2.data[]; // hadamard product
        break;
    default:
        break;
    }
    return Matrix!T(m1.rows, m1.cols, data);
}

/// [2 x 3]@[3 x 2]-- > [2 x 2], slow
@fastmath Matrix!T matrixDotProduct(T)(Matrix!T m1, Matrix!T m2, Matrix!T initM)
in
{
    assert(m1.rows == m2.cols);
}
do
{
    /// This implementation requires opIndex in Matrix struct.
    for (int i; i < m1.rows; ++i)
    {
        for (int j; j < m2.cols; ++j)
        {
            for (int k; k < m2.rows; ++k)
            {
                initM.data[initM.cols * i + j] += m1[i, k] * m2[k, j];
            }
        }
    }
    return initM;
}

@fastmath double squareL2Norm(T)(Matrix!T m)
{
    return m.data.map!(a => a * a).sum.sqrt;
}

@fastmath Matrix!T columnWiseSort(T)(Matrix!T m)
{
    m.to2DLazy.each!(row => sort!("a < b", SwapStrategy.unstable)(row));
    return m;
}

/// Measure only op execution excluding the time for matrix/slice allocations.
long[][string] functions(in int nruns = 10)
{
    auto sw = StopWatch(AutoStart.no);
    const rows = 500;
    const cols = 600;
    const reduceRows = rows / 5;
    const reduceCols = cols / 6;

    int[][] smallIntArrOfArraysA = getRandomAArray!int(10, reduceRows, reduceCols);
    int[][] smallIntArrOfArraysB = getRandomAArray!int(10, reduceRows, reduceCols);
    double[][] smallArrOfArraysA = getRandomAArray!double(1.0, reduceRows, reduceCols);
    double[][] smallArrOfArraysB = getRandomAArray!double(1.0, reduceRows, reduceCols);
    auto smallIntMatrixA = Matrix!int(reduceRows, reduceCols,
            getRandomArray!int(10, reduceRows * reduceCols));
    auto smallIntMatrixB = Matrix!int(reduceRows, reduceCols,
            getRandomArray!int(10, reduceRows * reduceCols));
    auto smallMatrixA = Matrix!double(reduceRows, reduceCols,
            getRandomArray!double(1.0, reduceRows * reduceCols));

    auto smallMatrixB = Matrix!double(reduceRows, reduceCols,
            getRandomArray!double(1.0, reduceRows * reduceCols));

    auto arrayA = getRandomArray!double(1.0, rows * cols);
    auto arrayB = getRandomArray!double(1.0, rows * cols);

    auto matrixA = Matrix!double(rows, cols, getRandomArray!double(1.0, rows * cols));
    auto matrixB = Matrix!double(cols, rows, getRandomArray!double(1.0, rows * cols));

    auto matrixC = Matrix!double(rows, cols, getRandomArray!double(1.0, rows * cols));
    auto matrixD = Matrix!double(rows, cols, getRandomArray!double(1.0, rows * cols));

    long[][string] funcs;
    string name0 = format("Element-wise sum of two [%sx%s] arrays of arrays (int), (1000 loops)",
            reduceRows, reduceCols);
    for (int i; i < nruns; ++i)
    {
        sw.reset;
        sw.start;
        for (int j; j < 1000; ++j)
        {
            int[][] res = elementWiseOP!int(OPS.sum, smallIntArrOfArraysA, smallIntArrOfArraysB);
        }
        sw.stop;
        funcs[name0] ~= sw.peek.total!"nsecs";
    }

    string name1 = format("Element-wise multiplication of two [%sx%s] arrays of arrays (double), (1000 loops)",
            reduceRows, reduceCols);
    for (int i; i < nruns; ++i)
    {
        sw.reset;
        sw.start;
        for (int j; j < 1000; ++j)
        {
            double[][] res = elementWiseOP!double(OPS.mul, smallArrOfArraysA, smallArrOfArraysB);
        }
        sw.stop;
        funcs[name1] ~= sw.peek.total!"nsecs";
    }

    string name2 = format("Element-wise sum of two [%sx%s] struct matrices (int), (1000 loops)",
            reduceRows, reduceCols);
    for (int i; i < nruns; ++i)
    {
        sw.reset;
        sw.start;
        for (int j; j < 1000; ++j)
        {
            auto res = matrixElementWiseOp!int(OPS.sum, smallIntMatrixA, smallIntMatrixB).to2D;
        }
        sw.stop;
        funcs[name2] ~= sw.peek.total!"nsecs";
    }

    string name3 = format("Element-wise multiplication of two [%sx%s] struct matrices (double), (1000 loops)",
            reduceRows, reduceCols);
    for (int i; i < nruns; ++i)
    {
        sw.reset;
        sw.start;
        for (int j; j < 1000; ++j)
        {
            auto res = matrixElementWiseOp!double(OPS.mul, smallMatrixA, smallMatrixB).to2D;
        }
        sw.stop;
        funcs[name3] ~= sw.peek.total!"nsecs";
    }

    string name4 = format("Dot (scalar) product of two [%s] arrays (double), (1000 loops)",
            rows * cols);
    for (int i; i < nruns; ++i)
    {
        sw.reset;
        sw.start;
        for (int j; j < 1000; ++j)
            auto res = dotProduct(arrayA, arrayB);
        sw.stop;
        funcs[name4] ~= sw.peek.total!"nsecs";
    }

    /* 
        We pick smaller because straight calculation of matrices >5kk elems becomes prohibitevely slow 
        without using optimization techniques. 
    */
    string name5 = format("(Reference only) unoptimized matrix product of two [%sx%s] struct matrices (double)",
            rows, cols);
    for (int i; i < nruns; ++i)
    {
        sw.reset;
        sw.start;
        Matrix!double initMatrix = Matrix!double(matrixA.rows, matrixB.cols);
        auto res = matrixDotProduct!double(matrixA, matrixB, initMatrix);
        sw.stop;
        funcs[name5] ~= sw.peek.total!"nsecs";
    }

    string name6 = format("L2 norm of [%sx%s] struct matrix (double), (1000 loops)", rows, cols);
    for (int i; i < nruns; ++i)
    {
        sw.reset;
        sw.start;
        for (int j; j < 1000; ++j)
            auto res = squareL2Norm(matrixC);
        sw.stop;
        funcs[name6] ~= sw.peek.total!"nsecs";
    }

    string name7 = format("Sort of [%sx%s] struct matrix (double)", rows, cols);
    for (int i; i < nruns; ++i)
    {
        matrixD.to2D.each!randomShuffle; // reshuffle columns
        sw.reset;
        sw.start;
        auto res = columnWiseSort(matrixD);
        sw.stop;
        funcs[name7] ~= sw.peek.total!"nsecs";
    }

    return funcs;
}

void runStandardBenchmarks(int nruns)
{
    pragma(inline, false);
    writeln("---[Standard D]---");
    auto timings = functions(nruns);
    import mir.series; // for sorted output
    foreach (pair; timings.series)
    {
        // convert nsec. to sec. and compute the average
        const double secs = pair.value.map!(a => a / pow(1000.0, 3)).sum / pair.value.length;
        writeln(format("| %s | %s |", pair.key, secs));
    }
}

unittest
{
    int[][] arr1 = [[1, 2, 3], [4, 5, 6]];
    int[][] arr2 = [[1, 1, 1], [2, 2, 2]];
    int[][] res0 = [[2, 3, 4], [6, 7, 8]];
    assert(elementWiseOP(OPS.sum, arr1, arr2) == res0);

    auto ma = Matrix!int(2, 3, [2, 1, 1, 2, 2, 0]);
    auto mb = Matrix!int(2, 3, [-1, 0, 1, 0, 0, -1]);
    int[][] res1 = [[1, 1, 2], [2, 2, -1]];
    assert(matrixElementWiseOp!int(OPS.sum, ma, mb).to2D == res1);

    auto mc = Matrix!double(2, 3, [-2, 0, 1, 0, 0, -3]);
    auto md = Matrix!double(3, 2, [-1, 0, 2, 0, 0, -1]);
    auto initM = Matrix!double(2, 2);
    auto res2 = Matrix!double(2, 2, [2, -1, 0, 3]);
    assert(matrixDotProduct!double(mc, md, initM).data == res2.data);

    import std.math : approxEqual;

    auto me = Matrix!double(2, 3, [-1.5, 2.3, 1.1, 0.2, 0.5, -3.7]);
    double res3 = 4.7676;
    auto res4 = Matrix!double(2, 3, [-3.7, -1.5, 0.2, 0.5, 1.1, 2.3]);
    assert(approxEqual(squareL2Norm!double(me), res3));
    assert(standardSort!double(me).to2D == res4.to2D);
}
