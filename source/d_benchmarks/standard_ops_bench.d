module standard_ops_bench;
/*
This file contains several benchmarks for multidimensional D arrays:

    - multidimensional array element-wise sum
    - multidimensional array element-wise multiplication
    - single array struct (Matrix) element-wise sum
    - single array struct (Matrix) element-wise multiplication
    - dot product of two single array struct (Matrix)
    - L2 norm of single array struct (Matrix)
    - Standard sort of single array struct (Matrix)

The implementation is naive, so no D black magic optimization here.
There are two multidimensional array representations: array of arrays and a struct using a single array.
First two benchmarks show how slow array of arrays are in comparison to single array struct.
The remaining benchmarks use single array struct for multidimensional array representation.

RUN: dub run --compiler=ldc2 --build=release
TEST: dub run --compiler=ldc2 --build=tests
*/

import core.memory : GC;
import std.algorithm : joiner, fill, map, sort, sum, SwapStrategy;
import std.array : array;
import std.datetime.stopwatch : AutoStart, StopWatch;
import std.format : format;
import std.math : pow, sqrt;
import std.numeric : dotProduct;
import std.random : uniform, unpredictableSeed, Xorshift;
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
        return this.data.chunks(this.cols).array;
    }

    size_t length()
    {
        return this.data.length;
    }

    /// Allow element 2D indexing, e.g. Matrix[row, col]
    T opIndex(in int r, in int c)
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

static T[][] getRandom2DArray(T)(in T max, in int rows, in int cols)
{
    Xorshift rnd;
    rnd.seed(unpredictableSeed);
    const amount = rows * cols;
    return generate(() => uniform(0, max, rnd)).take(amount).array.chunks(cols).array;
}

T[][] elementWiseOP(T)(string op, T[][] arr1, T[][] arr2)
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

auto matrixElementWiseOp(T)(string op, Matrix!T m1, Matrix!T m2)
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
Matrix!T matrixDotProduct(T)(Matrix!T m1, Matrix!T m2, Matrix!T initM)
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

double squareL2Norm(T)(Matrix!T m)
{
    return m.data.map!(a => a.pow(2)).sum.sqrt;
}

Matrix!T standardSort(T)(Matrix!T m)
{
    m.data = sort!("a < b", SwapStrategy.unstable)(m.data).array;
    return m;
}

void reportTime(StopWatch sw, string msg)
{
    auto msecs = sw.peek.total!"msecs";
    auto usecs = sw.peek.total!"usecs";
    writeln(format(msg ~ ": %s sec. %s msec.", msecs / 1000.0, usecs / 1000.0));
}

void runStandardBenchmarks()
{
    auto sw = StopWatch(AutoStart.no);
    int rows = 5000;
    int cols = 6000;

    /// Element-wise sum of 2x[rows, cols] random 2D int arrays.
    int[][] arr2D1 = getRandom2DArray!int(10, rows, cols);
    int[][] arr2D2 = getRandom2DArray!int(10, rows, cols);
    sw.start;
    int[][] a = elementWiseOP!int(OPS.sum, arr2D1, arr2D2);
    sw.stop;
    reportTime(sw, format("Element-wise sum of two %sx%s int arrays", rows, cols));

    /// Element-wise multiplication of 2x[rows, cols] random 2D double arrays.
    double[][] arr2D3 = getRandom2DArray!double(1.0, rows, cols);
    double[][] arr2D4 = getRandom2DArray!double(1.0, rows, cols);
    sw.reset;
    sw.start;
    double[][] b = elementWiseOP(OPS.mul, arr2D3, arr2D4);
    sw.stop;
    reportTime(sw, format("Element-wise multiplication of two %sx%s double arrays", rows, cols));

    /// Element-wise sum of 2x[rows, cols] random int Matrices.
    auto m1 = Matrix!int(rows, cols, getRandomArray!int(10, rows * cols));
    auto m2 = Matrix!int(rows, cols, getRandomArray!int(10, rows * cols));
    sw.reset;
    sw.start;
    auto c = matrixElementWiseOp!int(OPS.sum, m1, m2).to2D;
    sw.stop;
    reportTime(sw, format("Element-wise sum of two %sx%s int Matrices", rows, cols));

    /// Element-wise multiplication of 2x[rows, cols] random double Matrices.
    auto m3 = Matrix!double(rows, cols, getRandomArray!double(1.0, rows * cols));
    auto m4 = Matrix!double(rows, cols, getRandomArray!double(1.0, rows * cols));
    sw.reset;
    sw.start;
    auto d = matrixElementWiseOp!double(OPS.mul, m3, m4).to2D;
    sw.stop;
    reportTime(sw, format("Element-wise multiplication of two %sx%s double Matrices", rows, cols));

    /// Scalar dot product of two random double arrays.
    auto vec0 = getRandomArray!double(1.0, rows * cols);
    auto vec1 = getRandomArray!double(1.0, rows * cols);
    sw.reset;
    sw.start;
    auto h = dotProduct(vec0, vec1);
    sw.stop;
    reportTime(sw, format("Scalar product of two %s double arrays", rows * cols));

    /// Dot product of two random double Matrices, we pick smaller arrays for speed.
    const int dotRows = 500;
    const int dotCols = 1000;
    auto m5 = Matrix!double(dotRows, dotCols, getRandomArray!double(1.0, dotRows * dotCols));
    auto m6 = Matrix!double(dotCols, dotRows, getRandomArray!double(1.0, dotRows * dotCols));
    sw.reset;
    sw.start;

    Matrix!double initMatrix = Matrix!double(m5.rows, m6.cols);
    auto e = matrixDotProduct!double(m5, m6, initMatrix);
    sw.stop;
    reportTime(sw, "Dot product of double Matrices");

    /// L2 norm of double Matrix.
    auto m7 = Matrix!double(rows, cols, getRandomArray!double(1.0, rows * cols));
    sw.reset;
    sw.start;
    auto f = squareL2Norm(m7);
    sw.stop;
    reportTime(sw, format("L2 norm of %sx%s double Matrix", rows, cols));

    /// Standard sort of double Matrix.
    auto m8 = Matrix!double(rows, cols, getRandomArray!double(10.0, rows * cols));
    sw.reset;
    sw.start;
    auto g = standardSort(m8).to2D;
    sw.stop;
    reportTime(sw, format("Standard sort of %sx%s double Matrix", rows, cols));
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
