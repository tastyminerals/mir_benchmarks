module basic_ops;

import std.stdio;
import mir.ndslice;
import std.array;

/// mir.ndslice object properties
void props()
{
    // Create a simple matrix slice.
    int err;
    // Does it reshape and return a new copy?
    auto matrix = 100.iota.array.sliced.reshape([20, 5], err);

    writeln("matrix strides: ", matrix.strides); // [5, 1]
    writeln("matrix shape: ", matrix.shape); // [20, 5]
    writeln("matrix length: ", matrix.length); // 20
    writeln("matrix: ", matrix);

}

/// D multidim arrays construction
void arrays()
{
    import std.range;

    // multidim array creation from jagged array
    int[][] jaggedArr = [[0, 1, 2], [3, 4, 5, 6], [7, 8]];

    // multidim dense array, more memory efficient
    int[2][3] denseArr = [[1, 2], [3, 4], [5, 6]];

    // 2-dim view of flat 1-dim buffer, view is not a copy!
    int rows1, cols1 = 3;
    int[] arr = new int[rows1 * cols1];
    auto arrView = arr.chunks(cols1);

    // dynamic dense arrays, all array elems are contiguous
    enum cols2 = 5;
    int rows2 = 3;
    double[cols2][] dynamicDense = new double[cols2][](rows2);

    writeln("jagged array: ", jaggedArr);
    writeln("dense array: ", denseArr);
    writeln("array view: ", arrView);
    writeln("dynamic dense array: ", dynamicDense);

}

/// mir.ndslice multidim array construction
void mirArrays()
{
    int[][] jaggedArr = [[0, 1, 2], [3, 4, 5]];
    auto arrSlice1 = sliced(jaggedArr);
    writeln(typeof(arrSlice1).stringof);
    auto arrSlice2 = jaggedArr.sliced;
    auto arrSlice3 = [0, 1, 2, 3, 4, 5].sliced;

    auto arr = 100.iota.array;
    auto arrSlice4 = arr.sliced;
    auto arrSlice5 = slice([2, 3], -0.1);
    auto arrSlice6 = slice!int([2, 3], 100);
    writeln(typeof(arrSlice6).stringof);

    writeln("slice1: ", arrSlice1);
    writeln("slice2: ", arrSlice2);
    writeln("slice3: ", arrSlice3);

    writeln("slice4: ", arrSlice4);
    writeln("slice5: ", arrSlice5);
    writeln("slice6: ", arrSlice6);

}

/// mir.ndslice basic operations on multidim arrays
void basicOps()
{
    import std.range : generate, take;
    import std.random : uniform;
    import std.math : exp, sqrt;
    import std.conv : to;
    import std.numeric : dotProduct;
    import mir.math.sum : sum;
    import mir.ndslice.topology : as;

    // create a matrix of random elements
    auto rndArr = generate!(() => uniform(0, 0.99)).take(10).array;
    int err;
    auto rndMatrix = rndArr.sliced.reshape([2, 5], err);
    writeln("reshaped matrix: ", rndMatrix);

    auto arrExp = rndArr.as!double
        .map!exp;
    writeln("exp on rndArr: ", arrExp);
    auto arrSqrt = rndArr.as!double
        .map!sqrt;
    writeln("sqrt on rndArr: ", arrSqrt);
    auto rndSum = rndArr.sum!"precise";
    writeln("sum of rdnArr: ", rndSum);

    alias dot = reduce!((a, b, c) => a + b * c);
    auto a = iota([4, 2], -1);
    auto b = [2.1, 3.4, 5.6, 7.8, 3.9, 9.0, 4.0, 2.0].sliced(2, 4);
    auto ab = dot(0.0, a, b.transposed);
    writeln("dot product: ", ab);
    assert(ab == dotProduct(a.flattened, b.transposed.flattened));
}
