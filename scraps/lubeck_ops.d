import std.stdio;
import std.random : Xorshift, unpredictableSeed, uniform;
import std.array: array;
import std.range: generate, take;

import mir.ndslice;
import mir.math.common : optmath;
import lubeck;


static @optmath T fmuladd(T, Z)(const T a, Z z){
    return a + z.a * z.b;
}

static @optmath auto rndMatrix(T)(const T maxN, in int dimA, in int dimB) {
    Xorshift rnd;
    rnd.seed(unpredictableSeed);
    const amount = dimA * dimB;
    return generate(() => uniform(0, maxN, rnd))
        .take(amount)
        .array
        .sliced(dimA, dimB);
}

void svdFun()
{
    Xorshift rnd; 
    rnd.seed(unpredictableSeed);
    auto matrix = generate(() => uniform(0, 1024, rnd)) 
        .take(5000000)
        .array
        .sliced(500, 10000); 
    auto svdResult = matrix.svd;
}

void dotFun() {
    auto matrixA = rndMatrix!double(1.0, 3000, 3000); 
    auto matrixB = rndMatrix!double(1.0, 3000, 3000); 
    auto zipped = zip!true(matrixA, matrixB);
    auto dot = reduce!fmuladd(0.0, zipped);

}

