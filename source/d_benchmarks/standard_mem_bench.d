module standard_mem_bench;

/*
Memory allocation benchmarks for standard D:

    - allocation, writing and deallocation of a huge array
    - allocation, writing and deallocation of several big arrays of different sizes
    - reallocation of one array into three arrays by indexing and slicing

*/

import core.memory : GC;
import mir.math.common : fastmath;
import std.algorithm : each, fill, joiner, map, sort, sum, SwapStrategy;
import std.array;
import std.format;
import std.stdio;
import std.random : randomShuffle, uniform, unpredictableSeed, Xorshift;
import std.range : chunks, generate, iota, take, zip;
import std.math : pow, sqrt;
import std.datetime.stopwatch : AutoStart, StopWatch;

static T[] getRandomArray(T)(in int elems)
{
    Xorshift rnd;
    rnd.seed(unpredictableSeed);
    return generate(() => uniform(0, 1.0, rnd)).take(elems).array;
}

long[][string] functions(in int nruns)
{
    auto sw = StopWatch(AutoStart.no);
    const size = 30_000_000;

    Xorshift rnd;
    rnd.seed(unpredictableSeed);
    auto lazyArr = generate(() => uniform(0, 1.0, rnd)).take(size);

    long[][string] funcs;
    string name0 = format("Allocation, writing and deallocation of a [%s] array ", size);
    for (int i; i < nruns; ++i)
    {
        sw.reset;
        sw.start;
        for (int j; j < 5; ++j)
        {
            // allocate
            double[] arr = lazyArr.array;
            // write
            arr[j] = 100.0;
            // destroy
            arr = null;
            GC.collect;
        }
        sw.stop;
        funcs[name0] ~= sw.peek.total!"nsecs";
    }

    string name1 = "Allocation, writing and deallocation of a several big arrays of different sizes";
    int[] mods = [10, 3, 5, 60, 1];
    for (int i; i < nruns; ++i)
    {
        sw.reset;
        sw.start;
        foreach (mod; mods)
        {
            // allocate
            double[] arr = getRandomArray!double(size / mod);
            // write
            arr[mod] = 100.0;
            // destroy
            arr = null;
            GC.collect;
        }
        sw.stop;
        funcs[name1] ~= sw.peek.total!"nsecs";
    }

    const int size1 = size / 100_000;
    const size2 = iota(1, size1 + 1).sum;
    string name2 = format("Slicing [%s] array into another array (%s loops)", size1, size1);
    double[size1] arr = getRandomArray!double(size1);
    for (int i; i < nruns; ++i)
    {
        sw.reset;
        sw.start;
        double[] a;
        a.reserve(size2);
        for (int j; j < size1; ++j)
        {
            a ~= arr[j .. $];
        }
        sw.stop;
        funcs[name2] ~= sw.peek.total!"nsecs";
        a = null;
        GC.collect;
    }

    return funcs;
}

void runStandardMemoryBenchmarks(int nruns)
{
    pragma(inline, false);
    writeln("---[Standard Memory D]---");
    auto timings = functions(nruns);
    import mir.series; // for sorted output
    foreach (pair; timings.series)
    {
        // convert nsec. to sec. and compute the average
        const double secs = pair.value.sum * 10.0.pow(-9) / pair.value.length;
        writeln(format("| %s | %s |", pair.key, secs));
    }
}
