module mir_mem_bench;

/*
Memory allocation benchmarks for standard D:

    - allocation, writing and deallocation of a huge array
    - allocation, writing and deallocation of several big arrays of different sizes
    - reallocation of one array into three arrays by indexing and slicing

*/

import core.memory : GC;
import mir.math.common : fastmath;
import mir.math.sum;
import mir.ndslice;
import mir.random.algorithm : randomSlice;
import mir.random.variable : uniformVar;
import std.array : join;
import std.datetime.stopwatch : AutoStart, StopWatch;
import std.format;
import std.math : pow, sqrt;
import std.random : randomShuffle, uniform, unpredictableSeed, Xorshift;
import std.stdio;

alias SliceArr = Slice!(double*)[];

long[][string] functions(in int nruns)
{
    auto sw = StopWatch(AutoStart.no);
    const size = 30_000_000;

    long[][string] funcs;
    string name0 = format("Allocation, writing and deallocation of a [%s] array ", size);
    for (int i; i < nruns; ++i)
    {
        sw.reset;
        sw.start;
        for (int j; j < 5; ++j)
        {
            // allocate
            auto arr = uniformVar!double(0, 1.0).randomSlice(size);
            // write
            arr[j] = 100.0;
            // destroy
            destroy(arr);
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
            auto arr = uniformVar!double(0, 1.0).randomSlice(size);
            // write
            arr[mod] = 100.0;
            // destroy
            destroy(arr);
            GC.collect;
        }
        sw.stop;
        funcs[name1] ~= sw.peek.total!"nsecs";
    }

    const int size1 = size / 1000;
    const size2 = iota([size1 + 1], 1).sum;
    string name2 = format("Slicing [%s] array into another array (%s loops)", size1, size1);
    auto arr = uniformVar!double(0, 1.0).randomSlice(size1);
    for (int i; i < nruns; ++i)
    {
        sw.reset;
        sw.start;
        SliceArr a;
        for (int j; j < size1; ++j)
        {
            a ~= arr[j .. $];
        }
        a.join;
        sw.stop;
        funcs[name2] ~= sw.peek.total!"nsecs";
        destroy(a);
        GC.collect;
    }

    return funcs;
}

void runMirMemoryBenchmarks(int nruns)
{
    pragma(inline, false);
    writeln("---[Mir Memory D]---");
    auto timings = functions(nruns);
    import mir.series; // for sorted output
    foreach (pair; timings.series)
    {
        // convert nsec. to sec. and compute the average
        const double secs = pair.value.sum * 10.0.pow(-9) / pair.value.length;
        writeln(format("| %s | %s |", pair.key, secs));
    }
}
