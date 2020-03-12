import std.stdio;
import basic_ops;
import standard_ops_bench;
import mir_ops_bench;

void main()
{
    // basic_ops.props;
    // basic_ops.arrays;
    // basic_ops.mirArrays;
    // basic_ops.basicOps;
    runStandardBenchmarks;
    runMirBenchmarks;
}
