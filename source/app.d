import std.stdio;
import basic_ops;
import standard_ops_bench;
import mir_ops_bench;

int nruns = 20;
void main()
{
    runStandardBenchmarks(nruns);
    runMirBenchmarks(nruns);
}
