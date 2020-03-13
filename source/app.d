import mir_ops_bench;
import standard_ops_bench;
import std.stdio;

int nruns = 20;
void main()
{
    runStandardBenchmarks(nruns);
    runMirBenchmarks(nruns);
}
