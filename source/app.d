import dataloader_bench;
import mir_mem_bench;
import mir_ops_bench;
import standard_mem_bench;
import standard_ops_bench;
import std.getopt;

void main(string[] args)
{
    int nruns = 20;
    string benchmarksId;

    auto opts = getopt(args, "n|nruns", "number of runs, the time will be averaged (default 20)", &nruns, "b|bench",
            "benchmark name to run [standard, mir, dataloader], runs all if not set",
            &benchmarksId);
    if (opts.helpWanted)
    {
        defaultGetoptPrinter(msg, opts.options);
        return;
    }

    switch (benchmarksId)
    {
    case "standard":
        // runStandardBenchmarks(nruns);
        runStandardMemoryBenchmarks(nruns);
        break;
    case "mir":
        // runMirBenchmarks(nruns);
        runMirMemoryBenchmarks(nruns);
        break;
    case "dataloader":
        runDataloaderBenchmark(nruns);
        break;
    default:
        runStandardBenchmarks(nruns);
        runStandardMemoryBenchmarks(nruns);
        runMirBenchmarks(nruns);
        runDataloaderBenchmark(nruns);
        break;
    }

}

static string msg = q"EOD
Run D benchmarks for standard D and Mir D.

Usage:
./app -n 20 -b standard
EOD";
