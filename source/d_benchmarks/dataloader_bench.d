module dataloader_bench;

/++
A naive reimplementation of NNER dataloader.py in D for comparison and benchmarking against Numpy version.

MOTIVATION: Some time ago for NER task we implemented BiLSTM + CRF binary token (word) classifier in Tensorflow.
In order to train it, we created a Dataloader class that was used to preprocess and feed the training data.
Dataloader was reading the data from tsv training file, preprocessing and converting to 3D numpy arrays.
Once the model was initialized it was using the class to iterate over the training dataset in mini-batches.
For example:

    data = Dataloader(filepath=train.tsv)
    data.load_input_data()
    for step in range(data.num_batches):
        words, pos_feats, added_feats, targets = data.next_batch()

Where "words" is a 3D tensor of [batch_size x sequence_length x embedding_dim], e.g. [32 x 25 x 100].
"pos_feats" and "added_feats" are another 3D tensors of shape [batch_size x sequence_length x features], e.g. [32 x 25 x 2].
Finally, "targets" is a 3D tensor of shape [batch_size x sequence_length x target_bin_value], e.g. [32 x 25 x 2].

We would like to reimplement the above functionality in D and compare both the memory footprint and execution speed.

REQUIREMENTS: Dataloader needs a dataset file generated from real documents in tsv format.
+/

import mir.ndslice;
import std.typecons : Tuple, tuple;
import std.stdio;

enum Delim = '\t';
enum Hyperparams
{
    batchSize = 32,
    seqlen = 25,
    inputSize = 100
}

/// Represents data from the training file
struct Data
{
    double[] tokenIdx;
    double[] positionalFeatures;
    double[] addedFeatures;
    double[] targets;

}

alias DoubleTensor = Slice!(double*, 3LU, cast(mir_slice_kind) 0);

struct Dataset
{

    int start;
    int end;
    int seqLength;
    Data data;
    DoubleTensor tokenTensor;
    DoubleTensor posFeatsTensor;
    DoubleTensor addedFeatsTensor;
    DoubleTensor targetsTensor;

    this(in int batchSize, in int seqLength, in int inputSize, Data data)
    {
        import std.math : ceil, floor;

        this.end = seqLength;
        this.seqLength = seqLength;

        // convert collected arrays into sliceable tensors
        int err;
        const ulong batchNum = cast(ulong)(
                (data.tokenIdx.length / (cast(float) batchSize * seqLength)).ceil - 1.0);
        assert(batchNum > 0);

        const ulong iterSize = data.tokenIdx.length / batchSize;

        const ulong maxSizeTokens = batchSize * iterSize;
        const ulong maxSize = batchSize * iterSize * 2;
        this.tokenTensor = data.tokenIdx[0 .. maxSizeTokens].sliced(batchSize,
                iterSize, 1).transposed(1, 0, 2);

        this.posFeatsTensor = data.positionalFeatures[0 .. maxSize].sliced(batchSize,
                iterSize, 2).transposed(1, 0, 2);

        this.addedFeatsTensor = data.addedFeatures[0 .. maxSize].sliced(batchSize,
                iterSize, 2).transposed(1, 0, 2);

        this.targetsTensor = data.targets[0 .. maxSize].sliced(batchSize,
                iterSize, 2).transposed(1, 0, 2);

    }

    Tuple!(DoubleTensor, DoubleTensor, DoubleTensor, DoubleTensor) next_batch()
    {
        // for some reason named arguments do not work with mir Slices (bug?)
        auto miniBatch = tuple!("tokensBatch", "posFeatsBatch", "addedFeatsBatch", "targetsBatch")(
                this.tokenTensor[this.start .. this.end],
                this.posFeatsTensor[this.start .. this.end],
                this.addedFeatsTensor[this.start .. this.end],
                this.targetsTensor[this.start .. this.end]);
        this.start += this.seqLength;
        this.end += this.seqLength;
        return miniBatch;
    }

}

void initializeDataloader()
{
    import std.string;
    import std.conv : to;

    string fileName = "test.tsv";

    // generate vocab
    auto file0 = File(fileName, "r");
    int[string] vocab;
    int idx = 1;
    string[8] lineForVocab;
    foreach (line; file0.byLineCopy)
    {
        lineForVocab = line.split(Delim);
        if (lineForVocab[1]!in vocab)
        {
            vocab[lineForVocab[1]] = idx;
            ++idx;
        }
    }

    // read the dataset tsv file, again (we follow Python implementation and repeat some tasks).
    auto file1 = File(fileName, "r");
    Data data;
    string[8] lineArr;
    string token;
    double left, top, isUpper, repeated;
    double[2] label;
    foreach (line; file1.byLineCopy)
    {
        lineArr = line.split(Delim);

        token = lineArr[1];
        left = to!double(lineArr[3]);
        top = to!double(lineArr[4]);
        isUpper = to!double(lineArr[5]);
        repeated = to!double(lineArr[6]);
        label = lineArr[7] == "0" ? [1.0, 0] : [0, 1.0];

        auto exists = (token in vocab);
        data.tokenIdx ~= exists ? vocab[token] : 0;
        data.positionalFeatures ~= [left, top];
        data.addedFeatures ~= [isUpper, repeated];
        data.targets ~= label;
    }

    auto dataset = Dataset(Hyperparams.batchSize, Hyperparams.seqlen, Hyperparams.inputSize, data);
    auto miniBatch = dataset.next_batch;

}

void runDataloaderBenchmark(int nruns)
{
    pragma(inline, false);

    import std.format;

    import std.datetime.stopwatch : AutoStart, StopWatch;
    import std.math : pow;
    import std.algorithm : sum;

    writeln("---[Dataloader]---");

    auto sw = StopWatch(AutoStart.no);
    long[][string] timings;
    foreach (i; 0 .. nruns)
    {
        sw.reset;
        sw.start;
        initializeDataloader;
        sw.stop;
        timings["dataloader"] ~= sw.peek.total!"nsecs";
    }

    import mir.series; // for sorted output

    foreach (pair; timings.series)
    {
        // convert nsec. to sec. and compute the average
        const double secs = pair.value.sum * 10.0.pow(-9) / pair.value.length;
        writeln(format("| %s | %s |", pair.key, secs));
    }
}

unittest
{

}
