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
    int[] tokenIdx;
    double[] positionalFeatures;
    int[] addedFeatures;
    int[] targets;

}

struct Dataset(T, U)
{
    T[] tokenTensor;
    U[] posFeatsTensor;
    T[] addedFeatsTensor;
    T[] targetsTensor;
}

void runDataloaderBenchmark(int nruns)
{
    import std.stdio;
    import std.string;
    import std.conv : to;
    import std.math : ceil, floor;

    string fileName = "test.tsv";
    // generate vocab
    auto file0 = File(fileName, "r");
    int[string] vocab;
    int idx = 0;
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
    double left, top;
    int isUpper, repeated;
    int[2] label;
    foreach (line; file1.byLineCopy)
    {
        lineArr = line.split(Delim);

        token = lineArr[1];
        left = to!double(lineArr[3]);
        top = to!double(lineArr[4]);
        isUpper = to!int(lineArr[5]);
        repeated = to!int(lineArr[6]);
        label = lineArr[7] == "0" ? [1, 0] : [0, 1];

        data.tokenIdx ~= vocab[token];
        data.positionalFeatures ~= [left, top];
        data.addedFeatures ~= [isUpper, repeated];
        data.targets ~= label;

    }

    // convert collected arrays into sliceable tensors
    const int batchNum = ((data.tokenIdx.length / (
            Hyperparams.batchSize.to!double * Hyperparams.seqlen)).ceil - 1.0).to!int;
    assert(batchNum > 0);

    const int interSize = (data.tokenIdx.length / Hyperparams.batchSize.to!double).floor.to!int;
    const int allocElems = (data.tokenIdx.length - (Hyperparams.batchSize.to!double * interSize))
        .to!int;
    int[] tokenIdxSlice;
    tokenIdxSlice.reserve(allocElems);
    tokenIdxSlice = data.tokenIdx[0 .. Hyperparams.batchSize * interSize];

    // # take only the amount of data sliceable into [seq_length x batch_size]
    // _tokens = tokens[: self.batch_size * itersize]
    // # reshape and transpose in order to create a sliceable (batch_size) dim
    // _tokens = np.reshape(_tokens, [self.batch_size, itersize])
    // _tokens = np.transpose(_tokens, [1, 0])  # [83448 x 4]
    // # make sure the data is contiguous
    // tokens = np.ascontiguousarray(_tokens)
}

unittest
{

}
