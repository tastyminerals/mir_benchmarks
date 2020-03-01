module dataloader;

/++
A naive reimplementation of NNER dataloader.py in D for comparison and benchmarking.
It needs a dataset file generated from dooku documents in tsv format.
The file is first read into memory and afterwards converted to a dense multidimensional array.
The array is wrapped into a class that provides dataset batch iteration.
+/

import std.stdio;

/// Represents data from the training file
struct Data
{
    int[] tokens;
    double[] positionalFeatures;
    double[] addedFeatures;
    int[] targets;
    double[] pretrainedEmbeddings;
    double[] documentVectors;
}

/// Represents dataset class that is capable of batch iteration
// class Dataset
// {

// }

/++
Read tsv dataset, use standard File.byLine.
A much faster and efficient way is to implement tsv-utilities' bufferedByLine.
+/
// Data readDatasetFile(in string filePath)
// {
//     ///
// }

// T[][][] toDenseArray(T)(T[] data)
// {
//     ///
// }

/// Generates token -> index mapping from the training tsv file.
// int[string] generateVocabulary(in strinf filePath)
// {

// }

void run(string[] args)
{
    // read the dataset tsv file
    // convert it to dense multidimensional array
    // construct Dataset instance
}

unittest
{

}
