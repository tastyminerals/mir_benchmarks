import csv
import io
import math
import pickle
from collections import Counter
from time import perf_counter as timer

import numpy as np
import pandas as pd


class Dataset:
    """
    This class serves as a wrapper around your training data.
    It preprocesses textual input into numpy arrays ready for iteration
    and further feeding into the neural network using predefined batch size
    and sequence length.
    """

    def __init__(self, file_path):
        self.fpath = file_path
        self.vocab = None
        self.data = []  # a list of extracted tokens, features and targets
        self.inputs = []  # ndarrays preprocessed for batch iteration
        self.batch_size = 32
        self.seq_length = 25
        self.input_size = 100
        self.nposfeats = 2
        self.naddfeats = 2
        self.onehot_arity = 2

    def load_input_data(self):
        self._read_input_data()
        self._transform_input_data()
        self.reset()

    def _create_vocab(self):
        with open(self.fpath, "r", encoding="utf-8") as fobj:
            reader = csv.reader(
                fobj, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"
            )
            tokens = [row[1] for row in reader]

        counter = Counter(tokens)
        sorted_tokens, _ = zip(
            *sorted(counter.items(), key=lambda x: x[1], reverse=True)
        )
        self.vocab = dict(zip(sorted_tokens, range(0, len(sorted_tokens))))

    def _read_input_data(self):
        def _is_oov(token):
            if self.vocab.get(token) is None:
                return True
            return False

        tokens = []  # tokens that were masked during feature generation
        pos_feats = []  # positional word features
        added_feats = []  # specific word binary features
        targets = []  # target word labels, (0, 1) for pay1

        # Loading training set as pandas dataframe: more efficient RAM usage
        # The first column is doc_id and it is not used
        df = pd.read_csv(
            self.fpath,
            header=None,
            sep="\t",
            quoting=csv.QUOTE_NONE,
            usecols=[0, 1, 2, 3, 4, 5, 6, 7],
        )
        df.columns = ["doc_id", "word", "orig", "l", "t", "isupper", "repeated", "gt"]
        size = len(df)
        # Transformation pandas dataframe into dictionary of lists with RAM cleaning
        dict_of_list = {}
        columns = df.columns
        for column in columns:
            dict_of_list[column] = df[column].tolist()
            del df[column]
        # Iterating dictionary of lists - this is much faster than to iterate pandas dataframe
        for i in range(size):
            doc_id = str(dict_of_list["doc_id"][i])
            token = dict_of_list["word"][i]
            tokens.append(token)
            pos_feats.append([dict_of_list["l"][i], dict_of_list["t"][i]])
            for_adding = [
                dict_of_list["isupper"][i],
                dict_of_list["repeated"][i],
            ]
            added_feats.append(for_adding)
            if dict_of_list["gt"][i] == 0:
                target = [1, 0]
            else:
                target = [0, 1]

            targets.append(target)

        self._create_vocab()

        # map vocab to tokens
        tokens = np.array(
            list(
                map(
                    lambda x: self.vocab["<unk>"] if _is_oov(x) else self.vocab[x],
                    tokens,
                )
            ),
            dtype=np.int32,
        )

        pos_feats = np.array(pos_feats, dtype=np.float32)
        added_feats = np.array(added_feats, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)

        self.data = {
            "tokens": tokens,
            "pos_feats": pos_feats,
            "added_feats": added_feats,
            "targets": targets,
        }

    def _transform_input_data(self):
        tokens = self.data["tokens"]
        pos_feats = self.data["pos_feats"]
        added_feats = self.data["added_feats"]
        targets = self.data["targets"]

        # -1 prevents feeding batch of shorter seq_length
        self.num_batches = (
            math.ceil(tokens.size / (self.batch_size * self.seq_length)) - 1
        )

        # print an error message when the data array is too small
        if self.num_batches == 0:
            raise Exception(
                "ERROR: Cannot create batches ==> data size={}, batch size={}, segment size={}".format(
                    tokens.size, self.batch_size, self.seq_length
                )
            )

        # here we loose some data due to floor
        itersize = math.floor(tokens.size / self.batch_size)
        max_size = self.batch_size * itersize

        # take only the amount of data sliceable into [seq_length x batch_size]
        _tokens = tokens[:max_size]
        # reshape and transpose in order to create a sliceable (batch_size) dim
        _tokens = np.reshape(_tokens, [self.batch_size, itersize])
        _tokens = np.transpose(_tokens, [1, 0])  # [83448 x 4]
        # make sure the data is contiguous
        tokens = np.ascontiguousarray(_tokens)

        _pos_feats = pos_feats[:max_size]
        _pos_feats = np.reshape(_pos_feats, [self.batch_size, itersize, self.nposfeats])
        _pos_feats = np.transpose(_pos_feats, [1, 0, 2])  # [83448 x 4 x 50]
        pos_feats = np.ascontiguousarray(_pos_feats)

        _added_feats = added_feats[:max_size]
        _added_feats = np.reshape(
            _added_feats, [self.batch_size, itersize, self.naddfeats]
        )
        _added_feats = np.transpose(_added_feats, [1, 0, 2])  # [83448 x 4 x 50]
        added_feats = np.ascontiguousarray(_added_feats)

        _targets = targets[:max_size]
        _targets = np.reshape(_targets, [self.batch_size, itersize, self.onehot_arity])
        _targets = np.transpose(_targets, [1, 0, 2])  # [83448 x 4]
        targets = np.ascontiguousarray(_targets)

        self.inputs = {
            "tokens": tokens,
            "pos_feats": pos_feats,
            "added_feats": added_feats,
            "targets": targets,
        }

    def reset(self):
        self.start = 0
        self.end = self.seq_length

    def next_batch(self):
        tokens = self.inputs["tokens"]
        pos_feats = self.inputs["pos_feats"]
        added_feats = self.inputs["added_feats"]
        targets = self.inputs["targets"]

        w = tokens[self.start : self.end]
        f1 = pos_feats[self.start : self.end]
        f2 = added_feats[self.start : self.end]
        t = targets[self.start : self.end]

        self.start += self.seq_length
        self.end += self.seq_length

        return w, f1, f2, t


if __name__ == "__main__":
    timings = []
    for _ in range(20):
        start = timer()
        dataset = Dataset("test.tsv")
        dataset.load_input_data()
        batch = dataset.next_batch()
        end = timer()
        timings.append(end - start)

    print("| {} | {} |".format("dataloader", sum(timings) / len(timings)))

