#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import pickle
import re
from collections import Counter

from utils import get_terminal_nodes, file_tqdm, tokenize


logging.basicConfig(level=logging.INFO)


UNK = "<unk_token>"
PAD = "<pad_token>"
PLACEHOLDER = "<placeholder_token>"


def get_value(line, vocab_type):
    if vocab_type == "token":
        return get_dfs(line)
    elif vocab_type == "subtoken":
        lst = []
        for node in get_terminal_nodes(line):
            lst += tokenize(node)
        return lst
    elif vocab_type == "output":
        return get_terminal_nodes(line)


def main():
    parser = argparse.ArgumentParser(
        description="Create vocab for code2seq model for py150 dataset"
    )
    parser.add_argument("--n_vocab", "-n", type=int, default=100000)
    parser.add_argument("--input_fp", "-i")
    parser.add_argument("--out_fp", "-o", default="/tmp/vocab.pkl")
    parser.add_argument(
        "--vocab_type",
        "-v",
        choices=["token", "subtoken", "output"],
        help="What type of vocab to get",
    )
    args = parser.parse_args()

    logging.info("Reading from: {}".format(args.input_fp))
    logging.info("Vocab type: {}".format(args.vocab_type))
    vocab = Counter()
    with open(args.input_fp, "r") as f:
        for line in file_tqdm(f):
            vocab.update(get_value(json.loads(line.strip()), args.vocab_type))
    vocab_to_keep = [i[0] for i in vocab.most_common(args.n_vocab)]
    top_total = sum(i[1] for i in vocab.most_common(args.n_vocab))
    total = sum(vocab.values())

    logging.info("Total # of vocab: {}".format(len(vocab)))
    logging.info(
        "Using {} top vocab covers: {:.2f}% of the entire dataset".format(
            args.n_vocab, 100 * top_total / total
        )
    )
    logging.info("Top 10 most common vocab:")
    for v, i in vocab.most_common(10):
        print(v, i)

    # add unk and pad tokens
    vocab_to_keep.append(UNK)
    vocab_to_keep.append(PAD)
    vocab_to_keep.append(PLACEHOLDER)
    logging.info("Added {} and {} and {}".format(UNK, PAD, PLACEHOLDER))

    # dump vocab to file
    with open(args.out_fp, "wb") as fout:
        pickle.dump(vocab_to_keep, fout)
    logging.info("Wrote {} vocab to: {}".format(len(vocab_to_keep), args.out_fp))


if __name__ == "__main__":
    main()