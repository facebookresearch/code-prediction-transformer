#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import pickle
import torch
from dataset.dataset import BaseDataset, BaseSetup


logging.basicConfig(level=logging.INFO)
UNK = "<unk_token>"
PAD = "<pad_token>"


class Setup(BaseSetup):
    def _add_extra_filepaths(self, base_dir):
        self.filepaths["rel_vocab"] = os.path.join(base_dir, "rel_vocab.pkl")

    def _create_vocab(self):
        return Vocab(self.filepaths["vocab"], self.filepaths["rel_vocab"])

    def _create_dataset(self, fp, ids_fp):
        return Dataset(fp, ids_fp)


class Vocab(object):
    def __init__(self, vocab_fp, rel_vocab_fp):
        super().__init__()
        self.unk_token = UNK
        self.pad_token = PAD
        self.pad_idx = None
        self.unk_idx = None

        if not os.path.exists(vocab_fp):
            raise Exception("Get the vocab from generate_vocab.py")

        # regular vocab
        with open(vocab_fp, "rb") as fin:
            self.idx2vocab = pickle.load(fin)
        logging.info("Loaded vocab from: {}".format(vocab_fp))
        self.vocab2idx = {token: i for i, token in enumerate(self.idx2vocab)}
        self.unk_idx = self.vocab2idx[self.unk_token]
        self.pad_idx = self.vocab2idx[self.pad_token]
        logging.info("Vocab size: {}".format(len(self.idx2vocab)))

        # open rel vocab
        with open(rel_vocab_fp, "rb") as fin:
            self.idx2rel = pickle.load(fin)
        logging.info("Loaded rel vocab from: {}".format(rel_vocab_fp))
        self.rel2idx = {token: i for i, token in enumerate(self.idx2rel)}
        self.rel_unk_idx = self.rel2idx[UNK]
        logging.info("Rel vocab sizes: {}".format(len(self.idx2rel)))

    def convert(self, line):
        dp, ext, rel_info = line
        dp_converted = [
            self.vocab2idx[token] if token in self.vocab2idx else self.unk_idx
            for token in dp
        ]
        rel_converted = [
            [
                self.rel2idx[token] if token in self.rel2idx else self.rel_unk_idx
                for token in rel.split()
            ]
            for rel in rel_info
        ]
        return [dp_converted, ext, rel_converted]

    def __len__(self):
        return len(self.idx2vocab)


class Dataset(BaseDataset):
    @staticmethod
    def collate(seqs, pad_idx):
        max_len = max(len(seq[0][0]) for seq in seqs)
        max_len = max(max_len, 2)
        input_seqs = []
        target_seqs = []
        extended = []
        rel_mask = torch.zeros((len(seqs), max_len - 1, max_len - 1)).long()
        ids = {name: [] for name in seqs[0][1].keys()}

        for i, ((seq, ext, mask), ids_lst) in enumerate(seqs):
            padding = [pad_idx] * (max_len - len(seq))
            input_seqs.append(seq[:-1] + padding)
            target_seqs.append(seq[1:] + padding)
            extended.append(ext)
            for j, rel in enumerate(mask):
                rel_mask[i][j][: len(rel)] = torch.tensor(rel)
            for name, lst in ids_lst.items():
                ids[name] += [j - 1 + (max_len - 1) * i for j in lst]

        return {
            "input_seq": torch.tensor(input_seqs),
            "target_seq": torch.tensor(target_seqs),
            "extended": torch.tensor(extended),
            "rel_mask": rel_mask,
            "ids": ids,
        }
