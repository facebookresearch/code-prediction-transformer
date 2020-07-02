#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from dataset.dataset import BaseDataset, BaseSetup, BaseVocab


class Setup(BaseSetup):
    def _create_vocab(self):
        return Vocab(self.filepaths["vocab"])

    def _create_dataset(self, fp, ids_fp):
        return Dataset(fp, ids_fp)


class Vocab(BaseVocab):
    def convert(self, line):
        dp, ext, root_paths = line
        dp_conv = [
            self.vocab2idx[token] if token in self.vocab2idx else self.unk_idx
            for token in dp
        ]
        root_paths_conv = [
            [
                self.vocab2idx[token] if token in self.vocab2idx else self.unk_idx
                for token in path
            ]
            for path in root_paths
        ]
        return [dp_conv, ext, root_paths_conv]


class Dataset(BaseDataset):
    @staticmethod
    def collate(seqs, pad_idx, bos_idx=None):
        def combine_root_paths(root_paths, max_len, max_path_len):
            paths = []
            for path in root_paths:
                paths.append(path + [pad_idx] * (max_path_len - len(path)))
            len_pad = torch.ones((max_len - len(paths), max_path_len)).long()
            return torch.cat((torch.tensor(paths), len_pad))

        max_len = max(len(seq[0][0]) for seq in seqs)
        max_len = max(2, max_len)
        max_path_len = max(max(len(path) for path in seq[0][2]) for seq in seqs)
        max_path_len - max(2, max_path_len)
        input_seqs = []
        target_seqs = []
        extended = []
        root_path_seqs = []
        ids = {name: [] for name in seqs[0][1].keys()}

        for i, ((seq, ext, root_paths), ids_lst) in enumerate(seqs):
            padding = [pad_idx] * (max_len - len(seq))
            input_seqs.append(seq[:-1] + padding)
            target_seqs.append(seq[1:] + padding)
            extended.append(ext)
            root_path_seqs.append(combine_root_paths(root_paths, max_len, max_path_len))
            for name, lst in ids_lst.items():
                ids[name] += [j - 1 + (max_len - 1) * i for j in lst]

        return {
            "input_seq": torch.tensor(input_seqs),
            "target_seq": torch.tensor(target_seqs),
            "extended": torch.tensor(extended),
            "root_paths": torch.stack(root_path_seqs)[:, 1:, :],
            "ids": ids,
        }
