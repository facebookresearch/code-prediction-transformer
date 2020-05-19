#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging

import torch
import utils


logging.basicConfig(level=logging.INFO)


UNK = "<unk_token>"
PAD = "<pad_token>"
PLACEHOLDER = "<placeholder_token>"


class Dataset(torch.utils.data.Dataset):
    def __init__(self, fp):
        super().__init__()
        self.fp = fp
        self._line_pos_dp = list(utils.line_positions(fp))

    def __len__(self):
        return len(self._line_pos_dp)

    def __getitem__(self, idx):
        line_pos = self._line_pos_dp[idx]
        with open(self.fp) as f:
            f.seek(line_pos)
            dp_line = json.loads(f.readline().strip())
        return dp_line

    @staticmethod
    def collate(batch, token_pad_idx, subtoken_pad_idx):
        def combine(seqs, max_len, max_path_len, pad_idx):
            if not seqs:
                return torch.ones((max_len, max_path_len)).long() * pad_idx
            paths = []
            for path in seqs:
                paths.append(path + [pad_idx] * (max_path_len - len(path)))
            len_pad = torch.ones((max_len - len(paths), max_path_len)).long()
            return torch.cat((torch.tensor(paths), len_pad))

        max_len = max(len(i[1]) for i in batch)
        max_start_len = max(
            max([len(start) for start in seq[1]], default=0) for seq in batch
        )
        max_path_len = max(
            max([len(path) for path in seq[2]], default=0) for seq in batch
        )
        max_end_len = max(
            max([len(start) for start in seq[3]], default=0) for seq in batch
        )
        all_targets = []
        all_starts = []
        all_paths = []
        all_ends = []

        for (target, starts, paths, ends) in batch:
            all_targets.append(target)
            starts = combine(starts, max_len, max_start_len, subtoken_pad_idx)
            paths = combine(paths, max_len, max_path_len, token_pad_idx)
            ends = combine(ends, max_len, max_end_len, subtoken_pad_idx)

            all_starts.append(starts)
            all_ends.append(ends)
            all_paths.append(paths)

        results = {
            "targets": torch.tensor(all_targets),
            "starts": torch.stack(all_starts),
            "paths": torch.stack(all_paths),
            "ends": torch.stack(all_ends),
        }
        return results
