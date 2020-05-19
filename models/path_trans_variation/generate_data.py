#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import os

from utils import file_tqdm, get_dfs, separate_dps


logging.basicConfig(level=logging.INFO)


def get_leaf_info(ast):
    leaf_tokens = []
    leaf_ids = []
    for i, node in enumerate(ast):
        if "value" in node:
            leaf_ids.append(i)
            leaf_tokens.append(node["value"])
    return leaf_tokens, leaf_ids


def get_dps(ast, max_len):
    leaf_tokens, leaf_ids = get_leaf_info(ast)
    if len(leaf_tokens) <= max_len:
        return [[leaf_tokens, 0]], [leaf_ids]

    half_len = int(max_len / 2)
    aug_tokens = [[leaf_tokens[:max_len], 0]]
    aug_leaf_ids = [leaf_ids[:max_len]]
    i = half_len
    while i < len(leaf_tokens) - max_len:
        aug_tokens.append([leaf_tokens[i : i + max_len], half_len])
        aug_leaf_ids.append(leaf_ids[i : i + max_len])
        i += half_len
    idx = max_len - (len(leaf_tokens) - (i + half_len))
    aug_tokens.append([leaf_tokens[-max_len:], idx])
    aug_leaf_ids.append(leaf_ids[-max_len:])
    return aug_tokens, aug_leaf_ids


def main():
    parser = argparse.ArgumentParser(description="Generate datapoints from AST")
    parser.add_argument("--ast_fp", "-a", help="Filepath with the ASTs to be parsed")
    parser.add_argument(
        "--n_ctx", "-c", type=int, default=1000, help="Number of contexts for each dp"
    )
    parser.add_argument(
        "--out_fp", "-o", default="/tmp/dps.txt", help="Filepath with the output dps"
    )

    args = parser.parse_args()
    if os.path.exists(args.out_fp):
        os.remove(args.out_fp)
    logging.info("Writing dps to: {}".format(args.out_fp))

    num_dps = 0
    with open(args.ast_fp, "r") as f, open(args.out_fp, "w") as fout:
        for line in file_tqdm(f):
            dp = json.loads(line.strip())
            aug_tokens, aug_leaf_ids = get_dps(dp, args.n_ctx)
            for (tokens, ext), leaf in zip(aug_tokens, aug_leaf_ids):
                if len(tokens) > 1:
                    json.dump([tokens, ext], fp=fout)
                    fout.write("\n")
                    num_dps += 1

    logging.info("Wrote {} datapoints to {}".format(num_dps, args.out_fp))


if __name__ == "__main__":
    main()
