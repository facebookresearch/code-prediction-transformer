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

from utils import file_tqdm


logging.basicConfig(level=logging.INFO)


def get_leaf_info(ast):
    leaf_tokens = []
    leaf_ids = []
    for i, node in enumerate(ast):
        if "value" in node:
            leaf_ids.append(i)
            leaf_tokens.append(node["value"])
    return leaf_tokens, leaf_ids


def get_ancestors(ast):
    ancestors = {0: []}
    node2parent = {0: 0}
    for i, node in enumerate(ast):
        if "children" in node:
            for child in node["children"]:
                node2parent[child] = i
        token = node["value"] if "value" in node else node["type"]
        ancestors[i] = [token] + ancestors[node2parent[i]]
    return ancestors


def get_root_paths(ancestors, leaf_ids, max_path_len):
    return [ancestors[i][1 :max_path_len + 1] for i in leaf_ids]


def get_dps(ast, max_len, max_path_len):
    leaf_tokens, leaf_ids = get_leaf_info(ast)
    ancestors = get_ancestors(ast)
    if len(leaf_tokens) <= max_len:
        return [[leaf_tokens, 0, get_root_paths(ancestors, leaf_ids, max_path_len)]]

    half_len = int(max_len / 2)
    aug_dps = [
        [
            leaf_tokens[:max_len],
            0,
            get_root_paths(ancestors, leaf_ids[:max_len], max_path_len),
        ]
    ]
    i = half_len
    while i < len(leaf_tokens) - max_len:
        aug_dps.append(
            [
                leaf_tokens[i : i + max_len],
                half_len,
                get_root_paths(ancestors, leaf_ids[i : i + max_len], max_path_len),
            ]
        )
        i += half_len
    idx = max_len - (len(leaf_tokens) - (i + half_len))
    aug_dps.append(
        [
            leaf_tokens[-max_len:],
            idx,
            get_root_paths(ancestors, leaf_ids[-max_len:], max_path_len),
        ]
    )
    return aug_dps


def main():
    parser = argparse.ArgumentParser(description="Generate datapoints from AST")
    parser.add_argument("--ast_fp", "-a", help="Filepath with the ASTs to be parsed")
    parser.add_argument(
        "--out_fp", "-o", default="/tmp/dps.txt", help="Filepath with the output dps"
    )
    parser.add_argument(
        "--n_ctx", "-c", type=int, default=1000, help="Number of contexts for each dp"
    )
    parser.add_argument(
        "--max_path_len",
        "-p",
        type=int,
        default=13,
        help="Max length of rootpath route",
    )

    args = parser.parse_args()
    if os.path.exists(args.out_fp):
        os.remove(args.out_fp)
    logging.info("Writing dps to: {}".format(args.out_fp))

    num_dps = 0
    with open(args.ast_fp, "r") as f, open(args.out_fp, "w") as fout:
        for line in file_tqdm(f):
            dp = json.loads(line.strip())
            for dp in get_dps(dp, args.n_ctx, args.max_path_len):
                if len(dp[0]) > 1:
                    json.dump(dp, fout)
                    fout.write("\n")
                    num_dps += 1

    logging.info("Wrote {} datapoints to {}".format(num_dps, args.out_fp))


if __name__ == "__main__":
    main()
