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

from utils import get_dfs, parallelize, separate_dps


logging.basicConfig(level=logging.INFO)


def separate_rel_mask(rel_mask, max_len):
    """
    Separate the mask by a sliding window to keep each dp at length max_len.
    For the masks, for each row, since we want the information to be relative
    to whatever is being predicted (ie. input_seq[i+1]), we are shifting
    everything by 1. Thus, the length of each mask will be len(seq) - 1.
    """
    if len(rel_mask) <= max_len:
        return [[" ".join(lst.split()[:-1]) for lst in rel_mask[1:]]]

    half_len = int(max_len / 2)
    rel_mask_aug = [[" ".join(lst.split()[:-1]) for lst in rel_mask[1:max_len]]]

    i = half_len
    while i < len(rel_mask) - max_len:
        rel_mask_aug.append(
            [" ".join(lst.split()[i:-1]) for lst in rel_mask[i + 1 : i + max_len]]
        )
        i += half_len
    rel_mask_aug.append(
        [
            " ".join(lst.split()[-(i + 2) : -1])
            for i, lst in enumerate(rel_mask[-max_len + 1 :])
        ]
    )
    return rel_mask_aug


def get_ud_masks(dp, max_len):
    def get_ancestors(dp):
        ancestors = {0: []}
        node2parent = {0: 0}
        levels = {0: 0}
        for i, node in enumerate(dp):
            if "children" in node:
                cur_level = levels[i]
                for child in node["children"]:
                    node2parent[child] = i
                    levels[child] = cur_level + 1
            ancestors[i] = [i] + ancestors[node2parent[i]]
        return ancestors, levels

    def get_path(i, j):
        if i == j:
            return "<self>"
        if i - j >= max_len:
            return "0"
        anc_i = set(ancestors[i])
        for node in ancestors[j][-(levels[i] + 1) :]:
            if node in anc_i:
                up_n = levels[i] - levels[node]
                down_n = levels[j] - levels[node]
                return str(up_n + 0.001 * down_n)

    ancestors, levels = get_ancestors(dp)
    path_rels = []
    for i in range(len(dp)):
        path_rels.append(" ".join([get_path(i, j) for j in range(i + 1)]))
    return path_rels


def get_udc_masks(dp, max_len):
    def get_ancestors(dp):
        ancestors = {0: []}
        node2parent = {0: 0}
        node2i = {0: "0"}
        levels = {0: 0}
        for i, node in enumerate(dp):
            if "children" in node:
                children = node["children"]
                for child in children:
                    node2parent[child] = i
                    node2i[child] = "1"
                    levels[child] = levels[i] + 1
                node2i[children[-1]] = "2"
                node2i[children[0]] = "0"
            ancestors[i] = [i] + ancestors[node2parent[i]]
        return ancestors, levels, node2i

    def get_path(i, j):
        if i == j:
            return "<self>"
        if i - j >= max_len:
            return "0"
        anc_i = set(ancestors[i])
        down = ""
        for node in ancestors[j]:
            if node in anc_i:
                up = str(levels[i] - levels[node])
                return "{}-{}".format(up, down)
            down += node2i[node]

    ancestors, levels, node2i = get_ancestors(dp)
    path_rels = []
    for i in range(len(dp)):
        path_rels.append(" ".join([get_path(i, j) for j in range(i + 1)]))
    return path_rels


def get_dp(dp, n_ctx, child=False):
    get_mask = get_udc_masks if child else get_ud_masks
    asts = separate_dps(dp, n_ctx)
    rel_masks = separate_rel_mask(get_mask(dp, n_ctx), n_ctx)
    aug_dps = []
    for (ast, ext), mask in zip(asts, rel_masks):
        aug_dps.append([get_dfs(ast), ext, mask])
    return aug_dps


def main():
    parser = argparse.ArgumentParser(description="Generate datapoints from AST")
    parser.add_argument("--ast_fp", "-a", help="Filepath with the ASTs to be parsed")
    parser.add_argument(
        "--out_fp", "-o", default="/tmp/dps.txt", help="Filepath for the output dps"
    )
    parser.add_argument(
        "--n_ctx", "-c", type=int, default=1000, help="Number of contexts for each dp"
    )
    parser.add_argument(
        "--child",
        action="store_true",
        help="Use flag to incorporate child (0,1,2) info",
    )
    args = parser.parse_args()
    if os.path.exists(args.out_fp):
        os.remove(args.out_fp)
    logging.info("Number of context: {}".format(args.n_ctx))

    data = []
    num_dps = 0
    i = 0
    with open(args.ast_fp, "r") as f, open(args.out_fp, "w") as fout:
        for _ in range(5):  # divide up into subparts
            i += 1
            print("Starting {}".format(i))
            for _ in range(1000):
                dp = json.loads(f.readline().strip())
                if len(dp) <= 1:
                    continue
                data.append(dp)
            print("  > Finished reading: {}".format(len(data)))
            dps = parallelize(data, get_dp, (args.n_ctx, args.child), n_cores=60)
            print("  > Finished getting the datasets")
            for dp in dps:
                for seq, extended, mask in dp:
                    if len(seq) > 1:
                        json.dump([seq, extended, mask], fout)
                        fout.write("\n")
                        num_dps += 1
            data = []
            print("  > Finished writing to file")
    logging.info("Wrote {} datapoints to {}".format(num_dps, args.out_fp))


if __name__ == "__main__":
    main()
