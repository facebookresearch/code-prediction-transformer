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


def convert(ast):
    increase_by = {}  # count of how many idx to increase the new idx by:
    # each time there is a value node
    cur = 0
    for i, node in enumerate(ast):
        increase_by[i] = cur
        if "value" in node:
            cur += 1

    new_dp = []
    for i, node in enumerate(ast):
        inc = increase_by[i]
        if "value" in node:
            child = [i + inc + 1]
            if "children" in node:
                child += [n + increase_by[n] for n in node["children"]]
            new_dp.append({"type": node["type"], "children": child})
            new_dp.append({"value": node["value"]})
        else:
            if "children" in node:
                node["children"] = [n + increase_by[n] for n in node["children"]]
            new_dp.append(node)

    # sanity check
    children = []
    for node in new_dp:
        if "children" in node:
            children += node["children"]
    assert len(children) == len(set(children))
    return new_dp


def main():
    parser = argparse.ArgumentParser(description="Generate datapoints from AST")
    parser.add_argument("--input_fp", "-i", help="Filepath with the ASTs to be parsed")
    parser.add_argument(
        "--out_fp",
        "-o",
        default="/tmp/new_trees.json",
        help="Filepath with the output dps",
    )

    args = parser.parse_args()
    if os.path.exists(args.out_fp):
        os.remove(args.out_fp)

    logging.info("Loading asts from: {}".format(args.input_fp))
    with open(args.input_fp, "r") as f, open(args.out_fp, "w") as fout:
        for line in file_tqdm(f):
            dp = json.loads(line.strip())
            print(json.dumps(convert(dp)), file=fout)
    logging.info("Wrote dps to: {}".format(args.out_fp))


if __name__ == "__main__":
    main()
