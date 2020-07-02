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


def main():
    parser = argparse.ArgumentParser(description="Generate datapoints from AST")
    parser.add_argument("--ast_fp", "-a", help="Filepath with the ASTs to be parsed")
    parser.add_argument(
        "--out_fp", "-o", default="/tmp/dps.txt", help="Filepath for the output dps"
    )
    parser.add_argument(
        "--n_ctx", "-c", type=int, default=1000, help="Number of contexts for each dp"
    )
    args = parser.parse_args()
    if os.path.exists(args.out_fp):
        os.remove(args.out_fp)
    logging.info("Number of context: {}".format(args.n_ctx))

    num_dps = 0
    logging.info("Loading asts from: {}".format(args.ast_fp))
    with open(args.ast_fp, "r") as f, open(args.out_fp, "w") as fout:
        for line in file_tqdm(f):
            dp = json.loads(line.strip())
            asts = separate_dps(dp, args.n_ctx)
            for ast, extended in asts:
                if len(ast) > 1:
                    json.dump([get_dfs(ast), extended], fp=fout)
                    fout.write("\n")
                    num_dps += 1

    logging.info("Wrote {} datapoints to {}".format(num_dps, args.out_fp))


if __name__ == "__main__":
    main()
