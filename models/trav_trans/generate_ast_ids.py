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

from utils import file_tqdm, separate_dps


logging.basicConfig(level=logging.INFO)


def get_leaf_ids(ast):
    ids = {"leaf_ids": [], "internal_ids": []}
    for i, node in enumerate(ast):
        if "value" in node:
            ids["leaf_ids"].append(i)
        else:
            ids["internal_ids"].append(i)
    return ids


def get_value_ids(ast):
    ids = {"attr_ids": [], "num_ids": [], "name_ids": [], "param_ids": []}
    for i, node in enumerate(ast):
        if "type" in node:
            if node["type"] == "attr":
                ids["attr_ids"].append(
                    i + 1
                )  # + 1 since i is the type, and we want the value
            elif node["type"] == "Num":
                ids["num_ids"].append(i + 1)
            elif node["type"] in {"NameLoad", "NameStore"}:
                ids["name_ids"].append(i + 1)
            elif node["type"] == "NameParam":
                ids["param_ids"].append(i + 1)
    return ids


def get_type_ids(ast):
    ids = {
        "call_ids": [],
        "assign_ids": [],
        "return_ids": [],
        "list_ids": [],
        "dict_ids": [],
        "raise_ids": [],
    }
    for i, node in enumerate(ast):
        if "type" in node:
            type_ = node["type"]
            if type_ == "Call":
                ids["call_ids"].append(i)
            elif type_ == "Assign":
                ids["assign_ids"].append(i)
            elif type_ == "Return":
                ids["return_ids"].append(i)
            elif type_ in {"ListComp", "ListLoad", "ListStore"}:
                ids["list_ids"].append(i)
            elif type_ in {"DictComp", "DictLoad", "DictStore"}:
                ids["dict_ids"].append(i)
            elif type_ == "Raise":
                ids["raise_ids"].append(i)
    return ids


def main():
    parser = argparse.ArgumentParser(
        description="Generate ids (leaf, values, types) from AST"
    )
    parser.add_argument(
        "--ast_fp", "-a", help="Filepath with the new ASTs to be parsed"
    )
    parser.add_argument(
        "--out_fp", "-o", default="/tmp/ids.txt", help="Filepath for the output ids"
    )
    parser.add_argument(
        "--n_ctx", "-c", type=int, default=1000, help="Number of contexts for each dp"
    )
    parser.add_argument(
        "id_type",
        choices=["leaf", "value", "type", "all"],
        default="leaf",
        help="Which ids to generate. Default = leaf",
    )

    args = parser.parse_args()
    if os.path.exists(args.out_fp):
        os.remove(args.out_fp)
    logging.info("Type of id to get: {}".format(args.id_type))

    logging.info("Loading dps from: {}".format(args.ast_fp))
    with open(args.ast_fp, "r") as f, open(args.out_fp, "w") as fout:
        for line in file_tqdm(f):
            dp = json.loads(line.strip())
            asts = separate_dps(dp, args.n_ctx)
            for ast, _ in asts:
                ids = {}
                if len(ast) > 1:
                    if args.id_type in {"leaf", "all"}:
                        ids.update(get_leaf_ids(ast))
                    if args.id_type in {"value", "all"}:
                        ids.update(get_value_ids(ast))
                    if args.id_type in {"type", "all"}:
                        ids.update(get_type_ids(ast))

                    json.dump(ids, fp=fout) 
                    fout.write("\n")
    logging.info("Wrote to: {}".format(args.out_fp))


if __name__ == "__main__":
    main()
